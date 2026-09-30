"""Teacher-forced instantaneous observation noise versus saved accumulated state noise."""
import argparse,time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import softmax
from threadpoolctl import threadpool_limits
from flying.training import whole_brain_memory as core
from alphabet_memory import read,check
from pathway_memory import seal
from context_memory import estimate
from frozen_state_probe import Budget
from relative_noise import read_table
import act5_robustness as old


def config():return read('configs/observation_noise.json')


def context(c):
    x=core.source_context(c)
    names=['observation_noise','verify_observation_noise','relative_noise','act5_robustness',
           'alphabet_memory','pathway_memory','context_memory','frozen_state_probe','verify_neuron_panel']
    x['adapter_sha256']={f'scripts/{n}.py':core.sha256(f'scripts/{n}.py') for n in names}
    return x


def noise_bank(ids,observed,seed,circuit,horizon,noise_seed):
    ix=pd.Index(ids).get_indexer(observed);assert (ix>=0).all() and len(np.unique(ix))==len(ix)
    rng=np.random.default_rng(np.random.SeedSequence([noise_seed,seed,circuit,1]))
    return np.stack([rng.standard_normal(len(ids))[ix] for _ in range(horizon)])


def instantaneous(features,noise,sigma):
    if features.shape!=noise.shape or sigma<0 or not np.isfinite(sigma) or not np.isfinite(features).all() or not np.isfinite(noise).all():
        raise ValueError('Invalid observation perturbation')
    return np.clip(features+sigma*noise,-1,1)


def history_gate(delta,co,c):
    n=4 if co=='discovery' else 3;d=np.asarray(delta)
    return bool(d.mean()>=c['history_cost_threshold'] and (d>0).sum()>=n)


def summarize(rows,c,out):
    f=pd.DataFrame(rows);f.to_csv(out/'raw-metrics.csv',index=False)
    metrics=['clean_accuracy','instantaneous_accuracy','full_accuracy','history_accuracy_cost',
             'instantaneous_accuracy_cost','full_accuracy_cost','history_feature_mse']
    b=f.groupby(['cohort','seed','level'])[metrics].mean().reset_index();b.to_csv(out/'seed-blocks.csv',index=False)
    d=f.groupby(['cohort','seed','level','strength'])[metrics].mean().reset_index();d.to_csv(out/'dose-seed-table.csv',index=False)
    stats={};gates={};near={}
    for co in c['cohorts']:
        for g in c['conditions']:
            z=b[(b.cohort==co)&(b.level==g)];key=f'{co}/{g}'
            stats[key]={k:estimate(z[k],c) for k in metrics}
            gates[key]=history_gate(z.history_accuracy_cost,co,c)
            near[key]=bool(abs(z.history_accuracy_cost.mean())<=c['near_mean_tolerance'] and (abs(z.history_accuracy_cost)<=c['near_seed_tolerance']).all())
    confirmed=[g for g in c['conditions'] if all(gates[f'{co}/{g}'] for co in c['cohorts'])]
    near_confirmed=[g for g in c['conditions'] if all(near[f'{co}/{g}'] for co in c['cohorts'])]
    result=dict(statistics=stats,history_cost_gates=gates,near_tolerance_gates=near,
                confirmed_history_cost=confirmed,within_registered_near_tolerance=near_confirmed,
                primary_confirmed='brain1' in confirmed,autonomous_recall_tested=False,cohorts_pooled=False)
    core.write_json(out/'summary.json',result);return result


@threadpool_limits.wrap(limits=1)
def run(c,out):
    started=time.monotonic();out.mkdir(parents=True,exist_ok=False);source=Path(c['source'])
    assert core.sha256(source/'manifest.json')==c['source_manifest_sha256'];parent=check(source)
    check(Path(c['source_validation']));validated=read(Path(c['source_validation'])/'checks.json')
    assert validated['all_checks_pass'] and validated['result_manifest_sha256']==c['source_manifest_sha256']
    assert not read(source/'verification.json')['smoke']
    for k in ['cohorts','conditions','circuit_seeds','relative_strengths','perturbation_seed']:assert c[k]==parent['config'][k]
    ctx=context(c);budget=Budget(dict(max_seconds=c['max_seconds'],max_rss_bytes=c['max_rss_bytes']))
    core.write_json(out/'config.json',c);core.write_json(out/'started.json',ctx)
    core.write_json(out/'environment.json',core.environment())
    prior={p.as_posix():core.sha256(p) for p in Path('results').rglob('*') if p.is_file() and out not in p.parents}
    core.write_json(out/'prior-artifacts.json',prior)
    with np.load(Path(c['cache'])/'nodes.npz') as z:ids=z['ids']
    assert core.sha256(Path(c['cache'])/'nodes.npz')==c['nodes_sha256']
    banks={};rows=[];clean_checks=0;first_checks=0
    for case in old.case_list(c):
        budget.check();src=source/old.stem(case);meta=read(src/'case.json');path=out/old.stem(case);path.mkdir()
        assert meta['horizon']==197 and meta['readout_parameters']==482 and meta['graph']['observed_features']==48
        with np.load(src/'checkpoint.npz') as z:cp=dict(z)
        head=core.NonlinearReadout(np.arange(48),8,core.nonlinear_seed(case['seed'],0));head.mean=cp['mean'];head.scale=cp['scale']
        head.parameters={k:cp[k] for k in ['w1','b1','w2','b2']};assert head.digest()==meta['head_sha256']
        with np.load(src/'dose0_teacher.npz') as z:clean=dict(z)
        pred=head.predict(clean['features']);probs=softmax(head.logits(clean['features']),axis=1)
        np.testing.assert_array_equal(pred,clean['prediction']);np.testing.assert_allclose(probs,clean['probabilities'],atol=1e-12,rtol=1e-10);clean_checks+=1
        target=cp['digits'][3:];observed=np.array(meta['graph']['observation_root_ids'],dtype=np.int64);key=(case['seed'],case['circuit_seed'])
        if key not in banks:banks[key]=(observed,noise_bank(ids,observed,*key,197,c['perturbation_seed']))
        np.testing.assert_array_equal(observed,banks[key][0]);noise=banks[key][1]
        np.savez_compressed(path/'noise.npz',observed_root_ids=observed,standard_normals=noise)
        q=float(np.median(cp['features'].std(axis=0)));np.testing.assert_allclose(q,meta['training_scale'],atol=0,rtol=1e-14)
        local=[]
        for j,r in enumerate(c['relative_strengths'],1):
            sigma=r*q;features=instantaneous(clean['features'],noise,sigma);probs=softmax(head.logits(features),axis=1);prediction=probs.argmax(1)
            with np.load(src/f'dose{j}_teacher.npz') as z:full=dict(z)
            # Identical clean prompt and current draw: history has no effect at t=0.
            np.testing.assert_allclose(features[0],full['features'][0],atol=1e-14,rtol=1e-12)
            np.testing.assert_allclose(probs[0],full['probabilities'][0],atol=1e-12,rtol=1e-10);first_checks+=1
            clean_acc=float(np.mean(clean['prediction']==target));instant_acc=float(np.mean(prediction==target));full_acc=float(np.mean(full['prediction']==target))
            a=dict(prediction=prediction,probabilities=probs,features=features,position_accuracy=prediction==target)
            name=f'instantaneous{j}.npz';np.savez_compressed(path/name,**a)
            row=dict(**case,strength=r,sigma=sigma,training_scale=q,artifact=name,source_artifact=f'dose{j}_teacher.npz',
                clean_accuracy=clean_acc,instantaneous_accuracy=instant_acc,full_accuracy=full_acc,
                history_accuracy_cost=instant_acc-full_acc,instantaneous_accuracy_cost=clean_acc-instant_acc,full_accuracy_cost=clean_acc-full_acc,
                history_feature_mse=float(np.mean(((full['features']-features)/head.scale)**2)),
                instantaneous_feature_mse=float(np.mean(((features-clean['features'])/head.scale)**2)),
                full_feature_mse=float(np.mean(((full['features']-clean['features'])/head.scale)**2)),
                instantaneous_target_probability=float(probs[np.arange(197),target].mean()),
                full_target_probability=float(full['probabilities'][np.arange(197),target].mean()))
            for label,p in [('instantaneous',prediction),('full',full['prediction'])]:
                row[label+'_accuracy_first32']=float(np.mean(p[:32]==target[:32]));row[label+'_accuracy_tail']=float(np.mean(p[32:]==target[32:]))
            local.append(row);rows.append(row)
        assert head.digest()==meta['head_sha256'];pd.DataFrame(local).to_csv(path/'metrics.csv',index=False)
        core.write_json(path/'case.json',dict(case=case,source_case=str(src).replace('\\','/'),source_manifest_sha256=core.sha256(src/'manifest.json'),head_sha256=head.digest(),
            training_scale=q,observed_neurons=48,positions=197,internal_dynamics_recomputed=False,head_unchanged=True))
        seal(path,c,ctx,purpose='Instantaneous teacher observation counterfactual')
        print(f'{len(rows)//3}/48 observation controls complete',flush=True)
    summarize(rows,c,out);assert context(c)==ctx
    for p,h in prior.items():assert core.sha256(p)==h,p
    core.write_json(out/'verification.json',dict(cases=len(rows)//3,new_instantaneous_evaluations=len(rows),archived_full_state_evaluations=len(rows),
        clean_replays=clean_checks,first_step_matches=first_checks,prior_files_unchanged=len(prior),budget=budget.close(),seconds=time.monotonic()-started))
    seal(out,c,ctx,purpose='Instantaneous versus accumulated teacher-state noise; no new autonomous runs')


def plot(root,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    m=check(root);c=m['config'];f=read_table(root/'dose-seed-table.csv');out.mkdir(parents=True,exist_ok=False)
    fig,axes=plt.subplots(2,3,figsize=(13,7),sharey=True,layout='constrained')
    for i,co in enumerate(c['cohorts']):
        for j,g in enumerate(c['conditions']):
            ax=axes[i,j];z=f[(f.cohort==co)&(f.level==g)]
            for name,color in [('instantaneous_accuracy','#28659c'),('full_accuracy','#c46a29')]:
                for seed,a in z.groupby('seed'):
                    a=a.sort_values('strength');ax.plot(range(4),[a.clean_accuracy.iloc[0],*a[name]],color=color,alpha=.18,lw=.8)
                a=z.groupby('strength')[['clean_accuracy',name]].mean();ax.plot(range(4),[a.clean_accuracy.iloc[0],*a[name]],marker='o',color=color,label=name.removesuffix('_accuracy'))
            ax.axhline(z.clean_accuracy.mean(),color='gray',ls=':',lw=.8,label='clean')
            ax.set_xticks(range(4),['0']+[f'{x:g}' for x in c['relative_strengths']]);ax.set_ylim(0,1);ax.set_title(co+': '+g);ax.set_xlabel('Relative noise r (unequal spacing)');ax.grid(alpha=.15)
            if j==0:ax.set_ylabel('Teacher-forced position accuracy')
    axes[0,0].legend();fig.suptitle('Current observation corruption versus accumulated all-state noise')
    fig.savefig(out/'observation-noise-curves.png',dpi=160);plt.close(fig)
    seal(out,c,m['context'],result_manifest_sha256=core.sha256(root/'manifest.json'),purpose='Observation-noise diagnostic figure')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['run','plot']);p.add_argument('--source',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.command=='run':run(config(),a.out)
    else:plot(a.source,a.out)
