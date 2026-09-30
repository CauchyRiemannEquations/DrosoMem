"""Independent stateless feature corruption, head scoring and paired statistics."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from flying.training import whole_brain_memory as core
from alphabet_memory import read,check
from pathway_memory import seal
from frozen_state_probe import Budget
from verify_neuron_panel import audit_stats
from relative_noise import read_table
import observation_noise as run


def independent_bank(ids,observed,seed,ci,h,base_seed):
    index={int(v):i for i,v in enumerate(ids)};ix=[index[int(v)] for v in observed]
    rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([base_seed,seed,ci,1])))
    result=np.empty((h,len(ix)))
    for t in range(h):result[t]=rng.standard_normal(len(ids))[ix]
    return result


def independent_head(x,cp):
    z=(x-cp['mean'])/cp['scale'];hidden=np.tanh(z.dot(cp['w1'])+cp['b1']);logits=hidden.dot(cp['w2'])+cp['b2']
    exponent=np.exp(logits-logits.max(axis=1,keepdims=True));probs=exponent/exponent.sum(axis=1,keepdims=True)
    return probs.argmax(1),probs


def verify_statistics(f,c,root):
    s=read(root/'summary.json');metrics=['clean_accuracy','instantaneous_accuracy','full_accuracy','history_accuracy_cost',
        'instantaneous_accuracy_cost','full_accuracy_cost','history_feature_mse'];blocks=[];gates={};near={}
    for co,seeds in c['cohorts'].items():
        for g in c['conditions']:
            vectors={k:[] for k in metrics}
            for seed in seeds:
                a=f[(f.cohort==co)&(f.level==g)&(f.seed==seed)];assert len(a)==6
                for k in metrics:vectors[k].append(float(np.sum(a[k].to_numpy())/6))
                blocks.append(dict(cohort=co,level=g,seed=seed,**{k:vectors[k][-1] for k in metrics}))
            key=f'{co}/{g}'
            for k in metrics:audit_stats(vectors[k],s['statistics'][key][k],c)
            d=np.asarray(vectors['history_accuracy_cost']);need=4 if co=='discovery' else 3
            gates[key]=bool(d.mean()>=.05 and (d>0).sum()>=need)
            near[key]=bool(abs(d.mean())<=.02 and np.all(abs(d)<=.05))
    assert gates==s['history_cost_gates'] and near==s['near_tolerance_gates']
    confirmed=[g for g in c['conditions'] if all(gates[f'{co}/{g}'] for co in c['cohorts'])]
    tol=[g for g in c['conditions'] if all(near[f'{co}/{g}'] for co in c['cohorts'])]
    assert confirmed==s['confirmed_history_cost'] and s['primary_confirmed']==('brain1' in confirmed)
    assert tol==s['within_registered_near_tolerance'] and not s['autonomous_recall_tested']
    saved=read_table(root/'seed-blocks.csv').sort_values(['cohort','level','seed']).reset_index(drop=True)
    expected=pd.DataFrame(blocks).sort_values(['cohort','level','seed']).reset_index(drop=True)
    pd.testing.assert_frame_equal(saved[expected.columns],expected,check_exact=False,atol=1e-12,rtol=1e-12)


@threadpool_limits.wrap(limits=1)
def verify(root,out):
    m=check(root);c=m['config'];current=run.context(c)
    for k in ['source_sha256','adapter_sha256','config_sha256']:assert current[k]==m['context'][k]
    out.mkdir(parents=True,exist_ok=False);source=Path(c['source']);check(source)
    assert core.sha256(source/'manifest.json')==c['source_manifest_sha256']
    with np.load(Path(c['cache'])/'nodes.npz') as z:ids=z['ids']
    assert core.sha256(Path(c['cache'])/'nodes.npz')==c['nodes_sha256']
    budget=Budget(dict(max_seconds=c['verification_seconds'],max_rss_bytes=c['max_rss_bytes']))
    count=0;clean_count=0;error=0.;banks={};allrows=[]
    for case in run.old.case_list(c):
        budget.check();path=root/run.old.stem(case);check(path);src=source/run.old.stem(case);meta=read(src/'case.json')
        with np.load(src/'checkpoint.npz') as z:cp=dict(z)
        target=cp['digits'][3:];assert len(target)==197
        np.testing.assert_array_equal(cp['digits'],core.SequenceDataset().symbols())
        with np.load(src/'dose0_teacher.npz') as z:clean=dict(z)
        pred,prob=independent_head(clean['features'],cp);np.testing.assert_array_equal(pred,clean['prediction'])
        np.testing.assert_allclose(prob,clean['probabilities'],atol=1e-12,rtol=1e-10);clean_count+=1
        observed=np.asarray(meta['graph']['observation_root_ids'],dtype=np.int64);key=(case['seed'],case['circuit_seed'])
        if key not in banks:banks[key]=independent_bank(ids,observed,*key,197,c['perturbation_seed'])
        noise=banks[key]
        with np.load(path/'noise.npz') as z:
            np.testing.assert_array_equal(z['observed_root_ids'],observed);np.testing.assert_array_equal(z['standard_normals'],noise)
        q=float(np.median(np.sqrt(np.mean((cp['features']-cp['features'].mean(0))**2,axis=0))))
        f=read_table(path/'metrics.csv');assert f.strength.tolist()==c['relative_strengths'];allrows.extend(f.to_dict('records'))
        for j,row in enumerate(f.to_dict('records'),1):
            assert row['source_artifact']==f'dose{j}_teacher.npz'
            np.testing.assert_allclose(row['sigma'],q*row['strength'],atol=1e-18,rtol=1e-13)
            with np.load(src/f'dose{j}_teacher.npz') as z:full=dict(z)
            x=np.maximum(-1,np.minimum(1,clean['features']+q*row['strength']*noise));pred,prob=independent_head(x,cp)
            with np.load(path/row['artifact']) as a:
                np.testing.assert_array_equal(a['prediction'],pred);np.testing.assert_array_equal(a['position_accuracy'],pred==target)
                np.testing.assert_allclose(a['features'],x,atol=1e-14,rtol=1e-12)
                np.testing.assert_allclose(a['probabilities'],prob,atol=1e-12,rtol=1e-10)
                error=max(error,float(np.max(np.abs(a['features']-x))),float(np.max(np.abs(a['probabilities']-prob))))
            np.testing.assert_allclose(x[0],full['features'][0],atol=1e-14,rtol=1e-12)
            np.testing.assert_allclose(prob[0],full['probabilities'][0],atol=1e-12,rtol=1e-10)
            ca=float(np.count_nonzero(clean['prediction']==target)/197);ia=float(np.count_nonzero(pred==target)/197);fa=float(np.count_nonzero(full['prediction']==target)/197)
            expected=dict(clean_accuracy=ca,instantaneous_accuracy=ia,full_accuracy=fa,history_accuracy_cost=ia-fa,
                instantaneous_accuracy_cost=ca-ia,full_accuracy_cost=ca-fa,
                history_feature_mse=float(np.mean(np.square((full['features']-x)/cp['scale']))),
                instantaneous_feature_mse=float(np.mean(np.square((x-clean['features'])/cp['scale']))),
                full_feature_mse=float(np.mean(np.square((full['features']-clean['features'])/cp['scale']))),
                instantaneous_target_probability=float(np.mean([prob[t,y] for t,y in enumerate(target)])),
                full_target_probability=float(np.mean([full['probabilities'][t,y] for t,y in enumerate(target)])))
            for label,p in [('instantaneous',pred),('full',full['prediction'])]:
                expected[label+'_accuracy_first32']=float(np.mean(p[:32]==target[:32]));expected[label+'_accuracy_tail']=float(np.mean(p[32:]==target[32:]))
            for k,val in expected.items():np.testing.assert_allclose(row[k],val,atol=1e-12,rtol=1e-10,err_msg=k)
            count+=1
        print(f'Audit {clean_count}/48 observation controls',flush=True)
    f=pd.DataFrame(allrows);saved=read_table(root/'raw-metrics.csv')
    pd.testing.assert_frame_equal(saved[f.columns],f,check_exact=False,atol=1e-12,rtol=1e-12)
    verify_statistics(f,c,root);prior=read(root/'prior-artifacts.json')
    for p,h in prior.items():assert core.sha256(p)==h,p
    core.write_json(out/'checks.json',dict(all_checks_pass=True,cases=clean_count,independent_instantaneous_evaluations=count,
        first_step_matches=count,maximum_numeric_error=error,new_full_neural_trajectory_replays=0,
        prior_files_unchanged=len(prior),result_manifest_sha256=core.sha256(root/'manifest.json'),budget=budget.close()))
    seal(out,c,m['context'],purpose='Independent instantaneous-observation counterfactual validation')
    print('Observation diagnostic verified',count,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args();verify(a.source,a.out)
