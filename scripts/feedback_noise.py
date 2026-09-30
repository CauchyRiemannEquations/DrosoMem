"""Target-free autonomous observation noise, with separately labeled prefix certificates."""
import argparse, json, os, subprocess, sys, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd
import psutil
from scipy.special import softmax
from threadpoolctl import threadpool_limits
from flying.training import whole_brain_memory as core
from alphabet_memory import read, check
from pathway_memory import seal
from context_memory import estimate
from relative_noise import read_table
import allocation_noise as alloc
import act5_robustness as old


def config():return read('configs/feedback_noise.json')


def context(c):
    x=core.source_context(c)
    names=['feedback_noise','verify_feedback_noise','allocation_noise','verify_allocation_noise',
           'act5_robustness','verify_act5_robustness','relative_noise','alphabet_memory',
           'pathway_memory','context_memory','frozen_state_probe','verify_neuron_panel']
    x['adapter_sha256']={f'scripts/{n}.py':core.sha256(f'scripts/{n}.py') for n in names}
    x['base_config_sha256']=core.sha256(c['base_config'])
    return x


def cases(c,smoke=False):
    allcases=old.case_list(c)
    return [x for x in allcases if x['seed']==7142 and x['circuit_seed']==701] if smoke else allcases


def load_head(cp,seed):
    head=core.NonlinearReadout(np.arange(48),8,core.nonlinear_seed(seed,0))
    head.mean=cp['mean'];head.scale=cp['scale'];head.parameters={k:cp[k] for k in ['w1','b1','w2','b2']}
    return head


def autonomous(model,observed,head,prompt,horizon,noise,amplitudes):
    """Only generated symbols enter the model after prompt; no target argument."""
    model.reset()
    for symbol in prompt:model.step(int(symbol))
    rows={k:[] for k in ['prediction','probabilities','features','state_features','active_counts',
                         'state_norm','clipped','injected_energy','clipped_energy']}
    for t in range(horizon):
        raw=model.state[observed].copy();delta=noise[t]*amplitudes
        x=np.clip(raw+delta,-1,1);p=softmax(head.logits(x)[0]);digit=int(p.argmax())
        values=[digit,p,x,raw,int(np.count_nonzero(np.abs(model.state)>1e-8)),float(np.linalg.norm(model.state)),
                int(np.count_nonzero(np.abs(raw+delta)>1)),float(delta@delta),float((x-raw)@(x-raw))]
        for key,value in zip(rows,values):rows[key].append(value)
        # Observation corruption is never written back into the neural state.
        model.step(digit)
    return {k:np.asarray(v) for k,v in rows.items()}


def certificate(prediction,target,clean_prefix):
    score=core.prefix_score(target,prediction)
    return dict(pi_memory_score=score,first_error_position=score+1 if score<len(target) else None,
        censored=score==len(target),exact_prefix_bits=float(score*np.log2(10)),
        clean_prefix=clean_prefix,retention=min(score/clean_prefix,1.) if clean_prefix else None)


def verify_pre_error(a,teacher_features,teacher_probs,teacher_pred,target):
    score=core.prefix_score(target,teacher_pred);stop=min(score+1,len(target))
    assert core.prefix_score(target,a['prediction'])==score
    np.testing.assert_array_equal(a['prediction'][:stop],teacher_pred[:stop])
    np.testing.assert_allclose(a['features'][:stop],teacher_features[:stop],atol=1e-12,rtol=1e-10)
    np.testing.assert_allclose(a['probabilities'][:stop],teacher_probs[:stop],atol=1e-12,rtol=1e-10)
    return stop


def rollout_metrics(a,target,head,clean,neurons):
    result=old.measures(a,target,head,clean,neurons)
    raw=a['state_features'];den=np.linalg.norm(raw[1:],axis=1)*np.linalg.norm(raw[:-1],axis=1)
    cos=np.divide(np.sum(raw[1:]*raw[:-1],axis=1),den,out=np.zeros(len(den)),where=den!=0)
    start=result['pi_memory_score']+1
    result.update(raw_state_effective_rank=core.state_diagnostics(raw)['effective_rank'],
        raw_mbon_mean_abs=float(np.abs(raw).mean()),raw_mbon_sparsity=float((np.abs(raw)<=1e-8).mean()),
        raw_temporal_cosine=float(cos.mean()),mean_injected_energy=float(a['injected_energy'].mean()),
        mean_clipped_energy=float(a['clipped_energy'].mean()),
        post_error_accuracy=float(np.mean(a['prediction'][start:]==target[start:])) if start<len(target) else None)
    return result


@threadpool_limits.wrap(limits=1)
def execute(c,case,out,smoke=False):
    out.mkdir(exist_ok=False);started=time.monotonic();ctx=context(c)
    src=Path(c['source'])/old.stem(case);parent=read(src/'case.json')
    allocation_src=Path(c['allocation_source'])/old.stem(case)
    with np.load(src/'checkpoint.npz') as z:cp=dict(z)
    h=8 if smoke else 197;target=cp['digits'][3:3+h];head=load_head(cp,case['seed'])
    assert head.digest()==parent['head_sha256'] and head.parameter_count==482
    base,obs,graph=core.build_model(c['cache'],core.NetworkCondition(case['level'],case['circuit_seed'],case['seed']),read(c['base_config']))
    assert graph==parent['graph'];weight_hash=core.weight_hash(base.weights)
    with np.load(src/'dose0_teacher.npz') as z:clean_teacher=z['features'][:h];clean_teacher_probs=z['probabilities'][:h];clean_teacher_pred=z['prediction'][:h]
    sd,q,weights=alloc.allocation(cp['features']);observed=np.array(graph['observation_root_ids'],dtype=np.int64)
    noise=np.stack([alloc.fresh_bank(observed,case['seed'],case['circuit_seed'],ns,h) for ns in c['fresh_noise_seeds']])
    clean=autonomous(base,obs,head,cp['digits'][:3],h,noise[0],np.zeros(48))
    with np.load(src/'dose0_autonomous.npz') as z:
        for key in ['prediction','features','probabilities','active_counts','state_norm']:
            np.testing.assert_array_equal(clean[key],z[key][:h])
    verify_pre_error(clean,clean_teacher,clean_teacher_probs,clean_teacher_pred,target)
    clean_prefix=core.prefix_score(target,clean['prediction']);allcert=[];rollrows=[]
    np.savez_compressed(out/'clean.npz',**clean,position_accuracy=clean['prediction']==target)
    rollrows.append(dict(**case,arm='clean',strength=0.,noise_seed=c['rollout_noise_seed'],artifact='clean.npz',
        **rollout_metrics(clean,target,head,clean,graph['neurons'])))
    with np.load(allocation_src/'evaluations.npz') as z:
        for k,ns in enumerate(c['archived_noise_seeds']):
            for j,r in enumerate(c['relative_strengths']):
                for a,arm in enumerate(['flat','training_sd']):
                    allcert.append(dict(**case,stage='archived',noise_seed=ns,strength=r,arm=arm,
                        endpoint_origin='archived_teacher_prefix_certificate',
                        **certificate(z['predictions'][k,j,a,:h],target,clean_prefix)))
    features=np.empty((3,3,2,h,48));probs=np.empty((3,3,2,h,10));amplitudes=np.empty((3,2,48));prechecks=1
    for k,ns in enumerate(c['fresh_noise_seeds']):
        for j,r in enumerate(c['relative_strengths']):
            amplitudes[j]=np.stack([np.full(48,r*q),r*q*weights])
            np.testing.assert_allclose(np.sum(amplitudes[j]**2,axis=1),48*(r*q)**2,atol=1e-18,rtol=1e-12)
            for a,arm in enumerate(['flat','training_sd']):
                x=alloc.perturb(clean_teacher,noise[k],amplitudes[j,a]);p=softmax(head.logits(x),axis=1)
                features[k,j,a]=x;probs[k,j,a]=p;prediction=p.argmax(1)
                allcert.append(dict(**case,stage='fresh',noise_seed=ns,strength=r,arm=arm,
                    endpoint_origin='fresh_teacher_prefix_certificate',**certificate(prediction,target,clean_prefix)))
                if ns==c['rollout_noise_seed']:
                    t=time.monotonic();aout=autonomous(base,obs,head,cp['digits'][:3],h,noise[k],amplitudes[j,a])
                    verify_pre_error(aout,x,p,prediction,target);prechecks+=1
                    score=core.prefix_score(target,prediction);stop=min(score+1,h)
                    np.testing.assert_allclose(aout['state_features'][:stop],clean_teacher[:stop],atol=1e-12,rtol=1e-10)
                    file=f'dose{j+1}_{arm}.npz';np.savez_compressed(out/file,**aout,position_accuracy=aout['prediction']==target)
                    rollrows.append(dict(**case,arm=arm,strength=r,noise_seed=ns,artifact=file,
                        teacher_accuracy=float(np.mean(prediction==target)),runtime_seconds=time.monotonic()-t,
                        **rollout_metrics(aout,target,head,clean,graph['neurons'])))
    np.savez_compressed(out/'fresh-teacher.npz',features=features,probabilities=probs,prediction=probs.argmax(-1),
        position_accuracy=probs.argmax(-1)==target,standard_normals=noise,amplitudes=amplitudes,training_sd=sd,
        allocation_weights=weights,observed_root_ids=observed,target=target)
    pd.DataFrame(allcert).to_csv(out/'certificates.csv',index=False);pd.DataFrame(rollrows).to_csv(out/'rollouts.csv',index=False)
    assert head.digest()==parent['head_sha256'] and core.weight_hash(base.weights)==weight_hash
    core.write_json(out/'case.json',dict(case=case,graph=graph,horizon=h,smoke=smoke,head_sha256=head.digest(),weight_sha256=weight_hash,
        source_case=src.as_posix(),source_manifest_sha256=core.sha256(src/'manifest.json'),
        allocation_case=allocation_src.as_posix(),allocation_manifest_sha256=core.sha256(allocation_src/'manifest.json'),
        clean_prefix=clean_prefix,certificates=len(allcert),actual_neural_trajectories=len(rollrows),pre_error_checks=prechecks,
        frozen_head_and_weights=True,environment=core.environment(),seconds=time.monotonic()-started))
    seal(out,c,ctx,purpose='Certified prefix versus actual autonomous feedback')


def gate(prefix,retention,co,c):
    p=np.asarray(prefix);r=np.asarray(retention)
    if not np.isfinite(r).all():return None
    need=4 if co=='discovery' else 3
    return bool(p.mean()>=c['curve_prefix_gain'] and r.mean()>=c['curve_retention_gain'] and (p>0).sum()>=need and (r>0).sum()>=need)


def summarize(cert,roll,c,out):
    f=pd.DataFrame(cert);r=pd.DataFrame(roll)
    f.to_csv(out/'raw-certificates.csv',index=False);r.to_csv(out/'raw-rollouts.csv',index=False)
    metrics=['pi_memory_score','retention'];mean=lambda x:np.mean(x.to_numpy())
    b=f.groupby(['cohort','stage','level','seed','arm'])[metrics].agg(mean).reset_index()
    b.to_csv(out/'seed-blocks.csv',index=False)
    f.groupby(['cohort','stage','level','seed','arm','strength'])[metrics].agg(mean).reset_index().to_csv(out/'dose-seed-table.csv',index=False)
    stats={};gates={};pairs=[]
    for co in c['cohorts']:
        for stage in ['archived','fresh']:
            for level in c['conditions']:
                z=b[(b.cohort==co)&(b.stage==stage)&(b.level==level)]
                flat=z[z.arm=='flat'].set_index('seed');allocated=z[z.arm=='training_sd'].set_index('seed')
                d=allocated[metrics]-flat[metrics];key=f'{co}/{stage}/{level}'
                stats[key]={k:estimate(d[k],c) if np.isfinite(d[k]).all() else None for k in metrics}
                for arm,values in [('flat',flat),('training_sd',allocated)]:
                    stats[key][arm]={k:estimate(values[k],c) if np.isfinite(values[k]).all() else None for k in metrics}
                gates[key]=gate(d.pi_memory_score,d.retention,co,c)
                pairs.extend(dict(cohort=co,stage=stage,level=level,seed=int(seed),**row) for seed,row in d.to_dict('index').items())
    pd.DataFrame(pairs).to_csv(out/'paired-differences.csv',index=False)
    confirmed=[g for g in c['conditions'] if all(gates[f'{co}/fresh/{g}'] is True for co in c['cohorts'])]
    core.write_json(out/'summary.json',dict(statistics=stats,descriptive_gates=gates,confirmed=confirmed,
        primary_confirmed='brain1' in confirmed,decision_stage='fresh',fresh_model_cohort=False,cohorts_pooled=False,
        prefix_certificates_are_not_full_rollouts=True))
    r[r.arm!='clean'].groupby(['cohort','level','seed','arm','strength'])[['pi_memory_score','accuracy','teacher_accuracy','post_error_accuracy']].mean().reset_index().to_csv(out/'rollout-seed-table.csv',index=False)


def supervise(c,case,out,smoke,deadline,stop):
    if stop.is_set() or time.monotonic()>deadline:raise RuntimeError('Run stopped before worker launch')
    command=[sys.executable,__file__,'worker','--out',str(out),'--case',json.dumps(case)]
    if smoke:command+=['--smoke']
    start=time.monotonic();peak=0
    with out.with_suffix('.log').open('x',encoding='utf-8') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
            env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
        proc=psutil.Process(process.pid)
        while process.poll() is None:
            try:
                tree=[proc]+proc.children(recursive=True);peak=max(peak,sum(p.memory_info().rss for p in tree if p.is_running()))
                if stop.is_set() or peak>c['worker_rss_bytes'] or time.monotonic()-start>c['worker_seconds'] or time.monotonic()>deadline:
                    for p in reversed(tree):
                        try:p.kill()
                        except psutil.NoSuchProcess:pass
                    process.wait();raise RuntimeError('Run stopped or resource budget exceeded; partial outputs retained')
            except psutil.NoSuchProcess:pass
            time.sleep(.2)
        if process.returncode:raise RuntimeError(f'Worker failed: {out.with_suffix(".log")}')
    return dict(case=old.stem(case),seconds=time.monotonic()-start,peak_sampled_rss_bytes=peak,sample_seconds=.2,scope='worker_process_tree')


def run(c,out,smoke=False):
    out.mkdir(parents=True,exist_ok=False);started=time.monotonic();ctx=context(c)
    core.write_json(out/'config.json',c);core.write_json(out/'started.json',ctx)
    try:
        for name,digest in [(c['source'],c['source_manifest_sha256']),(c['allocation_source'],c['allocation_manifest_sha256'])]:
            assert core.sha256(Path(name)/'manifest.json')==digest;check(Path(name))
        check(Path(c['source_validation']));v=read(Path(c['source_validation'])/'checks.json')
        assert v['all_checks_pass'] and v['result_manifest_sha256']==c['allocation_manifest_sha256']
        prior={p.as_posix():core.sha256(p) for p in Path('results').rglob('*') if p.is_file() and out not in p.parents}
        core.write_json(out/'prior-artifacts.json',prior);resources=[];selected=cases(c,smoke)
        stop=threading.Event()
        with ThreadPoolExecutor(max_workers=c['workers']) as pool:
            jobs=[pool.submit(supervise,c,case,out/old.stem(case),smoke,started+c['max_seconds'],stop) for case in selected]
            try:
                for job in as_completed(jobs):
                    resources.append(job.result());print(f'{len(resources)}/{len(selected)} feedback cases complete: {resources[-1]["case"]}',flush=True)
            except Exception:
                stop.set()
                for job in jobs:job.cancel()
                raise
        cert=[];roll=[];identities={};totals=dict(certificates=0,actual_neural_trajectories=0,pre_error_checks=0)
        for case in selected:
            path=out/old.stem(case);check(path);meta=read(path/'case.json');graph=meta['graph'];key=(case['seed'],case['circuit_seed'])
            identity={k:graph[k] for k in ['input_root_ids','observation_root_ids','input_mapping_sha256']}
            if key in identities:assert identity==identities[key]
            identities[key]=identity
            cert.extend(read_table(path/'certificates.csv').to_dict('records'));roll.extend(read_table(path/'rollouts.csv').to_dict('records'))
            for k in totals:totals[k]+=meta[k]
        if smoke:
            pd.DataFrame(cert).to_csv(out/'raw-certificates.csv',index=False);pd.DataFrame(roll).to_csv(out/'raw-rollouts.csv',index=False)
        else:summarize(cert,roll,c,out)
        core.write_json(out/'resources.json',resources);assert context(c)==ctx
        for p,digest in prior.items():assert core.sha256(p)==digest,p
        core.write_json(out/'verification.json',dict(cases=len(selected),smoke=smoke,**totals,
            prior_files_unchanged=len(prior),matched_inputs_observations=True,seconds=time.monotonic()-started))
        seal(out,c,ctx,purpose='Feedback smoke' if smoke else 'Observation-noise autonomous extension')
    except Exception as exc:
        core.write_json(out/'failure.json',dict(error=repr(exc),seconds=time.monotonic()-started))
        seal(out,c,ctx,purpose='Preserved feedback execution failure');raise


def plot(root,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    m=check(root);c=m['config'];d=read_table(root/'dose-seed-table.csv');out.mkdir(parents=True,exist_ok=False)
    fig,axes=plt.subplots(2,3,figsize=(13,7),sharey=True,layout='constrained')
    for i,co in enumerate(c['cohorts']):
        for j,g in enumerate(c['conditions']):
            ax=axes[i,j];z=d[(d.cohort==co)&(d.level==g)&(d.stage=='fresh')]
            for arm,color in [('flat','#b65b30'),('training_sd','#236ca5')]:
                a=z[z.arm==arm]
                for _,block in a.groupby('seed'):ax.plot(range(3),block.sort_values('strength').pi_memory_score,color=color,alpha=.2,lw=.8)
                mean=a.groupby('strength').pi_memory_score.mean();ax.plot(range(3),mean,marker='o',color=color,label=arm)
            ax.set_xticks(range(3),[f'{x:g}' for x in c['relative_strengths']]);ax.set_title(co+': '+g)
            ax.set_xlabel('Relative observation noise (unequal spacing)');ax.grid(alpha=.15)
            if j==0:ax.set_ylabel('Certified autonomous exact prefix')
    axes[0,0].legend();fig.suptitle('Three fresh noise streams: frozen-head autonomous prefix')
    fig.savefig(out/'feedback-prefix-curves.png',dpi=160);plt.close(fig)
    r=read_table(root/'rollout-seed-table.csv');fig,axes=plt.subplots(2,3,figsize=(13,7),sharey=True,layout='constrained')
    for i,co in enumerate(c['cohorts']):
        for j,g in enumerate(c['conditions']):
            ax=axes[i,j];z=r[(r.cohort==co)&(r.level==g)]
            for arm,color in [('flat','#b65b30'),('training_sd','#236ca5')]:
                a=z[z.arm==arm].groupby('strength')[['accuracy','teacher_accuracy']].mean()
                ax.plot(range(3),a.accuracy,marker='o',color=color,label=arm+' autonomous')
                ax.plot(range(3),a.teacher_accuracy,'--',color=color,label=arm+' teacher')
            ax.set_xticks(range(3),[f'{x:g}' for x in c['relative_strengths']]);ax.set_ylim(0,1)
            ax.set_title(co+': '+g);ax.set_xlabel('Relative observation noise');ax.grid(alpha=.15)
            if j==0:ax.set_ylabel('Position accuracy')
    axes[0,0].legend(fontsize=8);fig.suptitle('First fresh stream only: actual autonomous feedback versus teacher forcing')
    fig.savefig(out/'feedback-position-accuracy.png',dpi=160);plt.close(fig)
    seal(out,c,m['context'],result_manifest_sha256=core.sha256(root/'manifest.json'),purpose='Certified-prefix and actual-feedback figures')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['run','worker','plot']);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--source',type=Path);p.add_argument('--case');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.command=='run':run(config(),a.out,a.smoke)
    elif a.command=='worker':execute(config(),json.loads(a.case),a.out,a.smoke)
    else:plot(a.source,a.out)
