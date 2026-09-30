"""Sequential bounded sequence, wiring and correlated-observation experiments."""
import argparse, json, os, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd
import psutil
from scipy import sparse
from scipy.special import softmax
from threadpoolctl import threadpool_limits
from flying.training import whole_brain_memory as core
from flying.training import sequence_memory as seq
from flying.data.sequences import SequenceDataset
from alphabet_memory import read, check
from pathway_memory import seal
from context_memory import estimate
from relative_noise import read_table
import feedback_noise as feedback
import allocation_noise as allocation
import structural_controls as structural

CONFIG='configs/research_suite.json'
METRICS=['exact_prefix_symbols','retention']
GROUP=['cohort','family','level','topology','rho']


def context(c):
    ctx=core.source_context(c)
    names=['research_suite','verify_research_suite','feedback_noise','verify_feedback_noise',
           'allocation_noise','verify_allocation_noise','structural_controls','act5_robustness',
           'verify_act5_robustness','alphabet_memory','pathway_memory','context_memory',
           'relative_noise','frozen_state_probe','verify_neuron_panel']
    ctx['adapter_sha256']={f'scripts/{s}.py':core.sha256(f'scripts/{s}.py') for s in names}
    ctx['base_config_sha256']=core.sha256(c['base_config'])
    return ctx


def cases(c,stage,smoke=False):
    rows=[]
    cohorts=c['cohorts']
    if smoke and stage=='sequence':
        cohorts={'discovery':[c['cohorts']['discovery'][0]],'smoke':[c['smoke_block']]}
    elif smoke:cohorts={'discovery':[c['cohorts']['discovery'][0]]}
    families=['random'] if smoke else (['pi','random'] if stage=='structure' else c['families'])
    for cohort,blocks in cohorts.items():
        for seed,ds in blocks:
            for family in families:
                for circuit in ([701] if smoke else c['circuit_seeds']):
                    for level in (['legacy5'] if stage=='structure' else c['conditions']):
                        for topology in (c['topologies'] if stage=='structure' else ['intact']):
                            rows.append(dict(cohort=cohort,seed=seed,dataset_seed=ds,family=family,
                                             circuit_seed=circuit,level=level,topology=topology))
    return rows


def stem(case):
    return f"{case['cohort']}_{case['family']}_{case['level']}_{case['topology']}_c{case['circuit_seed']}_s{case['seed']}_d{case['dataset_seed']}"


def parent_stem(case):return stem(dict(case,topology='intact'))


def selected(c,case,stage,smoke=False):
    if smoke:return True
    first=case['seed']==c['cohorts'][case['cohort']][0][0] and case['circuit_seed']==701
    if stage=='correlation':return first and case['family']=='random'
    return stage=='structure' or case['level']=='legacy5' or first


def independently_selected(c,case,stage,smoke=False):
    if smoke or stage=='structure':return True
    if stage=='correlation':return selected(c,case,stage)
    return case['level']=='legacy5' or (selected(c,case,stage) and case['family']=='random')


def graph_seed(c,case):
    block=case['seed']-9142 if case['cohort']=='discovery' else case['seed']-440142+10
    return c['graph_seed_base']+10*block+2*(case['circuit_seed']-701)+(case['topology']=='role')


def build(c,case):
    bc=read(c['base_config']);condition=core.NetworkCondition(case['level'],case['circuit_seed'],case['seed'])
    model,obs,graph=core.build_model(c['cache'],condition,bc);raw=None
    if case['topology']!='intact':
        folder=Path(f"data/flywire_783_mb_left_kc512_s{case['circuit_seed']}")
        original,ids,_=core.load_connectome(folder);roles,_=core.load_roles(folder,ids);roles=np.asarray(roles)
        raw,log=structural.generate(original,roles,case['topology'],graph_seed(c,case),c['swaps_per_edge'])
        audit=structural.audit(original,raw,roles,case['topology'])
        w=core.normalize_condition(raw,bc['normalization'],bc['gain']);w.sort_indices()
        model=core.TimedReservoir(w,model.encoder,roles,bc['leak'],bc['schedule'])
        graph=dict(graph,weight_sha256=core.weight_hash(w),topology=case['topology'],
                   raw_sha256=core.weight_hash(raw),generation=log,structural_audit=audit)
    return model,obs,graph,raw


def noise_bank(c,case,ids,h):
    independent=np.stack([allocation.fresh_bank(ids,case['seed'],case['circuit_seed'],ns,h) for ns in c['noise_seeds']])
    common=np.stack([np.random.default_rng(np.random.SeedSequence([ns,case['seed'],case['circuit_seed'],3])).standard_normal(h)
                     for ns in c['common_noise_seeds']])
    return independent,common


def correlated(z,common,rho):
    if not 0<=rho<=1:raise ValueError('Invalid correlation')
    return np.sqrt(1-rho)*z+np.sqrt(rho)*common[...,None]


def endpoint(pred,target,clean_prefix):
    x=feedback.certificate(pred,target,clean_prefix)
    x['exact_prefix_symbols']=x.pop('pi_memory_score')
    return x


def rollout_metrics(a,target,head,clean,neurons):
    x=feedback.rollout_metrics(a,target,head,clean,neurons)
    x['exact_prefix_symbols']=x.pop('pi_memory_score')
    clean_prefix=core.prefix_score(target,clean['prediction'])
    x['clean_prefix']=clean_prefix
    x['retention']=min(x['exact_prefix_symbols']/clean_prefix,1.) if clean_prefix else None
    return x


def checkpoint(c,case,stage,model,obs,out):
    bc=read(c['base_config']);fresh=stage=='sequence' and case['cohort']!='discovery' or case['topology']!='intact'
    source=None;source_manifest=None
    if stage=='sequence' and not fresh:
        source=Path(c['archive'])/f"{case['family']}_{case['level']}_c{case['circuit_seed']}_s{case['seed']}_d{case['dataset_seed']}"
    elif stage!='sequence' and not fresh:source=Path(c['parent'])/parent_stem(case)
    if fresh:
        dataset=SequenceDataset(case['family'],case['dataset_seed'],**bc['dataset']);symbols=dataset.symbols()
        features,diagnostics=core.collect(model,obs,symbols[:-1],1e-8)
        head,history,weights=seq.fit(features,symbols,core.NetworkCondition(case['level'],case['circuit_seed'],case['seed']),bc)
        cp=dict(symbols=symbols,features=features,mean=head.mean,scale=head.scale,**head.parameters,
                teacher=head.predict(features),teacher_probabilities=softmax(head.logits(features),axis=1),**diagnostics)
        core.write_json(out/'training.json',history)
    else:
        check(source);source_manifest=core.sha256(source/'manifest.json')
        with np.load(source/'checkpoint.npz') as z:cp=dict(z)
        head=feedback.load_head(cp,case['seed'])
    expected=SequenceDataset(case['family'],case['dataset_seed'],**bc['dataset'])
    np.testing.assert_array_equal(cp['symbols'],expected.symbols())
    assert head.parameter_count==482
    return cp,head,fresh,source,source_manifest,expected.identity(cp['symbols'])


@threadpool_limits.wrap(limits=1)
def execute(c,case,stage,out,smoke=False):
    out.mkdir(exist_ok=False);start=time.monotonic();ctx=context(c);h=8 if smoke else 197
    actual=selected(c,case,stage,smoke)
    # Unselected correlation cases only decode previously verified teacher states.
    if stage=='correlation' and not actual:
        parent=Path(c['parent'])/parent_stem(case);graph=read(parent/'case.json')['graph'];model=obs=raw=None
    else:model,obs,graph,raw=build(c,case)
    cp,head,fresh,source,source_hash,dataset=checkpoint(c,case,stage,model,obs,out)
    if source and stage=='sequence':assert graph==read(source/'manifest.json')['graph']
    elif source:assert graph==read(source/'case.json')['graph']
    target=cp['symbols'][3:3+h];teacher=cp['features'][2:2+h]
    teacher_prob=softmax(head.logits(teacher),axis=1);head_hash=head.digest()
    ids=np.array(graph['observation_root_ids'],dtype=np.int64)
    z,common=noise_bank(c,case,ids,h);sd,own_q,weights=allocation.allocation(cp['features']);q=own_q
    if stage=='structure':
        with np.load(Path(c['parent'])/parent_stem(case)/'checkpoint.npz') as ref:q=allocation.allocation(ref['features'])[1]
    rolls=[];cert=[];pathcount=0;prechecks=0
    if stage!='correlation' or actual:
        clean=feedback.autonomous(model,obs,head,cp['symbols'][:3],h,z[0],np.zeros(48));pathcount+=1
        if source:
            if stage=='sequence':
                old={k:cp[v][:h] for k,v in [('prediction','prediction'),('probabilities','probabilities'),('features','recall_features')]}
            else:
                with np.load(source/'clean.npz') as archive:old={k:archive[k][:h] for k in clean}
            for k,v in old.items():np.testing.assert_array_equal(clean[k],v)
        feedback.verify_pre_error(clean,teacher,teacher_prob,teacher_prob.argmax(1),target);prechecks+=1
        np.savez_compressed(out/'clean.npz',**clean,position_accuracy=clean['prediction']==target)
        rolls.append(dict(**case,rho=0.,arm='clean',strength=0.,noise_seed=c['noise_seeds'][0],artifact='clean.npz',
                          **rollout_metrics(clean,target,head,clean,graph['neurons'])))
    else:
        with np.load(source/'clean.npz') as a:clean={k:a[k][:h] for k in a}
    clean_prefix=core.prefix_score(target,clean['prediction'])
    if stage!='correlation':
        cp.update(prediction=clean['prediction'],probabilities=clean['probabilities'],recall_features=clean['features'])
        np.savez_compressed(out/'checkpoint.npz',**cp)
    rhos=c['correlations'] if stage=='correlation' else [0.]
    probs=np.empty((len(rhos),3,3,2,h,10));amplitudes=np.empty((3,2,48))
    for ri,rho in enumerate(rhos):
        bank=correlated(z,common,rho)
        for ni,ns in enumerate(c['noise_seeds']):
            for di,dose in enumerate(c['relative_strengths']):
                amplitudes[di]=np.stack([np.full(48,dose*q),dose*q*weights])
                for ai,arm in enumerate(['flat','training_sd']):
                    amp=amplitudes[di,ai];np.testing.assert_allclose(amp@amp,48*(dose*q)**2,atol=1e-18,rtol=1e-12)
                    x=allocation.perturb(teacher,bank[ni],amp);p=softmax(head.logits(x),axis=1);pred=p.argmax(1)
                    probs[ri,ni,di,ai]=p
                    cert.append(dict(**case,rho=rho,noise_seed=ns,strength=dose,arm=arm,
                        reused_exposure=stage=='correlation' and rho==0.,
                        teacher_accuracy=float(np.mean(pred==target)),mean_confidence=float(p.max(1).mean()),
                        target_probability=float(p[np.arange(h),target].mean()),expected_energy=float(amp@amp),
                        injected_energy=float(np.mean(np.sum((bank[ni]*amp)**2,axis=1))),
                        clipped_energy=float(np.mean(np.sum((x-teacher)**2,axis=1))),
                        clipping_fraction=float(np.mean(np.abs(teacher+bank[ni]*amp)>1)),
                        **endpoint(pred,target,clean_prefix)))
                    if actual and ni==0 and (stage!='correlation' or rho>0):
                        a=feedback.autonomous(model,obs,head,cp['symbols'][:3],h,bank[ni],amp);pathcount+=1
                        stop=feedback.verify_pre_error(a,x,p,pred,target);prechecks+=1
                        np.testing.assert_allclose(a['state_features'][:stop],teacher[:stop],atol=1e-12,rtol=1e-10)
                        name=f'rho{ri}_dose{di}_{arm}.npz';np.savez_compressed(out/name,**a,position_accuracy=a['prediction']==target)
                        rolls.append(dict(**case,rho=rho,arm=arm,strength=dose,noise_seed=ns,artifact=name,
                            teacher_accuracy=float(np.mean(pred==target)),**rollout_metrics(a,target,head,clean,graph['neurons'])))
    if stage=='correlation':
        with np.load(source/'certificates.npz') as parent:
            np.testing.assert_array_equal(probs[0],parent['probabilities'][0,:,:,:,:h])
    np.savez_compressed(out/'certificates.npz',probabilities=probs,prediction=probs.argmax(-1),
        position_accuracy=probs.argmax(-1)==target,standard_normals=z,common_normals=common,
        amplitudes=amplitudes,training_sd=sd,allocation_weights=weights,observed_root_ids=ids,target=target)
    if raw is not None:sparse.save_npz(out/'raw-graph.npz',raw);sparse.save_npz(out/'weights.npz',model.weights)
    pd.DataFrame(cert).to_csv(out/'certificates.csv',index=False)
    if rolls:pd.DataFrame(rolls).to_csv(out/'rollouts.csv',index=False)
    assert head.digest()==head_hash
    if model is not None:assert core.weight_hash(model.weights)==graph['weight_sha256']
    loss,_,training=head.objective(cp['features'],cp['symbols'][1:],1e-5,core.sample_weights(199,3,32,4))
    core.write_json(out/'case.json',dict(case=case,stage=stage,smoke=smoke,horizon=h,graph=graph,dataset=dataset,
        head_sha256=head_hash,fresh_fit=fresh,training_trajectories=int(fresh),train_loss=loss,
        weighted_cross_entropy=training['cross_entropy'],training_accuracy=float(np.mean(head.predict(cp['features'])==cp['symbols'][1:])),
        teacher_state_diagnostics=core.state_diagnostics(cp['features']),own_q=own_q,used_q=q,
        parent=str(source) if source else None,parent_manifest_sha256=source_hash,clean_prefix=clean_prefix,
        actual_paths=pathcount,pre_error_checks=prechecks,certificates=len(cert),environment=core.environment(),seconds=time.monotonic()-start))
    seal(out,c,ctx,purpose=stage+' case')


def summarize(frame,c,out,stage):
    mean=lambda x:np.mean(x.to_numpy())
    blocks=frame.groupby(GROUP+['seed','arm'])[METRICS].agg(mean).reset_index()
    blocks.to_csv(out/'seed-blocks.csv',index=False)
    frame.groupby(GROUP+['seed','arm','strength'])[METRICS].agg(mean).reset_index().to_csv(out/'dose-seed-table.csv',index=False)
    stats={};gates={};pairs=[]
    for values,z in blocks.groupby(GROUP):
        key='/'.join(map(str,values));flat=z[z.arm=='flat'].set_index('seed');a=z[z.arm=='training_sd'].set_index('seed')
        delta=a[METRICS]-flat[METRICS]
        stats[key]={k:estimate(delta[k],c) if np.isfinite(delta[k]).all() else None for k in METRICS}
        for arm,v in [('flat',flat),('training_sd',a)]:stats[key][arm]={k:estimate(v[k],c) if np.isfinite(v[k]).all() else None for k in METRICS}
        gates[key]=feedback.gate(delta.exact_prefix_symbols,delta.retention,values[0],c)
        pairs.extend(dict(zip(GROUP,values),seed=int(seed),**row) for seed,row in delta.to_dict('index').items())
    pd.DataFrame(pairs).to_csv(out/'paired-differences.csv',index=False)
    structural_stats={};structural_gates={};structural_pairs=[]
    if stage=='structure':
        for (co,family),z in blocks[blocks.arm=='training_sd'].groupby(['cohort','family']):
            base=z[z.topology=='intact'].set_index('seed')
            for control in ['degree','role']:
                v=z[z.topology==control].set_index('seed');d=base[METRICS]-v[METRICS];key=f'{co}/{family}/{control}'
                structural_stats[key]={k:estimate(d[k],c) if np.isfinite(d[k]).all() else None for k in METRICS}
                structural_gates[key]=feedback.gate(d.exact_prefix_symbols,d.retention,co,c)
                structural_pairs.extend(dict(cohort=co,family=family,control=control,seed=int(seed),**row) for seed,row in d.to_dict('index').items())
        pd.DataFrame(structural_pairs).to_csv(out/'structural-paired-differences.csv',index=False)
    if stage=='sequence':primary_keys=[f'{co}/random/brain1/intact/0.0' for co in c['cohorts']]
    elif stage=='correlation':primary_keys=[f'{co}/random/brain1/intact/{rho}' for co in c['cohorts'] for rho in [.25,.75]]
    else:primary_keys=[f'{co}/{family}/role' for co in c['cohorts'] for family in ['pi','random']]
    decision=structural_gates if stage=='structure' else gates
    vals=[decision[k] for k in primary_keys]
    primary=None if any(v is None for v in vals) else all(vals)
    core.write_json(out/'summary.json',dict(stage=stage,statistics=stats,gates=gates,structural_statistics=structural_stats,
        structural_gates=structural_gates,primary_keys=primary_keys,primary_confirmed=primary,
        cohorts_pooled=False,undefined_retention_propagated=True))


def supervise(c,case,stage,out,smoke,deadline,stop):
    if stop.is_set():raise RuntimeError('Cancelled before launch')
    cmd=[sys.executable,__file__,'worker','--stage',stage,'--config',str(out.parent/'config.json'),'--out',str(out),'--case',json.dumps(case)]
    if smoke:cmd+=['--smoke']
    start=time.monotonic();peak=0
    with out.with_suffix('.log').open('x',encoding='utf-8') as log:
        child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
        proc=psutil.Process(child.pid)
        while child.poll() is None:
            try:
                tree=[proc]+proc.children(recursive=True);peak=max(peak,sum(p.memory_info().rss for p in tree if p.is_running()))
                if stop.is_set() or peak>c['worker_rss_bytes'] or time.monotonic()-start>c['worker_seconds'] or time.monotonic()>deadline:
                    for p in reversed(tree):
                        try:p.kill()
                        except psutil.NoSuchProcess:pass
                    child.wait();raise RuntimeError('Resource/cancellation stop; partial files preserved')
            except psutil.NoSuchProcess:pass
            time.sleep(.2)
        if child.returncode:raise RuntimeError(f'Worker failed: {out.with_suffix(".log")}')
    return dict(case=stem(case),seconds=time.monotonic()-start,peak_sampled_rss_bytes=peak,sample_seconds=.2,scope='worker_process_tree')


def run(c,stage,out,smoke=False):
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();ctx=context(c)
    core.write_json(out/'config.json',c);core.write_json(out/'started.json',ctx)
    try:
        assert core.sha256(Path(c['archive'])/'manifest.json')==c['archive_sha256'];check(Path(c['archive']))
        assert core.sha256(Path(c['archive_verification'])/'manifest.json')==c['archive_verification_sha256'];v=check(Path(c['archive_verification']))
        assert v['complete'] and v['source_manifest_sha256']==c['archive_sha256']
        parents={}
        if stage!='sequence':
            p=Path(c['parent']);check(p);vroot=Path(str(p)+'_validation');check(vroot);v=read(vroot/'checks.json')
            assert v['all_checks_pass'] and v['result_manifest_sha256']==core.sha256(p/'manifest.json')
            parents[str(p)]=core.sha256(p/'manifest.json')
        if stage=='correlation' and not smoke:
            p=Path(c['structural_validation']);check(p);assert read(p/'checks.json')['all_checks_pass']
            parents[str(p)]=core.sha256(p/'manifest.json')
        core.write_json(out/'parents.json',parents)
        prior={p.as_posix():core.sha256(p) for p in Path('results').rglob('*') if p.is_file() and out not in p.parents}
        core.write_json(out/'prior-artifacts.json',prior);usage=[];stop=threading.Event()
        chosen=cases(c,stage,smoke)
        with ThreadPoolExecutor(max_workers=c['workers']) as pool:
            jobs=[pool.submit(supervise,c,x,stage,out/stem(x),smoke,start+c['max_seconds'],stop) for x in chosen]
            try:
                for job in as_completed(jobs):
                    usage.append(job.result());print(f'{stage}: {len(usage)}/{len(chosen)} {usage[-1]["case"]}',flush=True)
            except Exception:
                stop.set()
                for job in jobs:job.cancel()
                raise
        cert=[];rolls=[];metas=[];identities={};datasets={}
        for case in chosen:
            root=out/stem(case);check(root);meta=read(root/'case.json');metas.append(meta)
            cert.extend(read_table(root/'certificates.csv').to_dict('records'))
            if (root/'rollouts.csv').exists():rolls.extend(read_table(root/'rollouts.csv').to_dict('records'))
            g=meta['graph'];identity={k:g[k] for k in ['input_root_ids','observation_root_ids','input_mapping_sha256']}
            key=(case['seed'],case['circuit_seed']);assert identities.setdefault(key,identity)==identity
            key=(case['family'],case['dataset_seed']);assert datasets.setdefault(key,meta['dataset'])==meta['dataset']
        frame=pd.DataFrame(cert);frame.to_csv(out/'raw-certificates.csv',index=False);pd.DataFrame(rolls).to_csv(out/'raw-rollouts.csv',index=False)
        if not smoke:summarize(frame,c,out,stage)
        assert context(c)==ctx
        for p,digest in prior.items():assert core.sha256(p)==digest,p
        assert time.monotonic()-start<c['max_seconds'],'Main stage deadline exceeded'
        core.write_json(out/'resources.json',usage)
        core.write_json(out/'verification.json',dict(stage=stage,smoke=smoke,cases=len(metas),certificates=len(cert),
            actual_paths=sum(x['actual_paths'] for x in metas),pre_error_checks=sum(x['pre_error_checks'] for x in metas),
            fresh_fits=sum(x['fresh_fit'] for x in metas),teacher_training_trajectories=sum(x['training_trajectories'] for x in metas),
            prior_files_unchanged=len(prior),seconds=time.monotonic()-start))
        seal(out,c,ctx,purpose=stage+(' smoke' if smoke else ' main'))
    except Exception as exc:
        core.write_json(out/'failure.json',dict(error=repr(exc),seconds=time.monotonic()-start));seal(out,c,ctx,purpose='Preserved execution failure');raise


def plot(root,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    m=check(root);c=m['config'];stage=read(root/'verification.json')['stage'];f=read_table(root/'dose-seed-table.csv')
    out.mkdir(parents=True,exist_ok=False)
    families=['pi','random'] if stage=='structure' else c['families']
    fig,axes=plt.subplots(2,len(families),figsize=(5*len(families),7),layout='constrained',squeeze=False)
    for i,co in enumerate(c['cohorts']):
        for j,family in enumerate(families):
            ax=axes[i,j];s=f[(f.cohort==co)&(f.family==family)]
            groups=['topology'] if stage=='structure' else (['rho','arm'] if stage=='correlation' else ['level','arm'])
            if stage=='structure':s=s[s.arm=='training_sd']
            if stage=='correlation':s=s[s.level=='brain1']
            for key,a in s.groupby(groups):
                y=a.groupby('strength').exact_prefix_symbols.mean();ax.plot(range(3),y,'o-',label=str(key))
            ax.set(title=co+' / '+family,xticks=range(3),xticklabels=c['relative_strengths'],xlabel='Dose (categorical spacing)',ylabel='Certified exact prefix')
            ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.savefig(out/'prefix-curves.png',dpi=150);plt.close(fig)
    seal(out,c,m['context'],result_manifest_sha256=core.sha256(root/'manifest.json'),purpose='Fixed dose curves')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['run','worker','plot']);p.add_argument('--stage',choices=['sequence','structure','correlation'])
    p.add_argument('--out',type=Path,required=True);p.add_argument('--source',type=Path);p.add_argument('--config',default=CONFIG);p.add_argument('--parent')
    p.add_argument('--structural-validation',default='results/research_suite_structure_validation');p.add_argument('--case');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    c=read(a.config)
    if a.command=='plot':plot(a.source,a.out)
    elif a.command=='worker':execute(c,json.loads(a.case),a.stage,a.out,a.smoke)
    else:
        c['parent']=a.parent or c['sequence_parent'];c['structural_validation']=a.structural_validation
        run(c,a.stage,a.out,a.smoke)
