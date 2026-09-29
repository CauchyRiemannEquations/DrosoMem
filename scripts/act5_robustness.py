"""Preregistered fixed-head five-family partial/whole graph robustness panel."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import psutil
from scipy import sparse
from scipy.special import softmax
from threadpoolctl import threadpool_limits
from flying.training import whole_brain_memory as core
from flying.brain.timed_reservoir import TimedReservoir
from alphabet_memory import read, check
from context_memory import estimate
from pathway_memory import seal

FAMILIES = ('pulse','ongoing','edge_dropout','neuron_dropout','weight_noise')


def config():
    return read('configs/act5_robustness.json')


def context(c):
    x=core.source_context(c)
    x['adapter_sha256']={str(p).replace('\\','/'):core.sha256(p) for p in
        [Path('scripts/act5_robustness.py'),Path('scripts/verify_act5_robustness.py'),Path('scripts/plot_act5_robustness.py')]}
    return x


def rng(c,seed,ci,tag):
    return np.random.default_rng(np.random.SeedSequence([c['perturbation_seed'],seed,ci,tag]))


def case_list(c,smoke=False):
    if smoke:
        return [dict(cohort='smoke',seed=7142,circuit_seed=701,level=g) for g in c['conditions']]
    return [dict(cohort=co,seed=s,circuit_seed=ci,level=g)
        for co,seeds in c['cohorts'].items() for s in seeds for ci in c['circuit_seeds'] for g in c['conditions']]


def stem(case):
    return f"{case['cohort']}_{case['level']}_c{case['circuit_seed']}_s{case['seed']}"


def maps(c,case,model):
    with np.load(Path(c['cache'])/'nodes.npz') as z:
        universe=z['ids']; roles=z['roles']
    if case['level']=='legacy5':
        _,ids,_=core.load_connectome(f"data/flywire_783_mb_left_kc512_s{case['circuit_seed']}")
        ix=pd.Index(universe).get_indexer(np.asarray(ids,dtype=np.int64));assert (ix>=0).all()
    else: ix=np.arange(len(universe))
    np.testing.assert_array_equal(np.flatnonzero(roles[ix]=='KC'),model.kc)
    np.testing.assert_array_equal(np.flatnonzero(roles[ix]=='MBON'),model.mbon)
    full=sparse.load_npz(Path(c['cache'])/'brain1.npz').tocsr();full.sort_indices();n=len(universe)
    if case['level']=='brain1':
        np.testing.assert_array_equal(model.weights.indptr,full.indptr)
        np.testing.assert_array_equal(model.weights.indices,full.indices)
        edge_ix=None
    else:
        keys=np.repeat(np.arange(n,dtype=np.int64),np.diff(full.indptr))*n+full.indices
        own=ix[np.repeat(np.arange(len(ix)),np.diff(model.weights.indptr))]*n+ix[model.weights.indices]
        edge_ix=np.searchsorted(keys,own);np.testing.assert_array_equal(keys[edge_ix],own)
    return dict(ids=universe[ix],node_ix=ix,roles=roles[ix],universe_size=n,edge_ix=edge_ix,edge_universe=full.nnz)


def bank(c,case,mapping):
    s,ci=case['seed'],case['circuit_seed'];e=mapping['edge_ix']
    u=rng(c,s,ci,3).random(mapping['edge_universe']);u=u if e is None else u[e]
    z=rng(c,s,ci,4).standard_normal(mapping['edge_universe']);z=z if e is None else z[e]
    return dict(neuron=rng(c,s,ci,2).random(mapping['universe_size'])[mapping['node_ix']],edge=u,weight=z)


def alter(weights,draws,kind,strength):
    if kind not in ('clean',*FAMILIES) or not np.isfinite(strength) or strength<0:
        raise ValueError('Invalid intervention')
    if kind=='clean' and strength!=0: raise ValueError('Nonzero clean condition')
    if kind.endswith('dropout') and strength>1: raise ValueError('Invalid dropout fraction')
    dead=np.zeros(weights.shape[0],dtype=bool);w=weights;removed=0
    if strength and kind=='edge_dropout':
        w=weights.copy();mask=draws['edge']<strength;removed=int(mask.sum());w.data[mask]=0;w.eliminate_zeros()
    elif strength and kind=='neuron_dropout': dead=draws['neuron']<strength
    elif strength and kind=='weight_noise':
        w=weights.copy();w.data*=np.exp(strength*draws['weight']-.5*strength**2)
    return w,dead,removed


def advance(model,digit,dead):
    if not dead.any(): return model.step(int(digit))
    old=model.state.copy();stimulation=model.encoder(int(digit));leak=model.leak
    following=(1-leak)*old+leak*np.tanh(model.weights@old+stimulation)
    following[dead]=0
    presynaptic=old.copy();presynaptic[model.kc]=following[model.kc]
    following[model.mbon]=(1-leak)*old[model.mbon]+leak*np.tanh(
        model.mbon_weights@presynaptic+stimulation[model.mbon])
    following[dead]=0;model.state=following
    return following


def _simulate(base,observed,head,prompt,horizon,mapping,weights,dead,kind,strength,random,feed):
    model=TimedReservoir(weights,base.encoder,mapping['roles'],base.leak,base.schedule)
    for x in prompt:advance(model,int(x),dead)
    pred=[];probs=[];features=[];active=[];norm=[];clipped=[]
    for t in range(horizon):
        clipped_now=0
        if strength and (kind=='ongoing' or (kind=='pulse' and t==0)):
            state=model.state+strength*random.standard_normal(mapping['universe_size'])[mapping['node_ix']]
            clipped_now=int((np.abs(state)>1).sum());model.state=np.clip(state,-1,1)
        x=model.state[observed].copy();p=softmax(head.logits(x)[0]);digit=int(p.argmax())
        pred.append(digit);probs.append(p);features.append(x);clipped.append(clipped_now)
        active.append(int(np.count_nonzero(np.abs(model.state)>1e-8)));norm.append(float(np.linalg.norm(model.state)))
        advance(model,digit if feed is None else int(feed[t]),dead)
    return dict(prediction=np.asarray(pred),probabilities=np.asarray(probs),features=np.asarray(features),
        active_counts=np.asarray(active),state_norm=np.asarray(norm),clipped=np.asarray(clipped))


def autonomous(base,observed,head,prompt,horizon,mapping,weights,dead,kind,strength,random):
    # No target/reference sequence accepted by this entry point.
    return _simulate(base,observed,head,prompt,horizon,mapping,weights,dead,kind,strength,random,None)


def teacher_forced(base,observed,head,symbols,horizon,mapping,weights,dead,kind,strength,random):
    return _simulate(base,observed,head,symbols[:3],horizon,mapping,weights,dead,kind,strength,random,symbols[3:])


def measures(a,target,head,clean,neurons):
    correct=a['prediction']==target;score=core.prefix_score(target,a['prediction']);x=a['features']
    denom=np.linalg.norm(x[1:],axis=1)*np.linalg.norm(x[:-1],axis=1)
    cos=np.divide(np.sum(x[1:]*x[:-1],axis=1),denom,out=np.zeros(len(denom)),where=denom!=0)
    return dict(pi_memory_score=score,exact_prefix_bits=float(score*np.log2(10)),
        first_error_position=None if correct.all() else score+1,accuracy=float(correct.mean()),
        prefix32=bool(score>=32),censored=bool(correct.all()),effective_rank=core.state_diagnostics(x)['effective_rank'],
        active_neurons_mean=float(a['active_counts'].mean()),activity_sparsity=float(1-a['active_counts'].mean()/neurons),
        mbon_mean_abs=float(np.abs(x).mean()),mbon_sparsity=float((np.abs(x)<=1e-8).mean()),
        temporal_cosine=float(cos.mean()),standardized_clean_mse=float(np.mean(((x-clean['features'])/head.scale)**2)),
        mean_target_probability=float(a['probabilities'][np.arange(len(target)),target].mean()),
        mean_confidence=float(a['probabilities'].max(axis=1).mean()),clipped_coordinates=int(a['clipped'].sum()),
        **{f'accuracy_{lo+1}_{min(hi,len(target))}':float(correct[lo:hi].mean())
           for lo,hi in [(0,32),(32,64),(64,128),(128,len(target))] if lo<len(target) and hi>lo})


def make_head(c,case,model,obs,base_config,out):
    digits=core.SequenceDataset(**base_config['dataset']).symbols();fresh=case['cohort']=='confirmation'
    head=core.NonlinearReadout(np.arange(48),8,core.nonlinear_seed(case['seed'],0))
    if fresh:
        features,diagnostics=core.collect(model,obs,digits[:-1],1e-8)
        _,history=head.fit(features,digits[1:],epochs=2000,learning_rate=.03,l2=1e-5,checkpoints=(2000,),
            sample_weight=core.sample_weights(199,3,32,4))
        a=dict(mean=head.mean,scale=head.scale,**head.parameters,features=features,digits=digits,
            teacher=head.predict(features),teacher_probabilities=softmax(head.logits(features),axis=1))
        core.write_json(out/'training.json',history)
    else:
        path=Path(c['source'])/f"{case['level']}_c{case['circuit_seed']}_s{case['seed']}"
        m=check(path)
        with np.load(path/'checkpoint.npz') as z:a={k:z[k] for k in z.files}
        np.testing.assert_array_equal(digits,a['digits']);head.mean=a['mean'];head.scale=a['scale']
        head.parameters={k:a[k] for k in ['w1','b1','w2','b2']}
        core.write_json(out/'source.json',dict(path=str(path).replace('\\','/'),manifest_sha256=core.sha256(path/'manifest.json'),
            checkpoint_sha256=core.sha256(path/'checkpoint.npz'),training_sha256=core.sha256(path/'training.json')))
    return head,digits,a


def execute(c,case,out,smoke=False):
    started=time.perf_counter();out=Path(out);out.mkdir(exist_ok=False)
    bc=read(Path(c['source'])/'config.json');cond=core.NetworkCondition(case['level'],case['circuit_seed'],case['seed'])
    with threadpool_limits(1):
        base,obs,graph=core.build_model(c['cache'],cond,bc);mp=maps(c,case,base);draws=bank(c,case,mp)
        head,digits,checkpoint=make_head(c,case,base,obs,bc,out);head_digest=head.digest();weight_digest=core.weight_hash(base.weights)
        h=8 if smoke else 197;target=digits[3:3+h];rows=[];cleans={};actual=0
        for mode in ['autonomous','teacher']:
            call=autonomous if mode=='autonomous' else teacher_forced
            a=call(base,obs,head,digits[:3] if mode=='autonomous' else digits,h,mp,base.weights,
                np.zeros(len(base.state),bool),'clean',0,rng(c,case['seed'],case['circuit_seed'],1))
            cleans[mode]=a;actual+=1
            if mode=='autonomous' and case['cohort']!='confirmation':
                for k,old in [('prediction','prediction'),('probabilities','probabilities'),('features','recall_features')]:
                    np.testing.assert_array_equal(a[k],checkpoint[old][:h])
            if mode=='teacher':
                np.testing.assert_array_equal(a['features'],checkpoint['features'][2:2+h])
                np.testing.assert_array_equal(a['prediction'],checkpoint['teacher'][2:2+h])
            np.savez_compressed(out/f'clean_{mode}.npz',**a,position_accuracy=a['prediction']==target)
            rows.append(dict(**case,kind='clean',strength=0.,mode=mode,artifact=f'clean_{mode}.npz',
                reused_clean=False,**measures(a,target,head,a,len(base.state))))
        checkpoint.update(prediction=cleans['autonomous']['prediction'],probabilities=cleans['autonomous']['probabilities'],
            recall_features=cleans['autonomous']['features'],observed_indices=obs)
        np.savez_compressed(out/'checkpoint.npz',**checkpoint)
        # Initial state for this auxiliary decay is the last teacher-forced scored state.
        base.reset()
        for digit in digits[:h+2]:base.step(int(digit))
        decay_full,decay_obs=core.decay_probe(base,obs,32)
        np.savez_compressed(out/'clean-decay.npz',full=decay_full,observed=decay_obs)
        interventions=[]
        for kind in FAMILIES:
            w0,d0,n0=alter(base.weights,draws,kind,0)
            assert core.weight_hash(w0)==weight_digest and not d0.any() and n0==0
            if smoke:
                a=autonomous(base,obs,head,digits[:3],h,mp,w0,d0,kind,0,rng(c,case['seed'],case['circuit_seed'],1))
                for k in a:np.testing.assert_array_equal(a[k],cleans['autonomous'][k])
                actual+=1
            rows.append(dict(**case,kind=kind,strength=0.,mode='autonomous',artifact='clean_autonomous.npz',reused_clean=True,
                **measures(cleans['autonomous'],target,head,cleans['autonomous'],len(base.state))))
            for j,strength in enumerate(c['strengths'][kind]):
                w,dead,removed=alter(base.weights,draws,kind,strength);key=f'{kind}_{j}'
                meta=dict(kind=kind,strength=strength,removed_edges=removed,dead_neurons=int(dead.sum()),
                    dead_observed=int(dead[obs].sum()),dead_input_neurons=int((dead & (base.encoder.patterns!=0).any(axis=0)).sum()),
                    weight_sha256=core.weight_hash(w),zero_renormalization=True,
                    relative_weight_l2=float(np.linalg.norm(w.data-base.weights.data)/np.linalg.norm(base.weights.data)) if w.nnz==base.weights.nnz else None,
                    state_noise_over_median_observed_train_sd=float(strength/np.median(checkpoint['features'].std(axis=0))) if kind in ('pulse','ongoing') else None)
                np.savez_compressed(out/(key+'_mask.npz'),dead_root_ids=mp['ids'][dead],
                    removed_csr_indices=np.flatnonzero(draws['edge']<strength) if kind=='edge_dropout' else np.array([],dtype=np.int64))
                interventions.append(meta)
                for mode in ['autonomous']+(['teacher'] if j==2 else []):
                    t=time.perf_counter();call=autonomous if mode=='autonomous' else teacher_forced
                    a=call(base,obs,head,digits[:3] if mode=='autonomous' else digits,h,mp,w,dead,kind,strength,
                        rng(c,case['seed'],case['circuit_seed'],1));actual+=1
                    file=f'{key}_{mode}.npz';np.savez_compressed(out/file,**a,position_accuracy=a['prediction']==target)
                    rows.append(dict(**case,kind=kind,strength=strength,mode=mode,artifact=file,reused_clean=False,
                        **measures(a,target,head,cleans[mode],len(base.state)),runtime_seconds=time.perf_counter()-t))
        assert head.digest()==head_digest and core.weight_hash(base.weights)==weight_digest
        pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
        core.write_json(out/'interventions.json',interventions)
        core.write_json(out/'case.json',dict(case=case,graph=graph,head_sha256=head_digest,readout_parameters=head.parameter_count,
            horizon=h,smoke=smoke,actual_trajectories=actual,zero_aliases=5,runtime_seconds=time.perf_counter()-started,
            environment=core.environment(),dataset_sha256=hashlib.sha256(digits.tobytes()).hexdigest(),checkpoint_sha256=core.sha256(out/'checkpoint.npz'),
            original_weight_sha256=weight_digest,unchanged_weights_and_head=True))
    seal(out,c,context(c),purpose='ACT V frozen-head perturbation case')


def summarize(rows,c,out):
    f=pd.DataFrame(rows);f.to_csv(out/'raw-metrics.csv',index=False)
    auto=f[(f['mode']=='autonomous') & ~f.reused_clean.astype(bool)].copy()
    keys=['cohort','seed','circuit_seed','level'];clean=auto[auto.kind=='clean'][keys+['pi_memory_score']].rename(columns={'pi_memory_score':'clean_prefix'})
    d=auto[auto.kind!='clean'].merge(clean,on=keys,validate='many_to_one')
    d['uncapped_retention']=d.pi_memory_score/d.clean_prefix.replace(0,np.nan)
    d['retention']=d.uncapped_retention.clip(upper=1)
    d.to_csv(out/'dose-head-table.csv',index=False)
    doses=d.groupby(['cohort','seed','level','kind','strength'])[['pi_memory_score','retention','accuracy']].agg(lambda x: np.mean(x.to_numpy())).reset_index()
    doses.to_csv(out/'dose-seed-table.csv',index=False)
    curves=d.groupby(keys+['kind'])[['pi_memory_score','retention']].agg(lambda x: np.mean(x.to_numpy())).reset_index()
    blocks=curves.groupby(['cohort','seed','level','kind'])[['pi_memory_score','retention']].agg(lambda x: np.mean(x.to_numpy())).reset_index()
    curves.to_csv(out/'curve-head-table.csv',index=False);blocks.to_csv(out/'seed-blocks.csv',index=False)
    invalid=set(tuple(r) for r in d[d.clean_prefix==0][['cohort','level']].drop_duplicates().to_numpy())
    statistics={};comparisons={};absolute={};pairs=[]
    for co,seeds in c['cohorts'].items():
        need=4 if co=='discovery' else 3
        for kind in FAMILIES:
            for level in c['conditions']:
                z=blocks[(blocks.cohort==co)&(blocks.level==level)&(blocks.kind==kind)]
                key=f'{co}/{level}/{kind}';bad=(co,level) in invalid
                statistics[key]={m:None if bad and m=='retention' else estimate(z[m],c) for m in ['pi_memory_score','retention']}
                hi=d[(d.cohort==co)&(d.level==level)&(d.kind==kind)&(d.strength==max(c['strengths'][kind]))]
                r=hi.groupby('seed').retention.mean();eligible=hi[hi.clean_prefix>=32]
                kept=float((eligible.pi_memory_score>=32).mean()) if len(eligible) else None
                absolute[key]=dict(identifiable=not bad,mean_retention=None if bad else float(r.mean()),
                    seed_blocks_passing=int((r>=c['retention_threshold']).sum()),eligible_heads=len(eligible),prefix32_retention=kept,
                    passed=None if bad else bool(r.mean()>=c['retention_threshold'] and (r>=c['retention_threshold']).sum()>=need
                        and kept is not None and kept>=c['prefix32_retention_threshold']))
            for hi,lo in [('brain1','legacy5'),('brain5','legacy5'),('brain1','brain5')]:
                x=blocks[(blocks.cohort==co)&(blocks.level==hi)&(blocks.kind==kind)].set_index('seed')
                y=blocks[(blocks.cohort==co)&(blocks.level==lo)&(blocks.kind==kind)].set_index('seed')
                delta=x[['pi_memory_score','retention']]-y[['pi_memory_score','retention']]
                bad=any((co,g) in invalid for g in [hi,lo]);key=f'{co}/{hi}-{lo}/{kind}'
                comparisons[key]=dict(identifiable=not bad,statistics={m:None if bad and m=='retention' else estimate(delta[m],c) for m in delta},
                    passed=None if bad else bool(delta.retention.mean()>=c['curve_retention_gain'] and delta.pi_memory_score.mean()>=c['curve_prefix_gain']
                        and (delta.retention>0).sum()>=need and (delta.pi_memory_score>0).sum()>=need))
                pairs.extend(dict(cohort=co,contrast=f'{hi}-{lo}',kind=kind,seed=int(s),**v) for s,v in delta.to_dict('index').items())
    pd.DataFrame(pairs).to_csv(out/'paired-differences.csv',index=False)
    confirmed=[f'{contrast}/{kind}' for contrast in ['brain1-legacy5','brain5-legacy5','brain1-brain5'] for kind in FAMILIES
        if all(comparisons[f'{co}/{contrast}/{kind}']['passed'] is True for co in c['cohorts'])]
    robust=[f'{g}/{kind}' for g in c['conditions'] for kind in FAMILIES
        if all(absolute[f'{co}/{g}/{kind}']['passed'] is True for co in c['cohorts'])]
    summary=dict(statistics=statistics,comparisons=comparisons,absolute_robustness=absolute,confirmed_comparisons=confirmed,
        confirmed_absolute_robustness=robust,primary_confirmed='brain1-legacy5/ongoing' in confirmed,cohorts_pooled=False,
        undefined_clean_conditions=[list(x) for x in sorted(invalid)])
    core.write_json(out/'summary.json',summary);return summary


def supervise(c,case,out,smoke,deadline):
    if time.monotonic()>deadline:raise RuntimeError("Total budget elapsed before worker start")
    args=[sys.executable,'scripts/act5_robustness.py','worker','--out',str(out),'--case',json.dumps(case)]
    if smoke:args+=['--smoke']
    t=time.monotonic();peak=0
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    with out.with_suffix('.log').open('w',encoding='utf-8') as log:
        process=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,env=env);p=psutil.Process(process.pid)
        while process.poll() is None:
            try:
                peak=max(peak,p.memory_info().rss)
                if peak>c['worker_rss_bytes'] or time.monotonic()-t>c['worker_seconds'] or time.monotonic()>deadline:
                    process.kill();process.wait();raise RuntimeError(f'Resource limit: {out}; partial outputs preserved')
            except psutil.NoSuchProcess:pass
            time.sleep(.2)
        if process.returncode:raise RuntimeError(f'Worker failed: {out.with_suffix(".log")}')
    return dict(case=stem(case),seconds=time.monotonic()-t,peak_sampled_rss_bytes=peak,sample_seconds=.2)


def run(c,out,smoke):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    assert core.sha256(Path(c['source'])/'manifest.json')==c['source_manifest_sha256'];check(Path(c['source']))
    ctx=context(c);core.write_json(out/'config.json',c);core.write_json(out/'started.json',ctx)
    prior={str(p).replace('\\','/'):core.sha256(p) for p in Path('results').rglob('*') if p.is_file() and out not in p.parents}
    core.write_json(out/'prior-artifacts.json',prior);resources=[];cases=case_list(c,smoke)
    with ThreadPoolExecutor(max_workers=c['workers']) as pool:
        pending={pool.submit(supervise,c,case,out/stem(case),smoke,started+c['max_seconds']):case for case in cases}
        for fut in as_completed(pending):
            resources.append(fut.result());print(f'{len(resources)}/{len(cases)} cases complete: {stem(pending[fut])}',flush=True)
            if time.monotonic()-started>c['max_seconds']:raise RuntimeError('Total budget exceeded; outputs preserved')
    rows=[];identities={}
    for case in cases:
        path=out/stem(case);check(path);m=read(path/'case.json');g=m['graph'];key=(case['seed'],case['circuit_seed'])
        mapping={k:g[k] for k in ['input_root_ids','observation_root_ids','input_mapping_sha256']}
        if key in identities:assert mapping==identities[key]
        identities[key]=mapping;rows.extend(pd.read_csv(path/'metrics.csv').to_dict('records'))
    core.write_json(out/'resources.json',resources)
    if smoke:pd.DataFrame(rows).to_csv(out/'raw-metrics.csv',index=False)
    else:summarize(rows,c,out)
    assert context(c)==ctx
    for p,h in prior.items():assert core.sha256(p)==h,p
    core.write_json(out/'verification.json',dict(prior_files_unchanged=len(prior),cases=len(cases),smoke=smoke,
        actual_trajectories=sum(read(out/stem(case)/'case.json')['actual_trajectories'] for case in cases),
        zero_aliases=5*len(cases),seconds=time.monotonic()-started,matched_inputs_observations=True))
    seal(out,c,ctx,purpose='ACT V smoke only' if smoke else 'ACT V main and unconditional fresh confirmation')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['run','worker']);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--case');a=p.parse_args()
    if a.command=='run':run(config(),a.out,a.smoke)
    else:execute(config(),json.loads(a.case),a.out,a.smoke)
