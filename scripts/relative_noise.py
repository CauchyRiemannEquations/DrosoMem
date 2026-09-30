"""Train-scale matched ongoing state noise; historical numerical sources untouched."""
import argparse,json,os,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits
from flying.training import whole_brain_memory as core
from alphabet_memory import read,check
from pathway_memory import seal
from context_memory import estimate
import act5_robustness as old


def config():return read('configs/relative_noise.json')


def context(c):
    x=core.source_context(c)
    names=['relative_noise','verify_relative_noise','plot_relative_noise','act5_robustness',
           'verify_act5_robustness','alphabet_memory','pathway_memory','context_memory',
           'verify_neuron_panel','frozen_state_probe']
    x['adapter_sha256']={f'scripts/{n}.py':core.sha256(f'scripts/{n}.py') for n in names}
    return x


def calibration(features,relative):
    x=np.asarray(features)
    if x.ndim!=2 or not np.isfinite(x).all() or not np.isfinite(relative) or relative<0:
        raise ValueError('Invalid training features or relative dose')
    q=float(np.median(x.std(axis=0,ddof=0)))
    if not np.isfinite(q) or q<=0:raise ValueError('Unidentifiable observed training scale')
    return q,q*relative


def mapping(c,case,model):
    with np.load(Path(c['cache'])/'nodes.npz') as z:ids=z['ids'];roles=z['roles']
    if case['level']=='legacy5':
        _,local,_=core.load_connectome(f"data/flywire_783_mb_left_kc512_s{case['circuit_seed']}")
        ix=pd.Index(ids).get_indexer(np.asarray(local,dtype=np.int64));assert (ix>=0).all()
    else:ix=np.arange(len(ids))
    np.testing.assert_array_equal(np.flatnonzero(roles[ix]=='KC'),model.kc)
    np.testing.assert_array_equal(np.flatnonzero(roles[ix]=='MBON'),model.mbon)
    return dict(ids=ids[ix],node_ix=ix,roles=roles[ix],universe_size=len(ids))


def execute(c,case,out,smoke=False):
    started=time.monotonic();out.mkdir(exist_ok=False);bc=read(Path(c['source'])/'config.json')
    with threadpool_limits(1):
        base,obs,g=core.build_model(c['cache'],core.NetworkCondition(case['level'],case['circuit_seed'],case['seed']),bc)
        mp=mapping(c,case,base);head,digits,cp=old.make_head(c,case,base,obs,bc,out)
        q,_=calibration(cp['features'],0);hh=head.digest();wh=core.weight_hash(base.weights)
        h=8 if smoke else 197;cleans={};rows=[];actual=0
        for j,r in enumerate([0,*c['relative_strengths']]):
            _,sigma=calibration(cp['features'],r)
            for mode in ['autonomous','teacher']:
                call=old.autonomous if mode=='autonomous' else old.teacher_forced
                a=call(base,obs,head,digits[:3] if mode=='autonomous' else digits,h,mp,base.weights,
                       np.zeros(len(base.state),bool),'ongoing',sigma,old.rng(c,case['seed'],case['circuit_seed'],1));actual+=1
                if r==0:
                    cleans[mode]=a
                    if mode=='autonomous' and case['cohort']!='confirmation':
                        for k,v in [('prediction','prediction'),('probabilities','probabilities'),('features','recall_features')]:
                            np.testing.assert_array_equal(a[k],cp[v][:h])
                    if mode=='teacher':np.testing.assert_array_equal(a['features'],cp['features'][2:2+h])
                name=f'dose{j}_{mode}.npz'
                np.savez_compressed(out/name,**a,position_accuracy=a['prediction']==digits[3:3+h])
                rows.append(dict(**case,kind='clean' if r==0 else 'ongoing',strength=r,sigma=sigma,training_scale=q,
                    mode=mode,artifact=name,reused_clean=False,**old.measures(a,digits[3:3+h],head,cleans[mode],len(base.state))))
        if smoke:
            zero=old.autonomous(base,obs,head,digits[:3],h,mp,base.weights,np.zeros(len(base.state),bool),'ongoing',0,
                                old.rng(c,case['seed'],case['circuit_seed'],1));actual+=1
            for k in zero:np.testing.assert_array_equal(zero[k],cleans['autonomous'][k])
        rows.append(dict(rows[0],kind='ongoing',reused_clean=True))
        cp.update(prediction=cleans['autonomous']['prediction'],probabilities=cleans['autonomous']['probabilities'],
                  recall_features=cleans['autonomous']['features'],observed_indices=obs)
        np.savez_compressed(out/'checkpoint.npz',**cp)
        assert head.digest()==hh and core.weight_hash(base.weights)==wh
        pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
        core.write_json(out/'case.json',dict(case=case,graph=g,head_sha256=hh,original_weight_sha256=wh,
            readout_parameters=head.parameter_count,training_scale=q,horizon=h,smoke=smoke,actual_trajectories=actual,
            zero_aliases=1,unchanged_weights_and_head=True,environment=core.environment(),seconds=time.monotonic()-started))
    seal(out,c,context(c),purpose='Relative ongoing noise case')


def gate(retention,prefix,need,c):
    if not np.isfinite(retention).all():return None
    return bool(np.mean(retention)>=c['curve_retention_gain'] and np.mean(prefix)>=c['curve_prefix_gain']
                and np.sum(np.asarray(retention)>0)>=need and np.sum(np.asarray(prefix)>0)>=need)


def summarize(rows,c,out):
    f=pd.DataFrame(rows);f.to_csv(out/'raw-metrics.csv',index=False)
    keys=['cohort','seed','circuit_seed','level'];a=f[(f['mode']=='autonomous')&~f.reused_clean]
    clean=a[a.kind=='clean'][keys+['pi_memory_score']].rename(columns={'pi_memory_score':'clean_prefix'})
    d=a[a.strength>0].merge(clean,on=keys,validate='many_to_one')
    d['uncapped_retention']=d.pi_memory_score/d.clean_prefix.replace(0,np.nan);d['retention']=d.uncapped_retention.clip(upper=1)
    d.to_csv(out/'dose-head-table.csv',index=False)
    mean=lambda x:np.mean(x.to_numpy())
    doses=d.groupby(['cohort','seed','level','strength'])[['pi_memory_score','retention','accuracy']].agg(mean).reset_index()
    blocks=d.groupby(['cohort','seed','level'])[['pi_memory_score','retention']].agg(mean).reset_index()
    doses.to_csv(out/'dose-seed-table.csv',index=False);blocks.to_csv(out/'seed-blocks.csv',index=False)
    stats={};pairs=[];comp={}
    for co in c['cohorts']:
        for g in c['conditions']:
            z=blocks[(blocks.cohort==co)&(blocks.level==g)]
            stats[f'{co}/{g}']={k:estimate(z[k],c) if np.isfinite(z[k]).all() else None for k in ['pi_memory_score','retention']}
        for hi,lo in [('brain1','legacy5'),('brain5','legacy5'),('brain1','brain5')]:
            x=blocks[(blocks.cohort==co)&(blocks.level==hi)].set_index('seed')
            y=blocks[(blocks.cohort==co)&(blocks.level==lo)].set_index('seed')
            delta=x[['pi_memory_score','retention']]-y[['pi_memory_score','retention']];key=f'{co}/{hi}-{lo}'
            comp[key]=dict(passed=gate(delta.retention,delta.pi_memory_score,4 if co=='discovery' else 3,c),
                statistics={k:estimate(delta[k],c) if np.isfinite(delta[k]).all() else None for k in delta})
            pairs.extend(dict(cohort=co,contrast=f'{hi}-{lo}',seed=int(s),**v) for s,v in delta.to_dict('index').items())
    pd.DataFrame(pairs).to_csv(out/'paired-differences.csv',index=False)
    confirmed=[p for p in ['brain1-legacy5','brain5-legacy5','brain1-brain5'] if all(comp[f'{co}/{p}']['passed'] is True for co in c['cohorts'])]
    result=dict(statistics=stats,comparisons=comp,confirmed_comparisons=confirmed,primary_confirmed='brain1-legacy5' in confirmed,cohorts_pooled=False)
    core.write_json(out/'summary.json',result);return result


def supervise(c,case,out,smoke,deadline):
    if time.monotonic()>deadline:raise RuntimeError('Main time budget elapsed')
    args=[sys.executable,__file__,'worker','--out',str(out),'--case',json.dumps(case)]
    if smoke:args+=['--smoke']
    started=time.monotonic();peak=0
    with out.with_suffix('.log').open('x',encoding='utf-8') as log:
        p=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
        process=psutil.Process(p.pid)
        while p.poll() is None:
            try:
                tree=[process]+process.children(recursive=True);peak=max(peak,sum(x.memory_info().rss for x in tree if x.is_running()))
                if peak>c['worker_rss_bytes'] or time.monotonic()-started>c['worker_seconds'] or time.monotonic()>deadline:
                    for child in reversed(tree):
                        try:child.kill()
                        except psutil.NoSuchProcess:pass
                    p.wait();raise RuntimeError('Resource budget exceeded; partial outputs preserved')
            except psutil.NoSuchProcess:pass
            time.sleep(.2)
        if p.returncode:raise RuntimeError(f'Worker failed; see {out.with_suffix(".log")}')
    return dict(case=old.stem(case),seconds=time.monotonic()-started,peak_sampled_rss_bytes=peak,sample_seconds=.2,scope='worker_process_tree')


def run(c,out,smoke):
    out.mkdir(parents=True,exist_ok=False);started=time.monotonic();ctx=context(c)
    assert core.sha256(Path(c['source'])/'manifest.json')==c['source_manifest_sha256'];check(Path(c['source']))
    core.write_json(out/'config.json',c);core.write_json(out/'started.json',ctx)
    prior={p.as_posix():core.sha256(p) for p in Path('results').rglob('*') if p.is_file() and out not in p.parents}
    core.write_json(out/'prior-artifacts.json',prior);cases=old.case_list(c,smoke);resources=[]
    with ThreadPoolExecutor(max_workers=c['workers']) as pool:
        futures=[pool.submit(supervise,c,case,out/old.stem(case),smoke,started+c['max_seconds']) for case in cases]
        for fut in as_completed(futures):resources.append(fut.result());print(f'{len(resources)}/{len(cases)} complete: {resources[-1]["case"]}',flush=True)
    rows=[];identities={}
    for case in cases:
        path=out/old.stem(case);check(path);g=read(path/'case.json')['graph'];key=(case['seed'],case['circuit_seed'])
        ids={k:g[k] for k in ['input_root_ids','observation_root_ids','input_mapping_sha256']}
        if key in identities:assert identities[key]==ids
        identities[key]=ids;rows.extend(pd.read_csv(path/'metrics.csv').to_dict('records'))
    if smoke:pd.DataFrame(rows).to_csv(out/'raw-metrics.csv',index=False)
    else:summarize(rows,c,out)
    core.write_json(out/'resources.json',resources);assert context(c)==ctx
    for p,h in prior.items():assert core.sha256(p)==h,p
    core.write_json(out/'verification.json',dict(cases=len(cases),smoke=smoke,actual_trajectories=sum(read(out/old.stem(x)/'case.json')['actual_trajectories'] for x in cases),
        zero_aliases=len(cases),prior_files_unchanged=len(prior),matched_inputs_observations=True,seconds=time.monotonic()-started))
    seal(out,c,ctx,purpose='Relative noise smoke' if smoke else 'Relative noise main and fresh confirmation')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['run','worker']);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--case');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.command=='run':run(config(),a.out,a.smoke)
    else:execute(config(),json.loads(a.case),a.out,a.smoke)
