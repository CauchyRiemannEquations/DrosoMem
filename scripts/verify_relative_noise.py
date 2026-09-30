"""Independent relative calibration, scalar subset, all metrics and decisions."""
import argparse,time
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from flying.training import whole_brain_memory as core
from alphabet_memory import read,check
from pathway_memory import seal
from frozen_state_probe import Budget
from verify_neuron_panel import audit_stats
from verify_act5_robustness import reference,metrics
import relative_noise as run


def statistics(frame,c,root):
    s=read(root/'summary.json');expected=[];gates={}
    for co,seeds in c['cohorts'].items():
        vectors={}
        for g in c['conditions']:
            ps=[];rs=[]
            for seed in seeds:
                pp=[];rr=[]
                for ci in c['circuit_seeds']:
                    a=frame[(frame.cohort==co)&(frame.seed==seed)&(frame.circuit_seed==ci)&(frame.level==g)&(frame['mode']=='autonomous')]
                    clean=float(a[a.kind=='clean'].pi_memory_score.iloc[0]);z=a[a.strength>0].sort_values('strength')
                    assert z.strength.tolist()==c['relative_strengths']
                    scores=z.pi_memory_score.to_numpy();pp.extend(scores);rr.extend(np.minimum(scores/clean,1) if clean else [np.nan]*3)
                ps.append(np.mean(pp));rs.append(np.mean(rr));expected.append(dict(cohort=co,seed=seed,level=g,pi_memory_score=ps[-1],retention=rs[-1]))
            vectors[g]=(np.array(ps),np.array(rs));key=f'{co}/{g}'
            audit_stats(ps,s['statistics'][key]['pi_memory_score'],c)
            if np.isfinite(rs).all():audit_stats(rs,s['statistics'][key]['retention'],c)
            else:assert s['statistics'][key]['retention'] is None
        for hi,lo in [('brain1','legacy5'),('brain5','legacy5'),('brain1','brain5')]:
            p=vectors[hi][0]-vectors[lo][0];r=vectors[hi][1]-vectors[lo][1];key=f'{co}/{hi}-{lo}'
            n=4 if co=='discovery' else 3
            passed=None if not np.isfinite(r).all() else bool(r.mean()>=.1 and p.mean()>=2 and (r>0).sum()>=n and (p>0).sum()>=n)
            assert s['comparisons'][key]['passed']==passed;gates[key]=passed
            audit_stats(p,s['comparisons'][key]['statistics']['pi_memory_score'],c)
            if np.isfinite(r).all():audit_stats(r,s['comparisons'][key]['statistics']['retention'],c)
    b=pd.read_csv(root/'seed-blocks.csv').sort_values(['cohort','seed','level']).reset_index(drop=True)
    e=pd.DataFrame(expected).sort_values(['cohort','seed','level']).reset_index(drop=True)
    pd.testing.assert_frame_equal(b[e.columns],e,check_exact=False,atol=1e-12,rtol=1e-12)
    confirmed=[x for x in ['brain1-legacy5','brain5-legacy5','brain1-brain5'] if all(gates[f'{co}/{x}'] is True for co in c['cohorts'])]
    assert s['confirmed_comparisons']==confirmed and s['primary_confirmed']==('brain1-legacy5' in confirmed)


@threadpool_limits.wrap(limits=1)
def verify(root,out):
    m=check(root);c=m['config'];ctx=run.context(c)
    for k in ['source_sha256','adapter_sha256','config_sha256']:assert ctx[k]==m['context'][k]
    out.mkdir(parents=True,exist_ok=False);budget=Budget(dict(max_seconds=c['verification_seconds'],max_rss_bytes=c['worker_rss_bytes']))
    smoke=read(root/'verification.json')['smoke'];counts=dict(cases=0,metric_rows=0,independent_trajectories=0,exact_fresh_refits=0);rows=[];error=0.
    for case in run.old.case_list(c,smoke):
        budget.check();path=root/run.old.stem(case);check(path);meta=read(path/'case.json')
        with np.load(path/'checkpoint.npz') as z:cp=dict(z)
        head=core.NonlinearReadout(np.arange(48),8,core.nonlinear_seed(case['seed'],0));head.mean=cp['mean'];head.scale=cp['scale']
        head.parameters={k:cp[k] for k in ['w1','b1','w2','b2']};assert head.digest()==meta['head_sha256']
        # Independent population variance calculation; train features, no flooring.
        x=cp['features'];q=float(np.median(np.sqrt(np.mean((x-np.mean(x,axis=0))**2,axis=0))))
        assert q>0;np.testing.assert_allclose(q,meta['training_scale'],rtol=1e-14)
        digits=cp['digits'];np.testing.assert_array_equal(digits,core.SequenceDataset().symbols());h=meta['horizon']
        selected=smoke or case['level']=='legacy5' or (case['seed'],case['circuit_seed']) in [(7142,701),(381142,702)]
        refit=case['cohort']=='confirmation' and case['seed']==381142
        if selected or refit:
            base,obs,g=core.build_model(c['cache'],core.NetworkCondition(case['level'],case['circuit_seed'],case['seed']),read(Path(c['source'])/'config.json'))
            assert g==meta['graph'];mp=run.mapping(c,case,base)
            if refit:
                x,_=core.collect(base,obs,digits[:-1],1e-8);np.testing.assert_array_equal(x,cp['features'])
                fitted=core.NonlinearReadout(np.arange(48),8,core.nonlinear_seed(case['seed'],0))
                fitted.fit(x,digits[1:],epochs=2000,learning_rate=.03,l2=1e-5,checkpoints=(2000,),sample_weight=core.sample_weights(199,3,32,4))
                assert fitted.digest()==head.digest();counts['exact_fresh_refits']+=1
        clean={}
        for mode in ['autonomous','teacher']:
            with np.load(path/f'dose0_{mode}.npz') as z:clean[mode]=dict(z)
        f=pd.read_csv(path/'metrics.csv');rows.extend(f.to_dict('records'))
        for row in f.to_dict('records'):
            np.testing.assert_allclose(row['sigma'],q*row['strength'],rtol=1e-13,atol=1e-18)
            with np.load(path/row['artifact']) as z:a=dict(z)
            metrics(a,digits[3:3+h],row,clean[row['mode']],head.scale,meta['graph']['neurons']);counts['metric_rows']+=1
            if selected and not row['reused_clean']:
                random=np.random.default_rng(np.random.SeedSequence([372001,case['seed'],case['circuit_seed'],1]))
                ref=reference(base,obs,head,digits[:3],h,mp,base.weights,np.zeros(len(base.state),bool),'ongoing',q*row['strength'],random,
                    digits[3:] if row['mode']=='teacher' else None)
                for k,v in ref.items():
                    if k in ['prediction','active_counts','clipped']:np.testing.assert_array_equal(v,a[k])
                    else:np.testing.assert_allclose(v,a[k],atol=1e-12,rtol=1e-10);error=max(error,float(np.max(np.abs(v-a[k]))))
                counts['independent_trajectories']+=1
        counts['cases']+=1;print(f'Audit {counts["cases"]}/{len(run.old.case_list(c,smoke))}: {run.old.stem(case)}',flush=True)
    frame=pd.DataFrame(rows);saved=pd.read_csv(root/'raw-metrics.csv')
    pd.testing.assert_frame_equal(saved[frame.columns],frame,check_dtype=False,check_exact=False,rtol=1e-12,atol=1e-12)
    if not smoke:statistics(frame,c,root)
    prior=read(root/'prior-artifacts.json')
    for p,h in prior.items():assert core.sha256(p)==h,p
    core.write_json(out/'checks.json',dict(all_checks_pass=True,**counts,maximum_numeric_error=error,
        all_whole_trajectories_replayed=False,prior_files_unchanged=len(prior),result_manifest_sha256=core.sha256(root/'manifest.json'),budget=budget.close()))
    seal(out,c,m['context'],purpose='Independent relative-noise verification');print('Verification complete',counts,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args();verify(a.source,a.out)
