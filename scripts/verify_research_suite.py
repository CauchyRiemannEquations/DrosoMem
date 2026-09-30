"""Independent noise/head/trajectory/statistical reconstruction for the suite."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits
from flying.training import whole_brain_memory as core
from flying.data.sequences import SequenceDataset
from flying.training import sequence_memory as seq
from alphabet_memory import read,check
from pathway_memory import seal
from relative_noise import read_table
from frozen_state_probe import Budget
from verify_allocation_noise import independent_head,independent_noise
from verify_act5_robustness import metrics as audit_metrics
from verify_feedback_noise import reference,additional_metrics,first_error
from verify_neuron_panel import audit_stats
import research_suite as run


def teacher_reference(base,obs,symbols):
    w=base.weights;wm=w[base.mbon];alpha=base.leak;state=np.zeros(w.shape[0]);rows=[]
    for symbol in symbols:
        u=base.encoder(int(symbol));following=(1-alpha)*state+alpha*np.tanh(w.dot(state)+u)
        staged=state.copy();staged[base.kc]=following[base.kc]
        following[base.mbon]=(1-alpha)*state[base.mbon]+alpha*np.tanh(wm.dot(staged)+u[base.mbon])
        state=following;rows.append(state[obs].copy())
    return np.asarray(rows)


def graph_invariants(root,case,c):
    folder=Path(f"data/flywire_783_mb_left_kc512_s{case['circuit_seed']}")
    original,ids,_=core.load_connectome(folder);roles,_=core.load_roles(folder,ids);roles=np.asarray(roles)
    stored=sparse.load_npz(root/'raw-graph.npz');a=original.toarray();b=stored.toarray()
    assert a.shape==b.shape and np.count_nonzero(a)==np.count_nonzero(b) and not np.diag(b).any()
    for i in range(len(a)):np.testing.assert_array_equal(sorted(a[i,a[i]!=0]),sorted(b[i,b[i]!=0]))
    for sign in [-1,1]:
        for axis in [0,1]:np.testing.assert_array_equal((np.sign(a)==sign).sum(axis),(np.sign(b)==sign).sum(axis))
        if case['topology']=='role':
            for role in np.unique(roles):
                np.testing.assert_array_equal((np.sign(a[:,roles==role])==sign).sum(1),(np.sign(b[:,roles==role])==sign).sum(1))
                np.testing.assert_array_equal((np.sign(a[roles==role,:])==sign).sum(0),(np.sign(b[roles==role,:])==sign).sum(0))
    totals=np.abs(b).sum(1);expected=np.divide(.9*b,totals[:,None],out=np.zeros_like(b),where=totals[:,None]!=0)
    np.testing.assert_allclose(sparse.load_npz(root/'weights.npz').toarray(),expected,atol=1e-14,rtol=1e-12)


def gate(p,r,co):
    if not np.isfinite(r).all():return None
    need=4 if co=='discovery' else 3
    return bool(np.mean(p)>=2 and np.mean(r)>=.1 and np.sum(p>0)>=need and np.sum(r>0)>=need)


def statistics(frame,c,root,stage):
    summary=read(root/'summary.json');records=[];doses=[];pairs=[];gates={};vectors={}
    for co,blocks in c['cohorts'].items():
        families=['pi','random'] if stage=='structure' else c['families']
        for family in families:
            for level in (['legacy5'] if stage=='structure' else c['conditions']):
                for topology in (c['topologies'] if stage=='structure' else ['intact']):
                    for rho in (c['correlations'] if stage=='correlation' else [0.]):
                        group=dict(cohort=co,family=family,level=level,topology=topology,rho=rho)
                        base=frame.copy()
                        for k,v in group.items():base=base[base[k]==v]
                        arms={}
                        key='/'.join(map(str,group.values()))
                        for arm in ['flat','training_sd']:
                            pv=[];rv=[]
                            for seed,ds in blocks:
                                a=base[(base.seed==seed)&(base.arm==arm)]
                                assert len(a)==18 and set(a.dataset_seed)=={ds}
                                p=float(np.mean(a.exact_prefix_symbols.to_numpy()));r=float(np.mean(a.retention.to_numpy()))
                                pv.append(p);rv.append(r);records.append(dict(**group,seed=seed,arm=arm,exact_prefix_symbols=p,retention=r))
                                for dose in c['relative_strengths']:
                                    z=a[a.strength==dose]
                                    doses.append(dict(**group,seed=seed,arm=arm,strength=dose,
                                        exact_prefix_symbols=float(np.mean(z.exact_prefix_symbols.to_numpy())),retention=float(np.mean(z.retention.to_numpy()))))
                            arms[arm]=(np.array(pv),np.array(rv));vectors[(co,family,level,topology,rho,arm)]=arms[arm]
                            for metric,data in zip(run.METRICS,arms[arm]):
                                if np.isfinite(data).all():audit_stats(data,summary['statistics'][key][arm][metric],c)
                                else:assert summary['statistics'][key][arm][metric] is None
                        p=arms['training_sd'][0]-arms['flat'][0];r=arms['training_sd'][1]-arms['flat'][1]
                        for metric,data in zip(run.METRICS,[p,r]):
                            if np.isfinite(data).all():audit_stats(data,summary['statistics'][key][metric],c)
                            else:assert summary['statistics'][key][metric] is None
                        gates[key]=gate(p,r,co)
                        pairs.extend(dict(**group,seed=b[0],exact_prefix_symbols=float(p[i]),retention=float(r[i])) for i,b in enumerate(blocks))
    sgate={};spairs=[]
    if stage=='structure':
        for co,blocks in c['cohorts'].items():
            for family in ['pi','random']:
                base=vectors[(co,family,'legacy5','intact',0.,'training_sd')]
                for topology in ['degree','role']:
                    other=vectors[(co,family,'legacy5',topology,0.,'training_sd')];p=base[0]-other[0];r=base[1]-other[1]
                    key=f'{co}/{family}/{topology}';sgate[key]=gate(p,r,co)
                    for metric,data in zip(run.METRICS,[p,r]):
                        if np.isfinite(data).all():audit_stats(data,summary['structural_statistics'][key][metric],c)
                        else:assert summary['structural_statistics'][key][metric] is None
                    spairs.extend(dict(cohort=co,family=family,control=topology,seed=b[0],exact_prefix_symbols=float(p[i]),retention=float(r[i])) for i,b in enumerate(blocks))
    tables=[('seed-blocks.csv',records,run.GROUP+['seed','arm']),('paired-differences.csv',pairs,run.GROUP+['seed']),
            ('dose-seed-table.csv',doses,run.GROUP+['seed','arm','strength'])]
    if spairs:tables.append(('structural-paired-differences.csv',spairs,['cohort','family','control','seed']))
    for name,rows,keys in tables:
        expected=pd.DataFrame(rows).sort_values(keys).reset_index(drop=True);actual=read_table(root/name).sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(actual[expected.columns],expected,check_exact=False,atol=1e-12,rtol=1e-12)
    assert summary['gates']==gates and summary['structural_gates']==sgate
    if stage=='sequence':keys=[f'{co}/random/brain1/intact/0.0' for co in c['cohorts']]
    elif stage=='correlation':keys=[f'{co}/random/brain1/intact/{rho}' for co in c['cohorts'] for rho in [.25,.75]]
    else:keys=[f'{co}/{family}/role' for co in c['cohorts'] for family in ['pi','random']]
    decision=sgate if stage=='structure' else gates;vals=[decision[k] for k in keys]
    assert summary['primary_keys']==keys and summary['primary_confirmed']==(None if any(v is None for v in vals) else all(vals))
    assert not summary['cohorts_pooled'] and summary['undefined_retention_propagated']


@threadpool_limits.wrap(limits=1)
def verify(root,out):
    m=check(root);c=m['config'];out.mkdir(parents=True,exist_ok=False);current=run.context(c)
    for k in ['source_sha256','adapter_sha256','config_sha256','base_config_sha256']:assert current[k]==m['context'][k],k
    v=read(root/'verification.json');stage=v['stage'];smoke=v['smoke'];h=8 if smoke else 197
    budget=Budget(dict(max_seconds=c['verification_seconds'],max_rss_bytes=c['worker_rss_bytes']))
    try:
        counts=dict(cases=0,certificates=0,actual_paths_checked=0,independent_full_paths=0,independent_teacher_paths=0,
                    exact_refits=0,structural_audits=0,reused_rho0_certificates=0)
        allcert=[];allroll=[];maxerr=0.;minmargin=1.;banks={};identities={}
        for case in run.cases(c,stage,smoke):
            budget.check();path=root/run.stem(case);check(path);meta=read(path/'case.json');assert meta['case']==case
            source=Path(meta['parent']) if meta['parent'] else None
            if source:assert core.sha256(source/'manifest.json')==meta['parent_manifest_sha256'];check(source)
            cp_path=source/'checkpoint.npz' if stage=='correlation' else path/'checkpoint.npz'
            with np.load(cp_path) as a:cp=dict(a)
            for value in cp.values():assert np.isfinite(value).all()
            dataset=SequenceDataset(case['family'],case['dataset_seed'],**read(c['base_config'])['dataset'])
            np.testing.assert_array_equal(cp['symbols'],dataset.symbols())
            import json
            assert json.loads(json.dumps(dataset.identity(cp['symbols'])))==meta['dataset']
            head=run.feedback.load_head(cp,case['seed']);assert head.digest()==meta['head_sha256'] and head.parameter_count==482
            teacher=cp['features'][2:2+h];target=cp['symbols'][3:3+h]
            clean_path=path/'clean.npz' if (path/'clean.npz').exists() else source/'clean.npz'
            with np.load(clean_path) as a:clean={k:a[k][:h] for k in a}
            clean_prefix=first_error(clean['prediction'],target);assert clean_prefix==meta['clean_prefix']
            if source and stage=='sequence':
                with np.load(source/'checkpoint.npz') as old:
                    for key in ['mean','scale','w1','b1','w2','b2','symbols','features']:np.testing.assert_array_equal(cp[key],old[key])
                    for k,t in [('prediction','prediction'),('probabilities','probabilities'),('features','recall_features')]:np.testing.assert_array_equal(clean[k],old[t][:h])
            elif source:
                with np.load(source/'checkpoint.npz') as old:
                    for key in ['mean','scale','w1','b1','w2','b2','symbols','features']:np.testing.assert_array_equal(cp[key],old[key])
                with np.load(source/'clean.npz') as old:
                    for key in clean:np.testing.assert_array_equal(clean[key],old[key][:h])
            ids=np.array(meta['graph']['observation_root_ids'],dtype=np.int64)
            z=np.stack([independent_noise(ids,case['seed'],case['circuit_seed'],ns,h) for ns in c['noise_seeds']])
            common=np.stack([np.random.Generator(np.random.PCG64(np.random.SeedSequence([ns,case['seed'],case['circuit_seed'],3]))).normal(size=h) for ns in c['common_noise_seeds']])
            key=(case['seed'],case['circuit_seed']);identity={k:meta['graph'][k] for k in ['input_root_ids','observation_root_ids','input_mapping_sha256']}
            assert identities.setdefault(key,identity)==identity
            if key in banks:np.testing.assert_array_equal(z,banks[key][0]);np.testing.assert_array_equal(common,banks[key][1])
            banks[key]=(z,common)
            sd=np.sqrt(np.mean((cp['features']-cp['features'].mean(0))**2,axis=0));own_q=float(np.median(sd));weights=np.sqrt(48)*sd/np.linalg.norm(sd);q=own_q
            if stage=='structure':
                with np.load(Path(c['parent'])/run.parent_stem(case)/'checkpoint.npz') as a:q=float(np.median(np.sqrt(np.mean((a['features']-a['features'].mean(0))**2,axis=0))))
            np.testing.assert_allclose([own_q,q],[meta['own_q'],meta['used_q']],atol=1e-18,rtol=1e-12)
            cert=read_table(path/'certificates.csv');assert len(cert)==(54 if stage=='correlation' else 18)
            allcert.extend(cert.to_dict('records'));rhos=c['correlations'] if stage=='correlation' else [0.];refs={}
            with np.load(path/'certificates.npz') as saved:
                np.testing.assert_array_equal(saved['target'],target);np.testing.assert_array_equal(saved['observed_root_ids'],ids)
                np.testing.assert_array_equal(saved['standard_normals'],z);np.testing.assert_array_equal(saved['common_normals'],common)
                np.testing.assert_allclose(saved['training_sd'],sd,atol=1e-18,rtol=1e-12);np.testing.assert_allclose(saved['allocation_weights'],weights,atol=1e-12,rtol=1e-12)
                for row in cert.to_dict('records'):
                    for k,vv in case.items():assert row[k]==vv
                    ri=rhos.index(row['rho']);ni=c['noise_seeds'].index(row['noise_seed']);di=c['relative_strengths'].index(row['strength']);ai=int(row['arm']=='training_sd')
                    noise=np.sqrt(1-row['rho'])*z[ni]+np.sqrt(row['rho'])*common[ni,:,None]
                    amp=q*row['strength']*(weights if ai else np.ones(48));x=np.maximum(-1,np.minimum(1,teacher+noise*amp));p=independent_head(x,cp);pred=p.argmax(1)
                    np.testing.assert_allclose(saved['amplitudes'][di,ai],amp,atol=1e-18,rtol=1e-12)
                    actual=saved['probabilities'][ri,ni,di,ai];np.testing.assert_allclose(actual,p,atol=1e-12,rtol=1e-10);maxerr=max(maxerr,float(np.max(abs(actual-p))))
                    np.testing.assert_array_equal(saved['prediction'][ri,ni,di,ai],pred);np.testing.assert_array_equal(saved['position_accuracy'][ri,ni,di,ai],pred==target)
                    ordered=np.sort(p,axis=1);minmargin=min(minmargin,float(np.min(ordered[:,-1]-ordered[:,-2])))
                    score=first_error(pred,target);assert row['exact_prefix_symbols']==score and row['clean_prefix']==clean_prefix
                    assert row['censored']==(score==h)
                    if score<h:assert row['first_error_position']==score+1
                    else:assert pd.isna(row['first_error_position'])
                    if clean_prefix:np.testing.assert_allclose(row['retention'],min(score/clean_prefix,1),atol=1e-12,rtol=1e-12)
                    else:assert pd.isna(row['retention'])
                    values=dict(exact_prefix_bits=score*np.log2(10),teacher_accuracy=np.mean(pred==target),mean_confidence=p.max(1).mean(),
                        target_probability=p[np.arange(h),target].mean(),expected_energy=48*(q*row['strength'])**2,
                        injected_energy=np.mean(np.sum((noise*amp)**2,axis=1)),clipped_energy=np.mean(np.sum((x-teacher)**2,axis=1)),
                        clipping_fraction=np.mean(np.abs(teacher+noise*amp)>1))
                    for k,value in values.items():np.testing.assert_allclose(row[k],value,atol=1e-12,rtol=1e-10)
                    assert row['reused_exposure']==(stage=='correlation' and row['rho']==0.)
                    if row['reused_exposure']:counts['reused_rho0_certificates']+=1
                    if ni==0:refs[(row['rho'],row['strength'],row['arm'])]=(x,p,pred,noise,amp)
                    counts['certificates']+=1
                if stage=='correlation':
                    with np.load(source/'certificates.npz') as old:np.testing.assert_array_equal(saved['probabilities'][0],old['probabilities'][0,:,:,:,:h])
            independent=run.independently_selected(c,case,stage,smoke)
            if independent:
                base,obs,graph,raw=run.build(c,case);assert graph==meta['graph']
                reference_teacher=teacher_reference(base,obs,cp['symbols'][:-1])
                np.testing.assert_allclose(reference_teacher,cp['features'],atol=1e-12,rtol=1e-10);counts['independent_teacher_paths']+=1
                if raw is not None:
                    assert core.weight_hash(sparse.load_npz(path/'raw-graph.npz'))==core.weight_hash(raw)
                    graph_invariants(path,case,c);counts['structural_audits']+=1
            old_refit=stage=='sequence' and case['cohort']=='discovery' and case['family']=='random' and case['seed']==9142 and case['circuit_seed']==701
            if meta['fresh_fit'] or old_refit:
                fitted,_,_=seq.fit(cp['features'],cp['symbols'],core.NetworkCondition(case['level'],case['circuit_seed'],case['seed']),read(c['base_config']))
                assert fitted.digest()==head.digest();counts['exact_refits']+=1
            loss,_,train=head.objective(cp['features'],cp['symbols'][1:],1e-5,core.sample_weights(199,3,32,4))
            assert loss==meta['train_loss'] and train['cross_entropy']==meta['weighted_cross_entropy']
            assert np.mean(head.predict(cp['features'])==cp['symbols'][1:])==meta['training_accuracy']
            if (path/'rollouts.csv').exists():
                rows=read_table(path/'rollouts.csv').to_dict('records');allroll.extend(rows)
                for row in rows:
                    with np.load(path/row['artifact']) as a:actual=dict(a)
                    for value in actual.values():assert np.isfinite(value).all()
                    proxy=dict(row,pi_memory_score=row['exact_prefix_symbols']);audit_metrics(actual,target,proxy,clean,head.scale,meta['graph']['neurons']);additional_metrics(actual,target,proxy)
                    if row['arm']=='clean':x=teacher;p=independent_head(x,cp);pred=p.argmax(1);noise=z[0];amp=np.zeros(48)
                    else:x,p,pred,noise,amp=refs[(row['rho'],row['strength'],row['arm'])]
                    score=first_error(pred,target);stop=min(score+1,h)
                    assert first_error(actual['prediction'],target)==score
                    np.testing.assert_array_equal(actual['prediction'][:stop],pred[:stop])
                    for k,value in [('state_features',teacher),('features',x),('probabilities',p)]:np.testing.assert_allclose(actual[k][:stop],value[:stop],atol=1e-12,rtol=1e-10)
                    expected=np.maximum(-1,np.minimum(1,actual['state_features']+noise*amp))
                    np.testing.assert_allclose(actual['features'],expected,atol=1e-12,rtol=1e-10)
                    np.testing.assert_allclose(actual['probabilities'],independent_head(expected,cp),atol=1e-12,rtol=1e-10)
                    np.testing.assert_allclose(actual['injected_energy'],np.sum((noise*amp)**2,axis=1),atol=1e-12,rtol=1e-10)
                    np.testing.assert_allclose(actual['clipped_energy'],np.sum((expected-actual['state_features'])**2,axis=1),atol=1e-12,rtol=1e-10)
                    np.testing.assert_array_equal(actual['clipped'],np.sum(np.abs(actual['state_features']+noise*amp)>1,axis=1))
                    if independent:
                        ref=reference(base,obs,cp,cp['symbols'][:3],h,noise,amp)
                        for k,value in ref.items():
                            if k in ['prediction','active_counts','clipped']:np.testing.assert_array_equal(actual[k],value)
                            else:np.testing.assert_allclose(actual[k],value,atol=1e-12,rtol=1e-10);maxerr=max(maxerr,float(np.max(abs(actual[k]-value))))
                        counts['independent_full_paths']+=1
                    counts['actual_paths_checked']+=1
            counts['cases']+=1;print(f'{stage} audit {counts["cases"]}/{v["cases"]}: {run.stem(case)}',flush=True)
        for name,rows in [('raw-certificates.csv',allcert),('raw-rollouts.csv',allroll)]:
            expected=pd.DataFrame(rows);actual=read_table(root/name)
            pd.testing.assert_frame_equal(actual[expected.columns],expected,check_exact=False,atol=1e-12,rtol=1e-12)
        if not smoke:statistics(pd.DataFrame(allcert),c,root,stage)
        assert counts['actual_paths_checked']==v['actual_paths']==v['pre_error_checks']
        assert counts['certificates']==v['certificates'] and counts['cases']==v['cases']
        prior=read(root/'prior-artifacts.json')
        for path,digest in prior.items():assert core.sha256(path)==digest,path
        budget.check()
        core.write_json(out/'checks.json',dict(all_checks_pass=True,stage=stage,smoke=smoke,**counts,
            maximum_numeric_error=maxerr,min_teacher_probability_margin=minmargin,prior_files_unchanged=len(prior),
            result_manifest_sha256=core.sha256(root/'manifest.json'),budget=budget.close()))
        seal(out,c,m['context'],purpose='Independent '+stage+' verification')
    except Exception as exc:
        core.write_json(out/'failure.json',dict(error=repr(exc),budget=budget.close()));seal(out,c,m['context'],purpose='Preserved independent-verification failure');raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args();verify(a.source,a.out)
