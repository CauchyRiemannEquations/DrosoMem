"""Independent prefix/head audit and preregistered full autonomous replay subset."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from flying.training import whole_brain_memory as core
from alphabet_memory import read,check
from pathway_memory import seal
from frozen_state_probe import Budget
from relative_noise import read_table
from verify_neuron_panel import audit_stats
from verify_allocation_noise import independent_noise,independent_head
from verify_act5_robustness import metrics as audit_metrics
import feedback_noise as run


def first_error(prediction,target):
    for i,(pred,truth) in enumerate(zip(prediction,target)):
        if pred!=truth:return i
    return len(target)


def reference(base,obs,cp,prompt,horizon,noise,amplitudes):
    w=base.weights;wm=w[base.mbon];alpha=base.leak;state=np.zeros(w.shape[0])
    def step(symbol,current):
        stimulation=base.encoder(int(symbol))
        following=(1-alpha)*current+alpha*np.tanh(w.dot(current)+stimulation)
        staged=current.copy();staged[base.kc]=following[base.kc]
        following[base.mbon]=(1-alpha)*current[base.mbon]+alpha*np.tanh(wm.dot(staged)+stimulation[base.mbon])
        return following
    for symbol in prompt:state=step(symbol,state)
    result={k:[] for k in ['prediction','probabilities','features','state_features','active_counts','state_norm','clipped','injected_energy','clipped_energy']}
    for t in range(horizon):
        raw=state[obs].copy();delta=np.multiply(noise[t],amplitudes);x=np.maximum(-1,np.minimum(1,raw+delta))
        prob=independent_head(x[None,:],cp)[0];digit=int(np.argmax(prob))
        values=[digit,prob,x,raw,np.count_nonzero(np.abs(state)>1e-8),np.linalg.norm(state),
                np.count_nonzero((raw+delta>1)|(raw+delta< -1)),np.sum(delta**2),np.sum((x-raw)**2)]
        for key,value in zip(result,values):result[key].append(value)
        state=step(digit,state)
    return {k:np.asarray(v) for k,v in result.items()}


def additional_metrics(a,target,row):
    x=a['state_features'];sing=np.linalg.svd(x-x.mean(0),compute_uv=False)
    p=sing[sing>0]/sing.sum() if sing.sum() else np.array([])
    rank=float(np.exp(-np.sum(p*np.log(p)))) if len(p) else 0.
    cos=[]
    for left,right in zip(x[:-1],x[1:]):
        den=np.linalg.norm(left)*np.linalg.norm(right);cos.append(float(left@right/den) if den else 0.)
    expected=dict(raw_state_effective_rank=rank,raw_mbon_mean_abs=float(np.abs(x).mean()),
        raw_mbon_sparsity=float(np.mean(np.abs(x)<=1e-8)),raw_temporal_cosine=float(np.mean(cos)),
        mean_injected_energy=float(np.mean(a['injected_energy'])),mean_clipped_energy=float(np.mean(a['clipped_energy'])))
    start=first_error(a['prediction'],target)+1
    if start<len(target):expected['post_error_accuracy']=float(np.mean(a['prediction'][start:]==target[start:]))
    else:assert pd.isna(row['post_error_accuracy'])
    for k,value in expected.items():np.testing.assert_allclose(row[k],value,atol=1e-11,rtol=1e-10,err_msg=k)


def stats(f,c,root):
    s=read(root/'summary.json');blocks=[];paired=[];gates={};doses=[]
    for co,seeds in c['cohorts'].items():
        for stage in ['archived','fresh']:
            for g in c['conditions']:
                armvectors={}
                for arm in ['flat','training_sd']:
                    pv=[];rv=[]
                    for seed in seeds:
                        a=f[(f.cohort==co)&(f.stage==stage)&(f.level==g)&(f.arm==arm)&(f.seed==seed)]
                        assert len(a)==(24 if stage=='archived' else 18)
                        pv.append(float(np.mean(a.pi_memory_score.to_numpy())));rv.append(float(np.mean(a.retention.to_numpy())))
                        blocks.append(dict(cohort=co,stage=stage,level=g,seed=seed,arm=arm,pi_memory_score=pv[-1],retention=rv[-1]))
                        for r in c['relative_strengths']:
                            z=a[a.strength==r];doses.append(dict(cohort=co,stage=stage,level=g,seed=seed,arm=arm,strength=r,
                                pi_memory_score=float(np.mean(z.pi_memory_score.to_numpy())),retention=float(np.mean(z.retention.to_numpy()))))
                    armvectors[arm]=(np.array(pv),np.array(rv));saved=s['statistics'][f'{co}/{stage}/{g}'][arm]
                    audit_stats(pv,saved['pi_memory_score'],c)
                    if np.isfinite(rv).all():audit_stats(rv,saved['retention'],c)
                    else:assert saved['retention'] is None
                p=armvectors['training_sd'][0]-armvectors['flat'][0];r=armvectors['training_sd'][1]-armvectors['flat'][1]
                key=f'{co}/{stage}/{g}';audit_stats(p,s['statistics'][key]['pi_memory_score'],c)
                if np.isfinite(r).all():
                    audit_stats(r,s['statistics'][key]['retention'],c)
                    need=4 if co=='discovery' else 3
                    gates[key]=bool(np.mean(p)>=2 and np.mean(r)>=.1 and np.count_nonzero(p>0)>=need and np.count_nonzero(r>0)>=need)
                else:gates[key]=None;assert s['statistics'][key]['retention'] is None
                paired.extend(dict(cohort=co,stage=stage,level=g,seed=seed,pi_memory_score=float(p[i]),retention=float(r[i])) for i,seed in enumerate(seeds))
    for filename,rows,keys in [('seed-blocks.csv',blocks,['cohort','stage','level','seed','arm']),
                              ('paired-differences.csv',paired,['cohort','stage','level','seed']),
                              ('dose-seed-table.csv',doses,['cohort','stage','level','seed','arm','strength'])]:
        expected=pd.DataFrame(rows).sort_values(keys).reset_index(drop=True);saved=read_table(root/filename).sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(saved[expected.columns],expected,check_exact=False,atol=1e-12,rtol=1e-12)
    confirmed=[g for g in c['conditions'] if all(gates[f'{co}/fresh/{g}'] is True for co in c['cohorts'])]
    assert gates==s['descriptive_gates'] and confirmed==s['confirmed'] and s['primary_confirmed']==('brain1' in confirmed)
    assert s['decision_stage']=='fresh' and not s['fresh_model_cohort'] and s['prefix_certificates_are_not_full_rollouts']


@threadpool_limits.wrap(limits=1)
def verify(root,out):
    m=check(root);c=m['config'];out.mkdir(parents=True,exist_ok=False);current=run.context(c)
    for key in ['source_sha256','adapter_sha256','config_sha256','base_config_sha256']:assert current[key]==m['context'][key]
    budget=Budget(dict(max_seconds=c['verification_seconds'],max_rss_bytes=c['worker_rss_bytes']))
    try:
        for p,digest in [(c['source'],c['source_manifest_sha256']),(c['allocation_source'],c['allocation_manifest_sha256'])]:
            assert core.sha256(Path(p)/'manifest.json')==digest;check(Path(p))
        smoke=read(root/'verification.json')['smoke'];h=8 if smoke else 197
        certrows=[];rollrows=[];count=replays=checks=0;error=0.;banks={}
        for number,case in enumerate(run.cases(c,smoke),1):
            budget.check();path=root/run.old.stem(case);check(path);meta=read(path/'case.json')
            src=Path(c['source'])/run.old.stem(case);arc=Path(c['allocation_source'])/run.old.stem(case)
            assert meta['source_manifest_sha256']==core.sha256(src/'manifest.json')
            assert meta['allocation_manifest_sha256']==core.sha256(arc/'manifest.json')
            with np.load(src/'checkpoint.npz') as z:cp=dict(z)
            target=cp['digits'][3:3+h];np.testing.assert_array_equal(cp['digits'],core.SequenceDataset().symbols())
            head=run.load_head(cp,case['seed']);assert head.digest()==meta['head_sha256']
            with np.load(src/'dose0_teacher.npz') as z:clean_teacher=z['features'][:h];clean_teacher_probs=z['probabilities'][:h];clean_teacher_pred=z['prediction'][:h]
            with np.load(path/'clean.npz') as z:clean=dict(z)
            with np.load(src/'dose0_autonomous.npz') as z:
                for key in ['prediction','features','probabilities','active_counts','state_norm']:np.testing.assert_array_equal(clean[key],z[key][:h])
            clean_prefix=first_error(clean['prediction'],target);assert clean_prefix==meta['clean_prefix']
            train=cp['features'];sd=np.sqrt(np.sum((train-train.mean(0))**2,axis=0)/len(train))
            q=float(np.median(sd));weights=np.sqrt(48)*sd/np.linalg.norm(sd)
            ids=np.asarray(meta['graph']['observation_root_ids'],dtype=np.int64)
            noise=np.stack([independent_noise(ids,case['seed'],case['circuit_seed'],ns,h) for ns in c['fresh_noise_seeds']])
            key=(case['seed'],case['circuit_seed'])
            if key in banks:
                np.testing.assert_array_equal(ids,banks[key][0]);np.testing.assert_array_equal(noise,banks[key][1])
            banks[key]=(ids,noise)
            cert=read_table(path/'certificates.csv');roll=read_table(path/'rollouts.csv');assert len(cert)==42 and len(roll)==7
            certrows.extend(cert.to_dict('records'));rollrows.extend(roll.to_dict('records'))
            teacher={}
            with np.load(path/'fresh-teacher.npz') as fresh,np.load(arc/'evaluations.npz') as archived:
                np.testing.assert_array_equal(fresh['standard_normals'],noise);np.testing.assert_array_equal(fresh['observed_root_ids'],ids)
                np.testing.assert_array_equal(fresh['target'],target);np.testing.assert_allclose(fresh['training_sd'],sd,atol=1e-18,rtol=1e-12)
                np.testing.assert_allclose(fresh['allocation_weights'],weights,atol=1e-12,rtol=1e-12)
                for row in cert.to_dict('records'):
                    for key,val in case.items():assert row[key]==val
                    j=c['relative_strengths'].index(row['strength']);a=0 if row['arm']=='flat' else 1
                    if row['stage']=='archived':
                        k=c['archived_noise_seeds'].index(row['noise_seed']);pred=archived['predictions'][k,j,a,:h]
                        assert row['endpoint_origin']=='archived_teacher_prefix_certificate'
                    else:
                        k=c['fresh_noise_seeds'].index(row['noise_seed']);amp=q*row['strength']*(np.ones(48) if a==0 else weights)
                        x=np.maximum(-1,np.minimum(1,clean_teacher+noise[k]*amp));p=independent_head(x,cp);pred=p.argmax(1)
                        np.testing.assert_allclose(fresh['amplitudes'][j,a],amp,atol=1e-18,rtol=1e-12)
                        np.testing.assert_allclose(np.sum(amp**2),48*(q*row['strength'])**2,atol=1e-18,rtol=1e-12)
                        for actual,ref in [(fresh['features'][k,j,a],x),(fresh['probabilities'][k,j,a],p)]:
                            np.testing.assert_allclose(actual,ref,atol=1e-12,rtol=1e-10);error=max(error,float(np.max(np.abs(actual-ref))))
                        np.testing.assert_array_equal(fresh['prediction'][k,j,a],pred);np.testing.assert_array_equal(fresh['position_accuracy'][k,j,a],pred==target)
                        if k==0:teacher[(row['strength'],row['arm'])]=(x,p,pred,amp)
                        assert row['endpoint_origin']=='fresh_teacher_prefix_certificate'
                    score=first_error(pred,target);assert row['pi_memory_score']==score and row['clean_prefix']==clean_prefix
                    np.testing.assert_allclose(row['exact_prefix_bits'],score*np.log2(10),atol=1e-12,rtol=1e-12)
                    if clean_prefix:np.testing.assert_allclose(row['retention'],min(score/clean_prefix,1),atol=1e-12,rtol=1e-12)
                    else:assert pd.isna(row['retention'])
                    assert row['censored']==(score==h)
                    if score<h:assert row['first_error_position']==score+1
                    else:assert pd.isna(row['first_error_position'])
                    count+=1
            selected=smoke or case['level']=='legacy5' or (case['seed'],case['circuit_seed']) in [(7142,701),(381142,702)]
            if selected:
                base,obs,graph=core.build_model(c['cache'],core.NetworkCondition(case['level'],case['circuit_seed'],case['seed']),read(c['base_config']))
                assert graph==meta['graph'] and core.weight_hash(base.weights)==meta['weight_sha256']
            for row in roll.to_dict('records'):
                with np.load(path/row['artifact']) as z:aout=dict(z)
                for key in aout:assert np.isfinite(aout[key]).all()
                for key,val in case.items():assert row[key]==val
                audit_metrics(aout,target,row,clean,head.scale,meta['graph']['neurons']);additional_metrics(aout,target,row)
                p=independent_head(aout['features'],cp);np.testing.assert_allclose(aout['probabilities'],p,atol=1e-12,rtol=1e-10)
                if row['arm']=='clean':x,prob,pred,amp=clean_teacher,clean_teacher_probs,clean_teacher_pred,np.zeros(48)
                else:
                    x,prob,pred,amp=teacher[(row['strength'],row['arm'])]
                    np.testing.assert_allclose(row['teacher_accuracy'],np.mean(pred==target),atol=1e-12,rtol=1e-12)
                score=first_error(pred,target);stop=min(score+1,h)
                assert first_error(aout['prediction'],target)==score
                np.testing.assert_array_equal(aout['prediction'][:stop],pred[:stop])
                np.testing.assert_allclose(aout['features'][:stop],x[:stop],atol=1e-12,rtol=1e-10)
                np.testing.assert_allclose(aout['probabilities'][:stop],prob[:stop],atol=1e-12,rtol=1e-10)
                np.testing.assert_allclose(aout['state_features'][:stop],clean_teacher[:stop],atol=1e-12,rtol=1e-10);checks+=1
                delta=noise[0]*amp;expected=np.maximum(-1,np.minimum(1,aout['state_features']+delta))
                np.testing.assert_allclose(aout['features'],expected,atol=1e-12,rtol=1e-10)
                np.testing.assert_allclose(aout['injected_energy'],np.sum(delta**2,axis=1),atol=1e-12,rtol=1e-10)
                np.testing.assert_allclose(aout['clipped_energy'],np.sum((expected-aout['state_features'])**2,axis=1),atol=1e-12,rtol=1e-10)
                np.testing.assert_array_equal(aout['clipped'],np.sum(np.abs(aout['state_features']+delta)>1,axis=1))
                if selected:
                    ref=reference(base,obs,cp,cp['digits'][:3],h,noise[0],amp)
                    for key,value in ref.items():
                        if key in ['prediction','active_counts','clipped']:np.testing.assert_array_equal(aout[key],value)
                        else:
                            np.testing.assert_allclose(aout[key],value,atol=1e-12,rtol=1e-10,err_msg=key)
                            error=max(error,float(np.max(np.abs(aout[key]-value))))
                    replays+=1
            print(f'Audit {number}/{len(run.cases(c,smoke))}: {run.old.stem(case)}',flush=True)
        f=pd.DataFrame(certrows);r=pd.DataFrame(rollrows)
        for actual,name in [(f,'raw-certificates.csv'),(r,'raw-rollouts.csv')]:
            pd.testing.assert_frame_equal(read_table(root/name)[actual.columns],actual,check_exact=False,atol=1e-12,rtol=1e-12)
        if not smoke:
            stats(f,c,root)
            expected=r[r.arm!='clean'].groupby(['cohort','level','seed','arm','strength'])[['pi_memory_score','accuracy','teacher_accuracy','post_error_accuracy']].mean().reset_index()
            pd.testing.assert_frame_equal(read_table(root/'rollout-seed-table.csv')[expected.columns],expected,check_exact=False,atol=1e-12,rtol=1e-12)
        prior=read(root/'prior-artifacts.json')
        for p,digest in prior.items():assert core.sha256(p)==digest,p
        budget.check()
        core.write_json(out/'checks.json',dict(all_checks_pass=True,smoke=smoke,cases=len(run.cases(c,smoke)),certificates=count,
            actual_rollouts_checked=checks,independent_full_rollouts=replays,all_whole_rollouts_independently_replayed=smoke,
            maximum_numeric_error=error,prior_files_unchanged=len(prior),result_manifest_sha256=core.sha256(root/'manifest.json'),budget=budget.close()))
        seal(out,c,m['context'],purpose='Independent certificate, metrics and full-rollout subset audit')
    except Exception as exc:
        core.write_json(out/'failure.json',dict(error=repr(exc),budget=budget.close()))
        seal(out,c,m['context'],purpose='Preserved feedback validation failure');raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args();verify(a.source,a.out)
