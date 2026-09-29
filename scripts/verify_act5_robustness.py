"""Independent scalar replay subset plus complete metric/hash/decision audit."""
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
import act5_robustness as run


def reference(base,obs,head,prompt,horizon,mapping,w,dead,kind,strength,random,feed=None):
    state=np.zeros(w.shape[0]);kc=base.kc;mb=base.mbon;wm=w[mb];alpha=base.leak
    def step(symbol,state):
        u=base.encoder(int(symbol));next_state=(1-alpha)*state+alpha*np.tanh(w.dot(state)+u)
        next_state[dead]=0;context=state.copy();context[kc]=next_state[kc]
        next_state[mb]=(1-alpha)*state[mb]+alpha*np.tanh(wm.dot(context)+u[mb])
        next_state[dead]=0;return next_state
    for symbol in prompt:state=step(symbol,state)
    rows={k:[] for k in ['prediction','probabilities','features','active_counts','state_norm','clipped']}
    for i in range(horizon):
        clipped=0
        if strength>0 and (kind=='ongoing' or kind=='pulse' and i==0):
            delta=random.standard_normal(mapping['universe_size'])[mapping['node_ix']]*strength
            state=state+delta;clipped=np.sum(np.abs(state)>1);state=np.minimum(1,np.maximum(-1,state))
        observed=state[obs];z=((observed-head.mean)/head.scale)[None,:]
        params=head.parameters;logits=(np.tanh(z.dot(params['w1'])+params['b1']).dot(params['w2'])+params['b2'])[0]
        ex=np.exp(logits-logits.max());prob=ex/ex.sum();digit=int(logits.argmax())
        values=[digit,prob,observed.copy(),np.sum(np.abs(state)>1e-8),np.linalg.norm(state),clipped]
        for key,value in zip(rows,values):rows[key].append(value)
        state=step(digit if feed is None else feed[i],state)
    return {k:np.asarray(v) for k,v in rows.items()}


def independent_alter(w,draws,kind,strength):
    modified=w.copy();dead=np.zeros(w.shape[0],bool)
    if kind=='edge_dropout':modified.data[np.less(draws['edge'],strength)]=0;modified.eliminate_zeros()
    if kind=='neuron_dropout':dead=np.less(draws['neuron'],strength)
    if kind=='weight_noise':modified.data=modified.data*np.exp(strength*draws['weight']-strength*strength/2)
    return modified,dead


def metrics(a,target,row,clean,scale,n):
    ok=np.equal(a['prediction'],target);np.testing.assert_array_equal(a['position_accuracy'],ok)
    errors=np.where(~ok)[0];score=int(errors[0]) if len(errors) else len(ok)
    x=a['features'];sing=np.linalg.svd(x-x.mean(0),compute_uv=False);p=sing[sing>0]/sing.sum() if sing.sum() else []
    rank=float(np.exp(-np.sum(p*np.log(p)))) if len(p) else 0.
    cos=[]
    for left,right in zip(x[:-1],x[1:]):
        denominator=np.linalg.norm(left)*np.linalg.norm(right);cos.append(float(left@right/denominator) if denominator else 0.)
    values=dict(pi_memory_score=score,exact_prefix_bits=score*np.log2(10),accuracy=ok.mean(),
        prefix32=score>=32,censored=ok.all(),effective_rank=rank,active_neurons_mean=a['active_counts'].mean(),
        activity_sparsity=1-a['active_counts'].mean()/n,mbon_mean_abs=np.abs(x).mean(),mbon_sparsity=(np.abs(x)<=1e-8).mean(),
        temporal_cosine=np.mean(cos),standardized_clean_mse=np.mean(((x-clean['features'])/scale)**2),
        mean_target_probability=np.mean([q[t] for q,t in zip(a['probabilities'],target)]),
        mean_confidence=a['probabilities'].max(1).mean(),clipped_coordinates=a['clipped'].sum())
    for lo,hi in [(0,32),(32,64),(64,128),(128,len(target))]:
        if lo<len(target) and hi>lo:values[f'accuracy_{lo+1}_{min(hi,len(target))}']=ok[lo:hi].mean()
    for k,v in values.items():np.testing.assert_allclose(row[k],v,atol=1e-11,rtol=1e-10,err_msg=k)
    if len(errors):assert row['first_error_position']==score+1
    else:assert pd.isna(row['first_error_position'])
    np.testing.assert_allclose(a['probabilities'].sum(1),1,atol=1e-14)
    np.testing.assert_array_equal(a['prediction'],a['probabilities'].argmax(1))


def statistics(f,c,root):
    summary=read(root/'summary.json');expected=[];absolute={};comparisons={}
    for co,seeds in c['cohorts'].items():
        need=4 if co=='discovery' else 3
        for kind in run.FAMILIES:
            vectors={}
            for g in c['conditions']:
                rs=[];ps=[];hi=[];eligible=[];bad=False
                for seed in seeds:
                    circuit_r=[];circuit_p=[];circuit_hi=[]
                    for ci in c['circuit_seeds']:
                        subset=f[(f.cohort==co)&(f.seed==seed)&(f.circuit_seed==ci)&(f.level==g)&(f['mode']=='autonomous')]
                        clean=float(subset[subset.kind=='clean'].pi_memory_score.iloc[0])
                        doses=subset[(subset.kind==kind)&(subset.strength>0)].sort_values('strength')
                        assert doses.strength.tolist()==c['strengths'][kind]
                        bad|=clean==0;scores=doses.pi_memory_score.to_numpy()
                        ret=np.minimum(scores/clean,1) if clean else np.full(3,np.nan)
                        circuit_r.append(ret.mean());circuit_p.append(scores.mean());circuit_hi.append(ret[-1])
                        if clean>=32:eligible.append(scores[-1]>=32)
                    rs.append(np.mean(circuit_r));ps.append(np.mean(circuit_p));hi.append(np.mean(circuit_hi))
                    expected.append(dict(cohort=co,seed=seed,level=g,kind=kind,retention=rs[-1],pi_memory_score=ps[-1]))
                vectors[g]=(np.array(rs),np.array(ps),bad);key=f'{co}/{g}/{kind}'
                audit_stats(ps,summary['statistics'][key]['pi_memory_score'],c)
                if not bad:audit_stats(rs,summary['statistics'][key]['retention'],c)
                outcome=None if bad else bool(np.mean(hi)>=.8 and np.sum(np.array(hi)>=.8)>=need and len(eligible)>0 and np.mean(eligible)>=.8)
                assert summary['absolute_robustness'][key]['passed']==outcome;absolute[key]=outcome
            for hi,lo in [('brain1','legacy5'),('brain5','legacy5'),('brain1','brain5')]:
                r=vectors[hi][0]-vectors[lo][0];p=vectors[hi][1]-vectors[lo][1];bad=vectors[hi][2] or vectors[lo][2]
                key=f'{co}/{hi}-{lo}/{kind}';saved=summary['comparisons'][key]
                audit_stats(p,saved['statistics']['pi_memory_score'],c)
                if not bad:audit_stats(r,saved['statistics']['retention'],c)
                passed=None if bad else bool(np.mean(r)>=.1 and np.mean(p)>=2 and np.sum(r>0)>=need and np.sum(p>0)>=need)
                assert saved['passed']==passed;comparisons[key]=passed
    saved=pd.read_csv(root/'seed-blocks.csv').sort_values(['cohort','seed','level','kind']).reset_index(drop=True)
    ref=pd.DataFrame(expected).sort_values(['cohort','seed','level','kind']).reset_index(drop=True)
    pd.testing.assert_frame_equal(saved[ref.columns],ref,check_exact=False,atol=1e-12,rtol=1e-12)
    confirmed=[f'{contrast}/{kind}' for contrast in ['brain1-legacy5','brain5-legacy5','brain1-brain5'] for kind in run.FAMILIES
        if all(comparisons[f'{co}/{contrast}/{kind}'] is True for co in c['cohorts'])]
    robust=[f'{g}/{kind}' for g in c['conditions'] for kind in run.FAMILIES if all(absolute[f'{co}/{g}/{kind}'] is True for co in c['cohorts'])]
    assert summary['confirmed_comparisons']==confirmed and summary['confirmed_absolute_robustness']==robust
    assert summary['primary_confirmed']==('brain1-legacy5/ongoing' in confirmed)


@threadpool_limits.wrap(limits=1)
def verify(root,out):
    started=time.perf_counter();m=check(root);c=m['config'];out.mkdir(parents=True,exist_ok=False)
    current=run.context(c)
    for key in ['source_sha256','adapter_sha256','config_sha256']:assert current[key]==m['context'][key],key
    budget=Budget(dict(max_seconds=c['verification_seconds'],max_rss_bytes=c['worker_rss_bytes']))
    counts=dict(cases=0,metric_rows=0,independent_trajectories=0,exact_fresh_refits=0,zero_constructions=0)
    smoke=read(root/'verification.json')['smoke'];allrows=[];maxerr=0.
    for case in run.case_list(c,smoke):
        budget.check();path=root/run.stem(case);check(path);meta=read(path/'case.json')
        with np.load(path/'checkpoint.npz') as z:checkpoint=dict(z)
        head=core.NonlinearReadout(np.arange(48),8,core.nonlinear_seed(case['seed'],0));head.mean=checkpoint['mean'];head.scale=checkpoint['scale']
        head.parameters={k:checkpoint[k] for k in ['w1','b1','w2','b2']};assert head.digest()==meta['head_sha256']
        h=meta['horizon'];digits=checkpoint['digits'];core_digits=core.SequenceDataset().symbols();np.testing.assert_array_equal(digits,core_digits)
        df=pd.read_csv(path/'metrics.csv');allrows.extend(df.to_dict('records'));cleans={}
        for mode in ['autonomous','teacher']:
            with np.load(path/f'clean_{mode}.npz') as z:cleans[mode]=dict(z)
        selected=smoke or case['level']=='legacy5' or (case['seed'],case['circuit_seed']) in [(7142,701),(371142,702)]
        refit=case['cohort']=='confirmation' and case['seed']==371142
        if selected or refit:
            bc=read(Path(c['source'])/'config.json');base,obs,graph=core.build_model(c['cache'],core.NetworkCondition(case['level'],case['circuit_seed'],case['seed']),bc)
            assert graph==meta['graph'];mp=run.maps(c,case,base);draws=run.bank(c,case,mp)
            for kind in run.FAMILIES:
                w,dead=independent_alter(base.weights,draws,kind,0)
                assert core.weight_hash(w)==meta['original_weight_sha256'] and not dead.any();counts['zero_constructions']+=1
            if refit:
                features,_=core.collect(base,obs,digits[:-1],1e-8);np.testing.assert_array_equal(features,checkpoint['features'])
                fitted=core.NonlinearReadout(np.arange(48),8,core.nonlinear_seed(case['seed'],0))
                fitted.fit(features,digits[1:],epochs=2000,learning_rate=.03,l2=1e-5,checkpoints=(2000,),sample_weight=core.sample_weights(199,3,32,4))
                assert fitted.digest()==head.digest();counts['exact_fresh_refits']+=1
        for row in df.to_dict('records'):
            with np.load(path/row['artifact']) as z:a=dict(z)
            metrics(a,digits[3:3+h],row,cleans[row['mode']],head.scale,meta['graph']['neurons']);counts['metric_rows']+=1
            if selected and not row['reused_clean']:
                w,dead=independent_alter(base.weights,draws,row['kind'],row['strength'])
                random=np.random.default_rng(np.random.SeedSequence([372001,case['seed'],case['circuit_seed'],1]))
                ref=reference(base,obs,head,digits[:3],h,mp,w,dead,row['kind'],row['strength'],random,
                    digits[3:] if row['mode']=='teacher' else None)
                for k,v in ref.items():
                    if k in ['prediction','active_counts','clipped']:np.testing.assert_array_equal(v,a[k])
                    else:
                        np.testing.assert_allclose(v,a[k],atol=1e-12,rtol=1e-10,err_msg=f'{path}/{row["artifact"]}/{k}')
                        maxerr=max(maxerr,float(np.max(np.abs(v-a[k]))))
                counts['independent_trajectories']+=1
        if selected:
            for intervention in read(path/'interventions.json'):
                w,dead=independent_alter(base.weights,draws,intervention['kind'],intervention['strength'])
                assert core.weight_hash(w)==intervention['weight_sha256']
                assert int(dead.sum())==intervention['dead_neurons']
            assert core.weight_hash(base.weights)==meta['original_weight_sha256']
        counts['cases']+=1;print(f"Audit {counts['cases']}/{len(run.case_list(c,smoke))}: {run.stem(case)}",flush=True)
    f=pd.DataFrame(allrows)
    saved=pd.read_csv(root/'raw-metrics.csv');pd.testing.assert_frame_equal(saved[f.columns],f,check_dtype=False,check_exact=False,rtol=1e-12,atol=1e-12)
    if not smoke:statistics(f,c,root)
    prior=read(root/'prior-artifacts.json')
    for p,h in prior.items():assert core.sha256(p)==h,p
    usage=budget.close();result=dict(all_checks_pass=True,**counts,maximum_numeric_error=maxerr,all_whole_trajectories_replayed=False,
        prior_files_unchanged=len(prior),result_manifest_sha256=core.sha256(root/'manifest.json'),budget=usage)
    core.write_json(out/'checks.json',result);seal(out,c,m['context'],purpose='Independent ACT V scalar subset, complete metrics/statistics and hashes')
    print('ACT V validation complete',counts,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args();verify(a.source,a.out)
