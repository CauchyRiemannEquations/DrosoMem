"""M9 equal-budget graph-specific training then frozen paired tracking."""
import argparse
from fractions import Fraction
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits
from flying.models.ridge import RidgeDecoder
from flying.training import whole_brain_memory as core
from temporal_memory_curve import build,collect,controls
from temporal_mechanism import FactorReservoir
from counterfactual_tracking import replay_window,scoring,target_margin,state_sha
from tdc_pool import pooled
from structural_tracking_support import config,read,write,sha,array_sha,attempt,seal,parent_audit,history_preserved,source_record,source_unchanged

WORLDS=('original','counterfactual','instantaneous_original','instantaneous_counterfactual')

class DelayReference:
    def __init__(self,d,code,prototypes,order,drive=.6,history=True):
        self.weights=d;self.code=code;self.prototypes=prototypes;self.order=order;self.drive=drive;self.history=history;self.reset()
    def reset(self):self.state=np.zeros(self.weights.shape[0])
    def step(self,s):
        following=self.weights@self.state if self.history else np.zeros_like(self.state)
        following[:512]=self.drive*np.tanh(self.code[int(s)])
        following[512:536]=self.prototypes[int(s),self.order[:24]]
        following[536:560]=self.prototypes[int(s),self.order[24:]]
        self.state=following;return following.copy()

def setup(c,b,l,g):
    p=Path(c['parent']);raw=None if g in ['real','delay'] else sparse.load_npz(p/'graphs'/f'{l}_{g}'/'raw.npz').tocsr()
    base,source_obs,info=build(c,l,b,raw);definition=p/'definitions'/l
    with np.load(definition/'definition.npz',allow_pickle=False) as d:
        pool=d['input_pool_indices'];np.testing.assert_array_equal(source_obs,d['observed_source_indices'])
        order=d['canonical_order'].copy();delay_obs=d['delay_observed_indices'].copy()
    source_info=read(definition/'definition.json')
    expected=source_info['source_normalized_sha256'] if g in ['real','delay'] else read(p/'graphs'/f'{l}_{g}'/'audit.json')['null_normalized_sha256']
    assert core.weight_hash(base.weights)==expected
    if g=='delay':
        transition=sparse.load_npz(definition/'delay-transition.npz').tocsr()
        with np.load(definition/f'prototypes-s{b["input_seed"]}.npz',allow_pickle=False) as a:prototypes=a['prototypes'].copy()
        model=DelayReference(transition,base.encoder.patterns[:,pool].copy(),prototypes,order);obs=delay_obs
        info.update(weight_sha256=core.weight_hash(transition),dynamics='engineered_delay',active_reference_coordinates=608,templates_sha256=sha(definition/f'prototypes-s{b["input_seed"]}.npz'))
    else:model=FactorReservoir(base,False,True);obs=source_obs;info.update(dynamics='zero_carry_source',raw_sha256=core.weight_hash(raw) if raw is not None else source_info['source_raw_sha256'])
    info.update(graph_id=g,source_observation_indices=source_obs.tolist(),parent_manifest_sha256=c['parent_manifest_sha256'],decoder_parameters_per_lag=490,heads_per_case=2,
        new_fitted_affine_parameters=980,new_recurrent_learning=False)
    return base,model,obs,source_obs,info

def window(base,model,observed,previous,symbols,history):
    if not isinstance(model,DelayReference):return replay_window(base,previous,symbols,observed,history)
    branch=DelayReference(model.weights,model.code,model.prototypes,model.order,model.drive,history);branch.state=previous.copy()
    rows=[];digest=hashlib.sha256()
    for s in symbols:
        state=branch.step(int(s));rows.append(state[observed]);digest.update(state.tobytes())
    return np.asarray(rows),branch.state.copy(),digest.hexdigest()

def count_pair(p,q,s,r):
    assert len(s)==64 and np.all(s!=r)
    a=p==s;b=q==r
    return dict(samples=64,original_correct=int(a.sum()),replacement_correct=int(b.sum()),joint_correct=int((a&b).sum()),
        original_accuracy=float(a.mean()),replacement_accuracy=float(b.mean()),joint_accuracy=float((a&b).mean()),prediction_changes=int((p!=q).sum()))

def trial_scoring(a,c,b,l,g,smoke):
    s,r=a['original_symbols'],a['alternative_symbols']
    for world in WORLDS:
        a['scores_'+world]=scoring(a['trial_'+world+'_features'][:,-1],a['mean'],a['scale'],a['coefficients'][:,10:20],a['intercept'][10:20])
        a['predictions_'+world]=a['scores_'+world].argmax(axis=1)
    np.testing.assert_array_equal(a['scores_instantaneous_original'],a['scores_instantaneous_counterfactual'])
    majority=int(a['majority'][1]);lookup=a['current_tables'][1].argmax(axis=1);lookup[a['current_tables'][1].sum(axis=1)==0]=majority
    a['frequency_trial_predictions']=np.full(64,majority,dtype=np.int64);a['current_trial_predictions']=lookup[a['current_symbols']]
    cell=dict(level=l,graph_id=g,seed=b['seed'],**count_pair(a['predictions_original'],a['predictions_counterfactual'],s,r),technical_checks_pass=True)
    for name in ['frequency','current','instantaneous']:
        x=a[name+'_trial_predictions'] if name!='instantaneous' else a['predictions_instantaneous_original']
        y=x if name!='instantaneous' else a['predictions_instantaneous_counterfactual'];cell[name+'_joint_correct']=count_pair(x,y,s,r)['joint_correct'];assert cell[name+'_joint_correct']==0
    j=np.arange(64);so,sc=a['scores_original'],a['scores_counterfactual'];delta=a['trial_counterfactual_features'][:,-1]-a['trial_original_features'][:,-1]
    a['original_contrast_margin']=so[j,r]-so[j,s];a['replacement_contrast_margin']=sc[j,r]-sc[j,s]
    a['contrast_margin_change']=a['replacement_contrast_margin']-a['original_contrast_margin']
    beta=a['coefficients'][:,10:20];a['projected_contrast_margin_change']=np.sum((delta/a['scale'])*(beta[:,r]-beta[:,s]).T,axis=1)
    np.testing.assert_allclose(a['contrast_margin_change'],a['projected_contrast_margin_change'],atol=1e-9,rtol=1e-9)
    a['observed_delta_norm']=np.linalg.norm(delta,axis=1)
    cell.update(tracking90_pass=None if smoke else 10*cell['joint_correct']>=576,current_correct=int(np.sum(a['predictions'][:,0]==a['ytest'][:,0])),test_samples=c['test_samples'])
    rows=[]
    for i in range(64):
        row=dict(level=l,graph_id=g,seed=b['seed'],probe_index=i,anchor=int(a['anchors'][i]),evaluation_time=int(a['evaluation_times'][i]),
            original_symbol=int(s[i]),replacement_symbol=int(r[i]),current_symbol=int(a['current_symbols'][i]),original_correct=bool(a['predictions_original'][i]==s[i]),
            replacement_correct=bool(a['predictions_counterfactual'][i]==r[i]),joint_correct=bool(a['predictions_original'][i]==s[i] and a['predictions_counterfactual'][i]==r[i]))
        for world in WORLDS:row['prediction_'+world]=int(a['predictions_'+world][i])
        for name in ['frequency','current']:row['prediction_'+name]=int(a[name+'_trial_predictions'][i])
        for name in ['original_contrast_margin','replacement_contrast_margin','contrast_margin_change','projected_contrast_margin_change','observed_delta_norm']:row[name]=float(a[name][i])
        rows.append(row)
    return cell,rows

def execute(c,b,l,g,resources,smoke):
    base,model,obs,source_obs,info=setup(c,b,l,g);w=c['warmup'];a=dict(input_patterns=base.encoder.patterns.copy(),observed_indices=obs,source_observed_indices=source_obs)
    train=np.random.default_rng(b['train_seed']).integers(0,10,w+c['train_samples'],dtype=np.int64)
    x,norm,last,digest=collect(model,obs,train,resources);times=np.arange(w,len(train))
    yt=train[times[:,None]-np.array([0,2])[None,:]]
    a.update(train_symbols=train,train_features=x,train_norms=norm,train_final_state=last,train_times=times,ytrain=yt)
    head=RidgeDecoder(1.).fit(x[w:],np.eye(10)[yt].reshape(c['train_samples'],20))
    a.update(mean=head.mean.copy(),scale=head.scale.copy(),coefficients=head.weights.copy(),intercept=head.target_mean.copy())
    for name in ['mean','scale','coefficients','intercept']:a[name].flags.writeable=False
    frozen={name:array_sha(a[name]) for name in ['mean','scale','coefficients','intercept']};info['train_full_state_trajectory_sha256']=digest
    test=np.random.default_rng(b['test_seed']).integers(0,10,w+c['test_samples'],dtype=np.int64)
    anchors=w+np.floor(np.linspace(0,c['test_samples']-3,64)).astype(np.int64);assert len(set(anchors))==64
    a.update(test_symbols=test,anchors=anchors,evaluation_times=anchors+2,original_symbols=test[anchors],alternative_symbols=(test[anchors]+1)%10,current_symbols=test[anchors+2])
    a['test_features']=np.empty((len(test),48));a['test_norms']=np.empty(len(test))
    for world in WORLDS:a['trial_'+world+'_features']=np.empty((64,3,48))
    a['trial_original_symbols']=np.empty((64,3),dtype=np.int64);a['trial_counterfactual_symbols']=np.empty((64,3),dtype=np.int64)
    model.reset();lookup={int(t):i for i,t in enumerate(anchors)};pending={};events=[];full=hashlib.sha256()
    for t,s in enumerate(test):
        if t in lookup:
            i=lookup[t];previous=model.state.copy();original=test[t:t+3].copy();changed=original.copy();changed[0]=(changed[0]+1)%10
            a['trial_original_symbols'][i]=original;a['trial_counterfactual_symbols'][i]=changed
            event=dict(probe_index=i,anchor=t,evaluation_time=t+2,pre_state_sha256=state_sha(previous));finals={}
            for world in WORLDS:
                rows,final,h=window(base,model,obs,previous,changed if world.endswith('counterfactual') else original,not world.startswith('instantaneous'))
                a['trial_'+world+'_features'][i]=rows;event[world+'_full_state_trajectory_sha256']=h;event[world+'_final_state_sha256']=state_sha(final);finals[world]=final
            np.testing.assert_array_equal(finals['instantaneous_original'],finals['instantaneous_counterfactual'])
            events.append(event);pending[i]=hashlib.sha256()
        state=model.step(int(s));a['test_features'][t]=state[obs];a['test_norms'][t]=np.linalg.norm(state);full.update(state.tobytes())
        for i in list(pending):
            pending[i].update(state.tobytes())
            if t==int(anchors[i])+2:
                e=events[i];e['baseline_window_sha256']=pending.pop(i).hexdigest();e['baseline_endpoint_full_state_sha256']=state_sha(state)
                assert e['baseline_window_sha256']==e['original_full_state_trajectory_sha256'] and e['baseline_endpoint_full_state_sha256']==e['original_final_state_sha256']
                np.testing.assert_array_equal(a['trial_original_features'][i],a['test_features'][t-2:t+1])
        if t%100==0:resources.check()
    assert not pending
    a.update(test_final_state=model.state.copy(),test_times=np.arange(w,len(test)));a['ytest']=test[a['test_times'][:,None]-np.array([0,2])[None,:]]
    scores=scoring(a['test_features'][w:],a['mean'],a['scale'],a['coefficients'],a['intercept']).reshape(c['test_samples'],2,10)
    a.update(scores=scores,predictions=scores.argmax(axis=2))
    fp,cp,tables,majority=controls(train,test,yt,a['ytest'],w,10);a.update(frequency_predictions=fp,current_predictions=cp,current_tables=tables,majority=majority)
    cell,rows=trial_scoring(a,c,b,l,g,smoke);metrics=[]
    for j,lag in enumerate([0,2]):
        target=a['ytest'][:,j];n=len(target);corr=int(np.sum(a['predictions'][:,j]==target));freq=int(np.sum(fp[:,j]==target));curr=int(np.sum(cp[:,j]==target))
        metrics.append(dict(level=l,graph_id=g,seed=b['seed'],lag=lag,samples=n,correct=corr,accuracy=corr/n,frequency_correct=freq,current_correct=curr,
            frequency_accuracy=freq/n,current_accuracy=curr/n,chance=.1,baseline=max(.1,freq/n,curr/n),baseline_excess=corr/n-max(.1,freq/n,curr/n)))
    assert all(array_sha(a[name])==h for name,h in frozen.items()) and core.weight_hash(model.weights)==info['weight_sha256']
    info.update(test_full_state_trajectory_sha256=full.hexdigest(),frozen_head_hashes=frozen,array_hashes={n:array_sha(v) for n,v in a.items()},heads_frozen_before_test=True)
    return a,cell,rows,metrics,events,info

def case_job(job):
    c,b,l,g,out=job['config'],job['block'],job['level'],job['graph_id'],Path(job['out'])
    with threadpool_limits(1),attempt(out,c,'m9-trained-tracking-case') as resources:
        a,cell,rows,metrics,events,info=execute(c,b,l,g,resources,job['smoke']);np.savez_compressed(out/'case.npz',**a)
        for name,data in [('cell',cell),('trials',rows),('stream-metrics',metrics),('events',events),('graph',info)]:write(out/(name+'.json'),data)
    seal(out,dict(complete=True,level=l,graph_id=g,block=b,smoke=job['smoke']))
    return dict(identity=f'{l}/{g}/s{b["seed"]}',cell=cell,rows=rows,metrics=metrics)

def describe(x,c):
    x=np.asarray(x,dtype=float);ix=np.random.default_rng(c['bootstrap_seed']).integers(len(x),size=(10000,len(x)));lo,hi=np.quantile(x[ix].mean(axis=1),[.025,.975])
    return dict(mean=float(x.mean()),median=float(np.median(x)),sd=float(x.std(ddof=1)) if len(x)>1 else 0.,bootstrap_lo=float(lo),bootstrap_hi=float(hi))

def summarize(cells,c,smoke):
    expected={(l,g,s) for l in c['levels'] for g in c['graph_ids'][l] for s in c['blocks_by_level'][l]}
    assert len(cells)==len(expected) and {(r['level'],r['graph_id'],r['seed']) for r in cells}==expected
    levels={};graph_rows=[];paired=[]
    for l in c['levels']:
        groups={g:sorted([r for r in cells if r['level']==l and r['graph_id']==g],key=lambda r:r['seed']) for g in c['graph_ids'][l]}
        for g,rows in groups.items():assert [r['seed'] for r in rows]==c['blocks_by_level'][l] and all(r['samples']==64 for r in rows)
        score={g:Fraction(sum(r['joint_correct'] for r in rows),64*len(rows)) for g,rows in groups.items()}
        null_ids=[g for g in groups if g not in ['real','delay']];null=[score[g] for g in null_ids];real=score['real'];mean=sum(null,Fraction(0))/len(null);gap=real-mean
        valid=all(10*r['current_correct']>=9*r['test_samples'] for g,rows in groups.items() if g!='delay' for r in rows)
        advantage=None if smoke or not valid else real>max(null) and gap>=Fraction(3,100)
        outcome='smoke' if smoke else 'assay-invalid' if not valid else 'PASS' if advantage else 'FAIL'
        dv=all(10*r['current_correct']>=9*r['test_samples'] for r in groups['delay']);dt=None if smoke or not dv else all(10*r['joint_correct']>=576 for r in groups['delay'])
        ns=np.asarray(list(map(float,null)));sd=float(ns.std(ddof=1)) if len(null)>1 else 0.
        block_delta=[Fraction(r['joint_correct'],64)-sum((Fraction(groups[g][i]['joint_correct'],64) for g in null_ids),Fraction(0))/len(null_ids) for i,r in enumerate(groups['real'])]
        delay_delta=[Fraction(r['joint_correct']-groups['delay'][i]['joint_correct'],64) for i,r in enumerate(groups['real'])]
        levels[l]=dict(current_access_valid=valid,registered_rule_pass=advantage,outcome=outcome,real_score=float(real),null_count=len(null),null_mean=float(mean),null_median=float(np.median(ns)),
            null_sd=sd,null_min=float(min(null)),null_max=float(max(null)),real_minus_null_mean=float(gap),real_minus_null_mean_exact=str(gap),real_above_all_nulls=real>max(null),
            empirical_percentile=100*(sum(s<real for s in null)+.5*sum(s==real for s in null))/len(null),standardized_effect=float(gap)/sd if sd else None,
            real_tracking90_pass=None if smoke else all(10*r['joint_correct']>=576 for r in groups['real']),delay_score=float(score['delay']),delay_current_access_valid=dv,
            delay_tracking90_pass=dt,delay_outcome='smoke' if smoke else 'assay-invalid' if not dv else 'PASS' if dt else 'FAIL',
            real_minus_delay=float(real-score['delay']),paired_real_minus_null=describe(list(map(float,block_delta)),c),paired_real_minus_delay=describe(list(map(float,delay_delta)),c))
        graph_rows += [dict(level=l,graph_id=g,score=float(s),numerator=s.numerator,denominator=s.denominator,blocks=len(groups[g])) for g,s in score.items()]
        paired += [dict(level=l,seed=r['seed'],real_score=r['joint_correct']/64,real_minus_null_mean=float(block_delta[i]),real_minus_delay=float(delay_delta[i])) for i,r in enumerate(groups['real'])]
    primary=levels[c['primary_level']]
    return dict(smoke=smoke,outcome=primary['outcome'],primary_pass=primary['registered_rule_pass'],levels=levels,cases=len(cells),lag_heads=2*len(cells),probe_pairs=64*len(cells),
        new_fitted_affine_parameters=980*len(cells),new_recurrent_learning=False,new_graphs=0,observation_noise=False,criterion_unchanged=True),graph_rows,paired

def plot(rows,summary,out):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,l in zip(axes,['legacy5','brain5']):
        selected=[r for r in rows if r['level']==l];colors=['#1469a1' if r['graph_id']=='real' else '#c57b25' if r['graph_id']=='delay' else '#777' for r in selected]
        ax.scatter(range(len(selected)),[100*r['score'] for r in selected],c=colors)
        ax.set(xticks=range(len(selected)),xticklabels=[r['graph_id'] for r in selected],ylim=(-2,102),ylabel='Joint tracking (% of all pairs)',title=l+': '+summary['levels'][l]['outcome'])
        ax.tick_params(axis='x',labelrotation=50)
    fig.suptitle('M9 graph-specific fitted then frozen lag2 tracking');fig.tight_layout();fig.savefig(out/'structural-tracking.png',dpi=180);plt.close(fig)

def run(out,smoke=False):
    c=config(smoke);captured=source_record(c)
    with attempt(out,c,'m9-smoke' if smoke else 'm9-main'):
        write(out/'config.json',c);write(out/'source.json',captured);write(out/'parent-audit.json',parent_audit(c));write(out/'historical-preservation.json',history_preserved())
        jobs=[dict(config=c,block=b,level=l,graph_id=g,out=str(out/'cases'/f'{l}_{g}_s{b["seed"]}'),smoke=smoke) for l in c['levels'] for g in c['graph_ids'][l] for b in c['blocks'] if b['seed'] in c['blocks_by_level'][l]]
        results,usage=pooled(case_job,jobs,4,c,'M9 training/tracking');cells=sorted([r['cell'] for r in results],key=lambda r:(r['level'],r['graph_id'],r['seed']))
        summary,graphs,paired=summarize(cells,c,smoke)
        pd.DataFrame([row for r in results for row in r['rows']]).sort_values(['level','graph_id','seed','probe_index']).to_csv(out/'raw-trials.csv',index=False)
        pd.DataFrame([row for r in results for row in r['metrics']]).sort_values(['level','graph_id','seed','lag']).to_csv(out/'stream-metrics.csv',index=False)
        pd.DataFrame(cells).to_csv(out/'block-scores.csv',index=False);pd.DataFrame(graphs).to_csv(out/'graph-scores.csv',index=False);pd.DataFrame(paired).to_csv(out/'paired-differences.csv',index=False)
        write(out/'cells.json',cells);write(out/'summary.json',summary);write(out/'process-tree-resources.json',usage);plot(graphs,summary,out);source_unchanged(captured)
    seal(out,dict(complete=True,smoke=smoke,source_commit=captured['source_commit']));print(dict(outcome=summary['outcome'],cases=len(cells)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    with threadpool_limits(1):run(a.out,a.smoke)
