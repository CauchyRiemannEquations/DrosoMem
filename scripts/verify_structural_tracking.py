"""Independent M9 source/manual-state/SVD-head/paired-label/criterion replay.

No import of M9 main reservoir, trajectory, fitter, scorer or summary helpers.
"""
import argparse
from fractions import Fraction
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits
from verify_temporal_memory_curve import independent_graph,validate_whole_source,partial_source
from verify_structural_specificity_design import invariants as bank_invariants
from tdc_pool import pooled
from structural_tracking_support import config,read,write,sha,array_sha,attempt,seal,check_manifest,parent_audit,history_preserved,source_record,source_unchanged

WORLDS=('original','counterfactual','instantaneous_original','instantaneous_counterfactual')

def matrix_hash(w):return hashlib.sha256(w.data.tobytes()+w.indices.tobytes()+w.indptr.tobytes()).hexdigest()
def state_hash(x):return hashlib.sha256(x.tobytes()).hexdigest()

def validate_bank(c,resources):
    source_check=validate_whole_source(c,resources);small,ids,roles=partial_source();members=[]
    for l in c['levels']:
        if l=='legacy5':raw,r=small,np.asarray(roles,dtype='U5')
        else:
            raw=sparse.load_npz(Path(c['cache'])/'brain5.npz').tocsr()
            with np.load(Path(c['cache'])/'nodes.npz') as a:r=a['roles']
        for g in config()['graph_ids'][l]:
            if g in ['real','delay']:continue
            root=Path(c['parent'])/'graphs'/f'{l}_{g}';audit=read(root/'audit.json');null=sparse.load_npz(root/'raw.npz').tocsr()
            names=['requested_swaps','accepted_swaps','attempts','eligible_delayed_edges','proposal_pool_edges','proposal_groups','proposal_pool_sha256','finite_budget_complete']
            report=bank_invariants(raw,null,r,{n:audit[n] for n in names},c);report.update(level=l,seed=int(g[1:]));compare(report,audit)
            members.append(dict(level=l,graph_id=g,all_checks_pass=True,raw_sha256=matrix_hash(null)))
            resources.check()
    return dict(source=source_check,members=members,all_checks_pass=True,no_new_graphs=True)

def model_data(c,b,l,g):
    p=Path(c['parent']);raw=None if g in ['real','delay'] else sparse.load_npz(p/'graphs'/f'{l}_{g}'/'raw.npz').tocsr()
    w,patterns,roles,source_obs,ids=independent_graph(c,l,b['input_seed'],raw)
    droot=p/'definitions'/l;source_info=read(droot/'definition.json')
    expected=source_info['source_normalized_sha256'] if g in ['real','delay'] else read(p/'graphs'/f'{l}_{g}'/'audit.json')['null_normalized_sha256']
    assert matrix_hash(w)==expected
    m=dict(n=len(ids),weights=w,patterns=patterns,kc=np.flatnonzero(roles=='KC'),mb=np.flatnonzero(roles=='MBON'),source_obs=source_obs,observed=source_obs,kind=g)
    m['mb_weights']=w[m['mb']]
    if g=='delay':
        with np.load(droot/'definition.npz',allow_pickle=False) as a:pool=a['input_pool_indices'];np.testing.assert_array_equal(ids,a['root_ids']);np.testing.assert_array_equal(source_obs,a['observed_source_indices'])
        order=np.asarray(sorted(range(48),key=lambda j:int(ids[source_obs[j]])),dtype=np.int64)
        obs=np.empty(48,dtype=np.int64)
        for k,j in enumerate(order):obs[j]=512+k if k<24 else 584+k-24
        with np.load(droot/f'prototypes-s{b["input_seed"]}.npz',allow_pickle=False) as a:templates=a['prototypes'].copy()
        code=patterns[:,pool];computed=np.stack([.6*np.tanh(w[source_obs][:,pool]@(.6*np.tanh(row))) for row in code])
        np.testing.assert_allclose(templates,computed,atol=1e-9,rtol=1e-9)
        rows=[];cols=[]
        for i in range(24):rows += [560+i,584+i];cols += [536+i,560+i]
        transition=sparse.csr_matrix((np.ones(48),(rows,cols)),shape=w.shape);transition.sort_indices()
        saved=sparse.load_npz(droot/'delay-transition.npz').tocsr();assert matrix_hash(saved)==matrix_hash(transition)
        m.update(weights=transition,observed=obs,order=order,code=code,templates=templates)
    return m

def manual_step(m,state,s,history=True):
    if m['kind']=='delay':
        following=np.zeros(m['n'])
        if history:
            # Accumulation into +0 reproduces sparse unit-shift signed-zero semantics.
            for i in range(24):following[560+i]+=state[536+i];following[584+i]+=state[560+i]
        following[:512]=.6*np.tanh(m['code'][int(s)])
        for i,j in enumerate(m['order']):following[512+i if i<24 else 536+i-24]=m['templates'][int(s),j]
        return following
    previous=state.copy();stimulus=m['patterns'][int(s)]
    drive=m['weights'].dot(previous) if history else np.zeros_like(previous)
    following=np.zeros_like(previous)+.6*np.tanh(drive+stimulus)
    mixed=previous.copy() if history else np.zeros_like(previous);mixed[m['kc']]=following[m['kc']]
    following[m['mb']]=np.zeros(len(m['mb']))+.6*np.tanh(m['mb_weights'].dot(mixed)+stimulus[m['mb']])
    return following

def replay(m,previous,symbols,history):
    state=previous.copy();digest=hashlib.sha256();features=[]
    for s in symbols:state=manual_step(m,state,int(s),history);digest.update(state.tobytes());features.append(state[m['observed']].copy())
    return np.asarray(features),state,digest.hexdigest()

def trajectory(m,symbols,resources,anchors=None):
    state=np.zeros(m['n']);features=np.empty((len(symbols),48));norms=np.empty(len(symbols));digest=hashlib.sha256()
    lookup={} if anchors is None else {int(t):i for i,t in enumerate(anchors)};events=[];pending={};trial={}
    if anchors is not None:
        for world in WORLDS:trial['trial_'+world+'_features']=np.empty((64,3,48))
        trial['trial_original_symbols']=np.empty((64,3),dtype=np.int64);trial['trial_counterfactual_symbols']=np.empty((64,3),dtype=np.int64)
    for t,s in enumerate(symbols):
        if t in lookup:
            i=lookup[t];original=symbols[t:t+3].copy();changed=original.copy();changed[0]=(changed[0]+1)%10
            trial['trial_original_symbols'][i]=original;trial['trial_counterfactual_symbols'][i]=changed
            event=dict(probe_index=i,anchor=t,evaluation_time=t+2,pre_state_sha256=state_hash(state));finals={}
            for world in WORLDS:
                rows,final,h=replay(m,state,changed if world.endswith('counterfactual') else original,not world.startswith('instantaneous'))
                trial['trial_'+world+'_features'][i]=rows;event[world+'_full_state_trajectory_sha256']=h;event[world+'_final_state_sha256']=state_hash(final);finals[world]=final
            np.testing.assert_array_equal(finals['instantaneous_original'],finals['instantaneous_counterfactual'])
            events.append(event);pending[i]=hashlib.sha256()
        state=manual_step(m,state,int(s));features[t]=state[m['observed']];norms[t]=np.linalg.norm(state);digest.update(state.tobytes())
        for i in list(pending):
            pending[i].update(state.tobytes())
            if t==int(anchors[i])+2:
                event=events[i];event['baseline_window_sha256']=pending.pop(i).hexdigest();event['baseline_endpoint_full_state_sha256']=state_hash(state)
                assert event['baseline_window_sha256']==event['original_full_state_trajectory_sha256'] and event['baseline_endpoint_full_state_sha256']==event['original_final_state_sha256']
                np.testing.assert_array_equal(trial['trial_original_features'][i],features[t-2:t+1])
        if t%100==0:resources.check()
    assert not pending
    return features,norms,state,digest.hexdigest(),trial,events

def fit_svd(x,y):
    mean=x.mean(axis=0);scale=np.maximum(x.std(axis=0),1e-5);z=(x-mean)/scale;target=np.eye(10)[y].reshape(len(y),20);bias=target.mean(axis=0)
    beta=np.linalg.lstsq(np.vstack([z,np.eye(48)]),np.vstack([target-bias,np.zeros((48,20))]),rcond=None)[0]
    return mean,scale,beta,bias

def independent_case(c,b,l,g,resources,smoke):
    m=model_data(c,b,l,g);w=c['warmup'];a=dict(input_patterns=m['patterns'],observed_indices=m['observed'],source_observed_indices=m['source_obs']);hashes={}
    for split in ['train','test']:
        symbols=np.random.Generator(np.random.PCG64(b[split+'_seed'])).integers(0,10,w+c[split+'_samples'],dtype=np.int64)
        anchors=None if split=='train' else w+np.floor(np.linspace(0,c['test_samples']-1-2,64)).astype(np.int64)
        x,norm,final,h,trial,events=trajectory(m,symbols,resources,anchors)
        times=np.arange(w,len(symbols));labels=np.column_stack([symbols[w:],symbols[w-2:len(symbols)-2]])
        a.update({split+'_symbols':symbols,split+'_features':x,split+'_norms':norm,split+'_final_state':final,split+'_times':times,'y'+split:labels});hashes[split]=h
        if split=='test':
            a.update(trial);a.update(anchors=anchors,evaluation_times=anchors+2,original_symbols=symbols[anchors],alternative_symbols=(symbols[anchors]+1)%10,current_symbols=symbols[anchors+2])
    mean,scale,beta,bias=fit_svd(a['train_features'][w:],a['ytrain']);a.update(mean=mean,scale=scale,coefficients=beta,intercept=bias)
    a['scores']=(((a['test_features'][w:]-mean)/scale).dot(beta)+bias).reshape(c['test_samples'],2,10);a['predictions']=a['scores'].argmax(axis=2)
    tables=np.zeros((2,10,10),dtype=np.int64);majority=np.zeros(2,dtype=np.int64);fp=np.zeros_like(a['ytest']);cp=np.zeros_like(fp);metrics=[]
    for j,lag in enumerate([0,2]):
        labels=a['ytrain'][:,j];majority[j]=np.bincount(labels,minlength=10).argmax()
        for current,target in zip(a['train_symbols'][w:],labels):tables[j,current,target]+=1
        lookup=np.array([row.argmax() if row.sum() else majority[j] for row in tables[j]],dtype=np.int64)
        fp[:,j]=majority[j];cp[:,j]=lookup[a['test_symbols'][w:]];target=a['ytest'][:,j];n=len(target)
        correct=int(np.sum(a['predictions'][:,j]==target));freq=int(np.sum(fp[:,j]==target));curr=int(np.sum(cp[:,j]==target))
        metrics.append(dict(level=l,graph_id=g,seed=b['seed'],lag=lag,samples=n,correct=correct,accuracy=correct/n,frequency_correct=freq,current_correct=curr,
            frequency_accuracy=freq/n,current_accuracy=curr/n,chance=.1,baseline=max(.1,freq/n,curr/n),baseline_excess=correct/n-max(.1,freq/n,curr/n)))
    a.update(frequency_predictions=fp,current_predictions=cp,current_tables=tables,majority=majority)
    for world in WORLDS:
        a['scores_'+world]=((a['trial_'+world+'_features'][:,-1]-mean)/scale).dot(beta[:,10:20])+bias[10:20]
        a['predictions_'+world]=a['scores_'+world].argmax(axis=1)
    np.testing.assert_array_equal(a['scores_instantaneous_original'],a['scores_instantaneous_counterfactual'])
    s,r=a['original_symbols'],a['alternative_symbols'];first=a['predictions_original']==s;second=a['predictions_counterfactual']==r;joint=int(np.count_nonzero(first&second))
    a['frequency_trial_predictions']=np.full(64,int(majority[1]),dtype=np.int64)
    lookup=np.array([row.argmax() if row.sum() else majority[1] for row in tables[1]],dtype=np.int64);a['current_trial_predictions']=lookup[a['current_symbols']]
    cell=dict(level=l,graph_id=g,seed=b['seed'],samples=64,original_correct=int(first.sum()),replacement_correct=int(second.sum()),joint_correct=joint,
        original_accuracy=float(first.mean()),replacement_accuracy=float(second.mean()),joint_accuracy=joint/64,prediction_changes=int(np.sum(a['predictions_original']!=a['predictions_counterfactual'])),
        technical_checks_pass=True,tracking90_pass=None if smoke else 10*joint>=576,current_correct=metrics[0]['correct'],test_samples=c['test_samples'])
    for name in ['frequency','current','instantaneous']:
        pred=a[name+'_trial_predictions'] if name!='instantaneous' else a['predictions_instantaneous_original']
        cell[name+'_joint_correct']=int(np.sum((pred==s)&(pred==r)));assert cell[name+'_joint_correct']==0
    indices=np.arange(64);so=a['scores_original'];sc=a['scores_counterfactual'];delta=a['trial_counterfactual_features'][:,-1]-a['trial_original_features'][:,-1]
    a['original_contrast_margin']=so[indices,r]-so[indices,s];a['replacement_contrast_margin']=sc[indices,r]-sc[indices,s]
    a['contrast_margin_change']=a['replacement_contrast_margin']-a['original_contrast_margin']
    a['projected_contrast_margin_change']=np.sum((delta/scale)*(beta[:,10:20][:,r]-beta[:,10:20][:,s]).T,axis=1)
    np.testing.assert_allclose(a['contrast_margin_change'],a['projected_contrast_margin_change'],atol=1e-9,rtol=1e-9);a['observed_delta_norm']=np.linalg.norm(delta,axis=1)
    rows=[]
    for i in range(64):
        row=dict(level=l,graph_id=g,seed=b['seed'],probe_index=i,anchor=int(a['anchors'][i]),evaluation_time=int(a['evaluation_times'][i]),original_symbol=int(s[i]),replacement_symbol=int(r[i]),
            current_symbol=int(a['current_symbols'][i]),original_correct=bool(first[i]),replacement_correct=bool(second[i]),joint_correct=bool(first[i] and second[i]))
        for world in WORLDS:row['prediction_'+world]=int(a['predictions_'+world][i])
        for name in ['frequency','current']:row['prediction_'+name]=int(a[name+'_trial_predictions'][i])
        for name in ['original_contrast_margin','replacement_contrast_margin','contrast_margin_change','projected_contrast_margin_change','observed_delta_norm']:row[name]=float(a[name][i])
        rows.append(row)
    return a,cell,rows,metrics,events,hashes,m

def compare(a,b):
    if isinstance(b,dict):
        assert set(a)==set(b)
        for k in b:compare(a[k],b[k])
    elif isinstance(b,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):compare(x,y)
    elif isinstance(b,float):np.testing.assert_allclose(a,b,atol=1e-9,rtol=1e-9)
    else:assert a==b,(a,b)

def case_job(job):
    c,b,l,g,result,out=job['config'],job['block'],job['level'],job['graph_id'],Path(job['result']),Path(job['out']);root=result/'cases'/f'{l}_{g}_s{b["seed"]}'
    manifest=check_manifest(root);assert manifest['block']==b and manifest['level']==l and manifest['graph_id']==g
    with threadpool_limits(1),attempt(out,c,'m9-independent-case') as resources:
        fresh,cell,rows,metrics,events,hashes,m=independent_case(c,b,l,g,resources,job['smoke']);info=read(root/'graph.json')
        assert info['weight_sha256']==matrix_hash(m['weights']) and info['heads_frozen_before_test'] and info['new_fitted_affine_parameters']==980
        assert not info['new_recurrent_learning'] and info['parent_manifest_sha256']==c['parent_manifest_sha256']
        for split,h in hashes.items():assert info[split+'_full_state_trajectory_sha256']==h
        with np.load(root/'case.npz',allow_pickle=False) as a:
            assert set(a.files)==set(fresh)==set(info['array_hashes'])
            for name,value in fresh.items():
                assert array_sha(a[name])==info['array_hashes'][name]
                numeric=name in ['mean','scale','coefficients','intercept','scores'] or name.startswith('scores_') or name.endswith('contrast_margin') or name in ['contrast_margin_change','projected_contrast_margin_change']
                if numeric:np.testing.assert_allclose(a[name],value,atol=1e-9,rtol=1e-9)
                else:np.testing.assert_array_equal(a[name],value)
            for n,h in info['frozen_head_hashes'].items():assert array_sha(a[n])==h
        compare(read(root/'cell.json'),cell);compare(read(root/'trials.json'),rows);compare(read(root/'stream-metrics.json'),metrics);assert read(root/'events.json')==events
        np.savez_compressed(out/'refit-head.npz',**{n:fresh[n] for n in ['mean','scale','coefficients','intercept']})
        write(out/'cell.json',cell);write(out/'checks.json',dict(all_checks_pass=True,train_test_state_hashes_exact=True,prestates=64,four_world_windows=256,four_world_finals=256,
            independently_refit_heads=2,predictions_exact=True,integer_counts_recomputed=True,all_anchors_counted=True,new_recurrent_learning=False,result_manifest_sha256=sha(root/'manifest.json')))
    seal(out,dict(complete=True));return dict(identity=f'{l}/{g}/s{b["seed"]}',cell=cell,rows=rows,metrics=metrics)

def descriptive(values,c):
    x=np.array(values,dtype=float);rng=np.random.Generator(np.random.PCG64(c['bootstrap_seed']));draws=rng.integers(0,len(x),(10000,len(x)));lo,hi=np.percentile(x[draws].mean(axis=1),[2.5,97.5])
    return dict(mean=float(np.mean(x)),median=float(np.median(x)),sd=float(np.std(x,ddof=1)) if len(x)>1 else 0.,bootstrap_lo=float(lo),bootstrap_hi=float(hi))

def summary(cells,c,smoke):
    expected={(l,g,s) for l in c['levels'] for g in c['graph_ids'][l] for s in c['blocks_by_level'][l]}
    assert len(cells)==len(expected) and {(r['level'],r['graph_id'],r['seed']) for r in cells}==expected
    levels={};graph_rows=[];paired=[]
    for l in c['levels']:
        groups={g:sorted([x for x in cells if x['level']==l and x['graph_id']==g],key=lambda x:x['seed']) for g in c['graph_ids'][l]}
        for rows in groups.values():assert [x['seed'] for x in rows]==c['blocks_by_level'][l] and all(x['samples']==64 for x in rows)
        scores={g:sum((Fraction(x['joint_correct'],64) for x in rows),Fraction(0))/len(rows) for g,rows in groups.items()}
        null_ids=[g for g in groups if g not in ['real','delay']];null=[scores[g] for g in null_ids];mean=sum(null,Fraction(0))/len(null);real=scores['real'];gap=real-mean
        valid=all(Fraction(x['current_correct'],x['test_samples'])>=Fraction(9,10) for g,rows in groups.items() if g!='delay' for x in rows)
        passed=None if smoke or not valid else all(real>s for s in null) and gap>=Fraction(3,100)
        dv=all(Fraction(x['current_correct'],x['test_samples'])>=Fraction(9,10) for x in groups['delay']);dt=None if smoke or not dv else all(10*x['joint_correct']>=576 for x in groups['delay'])
        sd=float(np.std([float(s) for s in null],ddof=1)) if len(null)>1 else 0.
        deltas=[Fraction(x['joint_correct'],64)-sum((Fraction(groups[g][i]['joint_correct'],64) for g in null_ids),Fraction(0))/len(null_ids) for i,x in enumerate(groups['real'])]
        delay_deltas=[Fraction(x['joint_correct']-groups['delay'][i]['joint_correct'],64) for i,x in enumerate(groups['real'])]
        outcome='smoke' if smoke else 'assay-invalid' if not valid else 'PASS' if passed else 'FAIL'
        levels[l]=dict(current_access_valid=valid,registered_rule_pass=passed,outcome=outcome,real_score=float(real),null_count=len(null),null_mean=float(mean),null_median=float(np.median(list(map(float,null)))),
            null_sd=sd,null_min=float(min(null)),null_max=float(max(null)),real_minus_null_mean=float(gap),real_minus_null_mean_exact=str(gap),real_above_all_nulls=all(real>s for s in null),
            empirical_percentile=100*(sum(s<real for s in null)+.5*sum(s==real for s in null))/len(null),standardized_effect=float(gap)/sd if sd else None,
            real_tracking90_pass=None if smoke else all(10*x['joint_correct']>=576 for x in groups['real']),delay_score=float(scores['delay']),delay_current_access_valid=dv,delay_tracking90_pass=dt,
            delay_outcome='smoke' if smoke else 'assay-invalid' if not dv else 'PASS' if dt else 'FAIL',real_minus_delay=float(real-scores['delay']),
            paired_real_minus_null=descriptive(list(map(float,deltas)),c),paired_real_minus_delay=descriptive(list(map(float,delay_deltas)),c))
        graph_rows += [dict(level=l,graph_id=g,score=float(s),numerator=s.numerator,denominator=s.denominator,blocks=len(groups[g])) for g,s in scores.items()]
        paired += [dict(level=l,seed=x['seed'],real_score=x['joint_correct']/64,real_minus_null_mean=float(deltas[i]),real_minus_delay=float(delay_deltas[i])) for i,x in enumerate(groups['real'])]
    primary=levels[c['primary_level']]
    return dict(smoke=smoke,outcome=primary['outcome'],primary_pass=primary['registered_rule_pass'],levels=levels,cases=len(cells),lag_heads=2*len(cells),probe_pairs=64*len(cells),new_fitted_affine_parameters=980*len(cells),
        new_recurrent_learning=False,new_graphs=0,observation_noise=False,criterion_unchanged=True),graph_rows,paired

def compare_csv(actual,expected,keys):
    assert set(actual.columns)==set(expected.columns) and not actual.duplicated(keys).any()
    actual=actual.sort_values(keys).reset_index(drop=True);expected=expected[actual.columns].sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual,expected,check_exact=False,check_dtype=False,atol=1e-9,rtol=1e-9)

def verify(result,out):
    manifest=check_manifest(result);smoke=manifest['smoke'];c=config(smoke);assert read(result/'config.json')==c;captured=source_record(c)
    with attempt(out,c,'m9-independent-validation') as resources:
        write(out/'config.json',c);write(out/'source.json',captured);write(out/'historical-preservation.json',history_preserved());write(out/'parent-audit.json',parent_audit(c))
        write(out/'source-bank-audit.json',validate_bank(c,resources))
        jobs=[dict(config=c,block=b,level=l,graph_id=g,result=str(result),out=str(out/'cases'/f'{l}_{g}_s{b["seed"]}'),smoke=smoke) for l in c['levels'] for g in c['graph_ids'][l] for b in c['blocks'] if b['seed'] in c['blocks_by_level'][l]]
        results,usage=pooled(case_job,jobs,4,c,'M9 independent refit/replay');cells=sorted([x['cell'] for x in results],key=lambda x:(x['level'],x['graph_id'],x['seed']))
        recomputed,graphs,paired=summary(cells,c,smoke);compare(read(result/'summary.json'),recomputed);compare(read(result/'cells.json'),cells)
        for name,frame,keys in [('raw-trials',pd.DataFrame([r for x in results for r in x['rows']]),['level','graph_id','seed','probe_index']),('stream-metrics',pd.DataFrame([r for x in results for r in x['metrics']]),['level','graph_id','seed','lag']),
            ('block-scores',pd.DataFrame(cells),['level','graph_id','seed']),('graph-scores',pd.DataFrame(graphs),['level','graph_id']),('paired-differences',pd.DataFrame(paired),['level','seed'])]:compare_csv(pd.read_csv(result/(name+'.csv')),frame,keys)
        write(out/'recomputed-summary.json',recomputed);write(out/'process-tree-resources.json',usage)
        write(out/'checks.json',dict(all_checks_pass=True,outcome=recomputed['outcome'],primary_pass=recomputed['primary_pass'],cases=len(cells),train_test_full_state_hashes=2*len(cells),
            prestates=64*len(cells),four_world_windows=256*len(cells),four_world_finals=256*len(cells),independently_refit_heads=2*len(cells),predictions_exact=True,all_csv_columns_replayed=True,
            scientific_tolerance_applied=False,result_manifest_sha256=sha(result/'manifest.json')));source_unchanged(captured)
    seal(out,dict(complete=True,source_commit=captured['source_commit'],result_manifest_sha256=sha(result/'manifest.json')));print(dict(all_checks_pass=True,outcome=recomputed['outcome'],cases=len(cells)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    with threadpool_limits(1):verify(a.result,a.out)
