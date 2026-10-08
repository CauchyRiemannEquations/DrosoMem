"""Independent source, factorial dynamics, label, SVD-ridge and endpoint replay.

No import of the main mechanism model, trajectory, fitter, controls or summary.
"""
import argparse
from fractions import Fraction
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from verify_temporal_memory_curve import independent_graph,validate_whole_source
from mechanism_support import read,write,sha,array_sha,attempt,seal,check_manifest,history_preserved
from tdc_support import environment,git
from tdc_pool import pooled


def analytic_prototypes(weights,patterns,roles,c):
    kc=np.flatnonzero(roles=='KC');mb=np.flatnonzero(roles=='MBON')
    outside=np.flatnonzero(roles!='KC')
    assert not patterns[:,outside].any(),'Input-access certificate assumes KC-only source encoding'
    result=np.zeros_like(patterns)
    result[:,kc]=c['leak']*np.tanh(patterns[:,kc])
    wm=weights[mb]
    for s in range(len(patterns)):
        result[s,mb]=c['leak']*np.tanh(wm@result[s]+patterns[s,mb])
    return result


def trajectory(weights,patterns,roles,obs,symbols,c,carry,history,resources):
    state=np.zeros(weights.shape[0]);kc=np.flatnonzero(roles=='KC');mb=np.flatnonzero(roles=='MBON')
    wm=weights[mb];b=c['leak']
    features=np.empty((len(symbols),len(obs)));norms=np.empty(len(symbols));digest=hashlib.sha256()
    instantaneous=not carry and not history
    prototype=analytic_prototypes(weights,patterns,roles,c) if instantaneous else None
    for t,s in enumerate(symbols):
        old=state.copy()
        # Independently specify the two update terms and the protected current path.
        retained=(1-b)*old if carry else np.zeros(len(state))
        drive=weights.dot(old) if history else np.zeros(len(state))
        state=retained+b*np.tanh(drive+patterns[s])
        pre=old.copy() if history else np.zeros(len(state))
        pre[kc]=state[kc]
        state[mb]=retained[mb]+b*np.tanh(wm.dot(pre)+patterns[s,mb])
        if instantaneous:
            np.testing.assert_array_equal(state,prototype[s])
        features[t]=state[obs];norms[t]=np.linalg.norm(state);digest.update(state.tobytes())
        if t%100==0:resources.check()
    return features,norms,state,digest.hexdigest()


def case_job(job):
    c,block,level,arm,result,out=job['config'],job['block'],job['level'],job['arm'],Path(job['result']),Path(job['out'])
    root=result/f'{level}_{arm}_s{block["seed"]}'
    m=check_manifest(root);assert m['block']==block and m['level']==level and m['arm']==arm
    info=read(root/'graph.json');switches=c['arms'][arm]
    with threadpool_limits(1):
        with attempt(out,c,'mechanism-independent-case') as resources:
            weights,patterns,roles,obs,ids=independent_graph(c,level,block['input_seed'])
            matrix_hash=hashlib.sha256(weights.data.tobytes()+weights.indices.tobytes()+weights.indptr.tobytes()).hexdigest()
            assert matrix_hash==info['weight_sha256'] and info['switches']==switches
            assert info['drive_multiplier']==.6 and info['same_step_kc_to_mbon_retained']
            proto=analytic_prototypes(weights,patterns,roles,c)
            rows=[];exact={};w,k=c['warmup'],c['alphabet_size'];rebuilt={};labels={};streams={}
            with np.load(root/'case.npz',allow_pickle=False) as saved:
                for n in saved.files:assert array_sha(saved[n])==info['array_hashes'][n],n
                np.testing.assert_array_equal(saved['input_patterns'],patterns)
                np.testing.assert_array_equal(saved['observed_indices'],obs)
                assert ids[obs].astype(str).tolist()==info['observation_root_ids']
                np.testing.assert_array_equal(saved['zero_state_observed_prototypes'],proto[:,obs])
                if arm=='instantaneous':np.testing.assert_array_equal(saved['instant_state_prototypes'],proto)
                for split in ['train','test']:
                    symbols=np.random.default_rng(block[split+'_seed']).integers(0,k,w+c[split+'_samples'],dtype=np.int64)
                    streams[split]=symbols;np.testing.assert_array_equal(saved[split+'_symbols'],symbols)
                    x,norm,last,h=trajectory(weights,patterns,roles,obs,symbols,c,switches['carry'],switches['synaptic_history'],resources)
                    np.testing.assert_allclose(x,saved[split+'_features'],atol=1e-9,rtol=1e-9)
                    np.testing.assert_allclose(norm,saved[split+'_norms'],atol=1e-9,rtol=1e-9)
                    np.testing.assert_allclose(last,saved[split+'_final_state'],atol=1e-9,rtol=1e-9)
                    exact[split]=h==info[split+'_full_state_trajectory_sha256']
                    if arm=='instantaneous':
                        np.testing.assert_array_equal(saved[split+'_features'],proto[symbols][:,obs])
                    times=np.arange(w,len(symbols));np.testing.assert_array_equal(saved[split+'_times'],times)
                    labels[split]=np.column_stack([symbols[w-lag:len(symbols)-lag] if lag else symbols[w:] for lag in c['lags']])
                    np.testing.assert_array_equal(saved['y'+split],labels[split]);rebuilt[split]=x[w:]
                mean=rebuilt['train'].mean(axis=0);scale=np.maximum(rebuilt['train'].std(axis=0),1e-5)
                np.testing.assert_allclose(mean,saved['mean'],atol=1e-9,rtol=1e-9)
                np.testing.assert_allclose(scale,saved['scale'],atol=1e-9,rtol=1e-9)
                x=(rebuilt['train']-mean)/scale;targets=np.eye(k)[labels['train']].reshape(len(x),-1);bias=targets.mean(axis=0)
                coef=np.linalg.lstsq(np.vstack([x,np.sqrt(c['alpha'])*np.eye(48)]),
                    np.vstack([targets-bias,np.zeros((48,targets.shape[1]))]),rcond=None)[0]
                np.testing.assert_allclose(coef,saved['coefficients'],atol=1e-9,rtol=1e-9)
                np.testing.assert_allclose(bias,saved['intercept'],atol=1e-9,rtol=1e-9)
                score=(((rebuilt['test']-mean)/scale)@coef+bias).reshape(c['test_samples'],21,k)
                np.testing.assert_allclose(score,saved['scores'],atol=1e-9,rtol=1e-9)
                pred=score.argmax(axis=2);np.testing.assert_array_equal(pred,saved['predictions'])
                for j,lag in enumerate(c['lags']):
                    yt,yv=labels['train'][:,j],labels['test'][:,j];n=len(yv)
                    majority=int(np.bincount(yt,minlength=k).argmax())
                    table=np.zeros((k,k),dtype=np.int64)
                    for current,target in zip(streams['train'][w:],yt):table[current,target]+=1
                    mapping=np.array([row.argmax() if row.sum() else majority for row in table])
                    cp=mapping[streams['test'][w:]];fp=np.full(n,majority)
                    np.testing.assert_array_equal(table,saved['current_tables'][j])
                    np.testing.assert_array_equal(cp,saved['current_predictions'][:,j])
                    np.testing.assert_array_equal(fp,saved['frequency_predictions'][:,j]);assert saved['majority'][j]==majority
                    correct=int(np.sum(pred[:,j]==yv));freq=int(np.sum(fp==yv));curr=int(np.sum(cp==yv));baseline=max(1/k,freq/n,curr/n)
                    rows.append(dict(level=level,arm=arm,seed=block['seed'],lag=lag,samples=n,correct=correct,frequency_correct=freq,
                                     current_correct=curr,accuracy=correct/n,frequency_accuracy=freq/n,current_accuracy=curr/n,
                                     chance=1/k,chance_adjusted=(correct/n-1/k)/(1-1/k),baseline=baseline,baseline_excess=correct/n-baseline))
            stored=pd.DataFrame(read(root/'metrics.json'));fresh=pd.DataFrame(rows)
            pd.testing.assert_frame_equal(stored,fresh[stored.columns],check_exact=False,atol=1e-12,rtol=0)
            write(out/'checks.json',dict(all_checks_pass=True,level=level,arm=arm,seed=block['seed'],
                                       exact_full_state_trace_hash=exact,instantaneous_no_history_certificate=arm=='instantaneous',
                                       preserved_zero_state_input_response=True,independently_refit_heads=21))
        seal(out,dict(complete=True,result_case_manifest_sha256=sha(root/'manifest.json')))
    return dict(identity=f'{level}/{arm}/s{block["seed"]}',rows=rows,exact=exact,resources=read(out/'resources.json'))


def score_fraction(part):
    values=[Fraction(int(r['correct']),int(r['samples'])) for r in part.to_dict('records')]
    return (sum(values,Fraction(0))/len(values)-Fraction(1,10))/Fraction(9,10)


def descriptive(values,c):
    x=np.array(list(map(float,values)))
    sample=np.random.default_rng(c['bootstrap_seed']).integers(0,len(x),(c['bootstrap_draws'],len(x)))
    q=np.quantile(np.mean(x[sample],axis=1),[.025,.975])
    return dict(mean=float(x.mean()),median=float(np.median(x)),sd=float(x.std(ddof=1)) if len(x)>1 else 0.,bootstrap_lo=float(q[0]),bootstrap_hi=float(q[1]))


def compare_numeric_dict(actual,expected):
    assert set(actual)==set(expected)
    for n,v in expected.items():assert abs(actual[n]-v)<1e-12,(n,actual[n],v)


def verify_summaries(f,result,c,smoke):
    saved=pd.read_csv(result/'raw-lags.csv')
    pd.testing.assert_frame_equal(saved,f[saved.columns],check_exact=False,atol=1e-12,rtol=0)
    exact={};score_rows=[]
    for (level,arm,seed),part in f[f.lag.between(1,20)].groupby(['level','arm','seed']):
        value=score_fraction(part);exact[(level,arm,int(seed))]=value
        score_rows.append(dict(level=level,arm=arm,seed=int(seed),score=float(value),numerator=value.numerator,denominator=value.denominator))
    sf=pd.read_csv(result/'seed-scores.csv')
    pd.testing.assert_frame_equal(sf,pd.DataFrame(score_rows)[sf.columns],check_exact=False,atol=1e-12,rtol=0)
    summary=read(result/'summary.json');endpoints={};contrast_rows=[]
    specifications={'synaptic_increment':('full','carry_only'),'carry_increment':('full','synaptic_only'),
                    'carry_only_history':('carry_only','instantaneous'),'synaptic_only_history':('synaptic_only','instantaneous')}
    for level in c['levels']:
        seeds=c['blocks_by_level'][level];status=summary['levels'][level]
        current=f[(f.level==level)&(f.lag==0)]
        valid=len(current)==4*len(seeds) and all(Fraction(int(r.correct),int(r.samples))>=Fraction(9,10) for r in current.itertuples())
        assert status['input_access_valid'] is bool(valid)
        endpoints[level]={}
        for name,(a,b) in specifications.items():
            delta=[exact[(level,a,s)]-exact[(level,b,s)] for s in seeds];mean=sum(delta,Fraction(0))/len(delta)
            gate=None if smoke or not valid else bool(mean>=Fraction(3,100) and min(delta)>0)
            reported=status['contrasts'][name]
            assert reported['registered_rule_pass'] is gate and reported['mean_exact']==str(mean)
            assert reported['all_blocks_positive'] is bool(min(delta)>0)
            compare_numeric_dict({k:reported[k] for k in ['mean','median','sd','bootstrap_lo','bootstrap_hi']},descriptive(delta,c))
            endpoints[level][name]=dict(registered_rule_pass=gate,mean_exact=str(mean))
            contrast_rows.extend(dict(level=level,contrast=name,seed=s,first=a,second=b,difference=float(v),numerator=v.numerator,denominator=v.denominator) for s,v in zip(seeds,delta))
        for arm in c['arms']:compare_numeric_dict(status['arm_scores'][arm],descriptive([exact[(level,arm,s)] for s in seeds],c))
        interaction=[exact[(level,'full',s)]-exact[(level,'carry_only',s)]-exact[(level,'synaptic_only',s)]+exact[(level,'instantaneous',s)] for s in seeds]
        compare_numeric_dict(status['interaction'],descriptive(interaction,c))
    cf=pd.read_csv(result/'paired-contrasts.csv')
    pd.testing.assert_frame_equal(cf,pd.DataFrame(contrast_rows)[cf.columns],check_exact=False,atol=1e-12,rtol=0)
    for row in pd.read_csv(result/'curve-summary.csv').to_dict('records'):
        part=f[(f.level==row['level'])&(f.arm==row['arm'])&(f.lag==row['lag'])].sort_values('seed')
        for metric in ['accuracy','baseline_excess','chance_adjusted','frequency_accuracy','current_accuracy']:
            for name,value in descriptive(part[metric],c).items():assert abs(row[metric+'_'+name]-value)<1e-12
    primary=endpoints['legacy5']['synaptic_increment']['registered_rule_pass'];assert summary['primary_pass'] is primary
    outcome='smoke' if smoke else 'assay-invalid' if not summary['levels']['legacy5']['input_access_valid'] else 'PASS' if primary else 'FAIL'
    assert summary['outcome']==outcome and summary['cases']==(8 if smoke else 36) and summary['lag_heads']==len(f)
    return endpoints,outcome


def verify(result,out):
    m=check_manifest(result);c=read(result/'config.json');assert c==m['config']
    for n,h in read(result/'source.json')['hashes'].items():assert sha(n)==h,n
    with attempt(out,c,'mechanism-independent-validation') as resources:
        write(out/'environment.json',environment());write(out/'source-graph-audit.json',validate_whole_source(c,resources))
        jobs=[dict(config=c,level=level,arm=arm,block=block,result=str(result),out=str(out/f'{level}_{arm}_s{block["seed"]}'))
              for level in c['levels'] for block in c['blocks'] if block['seed'] in c['blocks_by_level'][level] for arm in c['arms']]
        results,usage=pooled(case_job,jobs,c['neural_workers'],c,'Mechanism independent verification')
        f=pd.DataFrame([row for r in results for row in r['rows']]).sort_values(['level','arm','seed','lag']).reset_index(drop=True)
        assert len(f)==(168 if m['smoke'] else 756)
        endpoints,outcome=verify_summaries(f,result,c,m['smoke'])
        f.to_csv(out/'recomputed-lags.csv',index=False)
        write(out/'process-tree-resources.json',usage);write(out/'job-resources.json',[dict(identity=r['identity'],resources=r['resources']) for r in results])
        write(out/'checks.json',dict(all_checks_pass=True,cases=len(results),replayed_streams=2*len(results),independently_refit_lag_heads=len(f),
                                   exact_full_state_trace_hashes=sum(sum(r['exact'].values()) for r in results),
                                   instantaneous_certificates=sum('instantaneous' in r['identity'] for r in results),
                                   zero_state_input_access_certificates=len(results),outcome=outcome,endpoints=endpoints,
                                   result_manifest_sha256=sha(result/'manifest.json'),verifier_commit=git('rev-parse','HEAD'),verifier_sha256=sha(__file__),
                                   historical_preservation=history_preserved(c)))
    seal(out,dict(complete=True,result_manifest_sha256=sha(result/'manifest.json')))
    print(dict(all_checks_pass=True,outcome=outcome,cases=len(results)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('result',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    with threadpool_limits(1):verify(a.result,a.out)
