"""Registered 2x2 direct-carry / past-state synaptic-drive experiment."""
import argparse
from fractions import Fraction
from pathlib import Path
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from flying.models.ridge import RidgeDecoder
from flying.training import whole_brain_memory as core
from temporal_memory_curve import build,collect,controls
from mechanism_support import config,read,write,sha,array_sha,attempt,seal,check_manifest,source_record,history_preserved
from tdc_pool import pooled


class FactorReservoir:
    def __init__(self,base,carry,synaptic_history):
        self.weights=base.weights
        self.encoder=base.encoder
        self.kc,self.mbon=base.kc.copy(),base.mbon.copy()
        self.mbon_weights=base.mbon_weights
        self.leak=base.leak
        self.carry,self.synaptic_history=carry,synaptic_history
        self.reset()

    def reset(self):
        self.state=np.zeros(self.weights.shape[0])

    def step(self,symbol):
        previous=self.state.copy()
        stimulus=self.encoder(int(symbol))
        residual=(1-self.leak)*previous if self.carry else np.zeros_like(previous)
        synaptic=self.weights@previous if self.synaptic_history else np.zeros_like(previous)
        following=residual+self.leak*np.tanh(synaptic+stimulus)
        presynaptic=previous.copy() if self.synaptic_history else np.zeros_like(previous)
        presynaptic[self.kc]=following[self.kc]
        following[self.mbon]=residual[self.mbon]+self.leak*np.tanh(self.mbon_weights@presynaptic+stimulus[self.mbon])
        self.state=following
        return following.copy()


def zero_prototypes(base,model,obs):
    a,b=[],[]
    for symbol in range(10):
        base.reset()
        expected=base.step(symbol)
        model.reset()
        actual=model.step(symbol)
        np.testing.assert_array_equal(actual,expected)
        a.append(actual.copy());b.append(actual[obs].copy())
    base.reset();model.reset()
    return np.asarray(a),np.asarray(b)


def execute(c,block,level,arm,resources):
    begin=time.perf_counter()
    base,obs,info=build(c,level,block)
    switches=c['arms'][arm]
    model=base if arm=='full' else FactorReservoir(base,**switches)
    prototypes,observed_prototypes=zero_prototypes(base,model,obs)
    w,k=c['warmup'],c['alphabet_size']
    a=dict(observed_indices=obs,input_patterns=model.encoder.patterns.copy(),zero_state_observed_prototypes=observed_prototypes)
    if arm=='instantaneous':
        a['instant_state_prototypes']=prototypes
    trace={}
    for split in ['train','test']:
        symbols=np.random.default_rng(block[split+'_seed']).integers(0,k,w+c[split+'_samples'],dtype=np.int64)
        features,norms,final,h=collect(model,obs,symbols,resources)
        times=np.arange(w,len(symbols))
        a.update({split+'_symbols':symbols,split+'_features':features,split+'_norms':norms,
                  split+'_final_state':final,split+'_times':times,'y'+split:symbols[times[:,None]-np.asarray(c['lags'])[None,:]]})
        trace[split+'_full_state_trajectory_sha256']=h
        if arm=='instantaneous':
            np.testing.assert_array_equal(features,observed_prototypes[symbols])
            np.testing.assert_array_equal(final,prototypes[symbols[-1]])
    targets=np.eye(k)[a['ytrain']].reshape(c['train_samples'],-1)
    head=RidgeDecoder(c['alpha']).fit(a['train_features'][w:],targets)
    scores=head.scores(a['test_features'][w:]).reshape(c['test_samples'],len(c['lags']),k)
    a.update(mean=head.mean,scale=head.scale,coefficients=head.weights,intercept=head.target_mean,
             scores=scores,predictions=scores.argmax(axis=2))
    a['frequency_predictions'],a['current_predictions'],a['current_tables'],a['majority']=controls(a['train_symbols'],a['test_symbols'],a['ytrain'],a['ytest'],w,k)
    rows=[]
    for j,lag in enumerate(c['lags']):
        truth=a['ytest'][:,j];n=len(truth)
        correct=int(np.sum(a['predictions'][:,j]==truth));freq=int(np.sum(a['frequency_predictions'][:,j]==truth));current=int(np.sum(a['current_predictions'][:,j]==truth))
        accuracy=correct/n;baseline=max(1/k,freq/n,current/n)
        rows.append(dict(level=level,arm=arm,seed=block['seed'],lag=lag,samples=n,correct=correct,
                         frequency_correct=freq,current_correct=current,accuracy=accuracy,frequency_accuracy=freq/n,
                         current_accuracy=current/n,chance=1/k,chance_adjusted=(accuracy-1/k)/(1-1/k),baseline=baseline,baseline_excess=accuracy-baseline))
    info.update(arm=arm,switches=switches,drive_multiplier=c['leak'],direct_carry_multiplier=(1-c['leak']) if switches['carry'] else 0.,
                same_step_kc_to_mbon_retained=True,zero_state_all_arm_response_equal=True,
                instantaneous_certificate=arm=='instantaneous',**trace,
                array_hashes={n:array_sha(x) for n,x in a.items()},decoder_parameters_per_lag=490,seconds=time.perf_counter()-begin)
    assert core.weight_hash(model.weights)==info['weight_sha256']
    return a,rows,info


def case_job(job):
    c,block,level,arm,out=job['config'],job['block'],job['level'],job['arm'],Path(job['out'])
    with threadpool_limits(1):
        with attempt(out,c,'mechanism-case') as resources:
            arrays,rows,graph=execute(c,block,level,arm,resources)
            np.savez_compressed(out/'case.npz',**arrays)
            write(out/'metrics.json',rows);write(out/'graph.json',graph)
        seal(out,dict(complete=True,block=block,level=level,arm=arm))
    return dict(identity=f'{level}/{arm}/s{block["seed"]}',rows=rows,resources=read(out/'resources.json'))


def exact_score(data):
    acc=sum((Fraction(int(r.correct),int(r.samples)) for r in data.itertuples()),Fraction(0))/len(data)
    return (acc-Fraction(1,10))/Fraction(9,10)


def estimates(values,c):
    values=np.asarray(values,dtype=float)
    ix=np.random.default_rng(c['bootstrap_seed']).integers(0,len(values),(c['bootstrap_draws'],len(values)))
    lo,hi=np.quantile(values[ix].mean(axis=1),[.025,.975])
    return dict(mean=float(values.mean()),median=float(np.median(values)),sd=float(values.std(ddof=1)) if len(values)>1 else 0.,bootstrap_lo=float(lo),bootstrap_hi=float(hi))


def summarize(f,c,smoke):
    score_rows=[];exact={};contrast_rows=[];levels={}
    for (level,arm,seed),data in f[f.lag.isin(c['score_lags'])].groupby(['level','arm','seed']):
        s=exact_score(data);exact[(level,arm,int(seed))]=s
        score_rows.append(dict(level=level,arm=arm,seed=int(seed),score=float(s),numerator=s.numerator,denominator=s.denominator))
    names={'synaptic_increment':['full','carry_only'],'carry_increment':['full','synaptic_only'],
           'carry_only_history':['carry_only','instantaneous'],'synaptic_only_history':['synaptic_only','instantaneous']}
    for level in c['levels']:
        seeds=c['blocks_by_level'][level]
        current=f[(f.level==level)&(f.lag==0)]
        access=bool(len(current)==4*len(seeds) and all(Fraction(int(r.correct),int(r.samples))>=Fraction(9,10) for r in current.itertuples()))
        contrasts={}
        for name,(first,second) in names.items():
            delta=[exact[(level,first,s)]-exact[(level,second,s)] for s in seeds]
            mean=sum(delta,Fraction(0))/len(delta)
            status=None if smoke or not access else bool(mean>=Fraction(3,100) and all(x>0 for x in delta))
            contrasts[name]=dict(**estimates(list(map(float,delta)),c),mean_exact=str(mean),all_blocks_positive=all(x>0 for x in delta),registered_rule_pass=status)
            contrast_rows.extend(dict(level=level,contrast=name,seed=s,first=first,second=second,difference=float(d),numerator=d.numerator,denominator=d.denominator) for s,d in zip(seeds,delta))
        interaction=[exact[(level,'full',s)]-exact[(level,'carry_only',s)]-exact[(level,'synaptic_only',s)]+exact[(level,'instantaneous',s)] for s in seeds]
        levels[level]=dict(input_access_valid=access,contrasts=contrasts,interaction=estimates(list(map(float,interaction)),c),
                           arm_scores={arm:estimates([float(exact[(level,arm,s)]) for s in seeds],c) for arm in c['arms']})
    primary=levels['legacy5']['contrasts']['synaptic_increment']['registered_rule_pass']
    outcome='smoke' if smoke else 'assay-invalid' if not levels['legacy5']['input_access_valid'] else 'PASS' if primary else 'FAIL'
    summary=dict(smoke=smoke,outcome=outcome,primary_pass=primary,levels=levels,cases=int(f[['level','arm','seed']].drop_duplicates().shape[0]),
                 lag_heads=len(f),criterion_unchanged=True,biological_plasticity_performed=False,anatomical_cycle_only_claim=False)
    curves=[]
    for (level,arm,lag),data in f.groupby(['level','arm','lag']):
        row=dict(level=level,arm=arm,lag=int(lag))
        for metric in ['accuracy','baseline_excess','chance_adjusted','frequency_accuracy','current_accuracy']:
            row.update({metric+'_'+k:v for k,v in estimates(data.sort_values('seed')[metric],c).items()})
        curves.append(row)
    return summary,pd.DataFrame(score_rows),pd.DataFrame(contrast_rows),pd.DataFrame(curves)


def plot(f,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.7))
    colors={'full':'#17679e','carry_only':'#d17e20','synaptic_only':'#188c72','instantaneous':'#8a6b9c'}
    for ax,level in zip(axes,['legacy5','brain5']):
        for arm,color in colors.items():
            part=f[(f.level==level)&(f.arm==arm)&(f.lag>0)]
            for _,block in part.groupby('seed'):
                ax.plot(block.lag,block.accuracy,color=color,alpha=.15,lw=.8)
            mean=part.groupby('lag').accuracy.mean()
            ax.plot(mean.index,mean,'o-',color=color,label=arm,lw=1.7,markersize=3)
        ax.axhline(.1,color='gray',ls=':',label='chance')
        ax.set(title=level,xlabel='Historical lag (symbol steps)',ylabel='Independent test decoding accuracy',ylim=(0,1.02),xticks=[1,2,3,4,5,8,12,16,20])
        ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'mechanism-tdc.png',dpi=180);plt.close(fig)


def run(out,smoke):
    c=config(smoke);source=source_record(c)
    with attempt(out,c,'mechanism-smoke' if smoke else 'mechanism-main'):
        write(out/'config.json',c);write(out/'source.json',source)
        write(out/'historical-preservation.json',history_preserved(c))
        jobs=[dict(config=c,level=level,arm=arm,block=block,out=str(out/f'{level}_{arm}_s{block["seed"]}'))
              for level in c['levels'] for block in c['blocks'] if block['seed'] in c['blocks_by_level'][level] for arm in c['arms']]
        results,usage=pooled(case_job,jobs,c['neural_workers'],c,'Mechanism TDC')
        f=pd.DataFrame([row for result in results for row in result['rows']]).sort_values(['level','arm','seed','lag']).reset_index(drop=True)
        assert len(f)==(168 if smoke else 756)
        summary,scores,contrasts,curves=summarize(f,c,smoke)
        f.to_csv(out/'raw-lags.csv',index=False);scores.to_csv(out/'seed-scores.csv',index=False)
        contrasts.to_csv(out/'paired-contrasts.csv',index=False);curves.to_csv(out/'curve-summary.csv',index=False)
        write(out/'summary.json',summary);write(out/'process-tree-resources.json',usage)
        write(out/'job-resources.json',[dict(identity=r['identity'],resources=r['resources']) for r in results])
        # All paired arms must use identical input patterns, coordinates, streams and labels.
        for level in c['levels']:
            for block in c['blocks']:
                if block['seed'] not in c['blocks_by_level'][level]:continue
                with np.load(out/f'{level}_full_s{block["seed"]}'/'case.npz',allow_pickle=False) as a:
                    for arm in c['arms']:
                        with np.load(out/f'{level}_{arm}_s{block["seed"]}'/'case.npz',allow_pickle=False) as b:
                            for n in ['input_patterns','observed_indices','train_symbols','test_symbols','ytrain','ytest','zero_state_observed_prototypes']:
                                np.testing.assert_array_equal(a[n],b[n])
        plot(f,out)
        for n,h in source['hashes'].items():assert sha(n)==h,n
        history_preserved(c)
    seal(out,dict(complete=True,smoke=smoke,config=c,source_commit=source['source_commit']))
    print(dict(outcome=summary['outcome'],primary_pass=summary['primary_pass'],cases=summary['cases']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    with threadpool_limits(1):run(a.out,a.smoke)
