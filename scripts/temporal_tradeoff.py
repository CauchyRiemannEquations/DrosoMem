"""Prospective joint current/history endpoint; M1 numerical experiment frozen."""
import argparse
from fractions import Fraction
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from temporal_mechanism import execute,estimates
from tdc_pool import pooled
from tradeoff_support import config,read,write,sha,attempt,seal,source_record,history_preserved

def current_diagnostics(a):
    y=a['ytest'][:,0];previous=a['ytest'][:,1];prediction=a['predictions'][:,0]
    correct=prediction==y;repeat=y==previous
    d={}
    for name,mask in [('all',np.ones(len(y),dtype=bool)),('repeat',repeat),('switch',~repeat)]:
        n=int(mask.sum());count=int(correct[mask].sum())
        d[name]=dict(samples=n,correct=count,accuracy=count/n if n else None)
    errors=~correct;den=int(errors.sum());matches=int(np.sum(errors&(prediction==previous)))
    matrix=np.zeros((10,10),dtype=np.int64);np.add.at(matrix,(y,prediction),1)
    d.update(errors_total=den,errors_equal_previous_symbol=matches,error_equals_previous_fraction=matches/den if den else None,
             current_confusion=matrix.tolist())
    return d

def flat_diagnostics(level,arm,seed,d):
    row=dict(level=level,arm=arm,seed=seed)
    for name,label in [('all','current'),('repeat','repeat'),('switch','switch')]:
        row.update({label+'_'+key:value for key,value in d[name].items()})
    row.update({n:d[n] for n in ['errors_total','errors_equal_previous_symbol','error_equals_previous_fraction']})
    return row

def case_job(job):
    c,block,level,arm,out=job['config'],job['block'],job['level'],job['arm'],Path(job['out'])
    with threadpool_limits(1):
        with attempt(out,c,'tradeoff-case') as resources:
            arrays,rows,graph=execute(c,block,level,arm,resources)
            np.savez_compressed(out/'case.npz',**arrays)
            d=current_diagnostics(arrays)
            write(out/'diagnostics.json',d);write(out/'metrics.json',rows);write(out/'graph.json',graph)
        seal(out,dict(complete=True,block=block,level=level,arm=arm))
    return dict(identity=f'{level}/{arm}/s{block["seed"]}',rows=rows,diagnostics=flat_diagnostics(level,arm,block['seed'],d),resources=read(out/'resources.json'))

def exact_metrics(data):
    def mean_accuracy(part):
        return sum((Fraction(int(r.correct),int(r.samples)) for r in part.itertuples()),Fraction(0))/len(part)
    def adjusted(part):return (mean_accuracy(part)-Fraction(1,10))/Fraction(9,10)
    return dict(current=mean_accuracy(data[data.lag==0]),history=adjusted(data[data.lag.between(1,5)]),long_history=adjusted(data[data.lag.between(1,20)]))

def summarize(f,c,smoke):
    values={};score_rows=[]
    for (level,arm,seed),part in f.groupby(['level','arm','seed']):
        m=exact_metrics(part);values[(level,arm,int(seed))]=m
        row=dict(level=level,arm=arm,seed=int(seed))
        for n,v in m.items():row.update({n:float(v),n+'_numerator':v.numerator,n+'_denominator':v.denominator})
        score_rows.append(row)
    levels={};pairs=[];descriptive=[]
    for level in c['levels']:
        seeds=c['blocks_by_level'][level]
        ref=bool(all(values[(level,'instantaneous',s)]['current']>=Fraction(99,100) for s in seeds))
        dh=[values[(level,'carry_only',s)]['history']-values[(level,'instantaneous',s)]['history'] for s in seeds]
        dc=[values[(level,'carry_only',s)]['current']-values[(level,'instantaneous',s)]['current'] for s in seeds]
        mh,mc=sum(dh,Fraction(0))/len(dh),sum(dc,Fraction(0))/len(dc)
        hp,cp=mh>=Fraction(3,100),mc<=-Fraction(1,20)
        ah,ac=min(dh)>0,max(dc)<0
        primary=dict(historical_gain=dict(**estimates(list(map(float,dh)),c),mean_exact=str(mh)),
            current_difference=dict(**estimates(list(map(float,dc)),c),mean_exact=str(mc)),historical_mean_pass=hp,current_mean_pass=cp,
            all_history_positive=ah,all_current_negative=ac,registered_joint_pass=None if smoke or not ref else bool(hp and cp and ah and ac))
        arm_metrics={arm:{n:estimates([float(values[(level,arm,s)][n]) for s in seeds],c) for n in ['current','history','long_history']} for arm in c['arms']}
        levels[level]=dict(reference_valid=ref,arm_metrics=arm_metrics,primary=primary)
        for s,h,cur in zip(seeds,dh,dc):
            pairs.append(dict(level=level,seed=s,historical_gain=float(h),current_difference=float(cur),history_numerator=h.numerator,
                history_denominator=h.denominator,current_numerator=cur.numerator,current_denominator=cur.denominator))
        for first,second in combinations(c['arms'],2):
            for n in ['current','history','long_history']:
                descriptive.append(dict(level=level,first=first,second=second,dimension=n,
                    **estimates([float(values[(level,first,s)][n]-values[(level,second,s)][n]) for s in seeds],c)))
    curves=[]
    for (level,arm,lag),part in f.groupby(['level','arm','lag']):
        row=dict(level=level,arm=arm,lag=int(lag))
        for metric in ['accuracy','baseline_excess','chance_adjusted','frequency_accuracy','current_accuracy']:
            row.update({metric+'_'+name:value for name,value in estimates(part.sort_values('seed')[metric],c).items()})
        curves.append(row)
    primary=levels['legacy5']['primary']['registered_joint_pass']
    outcome='smoke' if smoke else 'assay-invalid' if not levels['legacy5']['reference_valid'] else 'PASS' if primary else 'FAIL'
    summary=dict(smoke=smoke,outcome=outcome,primary_pass=primary,levels=levels,cases=len(score_rows),lag_heads=len(f),criterion_unchanged=True,
                 biological_plasticity_performed=False,m1_outcome_unchanged='assay-invalid',m1_informed_design=True)
    return summary,pd.DataFrame(score_rows),pd.DataFrame(pairs),pd.DataFrame(descriptive),pd.DataFrame(curves)

def paired_checks(out,c):
    for block in c['blocks']:
        identities=[]
        for level in c['levels']:
            if block['seed'] not in c['blocks_by_level'][level]:continue
            reference=out/f'{level}_full_s{block["seed"]}'
            with np.load(reference/'case.npz',allow_pickle=False) as a:
                for arm in c['arms']:
                    root=out/f'{level}_{arm}_s{block["seed"]}'
                    with np.load(root/'case.npz',allow_pickle=False) as b:
                        for n in ['input_patterns','observed_indices','train_symbols','test_symbols','ytrain','ytest','zero_state_observed_prototypes']:
                            np.testing.assert_array_equal(a[n],b[n])
                    graph=read(root/'graph.json');identities.append(graph)
                    assert graph['weight_sha256']==read(reference/'graph.json')['weight_sha256']
        for name in ['input_root_ids','observation_root_ids']:
            assert all(x[name]==identities[0][name] for x in identities)

def plot(f,scores,out,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'full':'#17679e','carry_only':'#d17e20','synaptic_only':'#188c72','instantaneous':'#8a6b9c'}
    fig,axes=plt.subplots(2,2,figsize=(12,9))
    for col,level in enumerate(['legacy5','brain5']):
        ax=axes[0,col]
        for arm,color in colors.items():
            p=f[(f.level==level)&(f.arm==arm)&(f.lag>0)]
            for _,block in p.groupby('seed'):ax.plot(block.lag,block.accuracy,color=color,alpha=.16,lw=.8)
            mean=p.groupby('lag').accuracy.mean();ax.plot(mean.index,mean,'o-',color=color,label=arm,markersize=3)
        ax.axhline(.1,color='gray',ls=':');ax.set(title=level,xlabel='Historical lag',ylabel='Independent test accuracy',ylim=(0,1.02));ax.legend(fontsize=8)
        ax=axes[1,col]
        for arm,color in colors.items():
            p=scores[(scores.level==level)&(scores.arm==arm)]
            ax.scatter(p.current,.1+.9*p.history,color=color,alpha=.3,s=22)
            ax.scatter(p.current.mean(),.1+.9*p.history.mean(),color=color,s=65,label=arm)
        ax.axhline(.1,color='gray',ls=':');ax.set(xlabel='Current-symbol accuracy (lag 0)',ylabel='Mean historical accuracy (lags 1-5)',xlim=(.7,1.02),ylim=(.05,1.02))
        ax.legend(fontsize=8)
    fig.suptitle('M2 current/history decoding - registered outcome: '+summary['outcome'])
    fig.tight_layout();fig.savefig(out/'tradeoff-tdc.png',dpi=180);plt.close(fig)

def run(out,smoke):
    c=config(smoke);source=source_record(c)
    with attempt(out,c,'tradeoff-smoke' if smoke else 'tradeoff-main'):
        write(out/'config.json',c);write(out/'source.json',source);write(out/'historical-preservation.json',history_preserved(c))
        jobs=[dict(config=c,level=l,arm=a,block=b,out=str(out/f'{l}_{a}_s{b["seed"]}')) for l in c['levels'] for b in c['blocks'] if b['seed'] in c['blocks_by_level'][l] for a in c['arms']]
        results,usage=pooled(case_job,jobs,c['neural_workers'],c,'M2 tradeoff')
        f=pd.DataFrame([row for r in results for row in r['rows']]).sort_values(['level','arm','seed','lag']).reset_index(drop=True)
        assert len(f)==(168 if smoke else 756)
        summary,scores,pairs,pairwise,curves=summarize(f,c,smoke)
        for name,frame in [('raw-lags',f),('arm-scores',scores),('paired-tradeoff',pairs),('descriptive-pairwise',pairwise),('curve-summary',curves)]:frame.to_csv(out/(name+'.csv'),index=False)
        pd.DataFrame([r['diagnostics'] for r in results]).sort_values(['level','arm','seed']).to_csv(out/'diagnostics.csv',index=False)
        write(out/'summary.json',summary);write(out/'process-tree-resources.json',usage)
        write(out/'job-resources.json',[dict(identity=r['identity'],resources=r['resources']) for r in results])
        paired_checks(out,c);plot(f,scores,out,summary)
        for n,h in source['hashes'].items():assert sha(n)==h,n
        history_preserved(c)
    seal(out,dict(complete=True,smoke=smoke,config=c,source_commit=source['source_commit']))
    print(dict(outcome=summary['outcome'],primary_pass=summary['primary_pass'],cases=summary['cases']),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    with threadpool_limits(1):run(a.out,a.smoke)
