"""Predeclared block inference; every lag and negative control retained."""
import argparse
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from flying.training import whole_brain_memory as core
from run_delayed_symbol import check, read, measures, independent_checks


def estimate(values):
    x=np.asarray(values,dtype=float);rng=np.random.default_rng(21399)
    draws=x[rng.integers(0,len(x),size=(10000,len(x)))].mean(axis=1)
    sd=float(x.std(ddof=1)) if len(x)>1 else 0.
    return dict(n=len(x),mean=float(x.mean()),median=float(np.median(x)),variance=sd**2 if len(x)>1 else None,
        bootstrap95=np.quantile(draws,[.025,.975]).tolist(),paired_dz=float(x.mean()/sd) if sd else None,
        positive=int((x>0).sum()),zero=int((x==0).sum()),negative=int((x<0).sum()))


@threadpool_limits.wrap(limits=1)
def analyze(source,out):
    m=check(source);c=m['config'];rows=[];matched={}
    for b in c['blocks']:
        for circuit in c['circuit_seeds']:
            pair=[]
            for level in c['conditions']:
                path=source/f'{level}_c{circuit}_s{b["seed"]}';meta=check(path)
                with np.load(path/'checkpoint.npz',allow_pickle=False) as a:
                    independent_checks(a,c);metrics=measures(a,c);assert metrics==meta['metrics']
                    pair.append(dict(graph=meta['graph'],train=a['train_symbols'].copy(),test=a['test_symbols'].copy()))
                rows.extend(dict(**meta['identity'],**r,**meta['neural']) for r in metrics)
            for k in ['observation_root_ids','input_root_ids','input_mapping_sha256']:
                assert pair[0]['graph'][k]==pair[1]['graph'][k],k
            for k in ['train','test']:np.testing.assert_array_equal(pair[0][k],pair[1][k])
    f=pd.DataFrame(rows);metrics=['test_accuracy','train_accuracy','frequency_accuracy','null_accuracy','r2_vs_training_frequency']
    f['frequency_excess']=f.test_accuracy-f.frequency_accuracy;f['null_excess']=f.test_accuracy-f.null_accuracy
    metrics+=['frequency_excess','null_excess']
    raw=f[f.lag.isin(c['primary_lags'])].groupby(['level','seed','train_seed','test_seed','circuit_seed'])[metrics].mean().reset_index()
    blocks=raw.groupby(['level','seed'])[metrics].mean()
    confirmation=c['study'].endswith('confirmation');required=3 if confirmation else 4
    conditions={}
    for level in c['conditions']:
        p=blocks.loc[level]
        conditions[level]={k:estimate(p[k]) for k in metrics}
        conditions[level]['past_decoding_gate']=bool(p.frequency_excess.mean()>=.05 and p.null_excess.mean()>=.05 and
            (p.frequency_excess>=.05).sum()>=required and (p.null_excess>=.05).sum()>=required and p.r2_vs_training_frequency.mean()>0)
    delta=blocks.loc['brain1']-blocks.loc['legacy5'];graph={k:estimate(delta[k]) for k in metrics}
    graph['superiority_gate']=bool(delta.test_accuracy.mean()>=.03 and (delta.test_accuracy>0).sum()>=required)
    curves=f.groupby(['level','seed','lag'])[metrics].mean().reset_index()
    lagrows=[]
    for lag in c['lags']:
        for level in c['conditions']:
            p=curves[(curves.level==level)&(curves.lag==lag)]
            for metric in metrics:lagrows.append(dict(level=level,lag=lag,metric=metric,**estimate(p[metric])))
    result=dict(study=c['study'],primary_lags=c['primary_lags'],conditions=conditions,graph_contrast=graph,
        inference='strata averaged within paired input/stream seed block; descriptive 10000 bootstrap resamples seed21399',
        source_manifest_sha256=core.sha256(source/'manifest.json'),matched_interface_and_streams=True,
        confirmation_triggered=bool(graph['superiority_gate'] and not confirmation and len(c['blocks'])==5))
    out.mkdir(parents=True,exist_ok=False)
    f.to_csv(out/'all-lag-metrics.csv',index=False);raw.to_csv(out/'raw-seed-table.csv',index=False)
    blocks.to_csv(out/'seed-blocks.csv');delta.to_csv(out/'paired-differences.csv');pd.DataFrame(lagrows).to_csv(out/'lag-summary.csv',index=False)
    core.write_json(out/'summary.json',result)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors={'legacy5':'#1976b5','brain1':'#e17430'};fig,axes=plt.subplots(1,3,figsize=(14,4.2))
    for level in c['conditions']:
        p=curves[curves.level==level];g=p.groupby('lag')[metrics].mean().reindex(c['lags']);ix=np.arange(len(c['lags']))
        axes[0].plot(ix,g.test_accuracy,'o-',label=level,color=colors[level])
        axes[0].plot(ix,g.null_accuracy,'--',color=colors[level],alpha=.5)
        axes[1].plot(ix,g.r2_vs_training_frequency,'o-',label=level,color=colors[level])
        for _,block in p.groupby('seed'):axes[0].plot(ix,block.set_index('lag').loc[c['lags'],'test_accuracy'],color=colors[level],alpha=.15)
    for ax in axes[:2]:ax.set(xticks=np.arange(len(c['lags'])),xticklabels=c['lags'],xlabel='Lag (symbol steps)');ax.legend(frameon=False)
    axes[0].axhline(.1,color='gray',ls=':');axes[0].set(ylabel='Independent test accuracy',ylim=(-.02,1.02),title='Solid: true labels; dashed: shifted labels')
    axes[1].axhline(0,color='gray',ls=':');axes[1].set(ylabel='R2 vs training frequency',title='Negative scores retained')
    for seed,row in delta.iterrows():axes[2].scatter(seed,row.test_accuracy*100,color='#7055a2')
    e=graph['test_accuracy'];axes[2].axhline(e['mean']*100,color='#7055a2');axes[2].axhspan(e['bootstrap95'][0]*100,e['bootstrap95'][1]*100,color='#7055a2',alpha=.15)
    axes[2].axhline(0,color='gray',ls='--');axes[2].axhline(3,color='gray',ls=':');axes[2].set(xlabel='Paired seed block',ylabel='Whole - partial (percentage points)',title='Primary past-lag aggregate');axes[2].ticklabel_format(useOffset=False,style='plain')
    fig.tight_layout();fig.savefig(out/'delay-curves.png',dpi=180);plt.close(fig)
    core.write_json(out/'manifest.json',dict(source=str(source),source_manifest_sha256=result['source_manifest_sha256'],script_sha256=core.sha256(__file__),
        artifacts={p.name:core.sha256(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();analyze(a.source,a.out)
