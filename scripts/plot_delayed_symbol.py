"""Readable publication rendering of saved statistics; no metric recomputation."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flying.training import whole_brain_memory as core
from run_delayed_symbol import read,check


def plot(source,out):
    check(source);s=read(source/'summary.json');f=pd.read_csv(source/'all-lag-metrics.csv')
    delta=pd.read_csv(source/'paired-differences.csv');lags=sorted(f.lag.unique());ix=np.arange(len(lags))
    curves=f.groupby(['level','seed','lag'])[['test_accuracy','null_accuracy','r2_vs_training_frequency']].mean().reset_index()
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(15,4.4));colors={'legacy5':'#1976b5','brain1':'#e17430'}
    for level,color in colors.items():
        p=curves[curves.level==level];g=p.groupby('lag').mean(numeric_only=True).reindex(lags)
        axes[0].plot(ix,g.test_accuracy,'o-',color=color,label=level)
        axes[0].plot(ix,g.null_accuracy,'--',color=color,alpha=.55)
        axes[1].plot(ix,g.r2_vs_training_frequency,'o-',color=color,label=level)
        for _,b in p.groupby('seed'):axes[0].plot(ix,b.set_index('lag').loc[lags,'test_accuracy'],color=color,alpha=.15)
    for ax in axes[:2]:ax.set(xticks=ix,xticklabels=lags,xlabel='Tested lag (categorical spacing)');ax.legend(frameon=False)
    axes[0].axhline(.1,color='gray',ls=':');axes[0].set(ylabel='Independent test accuracy',ylim=(-.02,1.02),title='True labels vs shifted-label controls')
    axes[1].axhline(0,color='gray',ls=':');axes[1].set(ylabel='R2 vs training frequency',title='Negative scores retained')
    e=s['graph_contrast']['test_accuracy'];x=np.arange(len(delta))
    axes[2].scatter(x,delta.test_accuracy*100,color='#7055a2');axes[2].axhline(e['mean']*100,color='#7055a2')
    axes[2].axhspan(e['bootstrap95'][0]*100,e['bootstrap95'][1]*100,color='#7055a2',alpha=.15)
    axes[2].axhline(0,color='gray',ls='--');axes[2].axhline(3,color='gray',ls=':')
    axes[2].set(xticks=x,xticklabels=delta.seed.astype(int),xlabel='Paired seed block',ylabel='Whole - partial (percentage points)',title='Primary past-lag difference; 95% interval')
    fig.tight_layout();out.mkdir(parents=True,exist_ok=False);fig.savefig(out/'delay-curves.png',dpi=180);plt.close(fig)
    core.write_json(out/'manifest.json',dict(source=str(source),source_manifest_sha256=core.sha256(source/'manifest.json'),
        script_sha256=core.sha256(__file__),rendering_only=True,original_plot_preserved=True,
        artifacts={p.name:core.sha256(p) for p in out.iterdir() if p.is_file()}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();plot(a.source,a.out)
