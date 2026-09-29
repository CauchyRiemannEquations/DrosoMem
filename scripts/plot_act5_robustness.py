"""Separate autonomous retention curves and maximum-dose teacher diagnostics."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from alphabet_memory import check
from pathway_memory import seal
from act5_robustness import FAMILIES


def plot(root,out):
    m=check(root);c=m['config'];out.mkdir(parents=True,exist_ok=False)
    doses=pd.read_csv(root/'dose-seed-table.csv');colors=dict(legacy5='#1f77b4',brain5='#e3912b',brain1='#15836d')
    fig,axes=plt.subplots(2,5,figsize=(17,7.5),sharey=True,layout='constrained')
    for i,co in enumerate(c['cohorts']):
        for j,kind in enumerate(FAMILIES):
            ax=axes[i,j]
            for level in c['conditions']:
                sub=doses[(doses.cohort==co)&(doses.kind==kind)&(doses.level==level)]
                for seed,s in sub.groupby('seed'):
                    s=s.sort_values('strength');ax.plot(range(4),[1,*s.retention],color=colors[level],alpha=.2,lw=.8)
                mean=sub.groupby('strength').retention.mean();ax.plot(range(4),[1,*mean],marker='o',color=colors[level],label=level)
            ax.axhline(.8,color='gray',ls=':',lw=1);ax.set_xticks(range(4),['0']+[f'{v:g}' for v in c['strengths'][kind]],rotation=35)
            ax.set_title(f'{co}: {kind}');ax.set_ylim(-.03,1.05);ax.set_xlabel('Registered dose (unequal spacing)');ax.grid(alpha=.15)
            if j==0:ax.set_ylabel('Capped prefix retention')
    axes[0,0].legend(fontsize=8);fig.suptitle('ACT V — frozen readouts; thin lines are paired model-seed blocks')
    fig.savefig(out/'robustness-curves.png',dpi=150);plt.close(fig)
    f=pd.read_csv(root/'raw-metrics.csv');f=f[~f.reused_clean & ((f.kind=='clean') | f.apply(lambda r:r.strength==max(c['strengths'][r.kind]) if r.kind!='clean' else True,axis=1))]
    fig,axes=plt.subplots(2,3,figsize=(13,7.5),sharey=True,layout='constrained')
    for i,co in enumerate(c['cohorts']):
        for j,g in enumerate(c['conditions']):
            ax=axes[i,j];sub=f[(f.cohort==co)&(f.level==g)];order=['clean',*FAMILIES]
            for mode,color in [('autonomous','#255886'),('teacher','#c96736')]:
                a=sub[sub['mode']==mode].groupby('kind').accuracy.mean().reindex(order)
                ax.plot(range(6),a*100,marker='o',color=color,label=mode)
            ax.set_xticks(range(6),order,rotation=35,ha='right');ax.set_title(f'{co}: {g}');ax.set_ylim(0,100);ax.grid(alpha=.15)
            if j==0:ax.set_ylabel('Position accuracy (%)')
    axes[0,0].legend();fig.suptitle('Clean and largest registered dose: teacher forcing is not autonomous recall')
    fig.savefig(out/'teacher-diagnostics.png',dpi=150);plt.close(fig)
    seal(out,c,m['context'],purpose='ACT V figures',result_manifest_sha256=__import__('hashlib').sha256((root/'manifest.json').read_bytes()).hexdigest())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args();plot(a.source,a.out)
