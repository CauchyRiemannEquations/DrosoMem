"""Registered relative-noise curves, per-seed lines and clean anchors."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from alphabet_memory import check
from pathway_memory import seal
from flying.training import whole_brain_memory as core


def plot(root,out):
    m=check(root);c=m['config'];out.mkdir(parents=True,exist_ok=False)
    f=pd.read_csv(root/'raw-metrics.csv',float_precision='round_trip');f=f[~f.reused_clean];colors=dict(legacy5='#235c91',brain5='#c07822',brain1='#15836d')
    auto=f[f['mode']=='autonomous'].copy();keys=['cohort','seed','circuit_seed','level']
    clean=auto[auto.strength==0][keys+['pi_memory_score']].rename(columns={'pi_memory_score':'baseline'})
    auto=auto.merge(clean,on=keys);auto['retention']=(auto.pi_memory_score/auto.baseline.replace(0,np.nan)).clip(upper=1)
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for i,co in enumerate(c['cohorts']):
        for j,(metric,label,frame) in enumerate([('pi_memory_score','Autonomous exact prefix',auto),('retention','Capped prefix retention',auto),('accuracy','Teacher-forced accuracy',f[f['mode']=='teacher'])]):
            ax=axes[i,j]
            for g in c['conditions']:
                z=frame[(frame.cohort==co)&(frame.level==g)].groupby(['seed','strength'])[metric].agg(lambda x:np.mean(x.to_numpy())).reset_index()
                for seed,a in z.groupby('seed'):ax.plot(range(4),a.sort_values('strength')[metric],color=colors[g],alpha=.2,lw=.8)
                mean=z.groupby('strength')[metric].mean();ax.plot(range(4),mean,color=colors[g],marker='o',label=g)
            ax.set_xticks(range(4),['0']+[f'{x:g}' for x in c['relative_strengths']]);ax.set_xlabel('Relative noise r (unequal spacing)')
            ax.set_title(co+': '+label);ax.grid(alpha=.15)
            if j>0:ax.set_ylim(-.03,1.05)
    axes[0,0].legend();fig.suptitle('Noise sigma = r × median clean training MBON SD; frozen heads')
    fig.savefig(out/'relative-noise-curves.png',dpi=160);plt.close(fig)
    seal(out,c,m['context'],result_manifest_sha256=core.sha256(root/'manifest.json'),purpose='Relative noise figure')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args();plot(a.source,a.out)
