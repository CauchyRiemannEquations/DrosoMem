"""Plots from sealed reward-direction data; no outcome-dependent selection."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from alphabet_memory import check
from flying.training import whole_brain_memory as core

p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
check(a.root);a.out.mkdir(parents=True,exist_ok=False)
b=pd.read_csv(a.root/'seed-blocks.csv');r=pd.read_csv(a.root/'radii.csv')
fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
for ax,cohort in zip(axes,['discovery','confirmation']):
    q=b[(b.cohort==cohort)&(b['mode']=='noisy')]
    for j,(key,threshold) in enumerate([('local_directional',.25),('above_random',.1),('forward_gain',.1)]):
        v=q[key].to_numpy()*100;ax.scatter(j+np.linspace(-.08,.08,len(v)),v,label=key)
        ax.plot([j-.18,j+.18],[v.mean()]*2,'k-');ax.plot([j-.2,j+.2],[threshold]*2,'r--')
    ax.axhline(0,color='gray');ax.set(xticks=range(3),xticklabels=['Local + minus -','Above random axes','Local + minus base'],ylabel='Accuracy change (pp)',title=cohort);ax.tick_params(axis='x',labelrotation=12)
fig.savefig(a.out/'paired-responses.png',dpi=170);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
for ax,mode in zip(axes,['noisy','clean']):
    for j,cohort in enumerate(['discovery','confirmation']):
        q=b[(b.cohort==cohort)&(b['mode']==mode)]
        for row in q.itertuples():ax.plot([0,1,2],np.asarray([row.local_minus,row.baseline,row.local_plus])*100,'o-',alpha=.7,label=f'{cohort} {row.seed}')
    ax.set(xticks=[0,1,2],xticklabels=['Local minus','Initial','Local plus'],ylabel='Fixed-code accuracy (%)',title=mode);ax.legend(fontsize=7)
fig.savefig(a.out/'directional-accuracy.png',dpi=170);plt.close(fig)
fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
q=r[r.cohort!='smoke'];ax.bar(range(len(q)),100*q.relative_radius);ax.axhline(1,color='red',ls='--',label='Nominal 1%')
ax.set(xticks=range(len(q)),xticklabels=[f'{x.seed}/c{x.circuit_seed}' for x in q.itertuples()],ylabel='Actual L2 change / initial plastic L2 (%)');ax.tick_params(axis='x',labelrotation=45);ax.legend()
fig.savefig(a.out/'actual-radii.png',dpi=170);plt.close(fig)
core.write_json(a.out/'manifest.json',dict(source_manifest_sha256=core.sha256(a.root/'manifest.json'),plot_script_sha256=core.sha256(__file__),artifacts={p.name:core.sha256(p) for p in a.out.iterdir() if p.is_file()}))
print('Saved three directional-response figures')
