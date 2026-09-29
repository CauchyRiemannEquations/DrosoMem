"""Descriptive paired clean/noisy accuracy and learning interactions."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from alphabet_memory import check
from flying.training import whole_brain_memory as core
p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args();check(a.root);a.out.mkdir(parents=True,exist_ok=False)
b=pd.read_csv(a.root/'seed-blocks.csv');pairs=pd.read_csv(a.root/'paired-differences.csv')
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for ax,cohort in zip(axes,['discovery','confirmation']):
    q=b[b.cohort==cohort]
    for arm,color in [('contingent','tab:orange'),('frozen','tab:blue'),('yoked','tab:green')]:
        for seed,rows in q[q.arm==arm].groupby('seed'):
            v=rows.set_index('mode').accuracy;ax.plot([0,1],[v['clean']*100,v['noisy']*100],'o-',color=color,alpha=.65,label=arm if seed==q.seed.min() else None)
    ax.set(xticks=[0,1],xticklabels=['Clean','Training noise SD .02'],ylabel='Fixed-code test accuracy (%)',title=cohort);ax.legend()
fig.savefig(a.out/'noise-accuracy.png',dpi=170);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for ax,cohort in zip(axes,['discovery','confirmation']):
    for j,ctrl in enumerate(['frozen','yoked']):
        v=pairs[(pairs.cohort==cohort)&(pairs.contrast=='interaction_vs_'+ctrl)].difference.to_numpy()*100;ax.scatter(j+np.linspace(-.08,.08,len(v)),v);ax.plot([j-.2,j+.2],[v.mean()]*2,'k-')
    ax.axhline(0,color='gray');ax.axhline(1,color='red',ls='--');ax.set(xticks=[0,1],xticklabels=['Benefit vs frozen','Benefit vs yoked'],ylabel='Noisy minus clean learning benefit (pp)',title=cohort)
fig.savefig(a.out/'noise-interactions.png',dpi=170);plt.close(fig)
core.write_json(a.out/'manifest.json',dict(source_manifest_sha256=core.sha256(a.root/'manifest.json'),plot_script_sha256=core.sha256(__file__),artifacts={p.name:core.sha256(p) for p in a.out.iterdir() if p.is_file()}))
print('Saved two noise comparison figures')
