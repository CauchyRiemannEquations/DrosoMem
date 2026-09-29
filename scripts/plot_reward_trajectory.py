"""All seed trajectories, without best-epoch selection."""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from alphabet_memory import check
from flying.training import whole_brain_memory as core
p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args();check(a.root);a.out.mkdir(parents=True,exist_ok=False)
b=pd.read_csv(a.root/'seed-blocks.csv');n=pd.read_csv(a.root/'neural.csv')
fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
for col,cohort in enumerate(['discovery','confirmation']):
    for row,metric in enumerate(['forward_gain','above_random']):
        ax=axes[row,col];q=b[(b.cohort==cohort)&(b['mode']=='noisy')]
        for seed,g in q.groupby('seed'):ax.plot(g.epoch,g[metric]*100,'o-',label=str(seed))
        ax.axhline(0,color='gray');ax.axhline(.1,color='red',ls='--');ax.set(xticks=[0,1,5,10],xlabel='Training epoch',ylabel=metric.replace('_',' ')+' (pp)',title=cohort);ax.legend(fontsize=8)
fig.savefig(a.out/'temporal-utility.png',dpi=170);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for ax,cohort in zip(axes,['discovery','confirmation']):
    q=b[(b.cohort==cohort)&(b['mode']=='noisy')]
    for seed,g in q.groupby('seed'):ax.plot(g.epoch,g.baseline*100,'o-',label=str(seed))
    ax.set(xticks=[0,1,5,10],xlabel='Training epoch',ylabel='Unperturbed noisy fixed-code accuracy (%)',title=cohort);ax.legend(fontsize=8)
fig.savefig(a.out/'temporal-accuracy.png',dpi=170);plt.close(fig)
core.write_json(a.out/'manifest.json',dict(source_manifest_sha256=core.sha256(a.root/'manifest.json'),plot_script_sha256=core.sha256(__file__),artifacts={p.name:core.sha256(p) for p in a.out.iterdir() if p.is_file()}))
