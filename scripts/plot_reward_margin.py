"""All seed margin sensitivities and fixed-code accuracy through time."""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from alphabet_memory import check
from flying.training import whole_brain_memory as core
p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args();check(a.root);a.out.mkdir(parents=True,exist_ok=False)
b=pd.read_csv(a.root/'seed-blocks.csv');cohorts=['discovery','confirmation','fresh_confirmation']
fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
for col,cohort in enumerate(cohorts):
 for row,metric in enumerate(['local','above_random']):
  ax=axes[row,col];q=b[(b.cohort==cohort)&(b['mode']=='noisy')]
  for seed,g in q.groupby('seed'):ax.plot(g.epoch,g[metric]*1e6,'o-',label=str(seed))
  ax.axhline(0,color='gray');ax.set(xticks=[0,1,5,10],xlabel='Training epoch',ylabel=metric.replace('_',' ')+' (micro score / reference step)',title=cohort);ax.legend(fontsize=8)
fig.suptitle('Infinitesimal margin sensitivity; not finite-step accuracy')
fig.savefig(a.out/'margin-sensitivity.png',dpi=170);plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
for ax,cohort in zip(axes,cohorts):
 q=b[(b.cohort==cohort)&(b['mode']=='noisy')]
 for seed,g in q.groupby('seed'):ax.plot(g.epoch,g.accuracy*100,'o-',label=str(seed))
 ax.set(xticks=[0,1,5,10],xlabel='Training epoch',ylabel='Unchanged fixed-code accuracy (%)',title=cohort);ax.legend(fontsize=8)
fig.savefig(a.out/'margin-baseline-accuracy.png',dpi=170);plt.close(fig)
core.write_json(a.out/'manifest.json',dict(source_manifest_sha256=core.sha256(a.root/'manifest.json'),plot_script_sha256=core.sha256(__file__),artifacts={p.name:core.sha256(p) for p in a.out.iterdir() if p.is_file()}))
