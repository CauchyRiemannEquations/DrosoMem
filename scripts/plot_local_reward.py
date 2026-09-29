"""Descriptive figures for the sealed local-reward comparison; no decision changes."""
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parent))
from alphabet_memory import read,check
from flying.training import whole_brain_memory as core
p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--out',type=Path,required=True)
a=p.parse_args();m=check(a.root);a.out.mkdir(exist_ok=False,parents=True)
f=pd.read_csv(a.root/'seed-blocks.csv');s=read(a.root/'summary.json');order=[f'{n}/{arm}' for n in ['real'] for arm in ['contingent','frozen','yoked']]
fig,axes=plt.subplots(1,2,figsize=(11,6),layout='constrained')
for ax,cohort in zip(axes,['discovery','confirmation']):
 values=np.array([[s[cohort]['cells'][key+'/'+decoder]['mean']*100 for decoder in ['ridge','fixed']] for key in order]);im=ax.imshow(values,vmin=0,vmax=100,cmap='viridis',aspect='auto')
 for (i,j),v in np.ndenumerate(values):ax.text(j,i,f'{v:.2f}',ha='center',va='center',color='white' if v<55 else 'black')
 ax.set(xticks=[0,1],xticklabels=['Trained ridge','Fixed code'],yticks=range(len(order)),yticklabels=order,title=cohort)
fig.colorbar(im,ax=axes,label='Lag2 test accuracy (%)',shrink=.8);fig.savefig(a.out/'decoder-comparison.png',dpi=170);plt.close(fig)
pairs=pd.read_csv(a.root/'paired-differences.csv');names=['fixed_vs_frozen','fixed_vs_yoked','ridge_vs_frozen','ridge_vs_yoked']
fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
for ax,cohort in zip(axes,['discovery','confirmation']):
 for j,name in enumerate(names):
  vals=pairs[(pairs.cohort==cohort)&(pairs.contrast==name)].difference.to_numpy()*100;ax.scatter(np.full(len(vals),j),vals);ax.plot([j-.15,j+.15],[vals.mean()]*2,'k-')
 ax.axhline(0,color='gray');ax.axhline(5,color='red',ls='--');ax.set(xticks=range(len(names)),xticklabels=names,ylabel='Paired accuracy difference (pp)',title=cohort);ax.tick_params(axis='x',labelrotation=25)
fig.savefig(a.out/'paired-effects.png',dpi=170);plt.close(fig)
history=[]
for path in a.root.glob('*/*/history.json'):
 ident=read(path.parent/'manifest.json')['identity']
 if ident['cohort']=='smoke':continue
 history.extend(dict(**ident,**r) for r in read(path))
h=pd.DataFrame(history);h.to_csv(a.out/'training-histories.csv',index=False)
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for ax,cohort in zip(axes,['discovery','confirmation']):
 for arm,style in [('frozen',':'),('contingent','-'),('yoked','--')]:
  q=h[(h.cohort==cohort)&(h.arm==arm)].groupby('epoch').online_accuracy.mean();ax.plot(q.index,q,style,label=arm)
 ax.set(title=cohort,xlabel='Epoch',ylabel='Noisy online correctness fraction');ax.legend()
fig.savefig(a.out/'training-curves.png',dpi=170);plt.close(fig)
core.write_json(a.out/'manifest.json',dict(source_manifest_sha256=core.sha256(a.root/'manifest.json'),scope='descriptive figures; all criteria unchanged',plot_script_sha256=core.sha256(__file__),artifacts={p.name:core.sha256(p) for p in a.out.iterdir() if p.is_file()}))
print('Wrote three figures and full training histories')
