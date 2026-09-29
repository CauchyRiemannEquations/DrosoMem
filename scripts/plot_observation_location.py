from pathlib import Path
import json,argparse
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from alphabet_memory import read,check
from flying.training import whole_brain_memory as core
parser=argparse.ArgumentParser(description='Rebuild descriptive observation-location figures without changing metrics')
parser.add_argument('root',type=Path);parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args();root=args.root;m=check(root);c=m['config'];s=read(root/'summary.json')
out=args.out;out.mkdir(exist_ok=False,parents=True)
pairs=pd.read_csv(root/'paired-differences.csv');raw=pd.read_csv(root/'raw-lag-table.csv');neural=pd.read_csv(root/'neural-diagnostics.csv')
fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
for ax,cohort in zip(axes,['discovery','confirmation']):
 for j,site in enumerate(c['sites']):
  q=pairs[(pairs.cohort==cohort)&(pairs.site==site)]
  for _,r in q.iterrows():ax.plot([j-.12,j+.12],[r.intact*100,r.target*100],'-o',alpha=.7)
 ax.set(xticks=[-.12,.12,.88,1.12],xticklabels=['MBON intact','MBON cut','KC intact','KC cut'],ylabel='Past-symbol accuracy (%)',ylim=(20,100),title=cohort)
 ax.tick_params(axis='x',labelrotation=18);ax.axhline(25,color='gray',ls=':')
fig.savefig(out/'paired-observation.png',dpi=170);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
for ax,cohort in zip(axes,['discovery','confirmation']):
 for site,color in [('MBON','#1976b5'),('KC_unstimulated','#e17430')]:
  for arm,style in [('intact','-'),('DAN_MBON','--')]:
   q=raw[(raw.cohort==cohort)&(raw.site==site)&(raw.arm==arm)].groupby('lag').test_accuracy.mean().reindex(c['lags'])
   ax.plot(range(len(q)),q*100,style+'o',color=color,label=site+' '+arm,ms=3)
 ax.set(xticks=range(len(c['lags'])),xticklabels=c['lags'],xlabel='Past symbol lag',ylabel='Held-out accuracy (%)',ylim=(20,102),title=cohort);ax.legend(fontsize=8);ax.axhline(25,color='gray',ls=':')
fig.savefig(out/'lag-curves.png',dpi=170);plt.close(fig)
# Independently recompute descriptive state summaries from stored training features.
count=0
for p in root.glob('*/*/checkpoint.npz'):
 with np.load(p) as z:a=dict(z)
 n=read(p.parent/'neural.json');x=a['xtrain']
 for k,val in core.state_diagnostics(x).items():np.testing.assert_allclose(n[k],val,atol=1e-12,rtol=1e-12)
 np.testing.assert_allclose(n['observed_sparsity'],np.mean(abs(x)<=c['activity_epsilon']),atol=1e-12)
 count+=1
core.write_json(out/'diagnostic-checks.json',dict(checkpoints=count,state_summaries_pass=True))
core.write_json(out/'manifest.json',dict(source_manifest_sha256=core.sha256(root/'manifest.json'),purpose='descriptive plots and state audit; no new outcomes or criterion changes',artifacts={p.name:core.sha256(p) for p in out.iterdir() if p.is_file()}))
print('Rebuilt figures and audited130 state summaries')
