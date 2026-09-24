"""Validate saved recall records and render measured results."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flying.evaluation.metrics import pi_memory_score
from flying.data.connectome import load_connectome
from flying.data.mushroom_body import load_roles
p=argparse.ArgumentParser(); p.add_argument('--output',default='results/phase3'); args=p.parse_args(); out=Path(args.output)
df=pd.read_csv(out/'results.csv'); records=[json.loads(x) for x in (out/'recalls.jsonl').read_text().splitlines()]
cfg=json.loads((out/'config.json').read_text()); keys=['circuit','seed','normalization','model','length','view']
expected=int(np.prod([len(cfg[k]) for k in ('circuits','seeds','normalizations','models','lengths','views')]))
assert len(df)==len(records)==expected and not df.duplicated(keys).any()
lookup=df.set_index(keys)
for r in records:
    score=pi_memory_score(list(r['target']),list(r['prediction']))
    assert score==r['pi_memory_score']==lookup.loc[tuple(r[k] for k in keys),'pi_memory_score']
    assert r['censored']==(score==r['horizon'])
    assert r['first_error_digit_index']==(None if r['censored'] else r['prompt_length']+score)
summary=df.groupby(['length','view','normalization','model']).agg(n=('seed','size'),score_mean=('pi_memory_score','mean'),score_min=('pi_memory_score','min'),score_max=('pi_memory_score','max'),train_accuracy=('train_accuracy','mean'),heldout_accuracy=('heldout_accuracy','mean')).reset_index()
summary.to_csv(out/'summary.csv',index=False)
models=cfg['models']; labels=['Fly','Role shuffled','Random','No DAN','No non-KC→KC','Leaky only']
fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
for row,length in enumerate(cfg['lengths']):
    for col,norm in enumerate(cfg['normalizations']):
        ax=axes[row,col]
        for i,model in enumerate(models):
            values=df.query('length==@length and normalization==@norm and model==@model and view=="mbon"').pi_memory_score.to_numpy()
            ax.scatter(i+np.linspace(-.13,.13,len(values)),values,c='#297b91',alpha=.65,s=35)
            ax.plot([i-.24,i+.24],[values.mean()]*2,c='#db6544',lw=2)
        ax.set_xticks(range(len(models)),labels,rotation=22,ha='right'); ax.set_ylabel('Pi Memory Score (generated digits)')
        ax.set_title(f'{length} training digits · {norm}'); ax.grid(axis='y',alpha=.2)
        if length==50: ax.axhline(47,ls=':',c='gray',label='47-digit evaluation cap'); ax.legend(fontsize=8)
fig.suptitle('KC stimulation → MBON readout | 2 circuits × 3 seeds\nDots: runs; orange line: mean. Prompt of 3 digits excluded.',fontsize=14)
fig.savefig(out/'overview.png',dpi=150); plt.close(fig)
activity=np.load(out/(Path(cfg['circuits'][0]).name+'_activity.npz'))
fig,axes=plt.subplots(2,1,figsize=(10,7),constrained_layout=True)
s=activity['states']; roles=activity['roles']; ix=np.concatenate([np.flatnonzero(roles==role)[:16] for role in ['KC','MBON','DAN','APL']])
im=axes[0].imshow(s[:,ix].T,aspect='auto',cmap='coolwarm',vmin=-1,vmax=1)
axes[0].set_ylabel('Selected neurons by role'); axes[0].set_xlabel('Digit index'); axes[0].set_title('Circuit 701 · seed 242 · spectral · fixed real connectivity'); fig.colorbar(im,ax=axes[0],label='Rate-model state')
for role in ['KC','MBON','DAN','APL']: axes[1].plot(s[:,roles==role].mean(axis=1),label=role)
axes[1].set(xlabel='Digit index',ylabel='Mean state by role'); axes[1].legend(); fig.savefig(out/'activity.png',dpi=150);plt.close(fig)
h=pd.read_csv(out/'training.csv'); fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
for ax,norm in zip(axes,cfg['normalizations']):
    for model in models:
        q=h.query('normalization==@norm and model==@model and view=="mbon"')
        ax.plot(q.epoch,q.accuracy,label=model)
    ax.set(title=norm,xlabel='Readout epoch',ylabel='Teacher-forced training accuracy');ax.set_ylim(0,1)
axes[1].legend(fontsize=8);fig.suptitle('200 training digits · circuit 701 · seed 242 · MBON readout');fig.savefig(out/'training.png',dpi=150);plt.close(fig)
r=next(r for r in records if r['seed']==242 and r['view']=='mbon' and r['model']=='fly' and r['length']==200 and r['normalization']=='spectral')
fig,ax=plt.subplots(figsize=(11,2.4));ax.axis('off'); target=r['prompt']+r['target'];pred=r['prompt']+r['prediction']
ax.text(.02,.8,'Target:',fontfamily='monospace',transform=ax.transAxes)
for i,digit in enumerate(target[:55]):
    ax.text(.165+i*.0133,.8,digit,fontfamily='monospace',transform=ax.transAxes)
ax.text(.02,.55,'Prediction:',fontfamily='monospace',transform=ax.transAxes)
for i,digit in enumerate(pred[:55]):
    error=i==r['first_error_digit_index']
    ax.text(.165+i*.0133,.55,digit,fontfamily='monospace',transform=ax.transAxes,
            color='white' if error else 'black',bbox=dict(facecolor='#b03528',pad=1) if error else None)
ax.text(.02,.22,f"Pi Memory Score: {r['pi_memory_score']} | first error at digit index {r['first_error_digit_index']} (0-based, includes leading 3)",transform=ax.transAxes,color='#b03528')
fig.savefig(out/'recall.png',dpi=150,bbox_inches='tight');plt.close(fig)
(out/'score_verification.json').write_text(json.dumps(dict(records_checked=len(records),all_scores_match=True,unique_conditions=True),indent=2))
print('Validated',len(records),'recall records; figures saved')
