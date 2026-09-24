"""Independent saved-result verification and temporal diagnostic figures."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flying.evaluation.metrics import pi_memory_score
p=argparse.ArgumentParser();p.add_argument('--output',default='results/phase3b');args=p.parse_args();out=Path(args.output)
cfg=json.loads((out/'config.json').read_text()); pi=pd.read_csv(out/'pi.csv');memory=pd.read_csv(out/'memory.csv')
keys=['circuit','seed','normalization','model','microsteps','view'];expected=int(np.prod([len(cfg[k]) for k in ['circuits','seeds','normalizations','models','microsteps','views']]))
assert len(pi)==expected*len(cfg['pi_readouts']) and not pi.duplicated(keys+['readout']).any()
assert len(memory)==expected*len(cfg['lags']) and not memory.duplicated(keys+['lag']).any()
recalls=[json.loads(line) for line in (out/'recalls.jsonl').read_text().splitlines()];assert len(recalls)==len(pi)
lookup=pi.set_index(keys+['readout'])
for r in recalls:
    score=pi_memory_score(list(r['target']),list(r['prediction']))
    assert score==r['pi_memory_score']==lookup.loc[tuple(r[k] for k in keys+['readout']),'pi_memory_score']
    assert r['censored']==(score==r['horizon'])
    assert r['first_error_digit_index']==(None if r['censored'] else r['prompt_length']+score)
with np.load(out/'delayed_predictions.npz') as d:
    predicted=d['predictions'];targets=d['targets']
assert predicted.shape==targets.shape==(expected,cfg['memory_test_count'],len(cfg['lags']))
for r in memory.itertuples():
    j=cfg['lags'].index(r.lag);actual=np.mean(predicted[r.prediction_index,:,j]==targets[r.prediction_index,:,j])
    assert np.isclose(actual,r.accuracy,rtol=0,atol=1e-14)
summary=pi.groupby(['view','normalization','microsteps','model','readout']).agg(n=('seed','size'),score_mean=('pi_memory_score','mean'),score_min=('pi_memory_score','min'),score_max=('pi_memory_score','max'),train_accuracy=('train_accuracy','mean'),heldout_accuracy=('heldout_accuracy','mean')).reset_index()
summary.to_csv(out/'pi_summary.csv',index=False)
mem=memory.groupby(['view','normalization','microsteps','model','lag']).agg(n=('seed','size'),accuracy=('accuracy','mean'),accuracy_min=('accuracy','min'),accuracy_max=('accuracy','max'),r2=('r2_vs_training_frequency','mean')).reset_index()
mem.to_csv(out/'memory_summary.csv',index=False)
colors=dict(fly='#24788a',role_shuffled='#d4723c',random='#7d60a9',leaky_only='#808080')
fig,axes=plt.subplots(2,4,figsize=(15,7),constrained_layout=True)
for row,norm in enumerate(cfg['normalizations']):
    for col,steps in enumerate(cfg['microsteps']):
        ax=axes[row,col]
        for model in cfg['models']:
            q=mem.query('view=="mbon" and normalization==@norm and microsteps==@steps and model==@model')
            ax.plot(q.lag,q.accuracy,'o-',ms=3,label=model,color=colors[model])
        ax.axhline(.1,ls=':',c='gray');ax.set_xscale('symlog',linthresh=1);ax.set_xticks([0,1,2,4,8,16,32],[0,1,2,4,8,16,32]);ax.set_ylim(0,1.03);ax.set_xlim(0,32)
        ax.set(title=f'{norm} · {steps} updates',xlabel='Input lag (digits)',ylabel='Held-out digit reconstruction accuracy');ax.grid(alpha=.15)
axes[0,0].legend(fontsize=8);fig.suptitle('Temporal information at MBON | iid train/test streams, 48 observed neurons\nMeans over 2 circuit samples × 3 seeds; dotted line: 10% chance',fontsize=13)
fig.savefig(out/'memory_curves.png',dpi=150);plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
for row,view in enumerate(cfg['views']):
    for col,norm in enumerate(cfg['normalizations']):
        ax=axes[row,col]
        for model in cfg['models']:
            q=summary.query('view==@view and normalization==@norm and model==@model and readout=="softmax"')
            ax.plot(q.microsteps,q.score_mean,'o-',label=model,color=colors[model])
            ax.fill_between(q.microsteps,q.score_min,q.score_max,alpha=.12,color=colors[model])
        ax.set(title=f'{view} (48 features) · {norm}',xlabel='Updates per digit',ylabel='Pi Memory Score');ax.set_xticks(cfg['microsteps']);ax.set_ylim(-.3,float(summary.query('readout=="softmax"').score_max.max())+1);ax.grid(alpha=.15)
axes[0,0].legend(fontsize=8);fig.suptitle('200 training digits | fixed 400-epoch softmax readout\nMean and min–max across 6 runs; supplied 314 excluded; cap = 197',fontsize=13)
fig.savefig(out/'pi_update_sweep.png',dpi=150);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
for ax,norm in zip(axes,cfg['normalizations']):
    for kind in cfg['pi_readouts']:
        q=summary.query('view=="mbon" and normalization==@norm and model=="fly" and readout==@kind')
        ax.plot(q.microsteps,q.train_accuracy,'o-',label=kind)
    ax.set(title=norm,xlabel='Updates per digit',ylabel='Teacher-forced training accuracy',ylim=(0,1.02));ax.set_xticks(cfg['microsteps']);ax.legend();ax.grid(alpha=.2)
fig.suptitle('Same MBON states, two affine readout fitting methods | 200 training digits');fig.savefig(out/'readout_fit.png',dpi=150);plt.close(fig)
(out/'verification.json').write_text(json.dumps(dict(pi_recalls_checked=len(recalls),delayed_accuracy_rows_checked=len(memory),saved_test_predictions=int(predicted.size),unique_conditions=True,all_recomputed_scores_match=True),indent=2))
print('Verified',len(recalls),'pi recalls and',len(memory),'delay measurements')
print(summary.query('view=="mbon" and model=="fly"').round(3).to_string(index=False))
