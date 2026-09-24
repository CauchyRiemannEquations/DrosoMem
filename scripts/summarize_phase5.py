"""Audit actual rewards, yoked signals, frozen weights and autonomous replay."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flying.brain.mushroom_body import CircuitReservoir,SelectedReadout
from flying.data.pi_digits import pi_digits
from flying.evaluation.free_recall import evaluate_recall
from flying.evaluation.metrics import pi_memory_score
p=argparse.ArgumentParser();p.add_argument('--output',default='results/phase5');args=p.parse_args();out=Path(args.output)
cfg=json.loads((out/'config.json').read_text());df=pd.read_csv(out/'results.csv');h=pd.read_csv(out/'reward_training.csv')
keys=['circuit','seed','normalization','model','reward_delay','condition','readout']
networks=int(np.prod([len(cfg[k]) for k in ['circuits','seeds','normalizations','models','reward_delays','conditions']]))
assert len(df)==networks*len(cfg['readouts']) and not df.duplicated(keys).any()
assert df.nonplastic_edges_unchanged.all() and df.evaluation_weights_frozen.all() and df.policy_frozen_during_reward.all()
assert df.max_budget_error.max()<1e-12
assert (df.query('condition=="frozen"').changed_edges==0).all()
assert (df.groupby(['circuit','seed','normalization','model']).initial_policy_sha256.nunique()==1).all()
records=[json.loads(line) for line in (out/'recalls.jsonl').read_text().splitlines()];assert len(records)==len(df)
lookup=df.set_index(keys)
for r in records:
    score=pi_memory_score(list(r['target']),list(r['prediction']))
    assert score==r['pi_memory_score']==lookup.loc[tuple(r[k] for k in keys),'pi_memory_score']
    assert r['censored']==(score==r['horizon'])
    assert r['first_error_digit_index']==(None if r['censored'] else cfg['prompt_length']+score)
event_keys=json.loads((out/'event_keys.json').read_text());events=np.load(out/'reward_events.npz')
labels=pi_digits(cfg['pi_length'])[1:]
assert events['actions'].shape==(len(event_keys),cfg['reward_epochs'],len(labels))
assert np.array_equal(events['true_rewards'],events['actions']==labels[None,None,:])
assert h.delivered_rewards.eq(len(labels)).all()
event_map={tuple(e[k] for k in keys[:-1]):e['event_index'] for e in event_keys}
for e in event_keys:
    ix=e['event_index'];actual=events['true_rewards'][ix];applied=events['applied_rewards'][ix]
    if e['condition']!='yoked_reward':assert np.array_equal(actual,applied)
    else:
        source=tuple(e[k] if k!='condition' else 'reward_trace' for k in keys[:-1])
        assert np.array_equal(applied.sum(axis=1),events['true_rewards'][event_map[source]].sum(axis=1))
    q=h
    for k in keys[:-1]:q=q[q[k]==e[k]]
    q=q.sort_values('epoch');assert len(q)==cfg['reward_epochs']
    assert np.allclose(q.actual_reward_rate,actual.mean(axis=1),rtol=0,atol=1e-14)
    assert np.allclose(q.applied_reward_rate,applied.mean(axis=1),rtol=0,atol=1e-14)
paired=df.pivot(index=[k for k in keys if k!='condition'],columns='condition',values='pi_memory_score').reset_index()
for condition in ['frozen','reward_no_trace','yoked_reward']:paired['trace_minus_'+condition]=paired.reward_trace-paired[condition]
paired.to_csv(out/'paired_scores.csv',index=False)
summary=df.groupby(['normalization','model','reward_delay','condition','readout']).agg(n=('seed','size'),score_mean=('pi_memory_score','mean'),score_min=('pi_memory_score','min'),score_max=('pi_memory_score','max'),train_accuracy=('train_accuracy','mean'),heldout_accuracy=('heldout_accuracy','mean'),relative_plastic_change=('relative_plastic_change','mean')).reset_index()
summary.to_csv(out/'summary.csv',index=False)
colors={'frozen':'#808080','reward_trace':'#247b8c','reward_no_trace':'#8663a8','yoked_reward':'#d47740'}
fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
for row,delay in enumerate(cfg['reward_delays']):
    for col,norm in enumerate(cfg['normalizations']):
        ax=axes[row,col]
        for i,model in enumerate(cfg['models']):
            for j,condition in enumerate(cfg['conditions']):
                q=df.query('readout=="frozen_policy" and reward_delay==@delay and normalization==@norm and model==@model and condition==@condition')
                x=i+(j-1.5)*.18;values=q.pi_memory_score.to_numpy()
                ax.scatter(x+np.linspace(-.035,.035,len(values)),values,s=25,alpha=.65,color=colors[condition],label=condition if i==0 else None)
                ax.plot([x-.06,x+.06],[values.mean()]*2,color=colors[condition],lw=3)
        ax.set_xticks(range(3),cfg['models']);ax.set(title=f'{norm} · reward delay {delay}',ylabel='Pi Memory Score',ylim=(-.3,df.query('readout=="frozen_policy"').pi_memory_score.max()+1));ax.grid(axis='y',alpha=.2)
axes[0,0].legend(fontsize=8);fig.suptitle('Reward adaptation | frozen warm-started policy | 200 training digits\nDots: runs; bars: means; supplied 314 excluded',fontsize=14)
fig.savefig(out/'recall_comparison.png',dpi=150);plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
for row,delay in enumerate(cfg['reward_delays']):
    for col,norm in enumerate(cfg['normalizations']):
        ax=axes[row,col]
        for condition in ['reward_trace','reward_no_trace','yoked_reward']:
            q=h.query('model=="fly" and reward_delay==@delay and normalization==@norm and condition==@condition').groupby('epoch').actual_reward_rate.mean()
            ax.plot(q.index,q.values,label=condition,color=colors[condition])
        ax.set(title=f'{norm} · delay {delay}',xlabel='Reward-training pass',ylabel='Actual sampled-action reward rate',ylim=(0,1));ax.grid(alpha=.2)
axes[0,0].legend(fontsize=8);fig.suptitle('True next-digit success during stochastic training (temperature 2)\nYoked control curve shows its own success, not the unrelated reward it receives')
fig.savefig(out/'reward_training.png',dpi=150);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
for ax,norm in zip(axes,cfg['normalizations']):
    for view in cfg['readouts']:
        q=summary.query('model=="fly" and reward_delay==3 and normalization==@norm and readout==@view').set_index('condition').loc[cfg['conditions']]
        ax.plot(range(4),q.train_accuracy,'o-',label=view)
    ax.set_xticks(range(4),cfg['conditions'],rotation=18,ha='right');ax.set(title=norm,ylabel='Teacher-forced training accuracy',ylim=(0,1));ax.legend(fontsize=8)
fig.suptitle('Frozen policy vs a separate supervised refit diagnostic | delay 3');fig.savefig(out/'readout_diagnostic.png',dpi=150);plt.close(fig)
replayed=[];digits=pi_digits(cfg['pi_length'])
with threadpool_limits(limits=1):
    for condition in cfg['conditions']:
        w=sparse.load_npz(out/'checkpoints'/f'{condition}_weights.npz')
        for view in cfg['readouts']:
            with np.load(out/'checkpoints'/f'{condition}_{view}_readout.npz') as d:
                patterns=d['patterns'];ro=SelectedReadout(d['indices']);ro.model.weights=d['weights'];ro.model.mean=d['mean'];ro.model.scale=d['scale']
            reservoir=CircuitReservoir(w,lambda digit:patterns[int(digit)],cfg['leak'],cfg['microsteps'])
            result=evaluate_recall(reservoir,ro,digits,cfg['prompt_length'],cfg['pi_length']-cfg['prompt_length'])
            original=next(r for r in records if r['circuit']==Path(cfg['circuits'][0]).name and r['seed']==cfg['seeds'][0] and r['normalization']=='incoming_l1' and r['model']=='fly' and r['reward_delay']==3 and r['condition']==condition and r['readout']==view)
            assert result['prediction']==original['prediction']
            replayed.append(dict(condition=condition,readout=view,score=result['pi_memory_score'],prediction_exact=True))
(out/'verification.json').write_text(json.dumps(dict(recall_records_checked=len(records),network_conditions=networks,
    actual_reward_events_checked=int(events['true_rewards'].size),yoked_reward_counts_match=True,all_scores_match=True,checkpoints_replayed=replayed),indent=2))
print('Verified',len(records),'recalls and',events['true_rewards'].size,'binary reward events')
print(summary.query('model=="fly" and readout=="frozen_policy"').round(3).to_string(index=False))
