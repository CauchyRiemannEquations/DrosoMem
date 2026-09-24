"""Recompute every recall score; replay fixed checkpoints; draw measured results."""
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
p=argparse.ArgumentParser();p.add_argument('--output',default='results/phase4');args=p.parse_args();out=Path(args.output)
cfg=json.loads((out/'config.json').read_text());df=pd.read_csv(out/'results.csv')
keys=['circuit','seed','normalization','model','microsteps','length','condition']
expected=int(np.prod([len(cfg[k]) for k in ['circuits','seeds','normalizations','models','microsteps','lengths','conditions']]))
assert len(df)==expected and not df.duplicated(keys).any()
assert df.nonplastic_edges_unchanged.all() and df.evaluation_weights_frozen.all()
assert df.max_budget_error.max()<1e-12
assert (df.query('condition=="frozen"').changed_edges==0).all()
assert (df.query('condition!="frozen"').changed_edges>0).all()
assert (df.groupby(['circuit','seed']).plastic_edges.nunique()==1).all()
records=[json.loads(line) for line in (out/'recalls.jsonl').read_text().splitlines()];assert len(records)==len(df)
lookup=df.set_index(keys)
for r in records:
    score=pi_memory_score(list(r['target']),list(r['prediction']))
    assert score==r['pi_memory_score']==lookup.loc[tuple(r[k] for k in keys),'pi_memory_score']
    assert r['censored']==(score==r['horizon'])
    assert r['first_error_digit_index']==(None if r['censored'] else r['prompt_length']+score)
# Check exact paired gains; frozen reference differs for each graph/seed/configuration.
paired=df.pivot(index=keys[:-1],columns='condition',values='pi_memory_score').reset_index()
paired['supervised_minus_frozen']=paired.supervised-paired.frozen
paired['supervised_minus_permuted']=paired.supervised-paired.permuted_teacher
paired.to_csv(out/'paired_scores.csv',index=False)
summary=df.groupby(['normalization','model','microsteps','length','condition']).agg(n=('seed','size'),score_mean=('pi_memory_score','mean'),score_min=('pi_memory_score','min'),score_max=('pi_memory_score','max'),train_accuracy=('train_accuracy','mean'),heldout_accuracy=('heldout_accuracy','mean'),relative_plastic_change=('relative_plastic_change','mean')).reset_index()
summary.to_csv(out/'summary.csv',index=False)
colors={'frozen':'#808080','supervised':'#257b8c','permuted_teacher':'#cd7545'}
fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
for row,steps in enumerate(cfg['microsteps']):
    for col,norm in enumerate(cfg['normalizations']):
        ax=axes[row,col]
        for i,model in enumerate(cfg['models']):
            for j,condition in enumerate(cfg['conditions']):
                q=df.query('length==200 and microsteps==@steps and normalization==@norm and model==@model and condition==@condition')
                x=i+(j-1)*.23;values=q.pi_memory_score.to_numpy()
                ax.scatter(x+np.linspace(-.045,.045,len(values)),values,s=24,alpha=.65,color=colors[condition],label=condition if i==0 else None)
                ax.plot([x-.08,x+.08],[values.mean()]*2,color=colors[condition],lw=3)
        ax.set_xticks(range(3),cfg['models']);ax.set(title=f'{norm} · {steps} updates/digit',ylabel='Pi Memory Score');ax.grid(axis='y',alpha=.2);ax.set_ylim(-.3,float(df.query('length==200').pi_memory_score.max())+1)
axes[0,0].legend(fontsize=8);fig.suptitle('KC→MBON plasticity | 200 training digits | MBON readout\nDots: paired runs; bars: means; supplied 314 excluded',fontsize=14)
fig.savefig(out/'recall_comparison.png',dpi=150);plt.close(fig)
h=pd.read_csv(out/'plastic_training.csv');h=h.query('model=="fly" and length==200')
fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
for ax,norm in zip(axes,cfg['normalizations']):
    for condition in ['supervised','permuted_teacher']:
        for steps in cfg['microsteps']:
            q=h.query('normalization==@norm and condition==@condition and microsteps==@steps').groupby('epoch').teacher_mse.mean()
            ax.plot(q.index,q.values,label=f'{condition}, {steps} updates',color=colors[condition],ls='-' if steps==1 else '--')
    ax.set(title=norm,xlabel='Plasticity pass',ylabel='Online teacher-pattern MSE');ax.legend(fontsize=8)
fig.suptitle('Local teaching objective, not next-digit classification accuracy');fig.savefig(out/'plastic_training.png',dpi=150);plt.close(fig)
replay=[];histories=[];digits=pi_digits(200)
with threadpool_limits(limits=1):
    for model in cfg['models']:
        for condition in cfg['conditions']:
            name=f'{model}_{condition}';w=sparse.load_npz(out/'checkpoints'/f'{name}_weights.npz')
            with np.load(out/'checkpoints'/f'{name}_readout.npz') as d:
                patterns=d['patterns'];ro=SelectedReadout(d['indices']);ro.model.weights=d['weights'];ro.model.mean=d['mean'];ro.model.scale=d['scale']
            reservoir=CircuitReservoir(w,lambda digit:patterns[int(digit)],cfg['leak'],1)
            result=evaluate_recall(reservoir,ro,digits,cfg['prompt_length'],200-cfg['prompt_length'])
            original=next(r for r in records if r['circuit']==Path(cfg['circuits'][0]).name and r['seed']==cfg['seeds'][0] and r['normalization']=='incoming_l1' and r['microsteps']==1 and r['length']==200 and r['model']==model and r['condition']==condition)
            assert result['prediction']==original['prediction'] and result['pi_memory_score']==original['pi_memory_score']
            replay.append(dict(checkpoint=name,pi_memory_score=result['pi_memory_score'],prediction_exact=True))
            # Independently refit on the saved frozen network; compare classifier parameters.
            states=reservoir.states(digits[:-1]);refit=SelectedReadout(ro.indices)
            history=refit.fit(states,digits[1:],epochs=cfg['readout_epochs'],learning_rate=cfg['readout_learning_rate'],l2=cfg['readout_l2'])
            assert np.array_equal(refit.model.weights,ro.model.weights)
            histories.extend(dict(model=model,condition=condition,**r) for r in history)
pd.DataFrame(histories).to_csv(out/'readout_training.csv',index=False)
fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True)
for condition in cfg['conditions']:
    q=pd.DataFrame(histories).query('model=="fly" and condition==@condition');ax.plot(q.epoch,q.accuracy,label=condition,color=colors[condition])
ax.set(xlabel='Readout epoch',ylabel='Teacher-forced training accuracy',ylim=(0,1),title='Saved real circuit 701 · seed 442 · incoming-L1 · 1 update · 200 digits');ax.legend();fig.savefig(out/'readout_training.png',dpi=150);plt.close(fig)
(out/'verification.json').write_text(json.dumps(dict(records_checked=len(records),weight_invariants_checked=True,all_recall_scores_match=True,checkpoints_replayed=replay,checkpoint_readouts_refit_exact=True),indent=2))
print('Verified',len(records),'runs and',len(replay),'reloaded checkpoints')
print(summary.query('model=="fly" and length==200').round(3).to_string(index=False))
