import argparse,json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def summarize(directory):
    root=Path(directory);frame=pd.read_csv(root/'results.csv');rows=[];pairs=[]
    groupcols=['circuit','seed','normalization','model','view']
    for keys,group in frame.groupby(groupcols):
        lookup=group.set_index('condition');full=lookup.loc['full_bptt']
        for control in ['frozen','local_gradient','permuted_bptt']:
            c=lookup.loc[control]
            pairs.append(dict(zip(groupcols,keys),control=control,full_score=int(full.pi_memory_score),control_score=int(c.pi_memory_score),
                              gain=int(full.pi_memory_score-c.pi_memory_score),ce_gain=float(c.true_train_cross_entropy-full.true_train_cross_entropy)))
    pd.DataFrame(pairs).to_csv(root/'paired.csv',index=False)
    for keys,g in frame.groupby(['normalization','model','condition','view']):
        rows.append(dict(zip(['normalization','model','condition','view'],keys),n=len(g),score_mean=g.pi_memory_score.mean(),
                         score_min=int(g.pi_memory_score.min()),score_max=int(g.pi_memory_score.max()),
                         train_accuracy=g.train_accuracy.mean(),true_train_cross_entropy=g.true_train_cross_entropy.mean(),
                         mean_relative_plastic_change=g.relative_plastic_change.mean()))
    summary=pd.DataFrame(rows);summary.to_csv(root/'summary.csv',index=False)
    descriptions=[]
    for keys,g in pd.DataFrame(pairs).groupby(['normalization','model','view','control']):
        descriptions.append(dict(zip(['normalization','model','view','control'],keys),n=len(g),mean_gain=g.gain.mean(),
                                 wins=int((g.gain>0).sum()),ties=int((g.gain==0).sum()),losses=int((g.gain<0).sum()),mean_ce_gain=g.ce_gain.mean()))
    (root/'paired_summary.json').write_text(json.dumps(descriptions,indent=2)+'\n')
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained',sharey=True)
    conditions=['frozen','full_bptt','local_gradient','permuted_bptt']
    for i,norm in enumerate(['spectral','incoming_l1']):
        for j,view in enumerate(['fixed_policy','refit_readout']):
            ax=axes[i,j]
            for model,shift,color in [('fly',-.18,'#3872a8'),('role_shuffled',.18,'#de9446')]:
                g=summary[(summary.normalization==norm)&(summary.view==view)&(summary.model==model)].set_index('condition')
                ax.bar([k+shift for k in range(4)],[g.loc[c,'score_mean'] for c in conditions],width=.36,label=model,color=color)
                raw=frame[(frame.normalization==norm)&(frame.view==view)&(frame.model==model)]
                for k,c in enumerate(conditions):
                    values=raw[raw.condition==c].pi_memory_score.to_list()
                    ax.scatter([k+shift+(n-2.5)*.018 for n in range(len(values))],values,s=13,color='#222222',alpha=.6,zorder=3)
            ax.set_xticks(range(4),['Frozen','Full BPTT','Local gradient','Permuted'],rotation=12)
            ax.set_title(norm+' / '+view);ax.set_ylabel('Consecutive generated digits');ax.grid(axis='y',alpha=.2)
    axes[0,0].legend();fig.suptitle('Constrained KC→MBON learning: 6 runs per bar, points show every run')
    fig.savefig(root/'recall_comparison.png',dpi=140);plt.close(fig)
    print(summary.to_string(index=False))
    print('\nREAL paired comparisons:')
    print(pd.DataFrame(descriptions).query("model=='fly'").to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');a=p.parse_args();summarize(a.directory)
