import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from flying.training.phase5_diagnostic import paired_table


def summarize(root):
    root=Path(root);frames=[];summaries=[]
    selection=json.loads((root/'discovery/selection.json').read_text())
    for stage in ['discovery','confirmation']:
        directory=root/stage
        if not (directory/'manifest.json').exists():continue
        frame=pd.read_csv(directory/'results.csv');paired=paired_table(frame);paired['stage']=stage
        frames.append(paired)
        for keys,g in paired.groupby(['model','normalization','reward_delay','alpha','view']):
            row=dict(zip(['model','normalization','reward_delay','alpha','view'],keys));row['stage']=stage
            row.update(n_pairs=len(g),trace_mean=g.pi_memory_score.mean(),frozen_mean=g.frozen_score.mean(),
                       yoked_mean=g.yoked_score.mean(),mean_gain_frozen=g.gain_frozen.mean(),mean_gain_yoked=g.gain_yoked.mean(),
                       wins_frozen=int((g.gain_frozen>0).sum()),ties_frozen=int((g.gain_frozen==0).sum()),losses_frozen=int((g.gain_frozen<0).sum()),
                       wins_yoked=int((g.gain_yoked>0).sum()),ties_yoked=int((g.gain_yoked==0).sum()),losses_yoked=int((g.gain_yoked<0).sum()))
            summaries.append(row)
    summary=pd.DataFrame(summaries);summary.to_csv(root/'summary.csv',index=False)
    candidates=[]
    if 'confirmation' in set(summary.stage):
        for candidate in selection['candidates']:
            if candidate['alpha'] is None:continue
            for model in ['fly','role_shuffled']:
                match=summary[(summary.stage=='confirmation')&(summary.model==model)]
                for k in ['normalization','reward_delay','view','alpha']:match=match[match[k]==candidate[k]]
                assert len(match)==1
                row=match.iloc[0].to_dict()
                row['positive_mean_vs_both']=bool(row['mean_gain_frozen']>0 and row['mean_gain_yoked']>0)
                candidates.append(row)
        (root/'candidate_confirmation.json').write_text(json.dumps(candidates,indent=2)+'\n')
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharey=True,layout='constrained')
    discovery=summary[(summary.stage=='discovery')&(summary.model=='fly')]
    for ax,view in zip(axes.flat,['fixed','stats_only','coefficients_only','full_refit']):
        part=discovery[discovery.view==view]
        for (norm,delay),g in part.groupby(['normalization','reward_delay']):
            g=g.sort_values('alpha')
            ax.plot(g.alpha,g.mean_gain_frozen,marker='o',label=f'{norm}, delay {delay}')
        ax.axhline(0,color='#222222',linewidth=1)
        ax.set_xscale('symlog',linthresh=.01);ax.set_title(view.replace('_',' '))
        ax.set_xlim(0,1.05)
        ax.set_xlabel('Fraction of final synaptic change (alpha)')
        ax.set_ylabel('Mean recall gain over frozen (digits)');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Phase 5 diagnostic: real circuit, discovery seeds\nPost-training interpolation; each point = 2 circuits × 3 seeds')
    fig.savefig(root/'readout_decomposition.png',dpi=150);plt.close(fig)
    if candidates:
        real=[r for r in candidates if r['model']=='fly']
        labels=[r['view'].replace('_',' ') for r in real]
        discovery_gains=[next(c['discovery_gain_frozen'] for c in selection['candidates']
            if c['view']==r['view'] and c['normalization']==r['normalization'] and c['reward_delay']==r['reward_delay']) for r in real]
        x=list(range(len(real)))
        fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained')
        ax.bar([v-.18 for v in x],discovery_gains,width=.36,label='Discovery: 3 seeds',color='#3973ac')
        ax.bar([v+.18 for v in x],[r['mean_gain_frozen'] for r in real],width=.36,label='Confirmation: 5 new seeds',color='#d05a42')
        ax.set_xticks(x,labels);ax.axhline(0,color='#222222',linewidth=1)
        ax.set_ylabel('Mean recall gain over frozen (digits)')
        ax.set_title('Three selected real-network candidates failed to confirm\nSpectral normalization, immediate reward, alpha = 0.03')
        ax.legend();ax.grid(axis='y',alpha=.2)
        fig.savefig(root/'candidate_confirmation.png',dpi=150);plt.close(fig)
    print('Selected discovery candidates:',sum(c['alpha'] is not None for c in selection['candidates']))
    print(pd.DataFrame(candidates).to_string(index=False) if candidates else 'No candidates confirmed yet')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='results/phase5_diagnostic')
    args=parser.parse_args();summarize(args.output)
