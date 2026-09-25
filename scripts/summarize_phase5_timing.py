import argparse,json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def summarize(directory):
    root=Path(directory);pi=pd.read_csv(root/'pi.csv');memory=pd.read_csv(root/'memory.csv');rows=[];pairs=[]
    for keys,g in pi.groupby(['normalization','model','schedule']):
        selected=memory
        for k,v in zip(['normalization','model','schedule'],keys):selected=selected[selected[k]==v]
        row=dict(zip(['normalization','model','schedule'],keys),n=len(g),score_mean=g.pi_memory_score.mean(),score_min=int(g.pi_memory_score.min()),score_max=int(g.pi_memory_score.max()),train_accuracy=g.train_accuracy.mean())
        row.update({f'lag{lag}_accuracy':float(selected[selected.lag==lag].accuracy.mean()) for lag in sorted(selected.lag.unique())})
        rows.append(row)
    summary=pd.DataFrame(rows);summary.to_csv(root/'summary.csv',index=False)
    for keys,g in pi.groupby(['circuit','seed','normalization','model']):
        table=g.set_index('schedule')
        for control in ['sync_one','sync_two']:
            pairs.append(dict(zip(['circuit','seed','normalization','model'],keys),control=control,
                              gain=int(table.loc['mbon_after_kc','pi_memory_score']-table.loc[control,'pi_memory_score'])))
    paired=pd.DataFrame(pairs);paired.to_csv(root/'paired.csv',index=False)
    description=[]
    for keys,g in paired.groupby(['normalization','model','control']):
        description.append(dict(zip(['normalization','model','control'],keys),n=len(g),mean_gain=g.gain.mean(),wins=int((g.gain>0).sum()),ties=int((g.gain==0).sum()),losses=int((g.gain<0).sum())))
    (root/'paired_summary.json').write_text(json.dumps(description,indent=2)+'\n')
    fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
    schedules=['sync_one','mbon_after_kc','sync_two']
    for i,norm in enumerate(['spectral','incoming_l1']):
        for j,(metric,label) in enumerate([('score_mean','Mean consecutive correct digits'),('lag0_accuracy','Current-input decoding accuracy'),('lag3_accuracy','Lag-3 decoding accuracy')]):
            ax=axes[i,j]
            for model,color in [('fly','#3872a8'),('role_shuffled','#db9447')]:
                g=summary[(summary.normalization==norm)&(summary.model==model)].set_index('schedule')
                ax.plot(range(3),[g.loc[s,metric] for s in schedules],marker='o',color=color,label=model)
            ax.set_xticks(range(3),['Sync ×1','KC → MBON','Sync ×2'],rotation=15)
            ax.set_title(norm+'\n'+label);ax.grid(alpha=.2)
            if j>0:ax.set_ylim(0,1.03);ax.axhline(.1,color='#777777',linestyle='--',linewidth=1)
    axes[0,0].legend();fig.suptitle('Timing diagnosis: same weights and 48-MBON readout budget (6 runs per point)')
    fig.savefig(root/'timing_comparison.png',dpi=140);plt.close(fig)
    print(summary.to_string(index=False));print(pd.DataFrame(description).to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');a=p.parse_args();summarize(a.directory)
