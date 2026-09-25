import argparse,json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flying.training.phase5_readout import KEYS

def summarize(directory):
    root=Path(directory);df=pd.read_csv(root/'evaluations.csv');cfg=json.loads((root/'config.json').read_text())
    groups=['normalization','model','schedule','epochs','readout_type']
    summary=df.groupby(groups).agg(n=('pi_memory_score','size'),recall_mean=('pi_memory_score','mean'),recall_min=('pi_memory_score','min'),recall_max=('pi_memory_score','max'),train_accuracy=('train_accuracy','mean'),cross_entropy=('cross_entropy','mean')).reset_index()
    summary.to_csv(root/'summary.csv',index=False)
    primary=df[df.epochs==cfg['primary_checkpoint']]
    linear=primary[primary.readout_type=='affine'][KEYS+['pi_memory_score']].rename(columns={'pi_memory_score':'affine'})
    nonlinear=primary[primary.readout_type=='nonlinear'].groupby(KEYS).agg(nonlinear_mean=('pi_memory_score','mean'),nonlinear_min=('pi_memory_score','min'),nonlinear_max=('pi_memory_score','max')).reset_index()
    pairs=linear.merge(nonlinear,on=KEYS,validate='one_to_one');pairs['delta']=pairs.nonlinear_mean-pairs.affine
    pairs['all_initializations_beat_affine']=pairs.nonlinear_min>pairs.affine;pairs.to_csv(root/'paired.csv',index=False)
    rows=[]
    for keys,g in pairs.groupby(['normalization','model','schedule']):
        rows.append(dict(zip(['normalization','model','schedule'],keys),n=len(g),affine_mean=float(g.affine.mean()),nonlinear_mean=float(g.nonlinear_mean.mean()),wins=int((g.delta>0).sum()),ties=int((g.delta==0).sum()),losses=int((g.delta<0).sum()),all_initializations_beat_affine=int(g.all_initializations_beat_affine.sum())))
    (root/'paired_summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(10,4.2),sharex=True,sharey=True)
    for ax,model in zip(axes,['fly','role_shuffled']):
        for schedule,marker in [('sync_one','o'),('mbon_after_kc','s')]:
            for norm,color in [('spectral','#2864ba'),('incoming_l1','#d46c24')]:
                g=pairs[(pairs.model==model)&(pairs.schedule==schedule)&(pairs.normalization==norm)]
                ax.errorbar(g.affine,g.nonlinear_mean,yerr=[g.nonlinear_mean-g.nonlinear_min,g.nonlinear_max-g.nonlinear_mean],fmt=marker,color=color,alpha=.75,capsize=2,label=f'{norm}, {schedule}')
        lim=max(20,pairs.nonlinear_max.max()+5,pairs.affine.max()+5);ax.plot([0,lim],[0,lim],':',color='gray')
        ax.set(xlim=(-1,lim),ylim=(-1,lim),xlabel='Affine recall (490 parameters)',title=model);ax.grid(alpha=.15)
    axes[0].set_ylabel('Nonlinear recall (482 parameters)');axes[1].legend(fontsize=7)
    fig.suptitle('Frozen states, 2,000 epochs: nonlinear mean and range over 3 initializations')
    fig.tight_layout();fig.savefig(root/'overview.png',dpi=160);plt.close(fig)
    print(summary.to_string(index=False));print(json.dumps(rows,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');a=p.parse_args();summarize(a.directory)
