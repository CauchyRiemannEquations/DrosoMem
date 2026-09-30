"""Descriptive figures/tables only after independent calibration verification."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flying.training import whole_brain_memory as core
from alphabet_memory import read,check
from relative_noise import read_table
from pathway_memory import seal


def analyze(root,validation,out):
    m=check(root);check(validation);v=read(validation/'checks.json')
    assert v['all_checks_pass'] and not v['smoke'] and v['result_manifest_sha256']==core.sha256(root/'manifest.json')
    c=m['config'];out.mkdir(parents=True,exist_ok=False)
    pair=read_table(root/'paired-differences.csv');blocks=read_table(root/'seed-blocks.csv')
    dose=read_table(root/'dose-seed-table.csv');raw=read_table(root/'raw-rollouts.csv')
    scales=[dict(**read(p)['case'],common_q=read(p)['common_q'],own_q=read(p)['own_q'],q_ratio=read(p)['q_ratio']) for p in sorted(root.glob('*/case.json'))]
    pd.DataFrame(scales).to_csv(out/'scale-raw.csv',index=False)
    scale_summary={top:dict(n=len(z),minimum=float(z.q_ratio.min()),median=float(z.q_ratio.median()),maximum=float(z.q_ratio.max()))
                   for top,z in pd.DataFrame(scales).groupby('topology')}
    mean=lambda x:np.mean(x.to_numpy())
    blocks.groupby(['cohort','family','topology','calibration'])[ ['exact_prefix_symbols','retention']].agg(mean).reset_index().to_csv(out/'condition-means.csv',index=False)
    raw[raw.calibration=='clean'].groupby(['cohort','family','topology'])[['exact_prefix_symbols','retention']].agg(mean).reset_index().to_csv(out/'clean-means.csv',index=False)
    descriptive=['accuracy','teacher_accuracy','post_error_accuracy','raw_state_effective_rank','active_neurons_mean',
        'raw_mbon_sparsity','raw_temporal_cosine','standardized_clean_mse','mean_injected_energy',
        'mean_clipped_energy','clipped_coordinates']
    raw[raw.calibration!='clean'].groupby(['cohort','family','topology','calibration','strength'])[descriptive].agg(mean).reset_index().to_csv(out/'trajectory-diagnostics.csv',index=False)
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True,constrained_layout=True)
    colors={'degree':'#006f98','role':'#c76b16'}
    for ax,(co,family) in zip(axes.flat,[(co,f) for co in c['cohorts'] for f in c['families']]):
        z=dose[(dose.cohort==co)&(dose.family==family)]
        for top in ['degree','role']:
            for cal,style in [('common','--'),('own','-')]:
                b=z[(z.topology==top)&(z.calibration==cal)].set_index(['seed','strength'])
                a=z[(z.topology=='intact')&(z.calibration==cal)].set_index(['seed','strength'])
                values=(b.exact_prefix_symbols-a.exact_prefix_symbols).groupby('strength').mean()
                ax.plot(values.index,values.values,style,marker='o',color=colors[top],label=f'{top}, {cal}')
        ax.axhline(0,color='gray',lw=.8);ax.set_xscale('log');ax.set_title(f'{co} / {family}')
        ax.set_xlabel('Relative noise strength');ax.set_ylabel('Control minus intact (symbols)');ax.grid(alpha=.2)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside upper center',ncol=4)
    fig.savefig(out/'gap-curves.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,co in zip(axes,c['cohorts']):
        z=pair[(pair.cohort==co)&(pair.family=='random')&(pair.control=='degree')].sort_values('seed')
        x=np.arange(len(z));ax.bar(x,z.exact_prefix_symbols_attenuation,color='#006f98')
        ax.axhline(2,color='#c76b16',ls='--',label='Mean magnitude criterion = 2')
        ax.axhline(0,color='black',lw=.8);ax.set_xticks(x,z.seed.astype(str),rotation=20)
        ax.set_ylabel('Gap attenuation (symbols)');ax.set_title(f'Reused {co} models');ax.grid(axis='y',alpha=.2)
        ax.legend(fontsize=8,loc='upper right')
    fig.suptitle('Random / degree primary contrast: all paired model blocks')
    fig.savefig(out/'paired-attenuation.png',dpi=160);plt.close(fig)
    core.write_json(out/'checks.json',dict(all_checks_pass=True,result_manifest_sha256=core.sha256(root/'manifest.json'),
        validation_manifest_sha256=core.sha256(validation/'manifest.json'),scale_summary=scale_summary,
        descriptive_only=True,primary_decision=read(root/'summary.json')['primary_confirmed']))
    seal(out,c,m['context'],analysis_script_sha256=core.sha256(__file__),purpose='Descriptive calibration figures and tables')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('--validation',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();analyze(a.source,a.validation,a.out)
