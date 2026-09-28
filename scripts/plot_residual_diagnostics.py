"""Plot all registered blocks without changing numerical result archives."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from alphabet_memory import check
from flying.training import whole_brain_memory as core


def run(modes,orientation,out):
    out.mkdir(parents=True,exist_ok=False);check(modes);check(orientation)
    frames=[pd.read_csv(p/'seed-blocks.csv') for p in [modes,orientation]]
    cells=[(co,d) for co in ['main','confirmation'] for d in ['renormalized_to_fixed_original','fixed_original_to_renormalized']]
    labels=['Main\nR→F','Main\nF→R','Archived\nR→F','Archived\nF→R']
    fig,axes=plt.subplots(1,2,figsize=(11,4.6))
    for i,(co,d) in enumerate(cells):
        b=frames[0].query('cohort == @co and direction == @d')
        for offset,col,color,label in [(-.13,'low_state_share','#2774b4','State energy'),(.13,'low_signed_score_share','#d96827','Signed score contribution')]:
            y=100*b[col].to_numpy();axes[0].scatter(np.full(len(y),i+offset),y,color=color,alpha=.7,label=label if i==0 else None)
            axes[0].plot([i+offset-.06,i+offset+.06],[y.mean()]*2,color=color,lw=3)
        b=frames[1].query('cohort == @co and direction == @d')
        y=b.log2_actual_over_reference.to_numpy();axes[1].scatter(np.full(len(y),i),y,color='#2774b4',alpha=.7)
        axes[1].plot([i-.12,i+.12],[y.mean()]*2,color='#d96827',lw=3)
    axes[0].axhline(0,color='gray',ls=':');axes[0].legend(fontsize=8)
    axes[0].set(ylabel='Low-half share (%)',title='Fixed source ranks 25–48')
    axes[1].axhline(0,color='gray',ls=':',label='Equal energy distortion')
    axes[1].axhline(1,color='red',ls='--',label='Registered ≥2× threshold')
    axes[1].set(ylabel='Mean log2(actual / orientation reference)',title='Energy-matched orientation control');axes[1].legend(fontsize=8)
    for ax in axes:ax.set_xticks(range(4),labels);ax.grid(axis='y',alpha=.15)
    fig.tight_layout();fig.savefig(out/'residual-diagnostics.png',dpi=180);plt.close(fig)
    core.write_json(out/'manifest.json',dict(sources={str(p):core.sha256(p/'manifest.json') for p in [modes,orientation]},
        plotter_sha256=core.sha256(__file__),artifacts={'residual-diagnostics.png':core.sha256(out/'residual-diagnostics.png')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    run(Path('results/residual_modes'),Path('results/residual_orientation'),args.out)
