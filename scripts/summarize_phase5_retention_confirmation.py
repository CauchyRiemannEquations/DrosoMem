"""Plot each segment separately; never hide a failed segment by pooling."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='results/phase5_retention_confirmation')
    args = parser.parse_args()
    out = Path(args.output)
    cfg = json.loads((out/'config.json').read_text())
    frame = pd.read_csv(out/'evaluations.csv')
    epochs = np.array([1,2,3])*cfg['stage_epochs']
    arms = [('fixed','Fixed 32','#2e77b8'),('curriculum','Ordinary curriculum','#d7782a'),
            ('anchored','Preserved first-32 share','#32936f')]
    means = frame.groupby(['offset','model','treatment','epoch']).pi_memory_score.mean()
    upper = max(70,10*np.ceil(means.max()/10))
    fig, axes = plt.subplots(2,3,figsize=(12,7),sharex=True,sharey=True)
    for row, model in enumerate(['fly','role_shuffled']):
        for col, offset in enumerate(cfg['offsets']):
            ax = axes[row,col]
            for arm,label,color in arms:
                ax.plot(epochs,means.loc[(offset,model,arm)].loc[epochs],marker='o',label=label,color=color)
            ax.set_title(f"{'Real connectome' if model=='fly' else 'Role-degree-shuffled'} / offset {offset}")
            ax.set_ylim(0,upper)
            ax.set_xticks(epochs)
            ax.axhline(32,linestyle=':',color='#999',linewidth=.8)
            ax.spines[['top','right']].set_visible(False)
            ax.grid(alpha=.15)
            ax.set_axisbelow(True)
            if row == 1:
                ax.set_xlabel('Adam updates')
            if col == 0:
                ax.set_ylabel('Mean consecutive generated digits')
    axes[0,0].legend(frameon=False,fontsize=8)
    fig.suptitle('Anchored prefix weighting: fresh-seed cross-segment confirmation',fontsize=14)
    fig.text(.5,.02,'Each panel includes all 18 fits per arm. Primary endpoint: 6,000 updates; horizon: 197.',
             ha='center',fontsize=9,color='#555')
    fig.tight_layout(rect=(0,.05,1,.95))
    fig.savefig(out/'overview.png',dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
