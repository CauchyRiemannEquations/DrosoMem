"""Plot every arm and stage in the locked game-opening retention study."""
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
    parser.add_argument('--output', default='results/phase5_retention')
    args = parser.parse_args(); out = Path(args.output)
    cfg = json.loads((out/'config.json').read_text())
    frame = pd.read_csv(out/'evaluations.csv')
    epochs = np.array([1,2,3])*cfg['stage_epochs']
    arms = [('fixed','Fixed 32','#2e77b8'), ('curriculum','Ordinary curriculum','#d7782a'),
            ('anchored','Preserved first-32 share','#32936f')]
    upper = max(70,10*np.ceil(frame.groupby(['model','treatment','epoch']).pi_memory_score.mean().max()/10))
    fig, axes = plt.subplots(2,2,figsize=(10,7),sharex=True)
    for col, model in enumerate(['fly','role_shuffled']):
        subset = frame[frame.model==model]
        for arm,label,color in arms:
            groups = subset[subset.treatment==arm].groupby('epoch').pi_memory_score
            axes[0,col].plot(epochs,groups.mean().loc[epochs],marker='o',label=label,color=color)
            axes[1,col].plot(epochs,groups.apply(lambda x:100*(x>=32).mean()).loc[epochs],marker='o',color=color)
        axes[0,col].set_title('Real connectome' if model=='fly' else 'Role-degree-shuffled')
        axes[0,col].set_ylim(0,upper); axes[1,col].set_ylim(0,105)
        axes[0,col].axhline(32,linestyle=':',color='#aaa',linewidth=.8)
        axes[1,col].set_xticks(epochs); axes[1,col].set_xlabel('Adam updates')
        for row in range(2):
            axes[row,col].spines[['top','right']].set_visible(False)
            axes[row,col].grid(alpha=.15); axes[row,col].set_axisbelow(True)
    axes[0,0].set_ylabel('Mean consecutive generated digits')
    axes[1,0].set_ylabel('Models completing first 32 digits (%)')
    axes[0,0].legend(frameon=False,fontsize=8)
    fig.suptitle('Preserving early-prefix loss share: game-opening discovery',fontsize=14)
    fig.text(.5,.02,'All seeds and initializations. Primary endpoint: 6,000 updates. Recall horizon: 197.',
             ha='center',fontsize=9,color='#555')
    fig.tight_layout(rect=(0,.05,1,.95)); fig.savefig(out/'overview.png',dpi=180); plt.close(fig)


if __name__ == '__main__':
    main()
