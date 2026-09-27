"""Plot final-stage outcomes without selecting seeds or checkpoints."""
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
    parser.add_argument('--output', default='results/phase5_curriculum')
    args = parser.parse_args(); out = Path(args.output)
    cfg = json.loads((out/'config.json').read_text())
    frame = pd.read_csv(out/'evaluations.csv')
    final = frame[frame.epoch == cfg['stage_epochs']*3]
    colors = ['#8b96a8', '#2e77b8', '#d7782a']
    labels = ['Uniform', 'Fixed 32', '32 → 64 → 128']
    arms = ['uniform', 'fixed', 'curriculum']
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for col, model in enumerate(['fly', 'role_shuffled']):
        subset = final[final.model == model]
        offsets = sorted(subset.offset.unique()); positions = np.arange(len(offsets))
        for arm, label, color, shift in zip(arms, labels, colors, [-.25, 0, .25]):
            means = subset[subset.treatment == arm].groupby('offset')[['pi_memory_score','later_accuracy']].mean()
            axes[0,col].bar(positions+shift, means.loc[offsets,'pi_memory_score'], width=.24, label=label, color=color)
            axes[1,col].bar(positions+shift, 100*means.loc[offsets,'later_accuracy'], width=.24, color=color)
        axes[0,col].set_title('Real connectome' if model == 'fly' else 'Role-degree-shuffled')
        axes[0,col].set_ylim(0,197)
        axes[0,col].axhline(64, color='#999', linestyle=':', linewidth=.8)
        axes[0,col].axhline(128, color='#999', linestyle=':', linewidth=.8)
        axes[1,col].set_ylim(0,100)
        axes[1,col].set_xticks(positions, [str(n) for n in offsets])
        axes[1,col].set_xlabel('Trained pi segment offset')
        for row in range(2):
            axes[row,col].spines[['top','right']].set_visible(False)
            axes[row,col].grid(axis='y', alpha=.15)
            axes[row,col].set_axisbelow(True)
    axes[0,0].set_ylabel('Consecutive generated digits')
    axes[1,0].set_ylabel('Later 165-target accuracy (%)')
    axes[0,0].legend(frameon=False, fontsize=9)
    fig.suptitle('Prefix curriculum at a matched 6,000-update budget', fontsize=15)
    fig.text(.5,.015,'All seeds and initializations; final stage only. Each segment is trained separately.',
             ha='center', fontsize=9, color='#555')
    fig.tight_layout(rect=(0,.04,1,.95))
    fig.savefig(out/'overview.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
