"""Separate cohort/control curves for the frozen-model robustness study."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',default='results/phase2_robustness')
    args = parser.parse_args(); out = Path(args.output)
    cfg = json.loads((out/'config.json').read_text())
    summary = json.loads((out/'summary.json').read_text())
    families = [('pulse','Single state perturbation'),('ongoing','Noise at every recall step'),
                ('edge_dropout','Permanent edge removal')]
    fig,axes = plt.subplots(2,3,figsize=(12,7),sharey=True)
    top = 1.1
    for row,cohort in enumerate(cfg['cohorts']):
        for col,(kind,title) in enumerate(families):
            ax = axes[row,col]
            for model,label,color in [('fly','Real connectome','#2e77b8'),('role_shuffled','Role-degree-shuffled','#d7782a')]:
                cells = summary['by_cohort'][cohort['name']][model][kind]
                strengths = sorted(float(k) for k in cells)
                ratios = [cells[str(s)]['mean_ratio'] for s in strengths]
                ax.plot(strengths,ratios,marker='o',color=color,label=label)
                top = max(top,max(ratios)*1.05)
            ax.axhline(.8,color='#999',linestyle=':',linewidth=1)
            ax.axvline(cfg['primary'][kind],color='#aaa',linestyle='--',linewidth=.8)
            ax.set_title(f"{cohort['name'].capitalize()} / {title}",fontsize=10)
            if kind != 'edge_dropout':
                ax.set_xscale('symlog',linthresh=1e-6)
                ax.set_xlim(-1e-7,1.2e-2)
                ax.set_xticks(cfg['strengths'][kind],['0','1e-6','1e-4','1e-3','1e-2'])
                ax.set_xlabel('Absolute state-noise SD')
            else:
                ax.set_xticks([0,.01,.05,.1],['0','1%','5%','10%'])
                ax.set_xlabel('Fraction of edges removed')
            if col == 0: ax.set_ylabel('Mean recall / matched clean mean')
            ax.spines[['top','right']].set_visible(False)
            ax.grid(alpha=.15); ax.set_axisbelow(True)
    for ax in axes.flat: ax.set_ylim(0,top)
    axes[0,0].legend(frameon=False,fontsize=8)
    fig.suptitle('Frozen fixed-32 models: perturbation robustness',fontsize=15)
    fig.text(.5,.015,'18 heads per graph/cohort; 20 seeds per nonzero level. Dotted: 80%; dashed: primary strength. No refitting.',
             ha='center',fontsize=9,color='#555')
    fig.tight_layout(rect=(0,.045,1,.95)); fig.savefig(out/'overview.png',dpi=180); plt.close(fig)


if __name__ == '__main__':
    main()
