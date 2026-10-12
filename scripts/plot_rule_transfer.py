"""Read-only M11 finite comparison figure, with full-scale preregistered gates."""
import argparse
from fractions import Fraction
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from rule_transfer_support import config, source, unchanged, coverage, attempt, write, read, seal, sha


def run(out):
    c, record = config(), source(config())
    main = Path(c['namespace']) / 'main'
    with attempt(out, c, 'm11-results-figure'):
        coverage(main)
        write(out / 'source.json', record)
        write(out / 'input.json', dict(main_manifest_sha256=sha(main / 'manifest.json'), display_only=True))
        rows = read(main / 'metrics.json')
        panels = [('train', 'all', 'teacher_correct', 'teacher_predictions', 'Training teacher accuracy', 50),
                  ('parameter_test', 'nonperiod2', 'first_correct', 'episodes', 'Held-out first next accuracy', 25),
                  ('parameter_test', 'nonperiod2', 'completed', 'episodes', 'Held-out autonomous completion', 10)]
        fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4), sharey=True)
        positions = [0, 1, 2, 3.6, 4.6, 5.6]
        labels = ['Real', '10 nulls', 'Shift', 'Real', '10 nulls', 'Shift']
        plot_data = []
        for ax, (split, group, count, denom, title, threshold) in zip(axes, panels):
            for cohort_index, (cohort, seeds) in enumerate(c['cohorts'].items()):
                for kind_index, kind in enumerate(('real', 'nulls', 'shift')):
                    selected = [r for r in rows if r['family'] == 'affine' and r['split'] == split and r['group'] == group
                                and r['input_seed'] in seeds and (r['model'].startswith('g') if kind == 'nulls' else r['model'] == kind)]
                    selected.sort(key=lambda r: (r['model'], r['input_seed']))
                    exact = [Fraction(r[count], r[denom]) for r in selected]
                    values = np.array([float(v) * 100 for v in exact])
                    x = positions[cohort_index * 3 + kind_index]
                    color = ['#2874A6', '#8E969B', '#CB7D22'][kind_index]
                    ax.bar(x, values.mean(), width=.62, color=color, alpha=.50 if cohort_index == 0 else .85)
                    offset = np.linspace(-.2, .2, len(values)) if len(values) > 1 else np.zeros(1)
                    ax.scatter(x + offset, values, color=color, s=12, marker='o' if cohort_index == 0 else '^', zorder=3)
                    ax.text(x, max(3., values.max()) + 2.2, f'{values.mean():.1f}', ha='center', fontsize=8)
                    plot_data.append(dict(panel=title, cohort=cohort, kind=kind, scores_exact=list(map(str, exact))))
            ax.axhline(threshold, color='#555555', linewidth=.85, linestyle='--')
            ax.text(5.95, threshold + 1.7, f'Real gate {threshold}%', fontsize=8, ha='right', color='#555555')
            ax.set_xticks(positions, labels, rotation=35, ha='right')
            ax.set_ylim(0, 105)
            ax.set_xlim(-.55, 6.2)
            ax.set_title(title, fontsize=10, pad=11)
            ax.grid(axis='y', alpha=.16)
            ax.spines[['top', 'right']].set_visible(False)
            ax.text(1., -25, 'Discovery', ha='center', transform=ax.transData, fontsize=9)
            ax.text(4.6, -25, 'Confirmation', ha='center', transform=ax.transData, fontsize=9)
        axes[0].set_ylabel('Accuracy / completion (%)')
        fig.suptitle('M11: fixed reservoirs + equally fitted affine heads', fontsize=13)
        fig.text(.5, .015, 'Held-out panels exclude period-2 a=10. Shift repeats one canonical interface. Dots are paired computational conditions.',
                 ha='center', fontsize=8, color='#444444')
        fig.subplots_adjust(left=.065, right=.985, top=.82, bottom=.27, wspace=.12)
        fig.savefig(out / 'rule-transfer.png', dpi=180, facecolor='white')
        fig.savefig(out / 'rule-transfer.svg', facecolor='white')
        plt.close(fig)
        write(out / 'plotted-values.json', plot_data)
        unchanged(record)
    seal(out, dict(complete=True, study='M11 figure'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    run(p.parse_args().out)
