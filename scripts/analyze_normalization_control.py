"""Descriptive saved-state diagnostics; never changes experimental gates."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from alphabet_memory import read, check
from flying.training import whole_brain_memory as core


def run(sources, out):
    out.mkdir(parents=True, exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    rows, rolesummary, manifests = [], [], {}
    colors = dict(real='#1976b5', renormalized='#e17430', fixed_original='#239666')
    for root in sources:
        m = check(root); cohort = m['cohort']; manifests[root.as_posix()] = core.sha256(root/'manifest.json')
        curves = {}
        for group in sorted(root.glob('*_c*_s*')):
            record = check(group); identity = record['identity']; arm = identity['arm']
            neural = read(group/'neural.json'); rows.append(dict(cohort=cohort, **identity, **neural))
            directory = Path(f'data/flywire_783_mb_left_kc512_s{identity["circuit_seed"]}')
            _, ids, _ = core.load_connectome(directory); roles, _ = core.load_roles(directory, ids); roles = np.asarray(roles)
            base = root/f'real_c{identity["circuit_seed"]}_s{identity["seed"]}'/'checkpoint.npz'
            with np.load(group/'checkpoint.npz', allow_pickle=False) as a, np.load(base, allow_pickle=False) as b:
                for role in sorted(set(roles)):
                    mask = roles == role
                    rolesummary.append(dict(cohort=cohort, **identity, role=str(role), neurons=int(mask.sum()),
                        changed_factors=int(np.count_nonzero(a['normalization_factors'][mask] != b['normalization_factors'][mask])),
                        rows_l1_above_one=int(np.sum(a['row_l1'][mask] > 1)), mean_row_l1=float(a['row_l1'][mask].mean()), max_row_l1=float(a['row_l1'][mask].max())))
                curves.setdefault(arm, []).append((a['decay_observed']/a['decay_observed'][0],
                    a['perturbation_full_inf']/a['perturbation_full_inf'][0]))
        frame = pd.DataFrame(rows); current = frame[frame.cohort == cohort]
        style = '-' if cohort == 'main' else '--'
        for i, arm in enumerate(['real', 'renormalized', 'fixed_original']):
            b = current[current.arm == arm].groupby('seed').effective_rank.mean()
            offset = -.10 if cohort == 'main' else .10
            axes[0].scatter(np.full(len(b), i+offset), b, color=colors[arm], marker='o' if cohort == 'main' else 'x')
            for ax, column in [(axes[1], 0), (axes[2], 1)]:
                mean = np.mean([value[column] for value in curves[arm]], axis=0)
                ax.semilogy(np.maximum(mean, 1e-150), style, color=colors[arm], label=f'{arm} {cohort}')
    axes[0].set(xticks=[0, 1, 2], xticklabels=['real', 'renormalized', 'fixed'], ylabel='Centered effective rank', title='Paired-block rank: main o / confirm x')
    axes[1].set(xlabel='Silent steps', ylabel='MBON norm / initial norm', title='Input-off decay (descriptive mean)')
    axes[2].set(xlabel='Shared-input steps', ylabel='Full-state difference / initial', title='Small initial-state perturbation')
    axes[1].legend(fontsize=7); fig.tight_layout(); fig.savefig(out/'state-diagnostics.png', dpi=180); plt.close(fig)
    allrows = pd.DataFrame(rows); allrows.to_csv(out/'state-diagnostics.csv', index=False)
    pd.DataFrame(rolesummary).to_csv(out/'role-normalization.csv', index=False)
    summary = {}
    for (cohort, arm), b in allrows.groupby(['cohort', 'arm']):
        summary[f'{cohort}/{arm}'] = dict(runs=len(b), all_empirical_forgetting=bool(b.empirical_forgetting.all()),
            sufficient_contraction_count=int(b.sufficient_contraction.sum()), max_row_l1_range=[float(b.max_row_l1.min()), float(b.max_row_l1.max())],
            max_abs_state=float(b.max_abs_state.max()), effective_rank_mean=float(b.effective_rank.mean()),
            mean_active_neurons=float(b.mean_active_neurons.mean()), silent_decay_ratio32_mean=float(b.observed_decay_ratio32.mean()))
    core.write_json(out/'summary.json', summary)
    core.write_json(out/'manifest.json', dict(source_manifests=manifests, script_sha256=core.sha256(__file__),
        descriptive_only=True, artifacts={p.name: core.sha256(p) for p in out.iterdir() if p.is_file()}))
    print(summary)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--out', type=Path, required=True)
    p.add_argument('sources', type=Path, nargs='+'); a = p.parse_args(); run(a.sources, a.out)
