"""Derive coverage/activity/resource tables and a figure from verified archives.

Run from the repository root with PYTHONPATH=src. This analysis does not select
seeds, fit a readout or alter model parameters.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from flying.data.connectome import sha256
from flying.training.phase6 import verify, write_json


def summarize(source, destination):
    verify(source)
    destination.mkdir(parents=True, exist_ok=True)
    coverage = pd.read_csv(source/'coverage.csv', dtype={'root_id': str})
    left_mbon_ids = coverage.root_id.unique().astype(np.int64)
    pooled = []
    for seed, d in coverage.groupby('circuit_seed'):
        values = d[['total', 'retained', 'weak', 'outside', 'autapse',
                    'full_left_kc_contacts', 'retained_kc_contacts']].sum().to_dict()
        pooled.append(dict(circuit_seed=int(seed), **{k: int(v) for k, v in values.items()},
                           retained_fraction=values['retained']/values['total'],
                           retained_left_kc_fraction=values['retained_kc_contacts']/values['full_left_kc_contacts']))
    rows, resources = [], []
    inputs = {}
    levels = ['legacy5', 'left5', 'brain5', 'brain1']
    for level in levels:
        for seed in [701, 702]:
            stem = f'{level}_s{seed}'
            meta = json.loads((source/(stem+'.json')).read_text())
            resource = json.loads((source/(stem+'.resources.json')).read_text())
            resources.append(dict(level=level, circuit_seed=seed, neurons=meta['neurons'],
                                  edges=meta['edges'], sparse_csc_bytes=meta['sparse_csc_bytes'],
                                  delay_ring_bytes=meta['delay_ring_bytes'],
                                  graph_load_seconds=meta['graph_load_seconds'],
                                  **resource))
            with np.load(source/(stem+'.npz'), allow_pickle=False) as arrays:
                ids = arrays['node_ids']
                patterns = arrays['input_patterns']
                root_inputs = [sorted(ids[p].tolist()) for p in patterns]
                if seed not in inputs:
                    inputs[seed] = root_inputs
                assert inputs[seed] == root_inputs
                assert all(len(p) == 51 for p in root_inputs)
                left_mbon = np.isin(ids, left_mbon_ids)
                assert left_mbon.sum() == 48
                for row in meta['probes']:
                    counts = arrays[row['probe']+'_counts']
                    rows.append(dict(level=level, circuit_seed=seed, **row,
                                     left_mbon_spikes=int(counts[:, left_mbon].sum()),
                                     active_left_mbons=int(np.count_nonzero(counts[:, left_mbon].sum(axis=0)))))
    activity, resource_table = pd.DataFrame(rows), pd.DataFrame(resources)
    activity.to_csv(destination/'activity.csv', index=False)
    resource_table.to_csv(destination/'resources.csv', index=False)
    pd.DataFrame(pooled).to_csv(destination/'coverage-pooled.csv', index=False)
    oracle = json.loads((source/'oracle.json').read_text())
    summary = dict(coverage=pooled,
                   mbon11_coverage=coverage[coverage.root_id == '720575940623201833'].to_dict(orient='records'),
                   all_input_root_ids_matched=True, input_cells_per_digit=51,
                   oracle_max_v_error_mv=max(r['max_v_error_mv'] for r in oracle['cases']),
                   oracle_max_g_error_mv=max(r['max_g_error_mv'] for r in oracle['cases']),
                   all_probe_repeats_exact=bool(activity.exact_repeat.all()),
                   fullbrain_peak_rss_bytes=int(resource_table[resource_table.level == 'brain1'].sampled_peak_rss_bytes.max()),
                   source_manifest_sha256=sha256(source/'manifest.json'),
                   analysis_script_sha256=sha256(__file__))
    write_json(destination/'summary.json', summary)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    ax = axes[0, 0]
    x, bottom = np.arange(2), np.zeros(2)
    for key, label, color in [('retained', 'Retained', '#247c7b'),
                              ('weak', 'Below 5 contacts', '#e8af4f'),
                              ('outside', 'Outside subset', '#b8bdc7')]:
        height = np.array([r[key]/r['total']*100 for r in pooled])
        ax.bar(x, height, bottom=bottom, label=label, color=color)
        bottom += height
    ax.set(xticks=x, xticklabels=['s701', 's702'], ylabel='% incoming absolute contacts',
           title='Inputs to the same 48 left MBONs')
    ax.legend(fontsize=8, loc='lower right')

    ax = axes[0, 1]
    for i, (seed, probe) in enumerate((s, p) for s in [701, 702] for p in ['digit3', 'digit1', 'pi20']):
        values = [activity[(activity.level == l) & (activity.circuit_seed == seed)
                           & (activity.probe == probe)].mbon11_spikes.iloc[0] for l in levels]
        ax.plot(range(4), values, marker='o', label=f's{seed} {probe}',
                linestyle='-' if seed == 701 else '--', alpha=.8)
    ax.set(xticks=range(4), xticklabels=levels, ylabel='Spikes in 1 second', title='MBON11 response; fixed 51-KC input')
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    for seed in [701, 702]:
        d = resource_table[resource_table.circuit_seed == seed].set_index('level').loc[levels]
        ax.plot(range(4), d.sampled_peak_rss_bytes/2**20, marker='o', label=f's{seed}')
    ax.set(xticks=range(4), xticklabels=levels, ylabel='Sampled process-tree RSS (MiB)',
           title='Measured memory per worker')
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    for seed in [701, 702]:
        d = activity[activity.circuit_seed == seed]
        means = [d[d.level == l][['first_wall_seconds', 'repeat_wall_seconds']].to_numpy().mean() for l in levels]
        ax.plot(range(4), means, marker='o', label=f's{seed}')
    ax.set(xticks=range(4), xticklabels=levels, ylabel='Mean wall seconds / simulated second',
           title='Observed CPU runtime; no readout training')
    ax.legend(fontsize=8)
    fig.suptitle('Phase 6: whole-brain feasibility, not pi recall performance', fontsize=14)
    fig.savefig(destination/'coverage-and-scaling.png', dpi=170)
    plt.close(fig)
    write_json(destination/'analysis-manifest.json',
               {p.name: sha256(p) for p in sorted(destination.iterdir()) if p.name != 'analysis-manifest.json'})
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('results/phase6'))
    parser.add_argument('--out', type=Path, default=Path('outputs/phase6-analysis'))
    args = parser.parse_args()
    summarize(args.source, args.out)
