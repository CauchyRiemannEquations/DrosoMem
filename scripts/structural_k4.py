"""Preregistered independent-stream structural probe; frozen historical sources."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits

from flying.brain.diagnostics import graph_diagnostics
from flying.brain.mushroom_body import role_shuffled
from flying.training import whole_brain_memory as core
from alphabet_memory import SymbolEncoder, symbol_bank, read, check
from frozen_state_probe import Budget, baseline, labels, fit_fold, verify_fold, measures
from context_memory import estimate


def audit_shuffle(a, b, roles, log):
    """Fail closed on incomplete swaps or changed structural invariants."""
    assert a.shape == b.shape and a.nnz == b.nnz
    assert not b.diagonal().any() and (b.data != 0).all()
    assert log['accepted_swaps'] == log['requested_swaps'] > 0
    for axis in [0, 1]:
        np.testing.assert_array_equal((a != 0).sum(axis=axis), (b != 0).sum(axis=axis))
    ac, bc = a.tocsc(), b.tocsc()
    for i in range(a.shape[0]):
        np.testing.assert_array_equal(np.sort(ac.data[ac.indptr[i]:ac.indptr[i+1]]),
                                      np.sort(bc.data[bc.indptr[i]:bc.indptr[i+1]]))
    blocks = {}
    roles = np.asarray(roles)
    for pre in sorted(set(roles)):
        for post in sorted(set(roles)):
            ix, iy = np.flatnonzero(roles == post), np.flatnonzero(roles == pre)
            x, y = a[ix][:, iy], b[ix][:, iy]
            assert x.nnz == y.nnz
            if x.nnz:
                overlap = int((x != 0).multiply(y != 0).nnz)
                blocks[f'{pre}->{post}'] = dict(edges=x.nnz, common_edges=overlap, overlap=overlap/x.nnz)
    incoming_delta = np.asarray(abs(b).sum(axis=1)-abs(a).sum(axis=1)).ravel()
    return dict(**log, invariants_verified=True, role_blocks=blocks,
                raw_incoming_strength_changed_nodes=int(np.count_nonzero(incoming_delta)),
                raw_incoming_strength_mean_absolute_change=float(abs(incoming_delta).mean()),
                raw_incoming_strength_max_absolute_change=float(abs(incoming_delta).max()))


def build(c, circuit, seed, arm='real', rewire_seed=None, k=4):
    directory = Path(f'data/flywire_783_mb_left_kc512_s{circuit}')
    original, ids, provenance = core.load_connectome(directory)
    roles, _ = core.load_roles(directory, ids)
    raw, audit = original, None
    if arm == 'role_shuffled':
        raw, log = role_shuffled(original, roles, rewire_seed, c['swaps_per_edge'])
        audit = audit_shuffle(original, raw, roles, log)
    elif arm != 'real':
        raise ValueError(arm)
    weights = core.normalize_condition(raw, c['normalization'], c['gain'])
    weights.sort_indices()
    row_l1 = np.asarray(abs(weights).sum(axis=1)).ravel()
    np.testing.assert_allclose(row_l1[row_l1 != 0], c['gain'], atol=1e-14, rtol=0)
    bank = symbol_bank(roles, seed, c['input_fraction'], c['input_amplitude'])[:k]
    obs = np.flatnonzero(np.asarray(roles) == 'MBON')
    assert len(obs) == 48
    model = core.TimedReservoir(weights, SymbolEncoder(bank), roles, c['leak'], c['schedule'])
    graph = dict(arm=arm, circuit_seed=circuit, threshold=5, **graph_diagnostics(raw),
                 source=provenance, raw_sha256=core.weight_hash(raw),
                 weight_sha256=core.weight_hash(weights), shuffle=audit,
                 observed_root_ids=[ids[i] for i in obs],
                 input_root_ids=[[ids[i] for i in np.flatnonzero(p)] for p in bank],
                 input_mapping_sha256=core.fingerprint(dict(ids=ids, patterns=bank.tolist())),
                 raw_outgoing_weights_preserved=True, normalized_outgoing_weights_preserved=None)
    if audit is not None:
        original_w = core.normalize_condition(original, c['normalization'], c['gain']).tocsc()
        shuffled_w = weights.tocsc()
        changed = sum(not np.array_equal(np.sort(original_w[:, i].data), np.sort(shuffled_w[:, i].data))
                      for i in range(len(ids)))
        graph['normalized_outgoing_weight_changed_nodes'] = changed
        graph['normalized_outgoing_weights_preserved'] = changed == 0
    return model, obs, graph, raw


def reproduce_baseline(c):
    result = baseline()
    path = Path(result['source']); manifest = check(path)
    model, obs, graph, _ = build(manifest['config'], 701, 14142, k=10)
    with np.load(path/'checkpoint.npz', allow_pickle=False) as a:
        for split in ['train', 'test']:
            features, _ = core.collect(model, obs, a[split+'_symbols'], c['activity_epsilon'])
            np.testing.assert_array_equal(features, a[split+'_features'])
    result.update(exact_train_and_test_trajectories=True, graph_weight_sha256=graph['weight_sha256'])
    return result


def execute(c, circuit, block, arm):
    model, obs, graph, raw = build(c, circuit, block['seed'], arm, block['rewire_seed'])
    a = dict(observed_indices=obs, input_patterns=model.encoder.patterns)
    for split in ['train', 'test']:
        n = c['warmup']+c[split+'_samples']
        symbols = np.random.default_rng(block[split+'_seed']).integers(0, 4, n).astype(np.uint8)
        features, diagnostics = core.collect(model, obs, symbols, c['activity_epsilon'])
        a[split+'_symbols'], a[split+'_features'] = symbols, features
        a.update({split+'_'+key: value for key, value in diagnostics.items()})
        if split == 'train':
            a['decay_full'], a['decay_observed'] = core.decay_probe(model, obs, c['decay_steps'])
    w = c['warmup']
    a.update(fit_fold(a['train_features'][w:], a['test_features'][w:],
                     labels(a['train_symbols'], np.arange(w, len(a['train_symbols'])), c['lags']),
                     labels(a['test_symbols'], np.arange(w, len(a['test_symbols'])), c['lags']), c['alpha']))
    metrics = measures([a], c['lags'])
    for j, row in enumerate(metrics):
        row['training_mse'] = float(np.mean((a['train_scores'][:, j]-np.eye(4)[a['ytrain'][:, j]])**2))
    x = a['train_features'][w:]
    a['centered_singular_values'] = np.linalg.svd(x-x.mean(axis=0), compute_uv=False)
    neural = dict(**core.state_diagnostics(x),
                  observed_sparsity=float(np.mean(abs(x) <= c['activity_epsilon'])),
                  mean_mbon_norm=float(np.linalg.norm(x, axis=1).mean()),
                  mean_active_neurons=float(a['train_active_counts'][w:].mean()),
                  mean_observed_cosine=float(a['train_observed_cosine'][w-1:].mean()),
                  clipped_features=int(np.sum(x.std(axis=0) < 1e-5)),
                  observed_decay_ratio32=float(a['decay_observed'][-1]/a['decay_observed'][0]))
    assert core.weight_hash(model.weights) == graph['weight_sha256']
    for value in a.values():
        assert np.isfinite(value).all()
    return a, graph, metrics, neural, raw, model.weights


def gates(blocks, delta, cohort):
    required = 4 if cohort == 'main' else 3
    access = {}
    for arm, b in blocks.groupby('arm'):
        access[arm] = bool(b.frequency_excess.mean() >= .05 and b.null_excess.mean() >= .05 and
                          ((b.frequency_excess >= .05) & (b.null_excess >= .05)).sum() >= required and
                          b.r2_vs_frequency.mean() > 0)
    advantage = bool(access.get('real', False) and delta.mean() >= .03 and (delta > 0).sum() >= required)
    return dict(past_access=access, real_wiring_advantage=advantage)


def summarize(rows, neuralrows, out, c, cohort):
    f = pd.DataFrame(rows); f.to_csv(out/'raw-lag-table.csv', index=False)
    metric = ['test_accuracy', 'train_accuracy', 'frequency_accuracy', 'null_accuracy',
              'frequency_excess', 'null_excess', 'r2_vs_frequency', 'training_mse']
    agg = f[f.lag.isin(c['primary_lags'])].groupby(['arm', 'seed', 'circuit_seed'])[metric].mean().reset_index()
    agg.to_csv(out/'raw-past-table.csv', index=False)
    blocks = agg.groupby(['arm', 'seed'])[metric].mean().reset_index()
    blocks.to_csv(out/'seed-blocks.csv', index=False)
    real = blocks[blocks.arm == 'real'].set_index('seed')
    shuffled = blocks[blocks.arm == 'role_shuffled'].set_index('seed')
    delta = real[metric]-shuffled[metric]; delta.to_csv(out/'paired-differences.csv')
    # Smoke has one block; no inference or confidence intervals.
    summary = dict(cohort=cohort, gates=None, cells=None, real_minus_shuffled=None)
    if cohort != 'smoke':
        summary.update(gates=gates(blocks, delta.test_accuracy, cohort),
                       cells={arm: {m: estimate(b[m], c) for m in metric} for arm, b in blocks.groupby('arm')},
                       real_minus_shuffled={m: estimate(delta[m], c) for m in metric})
    neural = pd.DataFrame(neuralrows); neural.to_csv(out/'neural-diagnostics.csv', index=False)
    core.write_json(out/'summary.json', summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for arm, color in [('real', '#1976b5'), ('role_shuffled', '#e17430')]:
        for seed, b in f[f.arm == arm].groupby('seed'):
            y = b.groupby('lag').test_accuracy.mean().reindex(c['lags'])
            axes[0].plot(range(len(y)), y, color=color, alpha=.25, lw=1)
        y = f[f.arm == arm].groupby('lag').test_accuracy.mean().reindex(c['lags'])
        axes[0].plot(range(len(y)), y, 'o-', color=color, label=arm)
    axes[0].axhline(.25, color='gray', ls=':')
    axes[0].set(xticks=range(len(c['lags'])), xticklabels=c['lags'], xlabel='Symbol lag',
                ylabel='Independent test accuracy', ylim=(0, 1.03)); axes[0].legend()
    for seed in real.index:
        axes[1].plot([0, 1], [real.loc[seed, 'test_accuracy'], shuffled.loc[seed, 'test_accuracy']], 'o-', label=str(seed))
    axes[1].set(xticks=[0, 1], xticklabels=['real', 'role-shuffled'], ylabel='Primary past accuracy')
    axes[1].legend(fontsize=8); fig.tight_layout(); fig.savefig(out/'structural-curve.png', dpi=180); plt.close(fig)
    return summary


def validate(c):
    assert c['conditions'] == ['real', 'role_shuffled'] and c['alphabet_size'] == 4
    assert c['lags'] == [0, 1, 2, 3, 4, 5, 8, 12, 16, 24, 32]
    assert c['primary_lags'] == [1, 2, 3, 4, 5, 8] and c['warmup'] == 100
    assert (c['gain'], c['leak'], c['input_fraction'], c['input_amplitude'], c['alpha']) == (.9, .6, .1, .5, 1.)
    assert c['schedule'] == 'mbon_after_kc' and c['normalization'] == 'incoming_l1'
    assert c['swaps_per_edge'] == 5 and (c['train_samples'], c['test_samples']) in [(2000, 1000), (200, 100)]
    seeds = [value for b in c['blocks'] for value in b.values()]
    assert len(set(seeds)) == len(seeds)


@threadpool_limits.wrap(limits=1)
def run(config, out, cohort, discovery=None):
    frozen = read(config); c = dict(frozen)
    if cohort == 'smoke':
        c.update(c['smoke'])
    elif cohort == 'confirmation':
        if discovery is None:
            raise ValueError('Confirmation requires verified main manifest')
        main = check(discovery)
        assert main['cohort'] == 'main' and main['config'] == frozen
        assert read(discovery/'summary.json')['gates']['real_wiring_advantage']
        c['blocks'] = c['confirmation_blocks']
    validate(c); out.mkdir(parents=True, exist_ok=False); budget = Budget(c)
    try:
        sources = core.source_context(c)
        for path in ['scripts/structural_k4.py', 'scripts/alphabet_memory.py', 'scripts/frozen_state_probe.py', 'scripts/context_memory.py', str(config)]:
            sources['source_sha256'][path] = core.sha256(path)
        prior = {p: core.sha256(p) for p in subprocess.check_output(['git', 'ls-files', 'results/'], text=True).splitlines()}
        core.write_json(out/'prior-results-sha256.json', prior)
        core.write_json(out/'config.json', c)
        core.write_json(out/'baseline.json', reproduce_baseline(c))
        print('Historical baseline: exact train/test trajectories and ridge refit', flush=True)
        rows, neuralrows = [], []
        for block in c['blocks']:
            for circuit in c['circuit_seeds']:
                paired = {}
                for arm in c['conditions']:
                    budget.check(); started = time.perf_counter()
                    dest = out/f'{arm}_c{circuit}_s{block["seed"]}'; dest.mkdir()
                    a, graph, metrics, neural, raw, weights = execute(c, circuit, block, arm)
                    np.savez_compressed(dest/'checkpoint.npz', **a)
                    sparse.save_npz(dest/'raw-graph.npz', raw); sparse.save_npz(dest/'weights.npz', weights)
                    core.write_json(dest/'graph.json', graph); core.write_json(dest/'metrics.json', metrics)
                    core.write_json(dest/'neural.json', neural)
                    # Regenerate graph, reset/replay both streams and independently refit.
                    b, graph2, metrics2, neural2, _, _ = execute(c, circuit, block, arm)
                    assert graph2 == graph and metrics2 == metrics and neural2 == neural
                    with np.load(dest/'checkpoint.npz', allow_pickle=False) as saved:
                        assert set(saved.files) == set(b)
                        for key, value in b.items():
                            np.testing.assert_array_equal(saved[key], value, err_msg=key)
                        verify_fold(dict(saved), c['alpha'])
                    assert core.weight_hash(sparse.load_npz(dest/'raw-graph.npz')) == graph['raw_sha256']
                    assert core.weight_hash(sparse.load_npz(dest/'weights.npz')) == graph['weight_sha256']
                    identity = dict(arm=arm, circuit_seed=circuit, **block)
                    rows.extend(dict(**identity, **row) for row in metrics)
                    neuralrows.append(dict(**identity, **neural))
                    paired[arm] = a, graph
                    core.write_json(dest/'manifest.json', dict(identity=identity, config=c, context=sources,
                        timestamp=datetime.now(timezone.utc).isoformat(), exact_replay=True,
                        independent_lstsq=True, coefficients_per_task=196, real_task_heads=len(c['lags']),
                        null_task_heads=len(c['lags']), seconds_including_replay=time.perf_counter()-started,
                        artifacts={p.name: core.sha256(p) for p in dest.iterdir() if p.is_file()}))
                    print(f'{cohort} {arm} c{circuit} s{block["seed"]}: saved, replayed, refitted', flush=True)
                real, rewire = paired['real'], paired['role_shuffled']
                for key in ['train_symbols', 'test_symbols', 'input_patterns', 'observed_indices', 'ytrain', 'ytest']:
                    np.testing.assert_array_equal(real[0][key], rewire[0][key])
                for key in ['input_root_ids', 'observed_root_ids', 'input_mapping_sha256']:
                    assert real[1][key] == rewire[1][key]
        summary = summarize(rows, neuralrows, out, c, cohort)
        for p, digest in {**prior, **sources['source_sha256']}.items():
            assert core.sha256(p) == digest, p
        budget.check(); usage = budget.close()
        core.write_json(out/'verification.json', dict(exact_replayed_graph_runs=len(neuralrows),
            independent_refits=len(neuralrows), independent_lstsq_runs=len(neuralrows),
            prior_result_files_unchanged=len(prior), budget=usage))
        core.write_json(out/'manifest.json', dict(cohort=cohort, config=frozen, effective_config=c, context=sources,
            timestamp=datetime.now(timezone.utc).isoformat(), environment=core.environment(),
            discovery_manifest_sha256=core.sha256(discovery/'manifest.json') if discovery else None,
            artifacts={p.relative_to(out).as_posix(): core.sha256(p) for p in out.rglob('*') if p.is_file()}))
        print(dict(gates=summary['gates'], usage=usage), flush=True)
    finally:
        budget.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/structural_k4.json'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cohort', choices=['smoke', 'main', 'confirmation'], required=True)
    parser.add_argument('--discovery', type=Path)
    args = parser.parse_args(); run(args.config, args.out, args.cohort, args.discovery)
