"""Fixed-raw-graph normalization intervention, with frozen historical code."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import time
import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits
from flying.training import whole_brain_memory as core
from alphabet_memory import read, check, SymbolEncoder
from frozen_state_probe import Budget, labels, fit_fold, verify_fold, measures
from context_memory import estimate
import structural_k4 as previous


def factors(raw, gain=.9):
    strength = np.asarray(abs(raw).sum(axis=1)).ravel()
    return np.divide(gain, strength, out=np.zeros_like(strength, dtype=float), where=strength > 0)


def normalized(original, raw, arm, gain=.9):
    if arm not in ['real', 'renormalized', 'fixed_original']:
        raise ValueError(arm)
    d = factors(raw if arm == 'renormalized' else original, gain)
    # Degree matching guarantees identical zero-input support.
    np.testing.assert_array_equal(factors(original) == 0, factors(raw) == 0)
    weights = (sparse.diags(d)@raw).tocsr(); weights.sort_indices()
    assert weights.nnz == raw.nnz
    return weights, d


def lipschitz_bound(weights, roles, leak):
    absolute = abs(weights)
    row_l1 = np.asarray(absolute.sum(axis=1)).ravel()
    bounds = 1-leak+leak*row_l1
    roles = np.asarray(roles); kc = roles == 'KC'; mbon = roles == 'MBON'
    input_bounds = np.ones(len(roles)); input_bounds[kc] = bounds[kc]
    bounds[mbon] = 1-leak+leak*np.asarray(absolute[mbon]@input_bounds).ravel()
    return row_l1, bounds


class MonitoredReservoir(core.TimedReservoir):
    def reset(self):
        super().reset(); self.maximum_abs = 0.
    def step(self, symbol):
        state = super().step(symbol)
        if not np.isfinite(state).all() or np.max(abs(state)) > 1+1e-12:
            raise RuntimeError('Nonfinite or out-of-box state; preserve failed cohort')
        self.maximum_abs = max(self.maximum_abs, float(np.max(abs(state))))
        return state


def build(c, circuit, block, arm, archive=None):
    oldarm = 'real' if arm == 'real' else 'role_shuffled'
    oldmodel, obs, graph, raw = previous.build(c, circuit, block['seed'], oldarm, block['rewire_seed'])
    directory = Path(f'data/flywire_783_mb_left_kc512_s{circuit}')
    original, ids, _ = core.load_connectome(directory); roles, _ = core.load_roles(directory, ids)
    weights, d = normalized(original, raw, arm, c['gain'])
    row_l1, bounds = lipschitz_bound(weights, roles, c['leak'])
    model = MonitoredReservoir(weights, SymbolEncoder(oldmodel.encoder.patterns), roles, c['leak'], c['schedule'])
    graph.update(arm=arm, weight_sha256=core.weight_hash(weights),
                 max_row_l1=float(row_l1.max()), rows_l1_above_one=int(np.sum(row_l1 > 1)),
                 sufficient_lipschitz_bound=float(bounds.max()), contraction_certified=bool(bounds.max() < 1),
                 original_raw_sha256=core.weight_hash(original), source_checkpoint=None)
    aw = core.normalize_condition(original, c['normalization'], c['gain']).tocsc(); bw = weights.tocsc()
    changed = sum(not np.array_equal(np.sort(aw[:, i].data), np.sort(bw[:, i].data)) for i in range(len(ids)))
    graph.update(normalized_outgoing_weight_changed_nodes=changed, normalized_outgoing_weights_preserved=changed == 0)
    if archive is not None:
        source = Path(archive)/f'{oldarm}_c{circuit}_s{block["seed"]}'
        check(source)
        assert core.weight_hash(sparse.load_npz(source/'raw-graph.npz')) == graph['raw_sha256']
        graph['source_checkpoint'] = dict(path=source.as_posix(), manifest_sha256=core.sha256(source/'manifest.json'))
    return model, obs, graph, raw, d, row_l1, bounds


def perturbation_probe(model, obs, symbols, seed, amplitude):
    model.reset(); reference = [model.state.copy()]
    for symbol in symbols:
        reference.append(model.step(int(symbol)))
    model.reset(); initial = np.random.default_rng(seed).uniform(-amplitude, amplitude, len(model.state))
    model.state = initial.copy(); full = []; observed = []
    for t, state in enumerate(reference):
        if t:
            model.step(int(symbols[t-1]))
        full.append(float(np.max(abs(model.state-state))))
        observed.append(float(np.linalg.norm((model.state-state)[obs])))
    return dict(initial_perturbation=initial, perturbation_symbols=np.asarray(symbols),
                perturbation_full_inf=np.asarray(full), perturbation_mbon_l2=np.asarray(observed))


def execute(c, circuit, block, arm, archive=None):
    model, obs, graph, raw, d, row_l1, bounds = build(c, circuit, block, arm, archive)
    a = dict(observed_indices=obs, input_patterns=model.encoder.patterns,
             normalization_factors=d, row_l1=row_l1, row_lipschitz_bounds=bounds)
    maxima = []
    for split in ['train', 'test']:
        n = c['warmup']+c[split+'_samples']
        symbols = np.random.default_rng(block[split+'_seed']).integers(0, 4, n).astype(np.uint8)
        features, diagnostics = core.collect(model, obs, symbols, c['activity_epsilon'])
        maxima.append(model.maximum_abs)
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
        row['test_mse'] = float(np.mean((a['scores'][:, j]-np.eye(4)[a['ytest'][:, j]])**2))
    x = a['train_features'][w:]
    a['centered_singular_values'] = np.linalg.svd(x-x.mean(axis=0), compute_uv=False)
    a['max_abs_by_stream'] = np.asarray(maxima)
    a.update(perturbation_probe(model, obs, a['train_symbols'][:c['perturbation_steps']], block['perturb_seed'], c['perturbation_amplitude']))
    ratio = float(a['perturbation_full_inf'][-1]/a['perturbation_full_inf'][0])
    neural = dict(**core.state_diagnostics(x), observed_sparsity=float(np.mean(abs(x) <= c['activity_epsilon'])),
                  mean_mbon_norm=float(np.linalg.norm(x, axis=1).mean()),
                  mean_active_neurons=float(a['train_active_counts'][w:].mean()),
                  mean_observed_cosine=float(a['train_observed_cosine'][w-1:].mean()),
                  clipped_features=int(np.sum(x.std(axis=0) < 1e-5)),
                  observed_decay_ratio32=float(a['decay_observed'][-1]/a['decay_observed'][0]),
                  max_abs_state=float(max(maxima)), initial_difference_ratio256=ratio,
                  empirical_forgetting=bool(ratio <= c['forgetting_ratio']),
                  sufficient_contraction=graph['contraction_certified'], max_row_l1=graph['max_row_l1'],
                  sufficient_lipschitz_bound=graph['sufficient_lipschitz_bound'])
    assert core.weight_hash(model.weights) == graph['weight_sha256']
    for value in a.values():
        assert np.isfinite(value).all()
    return a, graph, metrics, neural, raw, model.weights


def baseline(c):
    old = read('configs/structural_k4.json'); settings = dict(c, **old)
    block = dict(old['blocks'][0], perturb_seed=46142); records = []
    for arm, oldarm in [('real', 'real'), ('renormalized', 'role_shuffled')]:
        path = Path(c['archive'])/f'{oldarm}_c701_s34142'; manifest = check(path)
        a, graph, metrics, neural, _, _ = execute(settings, 701, block, arm)
        with np.load(path/'checkpoint.npz', allow_pickle=False) as saved:
            for key in saved.files:
                np.testing.assert_array_equal(saved[key], a[key], err_msg=key)
            arrays = len(saved.files)
        expected = read(path/'metrics.json')
        for original, actual in zip(expected, metrics):
            assert all(original[key] == actual[key] for key in original)
        oldgraph = read(path/'graph.json'); assert oldgraph['weight_sha256'] == graph['weight_sha256']
        oldneural = read(path/'neural.json'); assert all(oldneural[key] == neural[key] for key in oldneural)
        accuracy = float(np.mean([r['test_accuracy'] for r in metrics if r['lag'] in c['primary_lags']]))
        records.append(dict(source=path.as_posix(), manifest_sha256=core.sha256(path/'manifest.json'),
                            arrays_exact=arrays, expected_primary_accuracy=accuracy, actual_primary_accuracy=accuracy))
    return records


METRICS = ['test_accuracy', 'train_accuracy', 'frequency_accuracy', 'null_accuracy',
           'frequency_excess', 'null_excess', 'r2_vs_frequency', 'training_mse', 'test_mse']


def decision(blocks, delta, cohort, threshold=.01, main_direction=None):
    required = 4 if cohort == 'main' else 3
    access = {}
    for arm, b in blocks.groupby('arm'):
        access[arm] = bool(b.frequency_excess.mean() >= .05 and b.null_excess.mean() >= .05 and
                          ((b.frequency_excess >= .05) & (b.null_excess >= .05)).sum() >= required and b.r2_vs_frequency.mean() > 0)
    direction = int(np.sign(delta.mean())) if main_direction is None else main_direction
    passed = bool(direction != 0 and direction*delta.mean() >= threshold and (direction*delta > 0).sum() >= required
                  and access.get('renormalized', False) and access.get('fixed_original', False))
    return dict(direction=direction, past_access=access, material_normalization_effect=passed)


def summarize(rows, neuralrows, out, c, cohort, main_direction=None):
    frame = pd.DataFrame(rows); frame.to_csv(out/'raw-lag-table.csv', index=False)
    raw = frame[frame.lag.isin(c['primary_lags'])].groupby(['arm', 'seed', 'circuit_seed'])[METRICS].mean().reset_index()
    raw.to_csv(out/'raw-past-table.csv', index=False)
    blocks = raw.groupby(['arm', 'seed'])[METRICS].mean().reset_index(); blocks.to_csv(out/'seed-blocks.csv', index=False)
    deltas, stats = [], {}
    for name, left, right in [('renormalized_minus_fixed', 'renormalized', 'fixed_original'),
                               ('real_minus_renormalized', 'real', 'renormalized'), ('real_minus_fixed', 'real', 'fixed_original')]:
        a = blocks[blocks.arm == left].set_index('seed'); b = blocks[blocks.arm == right].set_index('seed')
        delta = a[METRICS]-b[METRICS]; delta['contrast'] = name; deltas.append(delta.reset_index())
        if cohort != 'smoke':
            stats[name] = {metric: estimate(delta[metric], c) for metric in METRICS}
    pd.concat(deltas).to_csv(out/'paired-differences.csv', index=False)
    cells = {}
    if cohort != 'smoke':
        for arm, b in blocks.groupby('arm'):
            cells[arm] = {metric: {k: v for k, v in estimate(b[metric], c).items() if k not in ['paired_dz', 'wins', 'ties', 'losses']} for metric in METRICS}
    result = dict(cohort=cohort, cells=cells, contrasts=stats,
                  gates=decision(blocks, deltas[0].test_accuracy, cohort, c['material_effect'], main_direction) if cohort != 'smoke' else None)
    core.write_json(out/'summary.json', result)
    pd.DataFrame(neuralrows).to_csv(out/'neural-diagnostics.csv', index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for arm, color in [('real', '#1976b5'), ('renormalized', '#e17430'), ('fixed_original', '#239666')]:
        y = frame[frame.arm == arm].groupby('lag').test_accuracy.mean().reindex(c['lags'])
        axes[0].plot(range(len(y)), y, 'o-', color=color, label=arm)
    axes[0].axhline(.25, color='gray', ls=':'); axes[0].legend(fontsize=8)
    axes[0].set(xticks=range(len(c['lags'])), xticklabels=c['lags'], xlabel='Symbol lag', ylabel='Independent test accuracy', ylim=(0, 1.03))
    for seed, b in blocks.groupby('seed'):
        v = b.set_index('arm'); axes[1].plot([0, 1], v.loc[['renormalized', 'fixed_original'], 'test_accuracy'], 'o-', label=str(seed))
    axes[1].set(xticks=[0, 1], xticklabels=['renormalized', 'fixed original factors'], ylabel='Primary past accuracy')
    axes[1].legend(fontsize=8); fig.tight_layout(); fig.savefig(out/'normalization-curve.png', dpi=180); plt.close(fig)
    return result


@threadpool_limits.wrap(limits=1)
def run(config, out, cohort, discovery=None):
    frozen = read(config); c = dict(frozen); direction = None
    archive = c['archive']
    if cohort == 'smoke':
        c.update(c['smoke']); archive = c['smoke_archive']
    elif cohort == 'confirmation':
        if discovery is None:
            raise ValueError('Verified main is required')
        parent = check(discovery); assert parent['config'] == frozen and parent['cohort'] == 'main'
        gates = read(discovery/'summary.json')['gates']; assert gates['material_normalization_effect']
        direction = gates['direction']; c['blocks'] = c['confirmation_blocks']; archive = None
    assert c['conditions'] == ['real', 'renormalized', 'fixed_original']
    assert c['lags'] == [0, 1, 2, 3, 4, 5, 8, 12, 16, 24, 32] and c['primary_lags'] == [1, 2, 3, 4, 5, 8]
    assert c['warmup'] == 100 and (c['train_samples'], c['test_samples']) in [(2000, 1000), (200, 100)]
    assert (c['gain'], c['leak'], c['input_fraction'], c['input_amplitude'], c['alpha']) == (.9, .6, .1, .5, 1.)
    assert c['schedule'] == 'mbon_after_kc' and c['normalization'] == 'incoming_l1'
    assert c['perturbation_steps'] == 256 and c['perturbation_amplitude'] == 1e-6 and c['swaps_per_edge'] == 5
    seeds = [v for block in c['blocks'] for v in block.values()]; assert len(set(seeds)) == len(seeds)
    out.mkdir(parents=True, exist_ok=False); budget = Budget(c)
    try:
        context = core.source_context(c)
        for path in [str(config), 'scripts/normalization_control.py', 'scripts/structural_k4.py', 'scripts/frozen_state_probe.py', 'scripts/alphabet_memory.py', 'scripts/context_memory.py']:
            context['source_sha256'][path] = core.sha256(path)
        prior = {p: core.sha256(p) for p in subprocess.check_output(['git', 'ls-files', 'results/'], text=True).splitlines()}
        core.write_json(out/'prior-results-sha256.json', prior); core.write_json(out/'config.json', c)
        core.write_json(out/'baseline.json', baseline(c)); print('Previous real/shuffled K4 baselines reproduced exactly', flush=True)
        rows, neuralrows = [], []
        for block in c['blocks']:
            for circuit in c['circuit_seeds']:
                paired = {}
                for arm in c['conditions']:
                    budget.check(); start = time.perf_counter(); dest = out/f'{arm}_c{circuit}_s{block["seed"]}'; dest.mkdir()
                    a, graph, metrics, neural, raw, weights = execute(c, circuit, block, arm, archive)
                    np.savez_compressed(dest/'checkpoint.npz', **a)
                    sparse.save_npz(dest/'raw-graph.npz', raw); sparse.save_npz(dest/'weights.npz', weights)
                    for name, value in [('graph', graph), ('metrics', metrics), ('neural', neural)]:
                        core.write_json(dest/(name+'.json'), value)
                    b, g2, m2, n2, _, _ = execute(c, circuit, block, arm, archive)
                    assert graph == g2 and metrics == m2 and neural == n2
                    with np.load(dest/'checkpoint.npz', allow_pickle=False) as saved:
                        for key in saved.files:
                            np.testing.assert_array_equal(saved[key], b[key], err_msg=key)
                        verify_fold(dict(saved), c['alpha'])
                    identity = dict(arm=arm, circuit_seed=circuit, **block)
                    rows.extend(dict(**identity, **r) for r in metrics); neuralrows.append(dict(**identity, **neural))
                    paired[arm] = a, graph
                    core.write_json(dest/'manifest.json', dict(identity=identity, config=c, context=context,
                        timestamp=datetime.now(timezone.utc).isoformat(), exact_replay=True, independent_refit=True,
                        independent_lstsq=True, coefficients_per_task=196, real_task_heads=11, null_task_heads=11,
                        seconds_including_replay=time.perf_counter()-start,
                        artifacts={p.name: core.sha256(p) for p in dest.iterdir() if p.is_file()}))
                    print(f'{cohort} {arm} c{circuit} s{block["seed"]}: saved and verified', flush=True)
                assert paired['renormalized'][1]['raw_sha256'] == paired['fixed_original'][1]['raw_sha256']
                for arm in ['renormalized', 'fixed_original']:
                    for key in ['train_symbols', 'test_symbols', 'input_patterns', 'observed_indices', 'ytrain', 'ytest', 'initial_perturbation']:
                        np.testing.assert_array_equal(paired['real'][0][key], paired[arm][0][key])
                np.testing.assert_array_equal(paired['real'][0]['normalization_factors'], paired['fixed_original'][0]['normalization_factors'])
        summary = summarize(rows, neuralrows, out, c, cohort, direction)
        for p, h in {**prior, **context['source_sha256']}.items():
            assert core.sha256(p) == h, p
        budget.check(); usage = budget.close()
        core.write_json(out/'verification.json', dict(exact_replayed_runs=len(neuralrows), independent_refits=len(neuralrows),
            independent_lstsq_runs=len(neuralrows), prior_result_files_unchanged=len(prior), budget=usage))
        core.write_json(out/'manifest.json', dict(cohort=cohort, config=frozen, effective_config=c, context=context,
            timestamp=datetime.now(timezone.utc).isoformat(), environment=core.environment(),
            discovery_manifest_sha256=core.sha256(discovery/'manifest.json') if discovery else None,
            artifacts={p.relative_to(out).as_posix(): core.sha256(p) for p in out.rglob('*') if p.is_file()}))
        print(dict(gates=summary['gates'], usage=usage), flush=True)
    finally:
        budget.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/normalization_control.json'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cohort', choices=['smoke', 'main', 'confirmation'], required=True)
    parser.add_argument('--discovery', type=Path)
    args = parser.parse_args(); run(args.config, args.out, args.cohort, args.discovery)
