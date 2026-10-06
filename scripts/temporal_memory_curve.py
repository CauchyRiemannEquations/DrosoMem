"""Prospective iid Temporal Decodability Curve; fixed reservoir, ridge heads."""
import argparse
import hashlib
from pathlib import Path
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from flying.models.ridge import RidgeDecoder
from flying.training import whole_brain_memory as core
from tdc_support import (read, write, sha, array_sha, attempt, seal, source_record,
                         old_tree_unchanged, check_manifest)

CONFIG = Path('configs/temporal_memory_curve.json')


def validate(c):
    assert c['alphabet_size'] == 10 and c['lags'] == list(range(21))
    assert c['primary_lags'] == list(range(1, 6)) and c['minimum_excess'] == .10
    assert c['warmup'] > max(c['lags']) and c['alpha'] == 1.
    assert (c['gain'], c['leak'], c['input_fraction'], c['input_amplitude']) == (.9, .6, .1, .5)
    assert c['normalization'] == 'incoming_l1' and c['schedule'] == 'mbon_after_kc'
    assert c['levels'] == ['legacy5', 'brain5'] and c['circuit_seeds'] == [701]
    values = [v for b in c['blocks'] for v in b.values()]
    assert len(values) == len(set(values)) and len(c['blocks']) == 6


def build(c, level, block, raw_override=None):
    condition = core.NetworkCondition(level, 701, block['input_seed'])
    model, obs, info = core.build_model(c['cache'], condition, c)
    if raw_override is not None:
        from flying.brain.timed_reservoir import TimedReservoir
        from flying.brain.diagnostics import normalize_condition
        from flying.data.mushroom_body import load_roles
        _, ids, _ = core.load_connectome(Path('data/flywire_783_mb_left_kc512_s701'))
        roles, _ = load_roles(Path('data/flywire_783_mb_left_kc512_s701'), ids)
        if level == 'brain5':
            with np.load(Path(c['cache'])/'nodes.npz', allow_pickle=False) as a:
                roles = a['roles'].copy()
        weights = normalize_condition(raw_override, c['normalization'], c['gain'])
        weights.sort_indices()
        model = TimedReservoir(weights, model.encoder, roles, c['leak'], c['schedule'])
        info = dict(info, weight_sha256=core.weight_hash(weights))
    return model, obs, info


def collect(model, obs, symbols, resources):
    model.reset()
    features = np.empty((len(symbols), len(obs)))
    norms = np.empty(len(symbols))
    h = hashlib.sha256()
    for t, s in enumerate(symbols):
        state = model.step(int(s))
        features[t] = state[obs]
        norms[t] = np.linalg.norm(state)
        h.update(state.tobytes())
        if t % 100 == 0:
            resources.check()
    assert np.isfinite(features).all()
    return features, norms, model.state.copy(), h.hexdigest()


def controls(train, test, yt, yv, w, k):
    # Only training labels enter frequency/table fitting.
    frequency = np.empty_like(yv)
    current = np.empty_like(yv)
    tables = np.zeros((yt.shape[1], k, k), dtype=np.int64)
    majority = np.empty(yt.shape[1], dtype=np.int64)
    for j in range(yt.shape[1]):
        majority[j] = np.bincount(yt[:, j], minlength=k).argmax()
        np.add.at(tables[j], (train[w:], yt[:, j]), 1)
        mapping = tables[j].argmax(axis=1)
        mapping[tables[j].sum(axis=1) == 0] = majority[j]
        frequency[:, j] = majority[j]
        current[:, j] = mapping[test[w:]]
    return frequency, current, tables, majority


def execute(c, block, level, resources, raw_override=None):
    started = time.perf_counter()
    model, obs, info = build(c, level, block, raw_override)
    w, k = c['warmup'], c['alphabet_size']
    a = dict(observed_indices=obs, input_patterns=model.encoder.patterns.copy())
    trace_hashes = {}
    for split in ['train', 'test']:
        symbols = np.random.default_rng(block[split+'_seed']).integers(0, k, w+c[split+'_samples'], dtype=np.int64)
        x, norms, final, h = collect(model, obs, symbols, resources)
        times = np.arange(w, len(symbols))
        a[split+'_symbols'] = symbols
        a[split+'_features'] = x
        a[split+'_norms'] = norms
        a[split+'_final_state'] = final
        a[split+'_times'] = times
        a['y'+split] = symbols[times[:, None] - np.asarray(c['lags'])[None, :]]
        trace_hashes[split+'_full_state_trajectory_sha256'] = h
    target = np.eye(k)[a['ytrain']].reshape(c['train_samples'], -1)
    decoder = RidgeDecoder(c['alpha']).fit(a['train_features'][w:], target)
    scores = decoder.scores(a['test_features'][w:]).reshape(c['test_samples'], len(c['lags']), k)
    a.update(mean=decoder.mean, scale=decoder.scale, coefficients=decoder.weights, intercept=decoder.target_mean,
             scores=scores, predictions=scores.argmax(axis=2))
    a['frequency_predictions'], a['current_predictions'], a['current_tables'], a['majority'] = controls(a['train_symbols'], a['test_symbols'], a['ytrain'], a['ytest'], w, k)
    rows = []
    for j, lag in enumerate(c['lags']):
        truth = a['ytest'][:, j]
        correct = int(np.count_nonzero(a['predictions'][:, j] == truth))
        freq = int(np.count_nonzero(a['frequency_predictions'][:, j] == truth))
        current = int(np.count_nonzero(a['current_predictions'][:, j] == truth))
        accuracy = correct/len(truth)
        baseline = max(1/k, freq/len(truth), current/len(truth))
        rows.append(dict(level=level, seed=block['seed'], lag=lag, samples=len(truth), correct=correct,
                         frequency_correct=freq, current_correct=current, accuracy=accuracy,
                         frequency_accuracy=freq/len(truth), current_accuracy=current/len(truth), chance=1/k,
                         chance_adjusted=(accuracy-1/k)/(1-1/k), baseline=baseline, baseline_excess=accuracy-baseline))
    info.update(**trace_hashes, array_hashes={n: array_sha(x) for n, x in a.items()},
                decoder_parameters_per_lag=(len(obs)+1)*k, seconds=time.perf_counter()-started)
    assert core.weight_hash(model.weights) == info['weight_sha256']
    return a, rows, info


def statistics(f, c):
    rows = []
    for (level, lag), data in f.groupby(['level', 'lag'], sort=True):
        data = data.sort_values('seed')
        rng = np.random.default_rng(c['bootstrap_seed'])
        ix = rng.integers(0, len(data), (c['bootstrap_draws'], len(data)))
        row = dict(level=level, lag=int(lag), n_blocks=len(data))
        for m in ['accuracy', 'chance_adjusted', 'baseline_excess', 'frequency_accuracy', 'current_accuracy']:
            v = data[m].to_numpy()
            lo, hi = np.quantile(v[ix].mean(axis=1), [.025, .975])
            row.update({m+'_mean': float(v.mean()), m+'_median': float(np.median(v)),
                        m+'_sd': float(v.std(ddof=1)) if len(v)>1 else 0.,
                        m+'_bootstrap_lo': float(lo), m+'_bootstrap_hi': float(hi)})
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(f, c, smoke):
    selected = f[(f.level == 'legacy5') & f.lag.isin(c['primary_lags'])]
    gate = None if smoke else bool(len(selected) == len(c['blocks'])*len(c['primary_lags']) and
                                  (selected.baseline_excess >= c['minimum_excess']).all())
    failures = selected[selected.baseline_excess < c['minimum_excess']][['seed', 'lag', 'baseline_excess']].to_dict('records')
    return dict(primary_pass=gate, primary_endpoint='every legacy5 block at every lag1-5 has >=.10 excess over max(chance,frequency,current-only)',
                smoke=smoke, cases=int(f[['level', 'seed']].drop_duplicates().shape[0]),
                lag_heads=len(f), failed_primary_cells=[] if smoke else failures,
                criterion_unchanged=True, whole_brain_secondary=True)


def plot(f, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for level, color in [('legacy5', '#1469a1'), ('brain5', '#b76126')]:
        data = f[(f.level == level) & (f.lag > 0)]
        for seed, block in data.groupby('seed'):
            ax.plot(block.lag, block.accuracy, color=color, alpha=.15, linewidth=.8)
        mean = data.groupby('lag').accuracy.mean()
        ax.plot(mean.index, mean, 'o-', color=color, label=level)
    controls_mean = f[(f.level == 'legacy5') & (f.lag>0)].groupby('lag')[['frequency_accuracy', 'current_accuracy']].mean()
    ax.plot(controls_mean.index, controls_mean.frequency_accuracy, ':', color='#737373', label='frequency')
    ax.plot(controls_mean.index, controls_mean.current_accuracy, '--', color='#8e5e91', label='current symbol only')
    ax.axhline(.1, color='#444444', ls=':', label='uniform chance')
    ax.set(xlabel='Historical lag (symbol steps)', ylabel='Independent test decoding accuracy',
           xticks=[1,2,3,4,5,8,12,16,20], ylim=(0,1.02), title='Temporal Decodability Curve — iid K10')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out/'tdc.png', dpi=180)
    plt.close(fig)


def run(out, smoke=False):
    frozen = read(CONFIG)
    validate(frozen)
    c = dict(frozen)
    if smoke:
        c.update({n: frozen['smoke'][n] for n in ['train_samples', 'test_samples']})
        c['blocks'] = c['blocks'][:frozen['smoke']['blocks']]
    source = source_record(c, CONFIG)
    rows = []
    with attempt(out, c, 'p1-smoke' if smoke else 'p1-main') as resources:
        write(out/'config.json', c)
        write(out/'source.json', source)
        write(out/'historical-preservation.json', old_tree_unchanged(c['baseline_commit']))
        for block in c['blocks']:
            paired = {}
            for level in c['levels']:
                root = out/f'{level}_s{block["seed"]}'
                root.mkdir()
                a, metrics, graph = execute(c, block, level, resources)
                np.savez_compressed(root/'case.npz', **a)
                write(root/'metrics.json', metrics)
                write(root/'graph.json', graph)
                seal(root, dict(complete=True, block=block, level=level))
                rows.extend(metrics)
                paired[level] = (a['train_symbols'], a['test_symbols'], a['observed_indices'], graph)
                print(f'P1 {level} s{block["seed"]}: saved {len(metrics)} lag heads; {graph["seconds"]:.2f}s', flush=True)
            for j in [0, 1]:
                np.testing.assert_array_equal(paired['legacy5'][j], paired['brain5'][j])
            for n in ['observation_root_ids', 'input_root_ids', 'input_mapping_sha256']:
                assert paired['legacy5'][3][n] == paired['brain5'][3][n]
        f = pd.DataFrame(rows)
        f.to_csv(out/'raw-lags.csv', index=False)
        statistics(f, c).to_csv(out/'curve-summary.csv', index=False)
        write(out/'summary.json', summarize(f, c, smoke))
        plot(f, out)
        for n, h in source['hashes'].items():
            assert sha(n) == h, n
        old_tree_unchanged(c['baseline_commit'])
    seal(out, dict(complete=True, smoke=smoke, config=c, source_commit=source['source_commit'], protocol_sha256=source['hashes'][c['protocol']]))
    print(read(out/'summary.json'), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    with threadpool_limits(1):
        run(a.out, a.smoke)
