"""Independent TDC reconstruction: raw sources, manual rate update, QR/SVD ridge.

Does not import the experiment runner, model constructors, label builder,
normalizer, decoder, baselines or scientific summary functions.
"""
import argparse
import hashlib
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits

from tdc_support import (read, write, sha, array_sha, attempt, seal,
                         check_manifest, old_tree_unchanged, environment, git)


def partial_source():
    p = Path('data/flywire_783_mb_left_kc512_s701')
    meta = read(p/'provenance.json')
    for n, key in [('edges.csv', 'edges_sha256'), ('neurons.json', 'neurons_sha256'), ('annotations.csv', 'roles_sha256')]:
        assert sha(p/n) == meta[key]
    ids = np.asarray(read(p/'neurons.json'), dtype=np.int64)
    lookup = {str(v): i for i, v in enumerate(ids)}
    edges = pd.read_csv(p/'edges.csv', dtype={'pre': str, 'post': str})
    assert not edges.duplicated(['pre', 'post']).any() and not (edges.pre == edges.post).any()
    assert edges['sign'].isin([-1, 1]).all() and (edges['count'] >= 5).all()
    raw = sparse.csr_matrix((edges['count'].to_numpy(float)*edges['sign'].to_numpy(),
                            ([lookup[v] for v in edges.post], [lookup[v] for v in edges.pre])), shape=(len(ids), len(ids)))
    ann = pd.read_csv(p/'annotations.csv', dtype=str).set_index('root_id').loc[ids.astype(str)]
    roles = ann.cell_class.map({'Kenyon_Cell': 'KC', 'MBON': 'MBON', 'DAN': 'DAN'}).mask(ann.cell_type == 'APL', 'APL').to_numpy()
    np.testing.assert_array_equal(roles, ann.role.to_numpy())
    return raw, ids, roles


def validate_whole_source(c, resources):
    import pyarrow.parquet as pq
    cache = Path(c['cache'])
    rawdir = cache.parent/'raw'
    provenance = read(cache/'provenance.json')
    for n, s in provenance['sources'].items():
        assert sha(rawdir/n) == s['sha256'], n
    for n in ['nodes.npz', 'brain5.npz']:
        assert sha(cache/n) == provenance['files'][n]
    nodes = pd.read_csv(rawdir/'Completeness_783.csv', index_col=0)
    ids = nodes.index.to_numpy(dtype=np.int64)
    assert nodes.index.is_unique and nodes.Completed.eq(True).all()
    annotation = pd.read_csv(rawdir/'annotations.tsv', sep='\t', dtype=str).fillna('')
    assert not annotation.root_id.duplicated().any()
    role = annotation.cell_class.map({'Kenyon_Cell': 'KC', 'MBON': 'MBON', 'DAN': 'DAN'})
    annotation['reconstructed_role'] = role.mask(annotation.cell_type == 'APL', 'APL').fillna('OTHER')
    roles = annotation.set_index('root_id').reindex(ids.astype(str)).reconstructed_role.fillna('OTHER').to_numpy(dtype='U5')
    with np.load(cache/'nodes.npz', allow_pickle=False) as a:
        np.testing.assert_array_equal(ids, a['ids'])
        np.testing.assert_array_equal(roles, a['roles'])
    pre, post, values = [], [], []
    for batch in pq.ParquetFile(rawdir/'Connectivity_783.parquet').iter_batches(batch_size=250000):
        d = batch.to_pandas()
        for side in ['Presynaptic', 'Postsynaptic']:
            ix = d[side+'_Index'].to_numpy()
            np.testing.assert_array_equal(ids[ix], d[side+'_ID'].to_numpy())
        assert d.Excitatory.isin([-1, 1]).all()
        np.testing.assert_array_equal(d['Excitatory x Connectivity'], d.Excitatory*d.Connectivity)
        keep = (d.Connectivity >= 5) & (d.Presynaptic_Index != d.Postsynaptic_Index)
        pre.append(d.loc[keep, 'Presynaptic_Index'].to_numpy(dtype=np.int32))
        post.append(d.loc[keep, 'Postsynaptic_Index'].to_numpy(dtype=np.int32))
        values.append(d.loc[keep, 'Excitatory x Connectivity'].to_numpy(dtype=float))
        resources.check()
    raw = sparse.csr_matrix((np.concatenate(values), (np.concatenate(post), np.concatenate(pre))), shape=(len(ids), len(ids)))
    raw.sort_indices()
    cached = sparse.load_npz(cache/'brain5.npz').tocsr()
    assert raw.nnz == sum(map(len, values)) == 2700513
    assert (raw != cached).nnz == 0
    small, small_ids, small_roles = partial_source()
    local = pd.Index(ids).get_indexer(small_ids)
    assert (local >= 0).all() and (raw[local][:, local] != small).nnz == 0
    np.testing.assert_array_equal(roles[local], small_roles)
    return dict(full_graph_rebuilt_from_pinned_parquet=True, roles_rebuilt_from_pinned_annotations=True,
                partial_induced_graph_exact=True, neurons=len(ids), edges=raw.nnz,
                raw_source_sha256={n: sha(rawdir/n) for n in provenance['sources']})


def independent_graph(c, level, input_seed, raw_override=None):
    small, small_ids, small_roles = partial_source()
    if level == 'legacy5':
        raw, ids, roles = small, small_ids, small_roles
    else:
        cache = Path(c['cache'])
        provenance = read(cache/'provenance.json')
        for n in ['nodes.npz', 'brain5.npz']:
            assert sha(cache/n) == provenance['files'][n]
        raw = sparse.load_npz(cache/'brain5.npz').tocsr()
        with np.load(cache/'nodes.npz', allow_pickle=False) as a:
            ids, roles = a['ids'].copy(), a['roles'].copy()
    if raw_override is not None:
        assert raw_override.shape == raw.shape
        raw = raw_override.copy()
    # Rebuild input patterns from seed; never use the saved patterns as input.
    kc = np.flatnonzero(small_roles == 'KC')
    local = pd.Index(ids).get_indexer(small_ids)
    assert (local >= 0).all()
    patterns = np.zeros((10, len(ids)))
    rng = np.random.default_rng(input_seed)
    count = max(1, int(len(kc)*c['input_fraction']))
    for s in range(10):
        selected = rng.choice(len(kc), count, replace=False)
        patterns[s, local[kc[selected]]] = c['input_amplitude']
    obs = local[np.flatnonzero(small_roles == 'MBON')]
    assert len(obs) == 48
    weights = raw.copy().tocsr()
    strength = np.asarray(abs(weights).sum(axis=1)).ravel()
    factor = np.zeros(len(ids))
    factor[strength > 0] = c['gain']/strength[strength > 0]
    weights.data *= np.repeat(factor, np.diff(weights.indptr))
    weights.sort_indices()
    row = np.asarray(abs(weights).sum(axis=1)).ravel()
    np.testing.assert_allclose(row[row>0], c['gain'], atol=1e-14, rtol=0)
    return weights, patterns, roles, obs, ids


def manual_trajectory(weights, patterns, roles, observed, symbols, c, resources):
    state = np.zeros(weights.shape[0])
    kc = np.flatnonzero(roles == 'KC')
    mbon = np.flatnonzero(roles == 'MBON')
    mbon_weights = weights[mbon]
    features, norms = np.empty((len(symbols), 48)), np.empty(len(symbols))
    h = hashlib.sha256()
    leak = c['leak']
    for t, s in enumerate(symbols):
        old = state.copy()
        stimulation = patterns[s]
        state = (1-leak)*old + leak*np.tanh(weights@old + stimulation)
        mixed = old.copy()
        mixed[kc] = state[kc]
        state[mbon] = (1-leak)*old[mbon] + leak*np.tanh(mbon_weights@mixed + stimulation[mbon])
        features[t], norms[t] = state[observed], np.linalg.norm(state)
        h.update(state.tobytes())
        if t % 100 == 0:
            resources.check()
    return features, norms, state, h.hexdigest()


def verify_case(root, c, block, level, resources, raw_override=None):
    begin = time.perf_counter()
    check_manifest(root)
    graph = read(root/'graph.json')
    weights, patterns, roles, obs, ids = independent_graph(c, level, block['input_seed'], raw_override)
    rows = []
    exact_trace = {}
    w, k, lags = c['warmup'], c['alphabet_size'], c['lags']
    with np.load(root/'case.npz', allow_pickle=False) as a:
        for n in a.files:
            assert array_sha(a[n]) == graph['array_hashes'][n], n
        np.testing.assert_array_equal(a['input_patterns'], patterns)
        np.testing.assert_array_equal(a['observed_indices'], obs)
        assert ids[obs].astype(str).tolist() == graph['observation_root_ids']
        # Independent matrix hash, matching the specified CSR serialization.
        h = hashlib.sha256()
        for x in [weights.data, weights.indices, weights.indptr]:
            h.update(x.tobytes())
        assert h.hexdigest() == graph['weight_sha256']
        rebuilt, labels, symbols = {}, {}, {}
        for split in ['train', 'test']:
            s = np.random.default_rng(block[split+'_seed']).integers(0, k, w+c[split+'_samples'], dtype=np.int64)
            symbols[split] = s
            np.testing.assert_array_equal(s, a[split+'_symbols'])
            x, norms, final, h = manual_trajectory(weights, patterns, roles, obs, s, c, resources)
            np.testing.assert_allclose(x, a[split+'_features'], atol=1e-9, rtol=1e-9)
            np.testing.assert_allclose(norms, a[split+'_norms'], atol=1e-9, rtol=1e-9)
            np.testing.assert_allclose(final, a[split+'_final_state'], atol=1e-9, rtol=1e-9)
            exact_trace[split] = h == graph[split+'_full_state_trajectory_sha256']
            rebuilt[split] = x[w:]
            times = np.arange(w, len(s))
            np.testing.assert_array_equal(a[split+'_times'], times)
            # Explicit slicing rather than runner's vectorized index matrix.
            labels[split] = np.column_stack([s[w-lag:len(s)-lag] if lag else s[w:] for lag in lags])
            np.testing.assert_array_equal(labels[split], a['y'+split])
        mean = rebuilt['train'].mean(axis=0)
        scale = np.maximum(rebuilt['train'].std(axis=0), 1e-5)
        np.testing.assert_allclose(mean, a['mean'], atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(scale, a['scale'], atol=1e-9, rtol=1e-9)
        z = (rebuilt['train']-mean)/scale
        targets = np.eye(k)[labels['train']].reshape(len(z), -1)
        intercept = targets.mean(axis=0)
        design = np.vstack([z, np.sqrt(c['alpha'])*np.eye(48)])
        rhs = np.vstack([targets-intercept, np.zeros((48, targets.shape[1]))])
        coefficients = np.linalg.lstsq(design, rhs, rcond=None)[0]
        np.testing.assert_allclose(coefficients, a['coefficients'], atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(intercept, a['intercept'], atol=1e-9, rtol=1e-9)
        scores = (((rebuilt['test']-mean)/scale)@coefficients+intercept).reshape(c['test_samples'], len(lags), k)
        np.testing.assert_allclose(scores, a['scores'], atol=1e-9, rtol=1e-9)
        predictions = scores.argmax(axis=2)
        np.testing.assert_array_equal(predictions, a['predictions'])
        for j, lag in enumerate(lags):
            train_label, test_label = labels['train'][:, j], labels['test'][:, j]
            majority = np.bincount(train_label, minlength=k).argmax()
            table = np.zeros((k, k), dtype=np.int64)
            for current_symbol, label in zip(symbols['train'][w:], train_label):
                table[current_symbol, label] += 1
            mapping = np.array([row.argmax() if row.sum() else majority for row in table])
            current_pred = mapping[symbols['test'][w:]]
            frequency_pred = np.full(len(test_label), majority)
            np.testing.assert_array_equal(a['current_tables'][j], table)
            np.testing.assert_array_equal(a['current_predictions'][:, j], current_pred)
            np.testing.assert_array_equal(a['frequency_predictions'][:, j], frequency_pred)
            assert a['majority'][j] == majority
            correct = int(np.sum(predictions[:, j] == test_label))
            frequency = int(np.sum(frequency_pred == test_label))
            current = int(np.sum(current_pred == test_label))
            n = len(test_label)
            baseline = max(1/k, frequency/n, current/n)
            rows.append(dict(level=level, seed=block['seed'], lag=lag, samples=n, correct=correct,
                             frequency_correct=frequency, current_correct=current, accuracy=correct/n,
                             frequency_accuracy=frequency/n, current_accuracy=current/n, chance=1/k,
                             chance_adjusted=(correct/n-1/k)/(1-1/k), baseline=baseline, baseline_excess=correct/n-baseline))
    saved = pd.DataFrame(read(root/'metrics.json'))
    fresh = pd.DataFrame(rows)[saved.columns]
    np.testing.assert_allclose(saved.select_dtypes('number'), fresh.select_dtypes('number'), atol=1e-12, rtol=0)
    assert saved.level.tolist() == fresh.level.tolist()
    return rows, dict(level=level, seed=block['seed'], seconds=time.perf_counter()-begin,
                     exact_full_state_trace_hash=exact_trace, manual_state_update=True,
                     independently_refit_heads=len(lags), prediction_arrays_exact=True)


def verify_summary(f, result, c, smoke):
    raw = pd.read_csv(result/'raw-lags.csv')
    pd.testing.assert_frame_equal(raw, f[raw.columns], check_exact=False, atol=1e-12, rtol=0)
    chosen = f[(f.level == 'legacy5') & f.lag.between(1,5)]
    passed = None if smoke else bool(len(chosen) == 30 and all(x >= .10 for x in chosen.baseline_excess))
    summary = read(result/'summary.json')
    assert summary['primary_pass'] is passed
    failures = chosen[chosen.baseline_excess < .10][['seed', 'lag', 'baseline_excess']].to_dict('records')
    assert summary['failed_primary_cells'] == ([] if smoke else failures)
    assert summary['cases'] == f[['level', 'seed']].drop_duplicates().shape[0]
    assert summary['lag_heads'] == len(f)
    # Recalculate every reported mean/median/SD and descriptive bootstrap interval.
    curve = pd.read_csv(result/'curve-summary.csv')
    for _, row in curve.iterrows():
        b = f[(f.level == row.level) & (f.lag == row.lag)].sort_values('seed')
        rng = np.random.default_rng(c['bootstrap_seed'])
        index = rng.integers(0, len(b), (c['bootstrap_draws'], len(b)))
        assert row.n_blocks == len(b)
        for metric in ['accuracy', 'chance_adjusted', 'baseline_excess', 'frequency_accuracy', 'current_accuracy']:
            x = b[metric].to_numpy()
            lo, hi = np.quantile(x[index].mean(axis=1), [.025, .975])
            expected = {'mean': x.mean(), 'median': np.median(x), 'sd': x.std(ddof=1) if len(x)>1 else 0., 'bootstrap_lo': lo, 'bootstrap_hi': hi}
            for suffix, value in expected.items():
                assert abs(row[metric+'_'+suffix]-value) < 1e-12
    return passed


def verify(result, out):
    m = check_manifest(result)
    c = read(result/'config.json')
    assert c == m['config']
    source = read(result/'source.json')
    for n, h in source['hashes'].items():
        assert sha(n) == h, n
    assert source['source_commit'] == m['source_commit']
    rows, cases = [], []
    with attempt(out, c, 'p1-independent-validation') as resources:
        write(out/'environment.json', environment())
        write(out/'source-validation.json', dict(verifier_commit=git('rev-parse', 'HEAD'), result_source_commit=source['source_commit'], source_hashes_checked=len(source['hashes'])))
        full = validate_whole_source(c, resources)
        write(out/'source-graph-audit.json', full)
        for block in c['blocks']:
            for level in c['levels']:
                a, check = verify_case(result/f'{level}_s{block["seed"]}', c, block, level, resources)
                rows.extend(a)
                cases.append(check)
                print(f'Independently verified P1 {level} s{block["seed"]}', flush=True)
        f = pd.DataFrame(rows)
        passed = verify_summary(f, result, c, m['smoke'])
        f.to_csv(out/'recomputed-lags.csv', index=False)
        write(out/'case-checks.json', cases)
        write(out/'checks.json', dict(all_checks_pass=True, result_manifest_sha256=sha(result/'manifest.json'),
                                    primary_pass=passed, cases=len(cases), replayed_streams=2*len(cases),
                                    independently_refit_lag_heads=len(rows), source_graph_reconstruction=full,
                                    historical_preservation=old_tree_unchanged(c['baseline_commit'])))
    seal(out, dict(complete=True, result_manifest_sha256=sha(result/'manifest.json')))
    print(read(out/'checks.json'), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('result', type=Path)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    with threadpool_limits(1):
        verify(a.result, a.out)
