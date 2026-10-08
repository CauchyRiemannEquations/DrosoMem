"""Independent M3 source, acyclic mask, finite-window, ridge and criterion replay.

Imports no M3 runner, reservoir class, graph-mask, label, readout or summary
implementation. Frozen independent source/dynamics helpers are shared only with
the preceding independent verification pipeline.
"""
import argparse
from fractions import Fraction
import hashlib
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from threadpoolctl import threadpool_limits

from verify_temporal_memory_curve import independent_graph, validate_whole_source, partial_source
from verify_temporal_mechanism import analytic_prototypes, trajectory
from cycle_support import (config, read, write, sha, array_sha, attempt, seal,
                           check_manifest, history_preserved)
from tdc_support import environment, git
from tdc_pool import pooled


ESTIMATES = ['mean', 'median', 'sd', 'bootstrap_lo', 'bootstrap_hi']
CURVE_METRICS = ['accuracy', 'baseline_excess', 'chance_adjusted',
                 'frequency_accuracy', 'current_accuracy']
PROTOCOL_COMMIT = '438cae437e110dbeb9cb2a02d0a58cfe61e6d233'


def matrix_digest(matrix):
    digest = hashlib.sha256()
    for vector in [matrix.data, matrix.indices, matrix.indptr]:
        digest.update(vector.tobytes())
    return digest.hexdigest()


def independent_dag(original, roles, ids, observed):
    """Filter CSR rows using separately sorted biological source coordinates."""
    original = original.tocsr()
    roles, ids = np.asarray(roles), np.asarray(ids, dtype=np.int64)
    assert original.shape == (len(ids), len(ids)) and len(np.unique(ids)) == len(ids)
    def key(index):
        tier = 0 if roles[index] == 'KC' else 2 if roles[index] == 'MBON' else 1
        return tier, int(ids[index])
    order = sorted(range(len(ids)), key=key)
    rank = np.empty(len(ids), dtype=np.int64)
    rank[order] = np.arange(len(ids), dtype=np.int64)
    data, indices, pointers = [], [], [0]
    for target in range(len(ids)):
        start, end = original.indptr[target:target+2]
        sources = original.indices[start:end]
        retained = np.array([rank[int(source)] < rank[target] for source in sources], dtype=bool)
        data.append(original.data[start:end][retained])
        indices.append(sources[retained])
        pointers.append(pointers[-1]+int(retained.sum()))
    dag = sparse.csr_matrix((np.concatenate(data), np.concatenate(indices),
                             np.asarray(pointers, dtype=original.indptr.dtype)), shape=original.shape)
    dag.sort_indices()
    assert np.isfinite(dag.data).all() and not (dag.data == 0).any()
    depth = np.zeros(len(ids), dtype=np.int64)
    input_depth = np.full(len(ids), -1, dtype=np.int64)
    input_depth[roles == 'KC'] = 0
    for target in order:
        for source in dag.indices[dag.indptr[target]:dag.indptr[target+1]]:
            assert rank[int(source)] < rank[target]
            delay = 0 if roles[source] == 'KC' and roles[target] == 'MBON' else 1
            depth[target] = max(depth[target], depth[source]+delay)
            if input_depth[source] >= 0:
                input_depth[target] = max(input_depth[target], input_depth[source]+delay)
    count, labels = connected_components(dag, directed=True, connection='strong')
    assert count == len(ids) and len(np.unique(labels)) == len(ids)
    kc, mb = np.flatnonzero(roles == 'KC'), np.flatnonzero(roles == 'MBON')
    assert (dag[mb][:, kc] != original[mb][:, kc]).nnz == 0
    arrays = dict(root_ids=ids, roles=roles.astype('U5'), rank=rank,
                  observed_indices=np.asarray(observed, dtype=np.int64),
                  global_dependency_depth=depth, input_pool_dependency_depth=input_depth)
    for name, matrix in [('original', original), ('dag', dag)]:
        post = np.repeat(np.arange(len(ids)), np.diff(matrix.indptr))
        arrays[name+'_indegree'] = np.diff(matrix.indptr).astype(np.int64)
        arrays[name+'_outdegree'] = np.bincount(matrix.indices, minlength=len(ids))
        arrays[name+'_incoming_abs_weight'] = np.bincount(post, weights=np.abs(matrix.data), minlength=len(ids))
        arrays[name+'_outgoing_abs_weight'] = np.bincount(matrix.indices, weights=np.abs(matrix.data), minlength=len(ids))
    mass = arrays['original_incoming_abs_weight']
    arrays['retained_incoming_abs_weight_fraction'] = np.divide(
        arrays['dag_incoming_abs_weight'], mass, out=np.zeros_like(mass), where=mass > 0)
    topology = dict(nodes=len(ids), original_edges=int(original.nnz), dag_edges=int(dag.nnz),
                    removed_edges=int(original.nnz-dag.nnz), dag_cycle_edges=0,
                    strict_rank_increase_on_every_retained_edge=True,
                    protected_kc_to_mbon_edges=int(original[mb][:, kc].nnz),
                    protected_kc_to_mbon_all_kept_exactly=True,
                    global_dependency_depth=int(depth.max(initial=0)),
                    observed_dependency_depth=int(depth[np.asarray(observed)].max(initial=0)),
                    original_weight_sha256=matrix_digest(original), weight_sha256=matrix_digest(dag),
                    dag_weight_sha256=matrix_digest(dag))
    topology.update(independent_topology(original, dag, roles, arrays))
    return dag, rank, arrays, topology


def independent_topology(original, dag, roles, arrays):
    """All structural summaries recomputed from the independently filtered CSR."""
    def describe(vector):
        vector = np.asarray(vector)
        assert len(vector)
        return dict(count=len(vector), min=float(np.min(vector)), max=float(np.max(vector)),
                    mean=float(np.mean(vector)), median=float(np.median(vector)),
                    sd=float(np.std(vector)), sum=float(np.sum(vector)))
    def components(matrix):
        count, labels = connected_components(matrix, directed=True, connection='strong')
        sizes = np.bincount(labels)
        targets = np.repeat(np.arange(matrix.shape[0]), np.diff(matrix.indptr))
        sources = matrix.indices
        same = labels[sources] == labels[targets]
        cycle = same & ((sizes[labels[sources]] > 1) | (sources == targets))
        return dict(components=count, largest_sizes=sorted(map(int, sizes), reverse=True)[:10],
                    nontrivial_components=int(np.count_nonzero(sizes > 1)),
                    cycle_participating_edges=int(np.count_nonzero(cycle)),
                    self_edges=int(np.count_nonzero(sources == targets)))
    unique, counts = np.unique(roles, return_counts=True)
    roles_count = {str(role): int(count) for role, count in zip(unique, counts)}
    blocks, per_neuron = {}, {}
    for name, matrix in [('original', original), ('dag', dag)]:
        target = np.repeat(np.arange(len(roles)), np.diff(matrix.indptr))
        source = matrix.indices
        blocks[name] = {}
        for pre in sorted(roles_count):
            for post in sorted(roles_count):
                selected = (roles[source] == pre) & (roles[target] == post)
                blocks[name][pre+'->'+post] = dict(edges=int(selected.sum()),
                    absolute_weight_sum=float(np.abs(matrix.data[selected]).sum()))
        per_neuron[name] = {metric: describe(arrays[name+'_'+metric]) for metric in
                            ['indegree', 'outdegree', 'incoming_abs_weight', 'outgoing_abs_weight']}
    observed = arrays['observed_indices']
    fraction = arrays['retained_incoming_abs_weight_fraction']
    input_depth = arrays['input_pool_dependency_depth'][observed]
    return dict(roles=roles_count, original_scc=components(original), dag_scc=components(dag),
                rank_definition='KC first; other roles second; MBON last; ascending numeric root ID within group',
                normalization='original incoming-L1 normalized weights masked without rescaling',
                role_blocks=blocks, per_neuron_statistics=per_neuron,
                retained_incoming_mass_fraction_active_rows=describe(
                    fraction[arrays['original_incoming_abs_weight'] > 0]),
                retained_incoming_mass_fraction_observed_rows=describe(fraction[observed]),
                delay_definition='0 for KC->MBON; 1 for every other edge under mbon_after_kc; no direct carry',
                input_pool_definition='all neurons with KC role, independent of per-symbol selected inputs',
                input_pool_count=int(np.count_nonzero(roles == 'KC')),
                input_pool_observed_depth=int(input_depth.max(initial=-1)),
                input_pool_observed_depths=input_depth.tolist(),
                input_pool_observed_reachable=int(np.count_nonzero(input_depth >= 0)),
                observed_count=len(observed))


def compare_topology(published, expected):
    """Published topology fields must all be derivable independently."""
    assert set(published) <= set(expected), set(published)-set(expected)
    def equal(actual, fresh):
        if isinstance(actual, dict):
            assert set(actual) == set(fresh)
            for key in actual:
                equal(actual[key], fresh[key])
        elif isinstance(actual, list):
            assert actual == fresh
        elif isinstance(actual, float):
            assert abs(actual-fresh) <= 1e-12, (actual, fresh)
        else:
            assert actual == fresh, (actual, fresh)
    for key in published:
        equal(published[key], expected[key])


def verify_registered_structure(c, resources, out):
    root = Path(c['namespace'])/'structural_audit'
    check_manifest(root)
    reports = {}
    for level in c['levels']:
        weights, _, roles, observed, ids = independent_graph(c, level, c['blocks'][0]['input_seed'])
        dag, _, arrays, topology = independent_dag(weights, roles, ids, observed)
        assert topology['global_dependency_depth'] == c['expected_depth'][level]
        compare_topology(read(root/(level+'.json')), topology)
        with np.load(root/(level+'-per-neuron.npz'), allow_pickle=False) as saved:
            assert set(saved.files) == set(arrays)
            for name, values in arrays.items():
                if values.dtype.kind in 'iuUb':
                    np.testing.assert_array_equal(saved[name], values)
                else:
                    np.testing.assert_allclose(saved[name], values, atol=1e-9, rtol=1e-9)
        with np.load(root/(level+'-dag.npz'), allow_pickle=False) as stored:
            stored_dag = sparse.csr_matrix((stored['data'], stored['indices'], stored['indptr']),
                                           shape=tuple(stored['shape']))
        assert matrix_digest(dag) == matrix_digest(stored_dag)
        np.savez_compressed(out/(level+'-independent-per-neuron.npz'), **arrays)
        write(out/(level+'-independent-topology.json'), topology)
        reports[level] = dict(all_checks_pass=True, nodes=topology['nodes'],
                              original_edges=topology['original_edges'], dag_edges=topology['dag_edges'],
                              global_dependency_depth=topology['global_dependency_depth'])
        resources.check()
    return dict(all_checks_pass=True, levels=reports, source_manifest_sha256=sha(root/'manifest.json'))


def certificate_replay(weights, patterns, roles, block, c, root, resources, horizon):
    """Independent arbitrary-initial-state finite-window probe and exact replay."""
    k, n = c['alphabet_size'], weights.shape[0]
    kc, mb = np.flatnonzero(roles == 'KC'), np.flatnonzero(roles == 'MBON')
    wm, b = weights[mb], c['leak']
    generated = {
        'initial_a': np.random.default_rng(block['initial_a_seed']).standard_normal(n),
        'initial_b': np.random.default_rng(block['initial_b_seed']).standard_normal(n),
        'prefix_a_symbols': np.random.default_rng(block['prefix_a_seed']).integers(
            0, k, c['certificate_prefix_steps'], dtype=np.int64),
        'prefix_b_symbols': np.random.default_rng(block['prefix_b_seed']).integers(
            0, k, c['certificate_prefix_steps'], dtype=np.int64),
        'suffix_symbols': np.random.default_rng(block['suffix_seed']).integers(
            0, k, horizon+1, dtype=np.int64),
    }
    hashes = {}
    for branch in ['a', 'b']:
        state = generated['initial_'+branch].copy()
        digest = hashlib.sha256()
        prefix = generated['prefix_'+branch+'_symbols']
        symbols = np.concatenate([prefix, generated['suffix_symbols']])
        for t, symbol in enumerate(symbols):
            previous = state.copy()
            state = np.zeros(n)+b*np.tanh(weights.dot(previous)+patterns[symbol])
            pre = previous.copy()
            pre[kc] = state[kc]
            state[mb] = np.zeros(len(mb))+b*np.tanh(wm.dot(pre)+patterns[symbol, mb])
            digest.update(state.tobytes())
            if t+1 == len(prefix):
                generated['prefix_'+branch+'_final_state'] = state.copy()
            if t % 100 == 0:
                resources.check()
        generated['final_'+branch+'_state'] = state.copy()
        hashes[branch] = digest.hexdigest()
    assert not np.array_equal(generated['initial_a'], generated['initial_b'])
    assert not np.array_equal(generated['prefix_a_final_state'], generated['prefix_b_final_state'])
    np.testing.assert_array_equal(generated['final_a_state'], generated['final_b_state'])
    report = read(root/'certificate.json')
    assert report['all_checks_pass'] is True
    assert report['global_dependency_depth'] == horizon
    assert report['common_suffix_steps'] == horizon+1
    assert report['prefix_steps'] == c['certificate_prefix_steps']
    assert report['distinct_prefix_states'] is True and report['exact_terminal_equality'] is True
    for branch in ['a', 'b']:
        assert report[branch+'_full_state_trajectory_sha256'] == hashes[branch]
    with np.load(root/'certificate.npz', allow_pickle=False) as saved:
        assert set(saved.files) == set(generated) == set(report['array_hashes'])
        for name, value in generated.items():
            assert array_sha(saved[name]) == report['array_hashes'][name]
            np.testing.assert_array_equal(saved[name], value)
    return dict(all_checks_pass=True, global_dependency_depth=horizon,
                common_suffix_steps=horizon+1, exact_terminal_equality=True,
                exact_certificate_full_state_trace_hashes=hashes,
                source_certificate_sha256=sha(root/'certificate.json'))


def case_job(job):
    c, block, level, arm = job['config'], job['block'], job['level'], job['arm']
    result, out = Path(job['result']), Path(job['out'])
    root = result/f'{level}_{arm}_s{block["seed"]}'
    manifest = check_manifest(root)
    assert manifest['block'] == block and manifest['level'] == level and manifest['arm'] == arm
    info, switches = read(root/'graph.json'), c['arms'][arm]
    assert switches['carry'] is False and info['switches'] == switches
    with threadpool_limits(1):
        with attempt(out, c, 'cycles-independent-case') as resources:
            original, patterns, roles, obs, ids = independent_graph(c, level, block['input_seed'])
            assert matrix_digest(original) == info['original_weight_sha256']
            assert info['drive_multiplier'] == .6 and info['direct_carry_multiplier'] == 0
            assert info['same_step_kc_to_mbon_retained'] is True
            weights, topology, rank = original, None, None
            if switches['dag']:
                weights, rank, structural_arrays, topology = independent_dag(original, roles, ids, obs)
                assert topology['global_dependency_depth'] == c['expected_depth'][level]
                assert info['global_dependency_depth'] == topology['global_dependency_depth']
                published = read(root/'topology.json')
                compare_topology(published, topology)
                np.savez_compressed(out/'independent-structure.npz', **structural_arrays)
                write(out/'independent-topology.json', topology)
            else:
                assert info['global_dependency_depth'] is None
            assert matrix_digest(weights) == info['weight_sha256']
            proto = analytic_prototypes(weights, patterns, roles, c)
            original_proto = analytic_prototypes(original, patterns, roles, c)
            np.testing.assert_array_equal(proto, original_proto)
            expected_inputs = [ids[pattern != 0].astype(str).tolist() for pattern in patterns]
            # Root IDs describe the original source input universe in its order.
            assert [set(row) for row in expected_inputs] == [set(row) for row in info['input_root_ids']]
            assert ids[obs].astype(str).tolist() == info['observation_root_ids']
            _, local_ids, _ = partial_source()
            local_indices = pd.Index(ids).get_indexer(local_ids)
            assert (local_indices >= 0).all()
            input_digest = hashlib.sha256(local_ids.tobytes()+patterns[:, local_indices].tobytes()).hexdigest()
            assert info['input_mapping_sha256'] == input_digest
            rows, exact = [], {}
            w, k = c['warmup'], c['alphabet_size']
            rebuilt, labels, streams = {}, {}, {}
            with np.load(root/'case.npz', allow_pickle=False) as saved:
                assert set(saved.files) == set(info['array_hashes'])
                for name in saved.files:
                    assert array_sha(saved[name]) == info['array_hashes'][name], name
                np.testing.assert_array_equal(saved['input_patterns'], patterns)
                np.testing.assert_array_equal(saved['observed_indices'], obs)
                np.testing.assert_array_equal(saved['zero_state_observed_prototypes'], proto[:, obs])
                if switches['dag']:
                    np.testing.assert_array_equal(saved['dag_rank'], rank)
                if arm == 'instantaneous':
                    np.testing.assert_array_equal(saved['instant_state_prototypes'], proto)
                for split in ['train', 'test']:
                    symbols = np.random.default_rng(block[split+'_seed']).integers(
                        0, k, w+c[split+'_samples'], dtype=np.int64)
                    streams[split] = symbols
                    np.testing.assert_array_equal(saved[split+'_symbols'], symbols)
                    features, norms, final, digest = trajectory(
                        weights, patterns, roles, obs, symbols, c, False,
                        switches['synaptic_history'], resources)
                    for name, value in [('features', features), ('norms', norms), ('final_state', final)]:
                        np.testing.assert_allclose(value, saved[split+'_'+name], atol=1e-9, rtol=1e-9)
                    exact[split] = digest == info[split+'_full_state_trajectory_sha256']
                    assert exact[split], 'Exact complete-state trajectory digest mismatch'
                    if arm == 'instantaneous':
                        np.testing.assert_array_equal(saved[split+'_features'], proto[:, obs][symbols])
                    np.testing.assert_array_equal(saved[split+'_times'], np.arange(w, len(symbols)))
                    labels[split] = np.column_stack([
                        symbols[w-lag:len(symbols)-lag] if lag else symbols[w:] for lag in c['lags']])
                    np.testing.assert_array_equal(saved['y'+split], labels[split])
                    rebuilt[split] = features[w:]
                mean = np.mean(rebuilt['train'], axis=0)
                scale = np.maximum(np.std(rebuilt['train'], axis=0), 1e-5)
                np.testing.assert_allclose(mean, saved['mean'], atol=1e-9, rtol=1e-9)
                np.testing.assert_allclose(scale, saved['scale'], atol=1e-9, rtol=1e-9)
                design = (rebuilt['train']-mean)/scale
                target = np.eye(k)[labels['train']].reshape(len(design), -1)
                intercept = target.mean(axis=0)
                coefficients = np.linalg.lstsq(
                    np.vstack([design, np.sqrt(c['alpha'])*np.eye(48)]),
                    np.vstack([target-intercept, np.zeros((48, target.shape[1]))]), rcond=None)[0]
                np.testing.assert_allclose(coefficients, saved['coefficients'], atol=1e-9, rtol=1e-9)
                np.testing.assert_allclose(intercept, saved['intercept'], atol=1e-9, rtol=1e-9)
                scores = (((rebuilt['test']-mean)/scale)@coefficients+intercept).reshape(
                    c['test_samples'], len(c['lags']), k)
                np.testing.assert_allclose(scores, saved['scores'], atol=1e-9, rtol=1e-9)
                prediction = scores.argmax(axis=2)
                np.testing.assert_array_equal(prediction, saved['predictions'])
                for j, lag in enumerate(c['lags']):
                    training, actual = labels['train'][:, j], labels['test'][:, j]
                    n = len(actual)
                    majority = int(np.bincount(training, minlength=k).argmax())
                    table = np.zeros((k, k), dtype=np.int64)
                    for current, past in zip(streams['train'][w:], training):
                        table[current, past] += 1
                    mapping = np.array([row.argmax() if row.sum() else majority for row in table])
                    current_predictions = mapping[streams['test'][w:]]
                    frequency_predictions = np.full(n, majority)
                    np.testing.assert_array_equal(table, saved['current_tables'][j])
                    np.testing.assert_array_equal(current_predictions, saved['current_predictions'][:, j])
                    np.testing.assert_array_equal(frequency_predictions, saved['frequency_predictions'][:, j])
                    assert saved['majority'][j] == majority
                    correct = int(np.count_nonzero(prediction[:, j] == actual))
                    freq = int(np.count_nonzero(frequency_predictions == actual))
                    current = int(np.count_nonzero(current_predictions == actual))
                    baseline = max(1/k, freq/n, current/n)
                    rows.append(dict(level=level, arm=arm, seed=block['seed'], lag=lag,
                                     samples=n, correct=correct, frequency_correct=freq,
                                     current_correct=current, accuracy=correct/n,
                                     frequency_accuracy=freq/n, current_accuracy=current/n,
                                     chance=1/k, chance_adjusted=(correct/n-1/k)/(1-1/k),
                                     baseline=baseline, baseline_excess=correct/n-baseline))
            compare_frame(pd.DataFrame(read(root/'metrics.json')), pd.DataFrame(rows), ['lag'])
            certificate = None
            if switches['dag']:
                certificate = certificate_replay(weights, patterns, roles, block, c, root,
                                                  resources, topology['global_dependency_depth'])
                write(out/'finite-window-checks.json', certificate)
            write(out/'checks.json', dict(all_checks_pass=True, level=level, arm=arm,
                                         seed=block['seed'], exact_full_state_trace_hash=exact,
                                         independently_refit_heads=len(c['lags']),
                                         instantaneous_no_history_certificate=arm == 'instantaneous',
                                         preserved_zero_state_input_response=True,
                                         finite_window_certificate=certificate))
        seal(out, dict(complete=True, result_case_manifest_sha256=sha(root/'manifest.json')))
    return dict(identity=f'{level}/{arm}/s{block["seed"]}', rows=rows, exact=exact,
                certificate=certificate, resources=read(out/'resources.json'))


def descriptive(values, c):
    data = np.array([float(value) for value in values])
    assert len(data) and np.isfinite(data).all()
    draws = np.random.default_rng(c['bootstrap_seed']).integers(
        0, len(data), (c['bootstrap_draws'], len(data)))
    lower, upper = np.quantile(np.mean(data[draws], axis=1), [.025, .975])
    return dict(mean=float(data.mean()), median=float(np.median(data)),
                sd=float(data.std(ddof=1)) if len(data) > 1 else 0.,
                bootstrap_lo=float(lower), bootstrap_hi=float(upper))


def compare_estimates(actual, expected):
    assert set(actual) == set(expected) == set(ESTIMATES)
    for name in ESTIMATES:
        assert abs(actual[name]-expected[name]) <= 1e-12, (name, actual, expected)


def compare_frame(actual, expected, keys):
    assert set(actual.columns) == set(expected.columns)
    assert not actual.duplicated(keys).any() and not expected.duplicated(keys).any()
    actual = actual.sort_values(keys).reset_index(drop=True)
    expected = expected[actual.columns].sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected, check_exact=False, atol=1e-12, rtol=0,
                                  check_dtype=False)
    for name in actual.columns:
        if name in ['seed', 'lag', 'samples', 'correct', 'frequency_correct', 'current_correct',
                    'n_blocks', 'numerator', 'denominator', 'meets_margin'] or name.endswith(
                        ('_numerator', '_denominator')):
            np.testing.assert_array_equal(actual[name].to_numpy(), expected[name].to_numpy())


def compare_csv(path, expected, keys):
    compare_frame(pd.read_csv(path), expected, keys)


def verify_pairing(result, c):
    checked = 0
    keys = ['input_patterns', 'observed_indices', 'train_symbols', 'test_symbols',
            'ytrain', 'ytest', 'train_times', 'test_times', 'zero_state_observed_prototypes']
    for block in c['blocks']:
        anchor_info, anchor_arrays = None, None
        for level in c['levels']:
            if block['seed'] not in c['blocks_by_level'][level]:
                continue
            root = result/f'{level}_intact_synaptic_s{block["seed"]}'
            info = read(root/'graph.json')
            with np.load(root/'case.npz', allow_pickle=False) as saved:
                streams = {n: saved[n] for n in ['train_symbols', 'test_symbols', 'ytrain', 'ytest']}
                if anchor_info is None:
                    anchor_info, anchor_arrays = info, streams
                else:
                    for name in ['input_root_ids', 'observation_root_ids', 'input_mapping_sha256']:
                        assert info[name] == anchor_info[name], (level, name)
                    for name in streams:
                        np.testing.assert_array_equal(streams[name], anchor_arrays[name])
                for arm in c['arms']:
                    other = result/f'{level}_{arm}_s{block["seed"]}'
                    paired = read(other/'graph.json')
                    for name in ['original_weight_sha256', 'input_root_ids', 'observation_root_ids',
                                 'input_mapping_sha256', 'decoder_parameters_per_lag']:
                        assert paired[name] == info[name], (level, arm, name)
                    assert paired['decoder_parameters_per_lag'] == 490
                    if not c['arms'][arm]['dag']:
                        assert paired['weight_sha256'] == info['weight_sha256']
                    with np.load(other/'case.npz', allow_pickle=False) as partner:
                        for name in keys:
                            np.testing.assert_array_equal(saved[name], partner[name])
                    checked += 1
    return dict(all_checks_pass=True, cases_checked=checked,
                source_inputs_observations_streams_labels_budget_paired=True,
                original_weights_paired_masked_weights_independently_reconstructed=True,
                cross_level_source_ids_and_stream_pairing=True)


def adjusted_fraction(part):
    mean = sum((Fraction(int(row.correct), int(row.samples)) for row in part.itertuples()),
               Fraction(0))/len(part)
    return (mean-Fraction(1, 10))/Fraction(9, 10)


def verify_summaries(frame, result, c, smoke):
    compare_csv(result/'raw-lags.csv', frame, ['level', 'arm', 'seed', 'lag'])
    expected_cases = 6 if smoke else 27
    assert len(frame) == expected_cases*len(c['lags'])
    assert len(frame[['level', 'arm', 'seed']].drop_duplicates()) == expected_cases
    long_scores, scores = {}, []
    for (level, arm, seed), part in frame.groupby(['level', 'arm', 'seed']):
        assert part.lag.tolist() == c['lags']
        value = adjusted_fraction(part[part.lag.between(1, 20)])
        long_scores[(level, arm, int(seed))] = value
        scores.append(dict(level=level, arm=arm, seed=int(seed), score=float(value),
                           numerator=value.numerator, denominator=value.denominator))
    compare_csv(result/'seed-scores.csv', pd.DataFrame(scores), ['level', 'arm', 'seed'])
    summary = read(result/'summary.json')
    assert summary['smoke'] is smoke and summary['criterion_unchanged'] is True
    assert summary['m1_outcome_unchanged'] == 'assay-invalid'
    assert summary['m2_outcome_unchanged'] == 'PASS'
    assert summary['cycle_effect_isolation_claim'] is False
    assert set(summary['levels']) == set(c['levels'])
    endpoints, primary_rows, differences = {}, [], []
    for level in c['levels']:
        seeds = c['blocks_by_level'][level]
        reported = summary['levels'][level]
        current = frame[(frame.level == level) & (frame.arm == 'instantaneous') & (frame.lag == 0)]
        assert set(current.seed) == set(seeds)
        valid = bool(all(Fraction(int(row.correct), int(row.samples)) >= Fraction(99, 100)
                         for row in current.itertuples()))
        assert reported['reference_valid'] is valid
        excesses = []
        for seed in seeds:
            selected = frame[(frame.level == level) & (frame.arm == c['primary_arm']) &
                             (frame.seed == seed) & (frame.lag == c['primary_lag'])]
            assert len(selected) == 1
            row = selected.iloc[0]
            baseline = max(Fraction(1, 10), Fraction(int(row.frequency_correct), int(row.samples)),
                           Fraction(int(row.current_correct), int(row.samples)))
            excess = Fraction(int(row.correct), int(row.samples))-baseline
            excesses.append(excess)
            primary_rows.append(dict(level=level, seed=seed, lag=c['primary_lag'],
                                     correct=int(row.correct), samples=int(row.samples),
                                     baseline_numerator=baseline.numerator,
                                     baseline_denominator=baseline.denominator,
                                     excess=float(excess), excess_numerator=excess.numerator,
                                     excess_denominator=excess.denominator,
                                     meets_margin=bool(excess >= Fraction(1, 10))))
        all_pass = bool(all(value >= Fraction(1, 10) for value in excesses))
        gate = None if smoke or not valid else all_pass
        primary = reported['primary']
        assert primary['lag'] == 2 and primary['arm'] == 'dag_synaptic'
        compare_estimates(primary['excess'], descriptive(excesses, c))
        assert abs(primary['minimum_excess']-float(min(excesses))) <= 1e-12
        assert primary['minimum_excess_exact'] == str(min(excesses))
        if 'mean_exact' in primary:
            assert primary['mean_exact'] == str(sum(excesses, Fraction(0))/len(excesses))
        assert primary['all_blocks_meet_margin'] is all_pass
        assert primary['registered_rule_pass'] is gate
        assert set(reported['arm_long_scores']) == set(c['arms'])
        for arm in c['arms']:
            compare_estimates(reported['arm_long_scores'][arm],
                              descriptive([long_scores[(level, arm, seed)] for seed in seeds], c))
        endpoints[level] = dict(reference_valid=valid, registered_rule_pass=gate,
                                minimum_excess_exact=str(min(excesses)),
                                all_blocks_meet_margin=all_pass)
        for lag in c['lags']:
            differences.append(dict(level=level, lag=lag, **descriptive([
                Fraction(int(frame[(frame.level == level) & (frame.arm == 'intact_synaptic') &
                                   (frame.seed == seed) & (frame.lag == lag)].iloc[0].correct),
                         c['test_samples'])-
                Fraction(int(frame[(frame.level == level) & (frame.arm == 'dag_synaptic') &
                                   (frame.seed == seed) & (frame.lag == lag)].iloc[0].correct),
                         c['test_samples']) for seed in seeds], c)))
    compare_csv(result/'primary-cells.csv', pd.DataFrame(primary_rows), ['level', 'seed', 'lag'])
    compare_csv(result/'descriptive-differences.csv', pd.DataFrame(differences), ['level', 'lag'])
    curves = []
    for (level, arm, lag), part in frame.groupby(['level', 'arm', 'lag']):
        part = part.sort_values('seed')
        row = dict(level=level, arm=arm, lag=int(lag))
        for metric in CURVE_METRICS:
            row.update({metric+'_'+name: value for name, value in descriptive(part[metric], c).items()})
        curves.append(row)
    compare_csv(result/'curve-summary.csv', pd.DataFrame(curves), ['level', 'arm', 'lag'])
    primary = endpoints[c['primary_level']]['registered_rule_pass']
    outcome = ('smoke' if smoke else 'assay-invalid'
               if not endpoints[c['primary_level']]['reference_valid'] else 'PASS' if primary else 'FAIL')
    assert summary['outcome'] == outcome and summary['primary_pass'] is primary
    assert summary['cases'] == expected_cases and summary['lag_heads'] == len(frame)
    assert summary['certificate_pairs'] == (2 if smoke else 9)
    return endpoints, outcome, primary


def verify(result, out):
    manifest = check_manifest(result)
    assert isinstance(manifest['smoke'], bool)
    c = read(result/'config.json')
    assert c == manifest['config'] == config(manifest['smoke'])
    assert subprocess.run(['git', 'diff', '--quiet', 'HEAD']).returncode == 0, 'Tracked edits before validation'
    recorded = read(result/'source.json')
    assert recorded['tracked_changes'] is False
    assert recorded['source_commit'] == manifest['source_commit']
    assert recorded['source_tree'] == git('rev-parse', recorded['source_commit']+'^{tree}')
    assert subprocess.run(['git', 'merge-base', '--is-ancestor', PROTOCOL_COMMIT,
                           recorded['source_commit']]).returncode == 0
    required = ['scripts/verify_temporal_cycles.py', 'scripts/cycle_support.py',
                'scripts/verify_temporal_mechanism.py', 'scripts/verify_temporal_memory_curve.py',
                'configs/temporal_cycles.json', c['protocol']]
    assert all(name in recorded['hashes'] for name in required)
    for name, digest in recorded['hashes'].items():
        assert sha(name) == digest, name
    for name in ['configs/temporal_cycles.json', c['protocol']]:
        frozen_bytes = subprocess.check_output(['git', 'show', PROTOCOL_COMMIT+':'+name])
        assert hashlib.sha256(frozen_bytes).hexdigest() == recorded['hashes'][name], name
    preservation = history_preserved(c)
    with attempt(out, c, 'cycles-independent-validation') as resources:
        write(out/'environment.json', environment())
        write(out/'source-graph-audit.json', validate_whole_source(c, resources))
        write(out/'registered-structure-checks.json', verify_registered_structure(c, resources, out))
        jobs = [dict(config=c, level=level, arm=arm, block=block, result=str(result),
                     out=str(out/f'{level}_{arm}_s{block["seed"]}'))
                for level in c['levels'] for block in c['blocks']
                if block['seed'] in c['blocks_by_level'][level] for arm in c['arms']]
        results, usage = pooled(case_job, jobs, c['neural_workers'], c,
                                'Cycles independent verification')
        frame = pd.DataFrame([row for replay in results for row in replay['rows']]).sort_values(
            ['level', 'arm', 'seed', 'lag']).reset_index(drop=True)
        endpoints, outcome, primary = verify_summaries(frame, result, c, manifest['smoke'])
        pairing = verify_pairing(result, c)
        frame.to_csv(out/'recomputed-lags.csv', index=False)
        write(out/'pairing-checks.json', pairing)
        write(out/'process-tree-resources.json', usage)
        write(out/'job-resources.json', [dict(identity=replay['identity'], resources=replay['resources'])
                                           for replay in results])
        assert history_preserved(c) == preservation
        for name, digest in recorded['hashes'].items():
            assert sha(name) == digest, name
        certificates = [replay['certificate'] for replay in results if replay['certificate'] is not None]
        assert len(certificates) == (2 if manifest['smoke'] else 9)
        write(out/'checks.json', dict(all_checks_pass=True, cases=len(results),
                                     replayed_streams=2*len(results), independently_refit_lag_heads=len(frame),
                                     exact_full_state_trace_hashes=sum(sum(replay['exact'].values()) for replay in results),
                                     instantaneous_certificates=sum('/instantaneous/' in replay['identity'] for replay in results),
                                     zero_state_input_access_certificates=len(results),
                                     finite_window_certificate_pairs=len(certificates),
                                     exact_certificate_trace_hashes=2*len(certificates),
                                     outcome=outcome, primary_pass=primary, endpoints=endpoints,
                                     result_manifest_sha256=sha(result/'manifest.json'),
                                     source_commit=recorded['source_commit'],
                                     protocol_commit=PROTOCOL_COMMIT,
                                     verifier_commit=git('rev-parse', 'HEAD'), verifier_sha256=sha(__file__),
                                     historical_preservation=preservation))
    seal(out, dict(complete=True, result_manifest_sha256=sha(result/'manifest.json')))
    print(dict(all_checks_pass=True, outcome=outcome, primary_pass=primary, cases=len(results)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('result', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(1):
        verify(args.result, args.out)
