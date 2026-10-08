"""Independent M6 source, discrete intervention and frozen-readout replay.

No main trajectory, reservoir, scorer, label builder or endpoint is imported.
The only scientific reuse is the previously independent raw-source graph
reconstruction. Old M5 ridge fitting here authenticates archived parameters;
all official new predictions use those exact archived bytes.
"""
import argparse
import hashlib
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from verify_temporal_memory_curve import independent_graph, validate_whole_source, partial_source
from counterfactual_support import (config, read, write, sha, array_sha, attempt,
    seal, check_manifest, history_preserved, source_record, assert_source_unchanged)
from tdc_support import environment, git
from tdc_pool import pooled


PROTOCOL_COMMIT = 'c2263270503c1d106dce7069a496c30c924e04f6'
WORLDS = ('original', 'counterfactual', 'instantaneous_original',
          'instantaneous_counterfactual')


def manual_step(weights, kc, mb, previous, stimulus, b=.6, history=True):
    """Explicit zero-carry scheduled update, including signed-zero semantics."""
    old = np.asarray(previous).copy()
    recurrent = weights.dot(old) if history else np.zeros_like(old)
    following = np.zeros_like(old) + b*np.tanh(recurrent+stimulus)
    mixed = old.copy() if history else np.zeros_like(old)
    mixed[kc] = following[kc]
    following[mb] = np.zeros(len(mb)) + b*np.tanh(weights[mb].dot(mixed)+stimulus[mb])
    return following


def frozen_scores(features, mean, scale, coefficients, intercept):
    assert np.asarray(features).shape[-1] == len(mean) == len(scale)
    assert np.isfinite(scale).all() and np.all(np.asarray(scale) > 0)
    return ((np.asarray(features)-mean)/scale) @ coefficients + intercept


def joint_endpoint(original_predictions, replacement_predictions, original, replacement):
    arrays = [np.asarray(x) for x in
              (original_predictions, replacement_predictions, original, replacement)]
    assert arrays[0].ndim == 1 and all(x.shape == arrays[0].shape for x in arrays)
    assert len(arrays[0]) and np.all(arrays[2] != arrays[3])
    original_ok = arrays[0] == arrays[2]
    replacement_ok = arrays[1] == arrays[3]
    count = int(np.count_nonzero(original_ok & replacement_ok))
    return dict(samples=len(arrays[0]), original_correct=int(original_ok.sum()),
                replacement_correct=int(replacement_ok.sum()), joint_correct=count,
                scientific_rule_pass=bool(10*count >= 9*len(arrays[0])))


def aggregate_endpoint(cells, smoke=False):
    assert cells and all(row['samples'] == 64 and row['technical_checks_pass'] for row in cells)
    passed = None if smoke else all(10*row['joint_correct'] >= 9*64 for row in cells)
    return dict(outcome='smoke' if smoke else 'PASS' if passed else 'FAIL',
                registered_rule_pass=passed)


def window_symbols(stream, anchor, lag=2, k=10):
    assert lag == 2 and 0 <= anchor < len(stream)-lag
    original = np.asarray(stream[anchor:anchor+lag+1], dtype=np.int64).copy()
    changed = original.copy()
    changed[0] = (int(original[0])+1) % k
    assert np.count_nonzero(original != changed) == 1
    assert np.array_equal(original[1:], changed[1:])
    return original, changed


def trajectory_window(weights, patterns, kc, mb, previous, symbols, observed,
                      b=.6, history=True):
    state = np.asarray(previous).copy()
    digest = hashlib.sha256()
    rows = []
    for symbol in symbols:
        state = manual_step(weights, kc, mb, state, patterns[int(symbol)], b, history)
        digest.update(state.tobytes())
        rows.append(state[observed].copy())
    return np.asarray(rows), state, digest.hexdigest()


def state_sha(state):
    return hashlib.sha256(np.asarray(state).tobytes()).hexdigest()


def matrix_sha(matrix):
    digest = hashlib.sha256()
    for array in (matrix.data, matrix.indices, matrix.indptr):
        digest.update(array.tobytes())
    return digest.hexdigest()


def close(actual, expected, c):
    np.testing.assert_allclose(actual, expected, atol=c['score_atol'], rtol=c['score_rtol'])


def compare_tree(actual, expected, c):
    if isinstance(expected, dict):
        assert set(actual) == set(expected), (set(actual), set(expected))
        for key in expected:
            compare_tree(actual[key], expected[key], c)
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            compare_tree(a, e, c)
    elif isinstance(expected, float):
        close(actual, expected, c)
    else:
        assert actual == expected, (actual, expected)


def estimates(values, c):
    values = np.asarray(values, dtype=float)
    assert len(values) and np.isfinite(values).all()
    draws = np.random.default_rng(c['bootstrap_seed']).integers(
        0, len(values), (c['bootstrap_draws'], len(values)))
    bounds = np.quantile(values[draws].mean(axis=1), [.025, .975])
    return dict(mean=float(values.mean()), median=float(np.median(values)),
        sd=float(values.std(ddof=1)) if len(values)>1 else 0.,
        bootstrap_lo=float(bounds[0]), bootstrap_hi=float(bounds[1]))


def compare_frame(actual, expected, keys, c):
    assert set(actual.columns) == set(expected.columns)
    assert not actual.duplicated(keys).any() and not expected.duplicated(keys).any()
    actual = actual.sort_values(keys).reset_index(drop=True)
    expected = expected[actual.columns].sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected, check_exact=False,
        atol=c['score_atol'], rtol=c['score_rtol'], check_dtype=False)


def authenticate_parent(c, level, block, patterns, roles, observed, ids):
    parent_root = Path(c['parent_namespace'])/f'{level}_s{block["parent_seed"]}'
    assert sha(Path(c['parent_namespace'])/'manifest.json') == c['parent_manifest_sha256']
    main_manifest = check_manifest(Path(c['parent_namespace']))
    assert main_manifest['source_commit'] == c['parent_source_commit']
    parent_manifest = check_manifest(parent_root)
    assert parent_manifest['level'] == level and parent_manifest['block']['seed'] == block['parent_seed']
    assert parent_manifest['block']['input_seed'] == block['input_seed']
    parent_config = read(Path(c['parent_namespace'])/'config.json')
    assert parent_config['warmup'] == 200 and parent_config['train_samples'] == 4000
    assert parent_config['alpha'] == 1 and parent_config['alphabet_size'] == 10
    assert parent_config['lags'] == list(range(21))
    graph = read(parent_root/'graph.json')
    names = ('input_patterns', 'roles', 'root_ids', 'observed_indices', 'mean', 'scale',
             'coefficients', 'intercept', 'current_tables', 'majority',
             'train_symbols', 'train_features', 'train_times', 'ytrain')
    with np.load(parent_root/'case.npz', allow_pickle=False) as archived:
        old = {name: archived[name].copy() for name in names}
    for name, value in old.items():
        assert array_sha(value) == graph['array_hashes'][name], name
        if value.dtype.kind in 'fc':
            assert np.isfinite(value).all()
    for name, actual in [('input_patterns', patterns), ('roles', roles),
                         ('root_ids', ids), ('observed_indices', observed)]:
        np.testing.assert_array_equal(old[name], actual)
    assert old['coefficients'].shape == (48, 210) and old['intercept'].shape == (210,)
    assert old['mean'].shape == old['scale'].shape == (48,)
    assert old['current_tables'].shape == (21, 10, 10) and old['majority'].shape == (21,)
    assert graph['direct_carry_multiplier'] == 0 and graph['drive_multiplier'] == .6
    assert graph['synaptic_history'] is True and graph['decoder_parameters_per_lag'] == 490
    # The old train symbols/labels are independently rebuilt from their old seed.
    train = np.random.default_rng(parent_manifest['block']['train_seed']).integers(
        0, 10, 4200, dtype=np.int64)
    np.testing.assert_array_equal(old['train_symbols'], train)
    np.testing.assert_array_equal(old['train_times'], np.arange(200, 4200))
    labels = train[198:4198]
    np.testing.assert_array_equal(old['ytrain'][:, 2], labels)
    features = old['train_features'][200:]
    mean = features.mean(axis=0)
    scale = np.maximum(features.std(axis=0), 1e-5)
    design = (features-mean)/scale
    targets = np.eye(10)[labels]
    intercept = targets.mean(axis=0)
    augmented = np.vstack([design, np.eye(48)])
    rhs = np.vstack([targets-intercept, np.zeros((48, 10))])
    coefficients = np.linalg.lstsq(augmented, rhs, rcond=None)[0]
    for a, e in [(old['mean'], mean), (old['scale'], scale),
                 (old['coefficients'][:,20:30], coefficients),
                 (old['intercept'][20:30], intercept)]:
        np.testing.assert_allclose(a, e, atol=1e-9, rtol=1e-9)
    majority = int(np.bincount(labels, minlength=10).argmax())
    table = np.zeros((10, 10), dtype=np.int64)
    for present, past in zip(train[200:], labels):
        table[present, past] += 1
    np.testing.assert_array_equal(old['current_tables'][2], table)
    assert int(old['majority'][2]) == majority
    return old, graph, dict(parent_case_npz_sha256=sha(parent_root/'case.npz'),
        parent_case_manifest_sha256=sha(parent_root/'manifest.json'),
        parent_graph_sha256=sha(parent_root/'graph.json'),
        old_head_independent_svd_authentications=1, old_training_samples=4000,
        new_trained_parameters=0, official_predictions_use_exact_archived_parameters=True)


def authenticate_source(source):
    assert source['protocol_commit'] == PROTOCOL_COMMIT
    assert source['source_tree'] == git('rev-parse', source['source_commit']+'^{tree}')
    assert source['tracked_changes'] is False
    assert source['every_tracked_hash_authenticated_at_capture'] is True
    subprocess.run(['git', 'merge-base', '--is-ancestor', PROTOCOL_COMMIT,
                    source['source_commit']], check=True)
    for name, digest in source['hashes'].items():
        assert sha(name) == digest, name
        if not name.startswith('outputs/'):
            blob = subprocess.check_output(['git', 'show', source['source_commit']+':'+name])
            assert hashlib.sha256(blob).hexdigest() == digest, name
    for name in ['configs/counterfactual_tracking.json', 'docs/counterfactual-tracking-protocol.md']:
        blob = subprocess.check_output(['git', 'show', PROTOCOL_COMMIT+':'+name])
        assert hashlib.sha256(blob).hexdigest() == source['hashes'][name]


def replay_case(saved, weights, patterns, roles, obs, old, c, block, graph,
                info, resources):
    """Stream replay keeps only one current full state, not64 full snapshots."""
    kc, mb = np.flatnonzero(roles == 'KC'), np.flatnonzero(roles == 'MBON')
    symbols = np.random.default_rng(block['stream_seed']).integers(
        0, 10, c['warmup']+c['test_samples'], dtype=np.int64)
    anchors = c['warmup'] + np.floor(np.linspace(
        0, c['test_samples']-1-2, c['probe_count'])).astype(np.int64)
    np.testing.assert_array_equal(saved['stream_symbols'], symbols)
    np.testing.assert_array_equal(saved['anchors'], anchors)
    assert len(np.unique(anchors)) == 64
    np.testing.assert_array_equal(saved['evaluation_times'], anchors+2)
    np.testing.assert_array_equal(saved['original_symbols'], symbols[anchors])
    np.testing.assert_array_equal(saved['alternative_symbols'], (symbols[anchors]+1)%10)
    np.testing.assert_array_equal(saved['current_symbols'], symbols[anchors+2])
    assert info['technical_checks_pass'] is True and len(info['trials']) == 64
    for key in ('every_fixed_anchor_counted','original_replay_equals_baseline',
                'instantaneous_full_endpoint_pair_equal','fixed_head_margin_projection_verified'):
        assert info[key] is True
    assert info['new_trained_parameters'] == 0
    state = np.zeros(weights.shape[0])
    features = np.empty((len(symbols), 48))
    norms = np.empty(len(symbols))
    stream_digest = hashlib.sha256()
    trial_index = {int(a): index for index, a in enumerate(anchors)}
    endpoint_checks = {}
    pre_hashes, window_hashes, final_hashes, independent_hashes = [], [], [], []
    for time, symbol in enumerate(symbols):
        if time in trial_index:
            index = trial_index[time]
            entry = info['trials'][index]
            assert entry['probe_index'] == index and entry['anchor'] == time
            assert entry['evaluation_time'] == time+2
            assert entry['pre_state_sha256'] == state_sha(state)
            pre_hashes.append(entry['pre_state_sha256'])
            hash_record = dict(probe_index=index,anchor=time,evaluation_time=time+2,
                               pre_state_sha256=state_sha(state))
            original, changed = window_symbols(symbols, time)
            np.testing.assert_array_equal(saved['trial_original_symbols'][index], original)
            np.testing.assert_array_equal(saved['trial_counterfactual_symbols'][index], changed)
            assert not np.array_equal(patterns[original[0]], patterns[changed[0]])
            finals = {}
            for world in WORLDS:
                rows, final, digest = trajectory_window(weights, patterns, kc, mb,
                    state, changed if world.endswith('counterfactual') else original,
                    obs, c['leak'], not world.startswith('instantaneous'))
                np.testing.assert_array_equal(saved['trial_'+world+'_features'][index], rows)
                assert entry[world+'_full_state_trajectory_sha256'] == digest
                assert entry[world+'_final_state_sha256'] == state_sha(final)
                window_hashes.append(digest)
                final_hashes.append(state_sha(final))
                hash_record[world+'_full_state_trajectory_sha256'] = digest
                hash_record[world+'_final_state_sha256'] = state_sha(final)
                finals[world] = final
                if world == 'original':
                    assert entry['baseline_window_sha256'] == digest
                    endpoint_checks[time+2] = (index, digest, state_sha(final))
                    hash_record['baseline_window_sha256'] = digest
            np.testing.assert_array_equal(finals['instantaneous_original'],
                                          finals['instantaneous_counterfactual'])
            assert state_sha(finals['instantaneous_original']) == state_sha(finals['instantaneous_counterfactual'])
            independent_hashes.append(hash_record)
        state = manual_step(weights, kc, mb, state, patterns[symbol], c['leak'])
        features[time], norms[time] = state[obs], np.linalg.norm(state)
        stream_digest.update(state.tobytes())
        if time in endpoint_checks:
            index, _, digest = endpoint_checks[time]
            assert state_sha(state) == digest == info['trials'][index]['baseline_endpoint_full_state_sha256']
            independent_hashes[index]['baseline_endpoint_full_state_sha256'] = state_sha(state)
            np.testing.assert_array_equal(features[time], saved['trial_original_features'][index,-1])
        if time % 100 == 0:
            resources.check()
    np.testing.assert_array_equal(saved['features'], features)
    compare_tree(info['trials'],independent_hashes,c)
    np.testing.assert_array_equal(saved['norms'], norms)
    np.testing.assert_array_equal(saved['final_state'], state)
    assert graph['baseline_full_state_trajectory_sha256'] == stream_digest.hexdigest()
    assert graph['final_state_sha256'] == state_sha(state)
    for index, anchor in enumerate(anchors):
        np.testing.assert_array_equal(saved['trial_original_features'][index], features[anchor:anchor+3])
    warmup = c['warmup']
    np.testing.assert_array_equal(saved['times'], np.arange(warmup, len(symbols)))
    labels = {0:symbols[warmup:], 2:symbols[warmup-2:len(symbols)-2]}
    for lag in (0,2):
        np.testing.assert_array_equal(saved['y'+str(lag)], labels[lag])
        scores = frozen_scores(features[warmup:], old['mean'], old['scale'],
            old['coefficients'][:,10*lag:10*(lag+1)], old['intercept'][10*lag:10*(lag+1)])
        close(saved['scores_lag'+str(lag)], scores, c)
        np.testing.assert_array_equal(saved['predictions_lag'+str(lag)], scores.argmax(axis=1))
    for world in WORLDS:
        scores = frozen_scores(saved['trial_'+world+'_features'][:,-1], old['mean'], old['scale'],
                               old['coefficients'][:,20:30], old['intercept'][20:30])
        close(saved['scores_'+world], scores, c)
        np.testing.assert_array_equal(saved['predictions_'+world], scores.argmax(axis=1))
    np.testing.assert_array_equal(saved['scores_instantaneous_original'], saved['scores_instantaneous_counterfactual'])
    np.testing.assert_array_equal(saved['predictions_instantaneous_original'], saved['predictions_instantaneous_counterfactual'])
    table, majority = old['current_tables'][2], int(old['majority'][2])
    lookup = np.asarray([row.argmax() if row.sum() else majority for row in table], dtype=np.int64)
    np.testing.assert_array_equal(saved['frequency_predictions'], np.full(c['test_samples'], majority))
    np.testing.assert_array_equal(saved['current_predictions'], lookup[symbols[warmup:]])
    np.testing.assert_array_equal(saved['frequency_trial_predictions'], np.full(64, majority))
    np.testing.assert_array_equal(saved['current_trial_predictions'], lookup[symbols[anchors+2]])
    indices = np.arange(64)
    original, replacement = symbols[anchors], (symbols[anchors]+1)%10
    score0, score1 = saved['scores_original'], saved['scores_counterfactual']
    masks0, masks1 = np.eye(10, dtype=bool)[original], np.eye(10, dtype=bool)[replacement]
    original_margin = score0[indices,original] - np.max(np.where(masks0,-np.inf,score0),axis=1)
    replacement_margin = score1[indices,replacement] - np.max(np.where(masks1,-np.inf,score1),axis=1)
    contrast0 = score0[indices,replacement]-score0[indices,original]
    contrast1 = score1[indices,replacement]-score1[indices,original]
    delta = saved['trial_counterfactual_features'][:,-1]-saved['trial_original_features'][:,-1]
    beta = old['coefficients'][:,20:30]
    projected = np.sum((delta/old['scale'])*(beta[:,replacement]-beta[:,original]).T,axis=1)
    for name, value in [('original_target_margin', original_margin),
            ('replacement_target_margin',replacement_margin),
            ('original_contrast_margin',contrast0),('replacement_contrast_margin',contrast1),
            ('contrast_margin_change',contrast1-contrast0),
            ('projected_contrast_margin_change',projected),('observed_delta_norm',np.linalg.norm(delta,axis=1))]:
        close(saved[name], value, c)
    close(contrast1-contrast0, projected, c)
    return dict(exact_baseline_full_state_trace_hash=stream_digest.hexdigest(),
        exact_full_state_prestate_hashes=len(pre_hashes),
        exact_four_world_window_trace_hashes=len(window_hashes),
        exact_four_world_final_state_hashes=len(final_hashes),
        instantaneous_final_state_pairs_exact=64, full_stream_replay_count=1,
        all_anchors_unconditionally_scored=True, one_actual_symbol_changed=True,
        prefix_and_future_identical=True, evaluation_lag=2,
        frozen_head_projection_of_margin_change_verified=True,
        independent_trial_hash_records=independent_hashes)


def recomputed_cell(saved, level, block, smoke=False):
    s, r = saved['original_symbols'], saved['alternative_symbols']
    counts = joint_endpoint(saved['predictions_original'], saved['predictions_counterfactual'], s, r)
    assert counts['samples'] == 64
    controls = {}
    for name, original_prediction, replacement_prediction in [
            ('frequency', saved['frequency_trial_predictions'], saved['frequency_trial_predictions']),
            ('current', saved['current_trial_predictions'], saved['current_trial_predictions']),
            ('instantaneous',saved['predictions_instantaneous_original'],saved['predictions_instantaneous_counterfactual'])]:
        count = joint_endpoint(original_prediction, replacement_prediction, s, r)['joint_correct']
        assert count == 0
        controls[name+'_joint_correct'] = count
    return dict(level=level, seed=block['seed'], parent_seed=block['parent_seed'],
        samples=64, original_correct=counts['original_correct'],
        replacement_correct=counts['replacement_correct'], joint_correct=counts['joint_correct'],
        original_accuracy=counts['original_correct']/64,
        replacement_accuracy=counts['replacement_correct']/64,
        joint_accuracy=counts['joint_correct']/64,
        prediction_changes=int(np.count_nonzero(saved['predictions_original'] != saved['predictions_counterfactual'])),
        scientific_rule_pass=None if smoke else counts['scientific_rule_pass'],
        technical_checks_pass=True, **controls)


def baseline_metric_rows(saved, level, block, c):
    rows = []
    for lag in (0, 2):
        truth = saved['y'+str(lag)]
        correct = int(np.count_nonzero(saved['predictions_lag'+str(lag)] == truth))
        n = len(truth)
        row = dict(level=level, seed=block['seed'], lag=lag, samples=n,
                   correct=correct, accuracy=correct/n, chance=.1)
        if lag == 2:
            frequency = int(np.count_nonzero(saved['frequency_predictions'] == truth))
            current = int(np.count_nonzero(saved['current_predictions'] == truth))
            row.update(frequency_correct=frequency,current_correct=current,
                       frequency_accuracy=frequency/n,current_accuracy=current/n)
        rows.append(row)
    return rows


def case_job(job):
    c, block, level = job['config'], job['block'], job['level']
    root = Path(job['result'])/f'{level}_s{block["seed"]}'
    out = Path(job['out'])
    case_manifest = check_manifest(root)
    assert case_manifest['block'] == block and case_manifest['level'] == level
    graph, info = read(root/'graph.json'), read(root/'trial-info.json')
    with threadpool_limits(1), attempt(out,c,'m6-independent-case') as resources:
        weights, patterns, roles, obs, ids = independent_graph(c,level,block['input_seed'])
        # The raw annotation mapper yields object strings for the partial graph;
        # frozen archives stipulate U5, so authenticate that documented encoding.
        roles = np.asarray(roles,dtype='U5')
        assert matrix_sha(weights) == graph['weight_sha256']
        assert graph['decoder_parameters_per_lag'] == 490 and graph['new_trained_parameters'] == 0
        assert ids[obs].astype(str).tolist() == graph['observation_root_ids']
        inputs = [set(ids[pattern != 0].astype(str)) for pattern in patterns]
        assert inputs == [set(row) for row in graph['input_root_ids']]
        _, small_ids, _ = partial_source()
        local = pd.Index(ids).get_indexer(small_ids)
        assert (local >= 0).all()
        assert hashlib.sha256(small_ids.tobytes()+patterns[:,local].tobytes()).hexdigest() == graph['input_mapping_sha256']
        for first in range(10):
            for second in range(first):
                assert not np.array_equal(patterns[first],patterns[second])
        old, parent_graph, old_checks = authenticate_parent(c,level,block,patterns,roles,obs,ids)
        assert parent_graph['weight_sha256'] == graph['weight_sha256']
        assert graph['parent_case_sha256'] == old_checks['parent_case_npz_sha256']
        assert graph['parent_case_manifest_sha256'] == old_checks['parent_case_manifest_sha256']
        assert graph['parent_graph_sha256'] == old_checks['parent_graph_sha256']
        assert graph['parent_main_manifest_sha256'] == c['parent_manifest_sha256']
        assert graph['parent_source_commit'] == c['parent_source_commit']
        assert graph['parent_protocol_commit'] == c['parent_protocol_commit']
        assert graph['parent_case_path'] == (Path(c['parent_namespace'])/f'{level}_s{block["parent_seed"]}').as_posix()
        for name,digest in graph['parent_array_hashes'].items():
            assert digest == parent_graph['array_hashes'][name]
        with np.load(root/'case.npz',allow_pickle=False) as saved:
            assert set(saved.files) == set(graph['array_hashes'])
            for name in saved.files:
                assert array_sha(saved[name]) == graph['array_hashes'][name]
                if saved[name].dtype.kind in 'fc':
                    assert np.isfinite(saved[name]).all(),name
            required = dict(input_patterns=patterns,roles=roles,root_ids=ids,observed_indices=obs,
                mean=old['mean'],scale=old['scale'],coefficients_lag2=old['coefficients'][:,20:30],
                intercept_lag2=old['intercept'][20:30],coefficients_lag0=old['coefficients'][:,:10],
                intercept_lag0=old['intercept'][:10],current_table_lag2=old['current_tables'][2],
                majority_lag2=np.asarray(old['majority'][2]))
            table, majority = old['current_tables'][2], int(old['majority'][2])
            required['current_lookup_lag2'] = np.asarray(
                [row.argmax() if row.sum() else majority for row in table],dtype=np.int64)
            for name,value in required.items():
                np.testing.assert_array_equal(saved[name],value)
            for name,digest in graph['frozen_array_hashes'].items():
                assert name in required and digest == array_sha(required[name])
            replay_checks = replay_case(saved,weights,patterns,roles,obs,old,c,block,graph,info,resources)
            hash_records = replay_checks.pop('independent_trial_hash_records')
            cell = recomputed_cell(saved,level,block,case_manifest['smoke'])
            rows = baseline_metric_rows(saved,level,block,c)
            compare_tree(read(root/'primary-cell.json'),cell,c)
            compare_tree(read(root/'metrics.json'),rows,c)
            trials = raw_trial_rows(saved,level,block)
            compare_tree(read(root/'trial-metrics.json'),dict(counts=cell,rows=trials),c)
        checks = dict(all_checks_pass=True,level=level,seed=block['seed'],
            independent_graph_mapping_and_normalization=True,
            exact_frozen_head_scaler_and_controls=True,
            old_parent_lineage_authenticated=True,
            frequency_current_instantaneous_joint_counts_exact_zero=True,
            exact_integer_primary_recalculated=True,
            scientific_threshold_tolerance_applied=False,
            official_cell=cell,**old_checks,**replay_checks)
        write(out/'checks.json',checks)
        write(out/'recomputed-trial-info.json',dict(technical_checks_pass=True,trials=hash_records,
            every_fixed_anchor_counted=True,new_trained_parameters=0,
            original_replay_equals_baseline=True,instantaneous_full_endpoint_pair_equal=True,
            fixed_head_margin_projection_verified=True))
        write(out/'recomputed-cell.json',cell)
        write(out/'recomputed-trials.json',trials)
    seal(out,dict(complete=True,result_case_manifest_sha256=sha(root/'manifest.json')))
    return dict(identity=f'{level}/s{block["seed"]}',cell=cell,rows=rows,trials=trials,
                checks=checks,resources=read(out/'resources.json'))


def raw_trial_rows(saved, level, block):
    rows = []
    for index,anchor in enumerate(saved['anchors']):
        source, alternate = int(saved['original_symbols'][index]),int(saved['alternative_symbols'][index])
        old_prediction = int(saved['predictions_original'][index])
        new_prediction = int(saved['predictions_counterfactual'][index])
        row = dict(level=level,seed=block['seed'],probe_index=index,anchor=int(anchor),
            evaluation_time=int(anchor)+2,original_symbol=source,replacement_symbol=alternate,
            current_symbol=int(saved['current_symbols'][index]),
            original_correct=old_prediction==source,replacement_correct=new_prediction==alternate,
            joint_correct=old_prediction==source and new_prediction==alternate,
            prediction_changed=old_prediction!=new_prediction)
        row.update({'prediction_'+world:int(saved['predictions_'+world][index]) for world in WORLDS})
        row.update({'prediction_'+name:int(saved[name+'_trial_predictions'][index]) for name in ('frequency','current')})
        for name in ('original_target_margin','replacement_target_margin','original_contrast_margin',
                     'replacement_contrast_margin','contrast_margin_change',
                     'projected_contrast_margin_change','observed_delta_norm'):
            row[name] = float(saved[name][index])
        rows.append(row)
    return rows


def summary(cells,c,smoke):
    levels = {}
    for level in c['levels']:
        selected = sorted((row for row in cells if row['level']==level),key=lambda row:row['seed'])
        assert [row['seed'] for row in selected] == c['blocks_by_level'][level]
        levels[level] = dict(**aggregate_endpoint(selected,smoke),cells=selected,
            **{name:estimates([row[name] for row in selected],c) for name in
               ('joint_accuracy','original_accuracy','replacement_accuracy')})
    primary = levels[c['primary_level']]
    return dict(smoke=bool(smoke),outcome=primary['outcome'],primary_pass=primary['registered_rule_pass'],
        levels=levels,cases=len(cells),probe_pairs=64*len(cells),frozen_heads_used=len(cells),
        new_trained_parameters=0,decoder_parameters_per_lag=490,criterion_unchanged=True,
        every_fixed_anchor_counted=True,performance_eligibility_gate=False,
        biological_plasticity_performed=False,unique_cycle_effect_claim=False,
        m1_outcome_unchanged='assay-invalid',m4_outcome_unchanged='INFEASIBLE')


def independent_pairing(result,c):
    paired = []
    for block in c['blocks']:
        cases = [result/f'{level}_s{block["seed"]}' for level in c['levels']
                 if block['seed'] in c['blocks_by_level'][level]]
        if len(cases)!=2:
            continue
        with np.load(cases[0]/'case.npz',allow_pickle=False) as first,np.load(cases[1]/'case.npz',allow_pickle=False) as second:
            for name in ('stream_symbols','times','y0','y2','anchors','evaluation_times',
                         'original_symbols','alternative_symbols','current_symbols',
                         'trial_original_symbols','trial_counterfactual_symbols'):
                np.testing.assert_array_equal(first[name],second[name])
        graphs = [read(root/'graph.json') for root in cases]
        for name in ('input_root_ids','observation_root_ids','input_mapping_sha256'):
            assert graphs[0][name]==graphs[1][name]
        paired.append(block['seed'])
    return dict(paired_block_seeds=paired,inputs_targets_anchors_identical=True,
                computational_blocks_are_not_biological_replicates=True)


def verify(result,out):
    manifest = check_manifest(result)
    smoke = manifest['smoke']
    assert type(smoke) is bool
    paths = {p.relative_to(result).as_posix() for p in result.rglob('*') if p.is_file()}
    assert paths == set(manifest['artifacts'])|{'manifest.json'}
    c = read(result/'config.json')
    assert c == config(smoke) == manifest['config']
    recorded_source = read(result/'source.json')
    assert recorded_source['source_commit'] == manifest['source_commit']
    authenticate_source(recorded_source)
    verifier_source = source_record(c)
    with attempt(out,c,'m6-independent-validation') as resources:
        write(out/'config.json',c)
        write(out/'source.json',verifier_source)
        write(out/'environment.json',environment())
        preservation = history_preserved(c)
        compare_tree(read(result/'historical-preservation.json'),preservation,c)
        write(out/'source-validation.json',dict(result_source_commit=recorded_source['source_commit'],
            verifier_source_commit=verifier_source['source_commit'],protocol_commit=PROTOCOL_COMMIT,
            recorded_source_hashes_authenticated=len(recorded_source['hashes']),
            independent_manual_dynamics_no_main_scorer_import=True))
        source_graph = validate_whole_source(c,resources)
        write(out/'source-graph-audit.json',source_graph)
        jobs = [dict(config=c,block=block,level=level,result=str(result),
                     out=str(out/f'{level}_s{block["seed"]}'))
                for level in c['levels'] for block in c['blocks']
                if block['seed'] in c['blocks_by_level'][level]]
        results,usage = pooled(case_job,jobs,c['neural_workers'],c,'M6 independent replay')
        cells = sorted((value['cell'] for value in results),key=lambda row:(row['level'],row['seed']))
        trials = pd.DataFrame([row for value in results for row in value['trials']])
        metrics = pd.DataFrame([row for value in results for row in value['rows']])
        for filename,frame,keys in [('raw-trials.csv',trials,['level','seed','probe_index']),
                                  ('stream-metrics.csv',metrics,['level','seed','lag']),
                                  ('block-scores.csv',pd.DataFrame(cells),['level','seed'])]:
            compare_frame(pd.read_csv(result/filename),frame,keys,c)
            frame.sort_values(keys).to_csv(out/('recomputed-'+filename),index=False)
        compare_tree(read(result/'primary-cells.json'),cells,c)
        recomputed_summary = summary(cells,c,smoke)
        compare_tree(read(result/'summary.json'),recomputed_summary,c)
        pairing = independent_pairing(result,c)
        compare_tree(read(result/'pairing.json'),pairing,c)
        write(out/'recomputed-summary.json',recomputed_summary)
        case_checks = [value['checks'] for value in results]
        write(out/'case-checks.json',case_checks)
        write(out/'process-tree-resources.json',usage)
        write(out/'job-resources.json',[dict(identity=value['identity'],resources=value['resources']) for value in results])
        write(out/'checks.json',dict(all_checks_pass=True,outcome=recomputed_summary['outcome'],
            primary_pass=recomputed_summary['primary_pass'],cases=len(cells),replayed_streams=len(cells),
            exact_full_state_hashes=len(cells),old_head_independent_svd_authentications=len(cells),
            frozen_heads_used=len(cells),new_trained_parameters=0,probe_pairs=64*len(cells),
            exact_full_state_prestate_hashes=64*len(cells),
            exact_four_world_window_trace_hashes=256*len(cells),
            exact_four_world_final_state_hashes=256*len(cells),
            instantaneous_final_state_pairs_exact=64*len(cells),
            all_anchors_unconditionally_scored=True,one_actual_symbol_changed=True,
            prefix_and_future_identical=True,evaluation_lag=2,
            frequency_current_instantaneous_joint_counts_exact_zero=True,
            frozen_head_projection_of_margin_change_verified=True,
            exact_frozen_head_scaler_and_controls=True,exact_integer_primary_recalculated=True,
            independent_manual_dynamics_no_main_scorer_import=True,
            scientific_threshold_tolerance_applied=False,performance_eligibility_gate=False,
            biological_plasticity_performed=False,unique_cycle_effect_claim=False,
            source_graph_reconstruction=source_graph,pairing=pairing,
            result_manifest_sha256=sha(result/'manifest.json'),
            source_commit=verifier_source['source_commit'],protocol_commit=PROTOCOL_COMMIT,
            historical_preservation=preservation))
        assert history_preserved(c) == preservation
        assert_source_unchanged(verifier_source)
        resources.check()
    seal(out,dict(complete=True,result_manifest_sha256=sha(result/'manifest.json'),
                  source_commit=verifier_source['source_commit']))
    print(dict(all_checks_pass=True,outcome=recomputed_summary['outcome'],cases=len(cells),
               exact_full_state_hashes=len(cells),old_head_independent_svd_authentications=len(cells)),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    with threadpool_limits(1):
        verify(args.result,args.out)


if __name__ == '__main__':
    main()
