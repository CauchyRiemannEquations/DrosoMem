"""M6 final artifact audit and unconditioned paired tracking decision replay.

No new head is fitted and no experiment reservoir/scoring/endpoint is imported.
Old M5 source/head provenance is authenticated; separately produced independent
replay evidence certifies complete states and old-training SVD authentication.
"""
import argparse
from fractions import Fraction
import hashlib
from pathlib import Path
import posixpath
import re
import subprocess

import numpy as np
import pandas as pd

from counterfactual_support import read, write, sha, array_sha, attempt, seal, git, history_preserved
from verify_temporal_memory_curve import independent_graph, partial_source

BASELINE = '43d6b068af853dd317241fc611f0df78b7110fca'
PREREGISTRATION = 'c2263270503c1d106dce7069a496c30c924e04f6'
NAMESPACE = Path('results/counterfactual_tracking_v1')
CONFIG = 'configs/counterfactual_tracking.json'
PROTOCOL = 'docs/counterfactual-tracking-protocol.md'
OVERVIEWS = {'README.md', 'docs/research-status.md', 'docs/research-roadmap.md', 'docs/next-work.md'}


def tree(revision):
    return {line.split('\t', 1)[1]: line.split('\t', 1)[0].split()[2]
            for line in git('ls-tree', '-r', revision).splitlines()}


def git_digest(blob):
    return hashlib.sha256(subprocess.check_output(['git', 'cat-file', 'blob', blob])).hexdigest()


def estimates(values, c):
    values = np.asarray(values, dtype=float)
    assert len(values) and np.isfinite(values).all()
    draws = np.random.default_rng(c['bootstrap_seed']).integers(0, len(values),
        (c['bootstrap_draws'], len(values)))
    lo, hi = np.quantile(values[draws].mean(axis=1), [.025, .975])
    return dict(mean=float(values.mean()), median=float(np.median(values)),
                sd=float(values.std(ddof=1)) if len(values)>1 else 0.,
                bootstrap_lo=float(lo), bootstrap_hi=float(hi))


def close(actual, expected, atol=1e-12, rtol=1e-12):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


def estimate_check(actual, expected):
    assert set(actual) == set(expected)
    for key, value in expected.items():
        close(actual[key], value)


def load_archive(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def hashed_arrays(record, arrays):
    assert record['array_hashes'] == {name: array_sha(value) for name, value in arrays.items()}


def manifest_and_failures(root, out, cached):
    manifests, incomplete, attempts = [], [], []
    for path in sorted(root.rglob('manifest.json')):
        if out.resolve() in path.resolve().parents:
            continue
        record = read(path)
        assert isinstance(record['artifacts'], dict)
        for name, digest in record['artifacts'].items():
            target = (path.parent/name).resolve()
            assert path.parent.resolve() in target.parents, 'Manifest path escape'
            assert cached(target) == digest, (path, name)
        actual = {value.relative_to(path.parent).as_posix() for value in path.parent.rglob('*') if value.is_file()}
        assert actual == set(record['artifacts']) | {'manifest.json'}, ('Unmanifested artifact', path)
        complete = record.get('complete', True)
        assert isinstance(complete, bool)
        manifests.append(dict(path=path.as_posix(), sha256=cached(path), artifacts=len(record['artifacts']), complete=complete))
        if not complete:
            incomplete.append(path.as_posix())
    registry_path = root/'known-incomplete-attempts.json'
    registry = read(registry_path) if registry_path.exists() else []
    assert isinstance(registry, list)
    assert sorted(row['manifest_path'] for row in registry) == sorted(incomplete)
    failures, failure_files = [], set()
    for row in registry:
        assert isinstance(row['reason'], str) and row['reason'].strip()
        path = Path(row['failure_path'])
        assert root.resolve() in path.resolve().parents
        assert path.parent/'manifest.json' == Path(row['manifest_path'])
        failure = read(path)
        assert failure['complete'] is False and failure['message']
        if 'parameters_changed' in row:
            assert row['parameters_changed'] is False
        failures.append(dict(row, failure_sha256=cached(path)))
        failure_files.add(path.as_posix())
    for path in sorted(root.rglob('attempt.json')):
        if out.resolve() in path.resolve().parents:
            continue
        record = read(path)
        assert isinstance(record['complete'], bool)
        if not record['complete']:
            assert (path.parent/'manifest.json').as_posix() in incomplete, 'Unsealed interruption'
        attempts.append(dict(path=path.as_posix(), complete=record['complete'], sha256=cached(path)))
    validation_path = root/'validation-failures.json'
    validation_registry = read(validation_path) if validation_path.exists() else []
    assert isinstance(validation_registry, list)
    validation_failures, invalid_sources = [], {}
    for row in validation_registry:
        # An exception is accepted only when explicitly registered with a
        # complete preserved snapshot; it never certifies that failed stage.
        assert isinstance(row['reason'], str) and row['reason'].strip()
        assert row['parameters_changed'] is False
        source_path = root/row['record']
        assert root.resolve() in source_path.resolve().parents
        assert cached(source_path.parent/'manifest.json') == row['manifest_sha256']
        source = read(source_path)
        assert row['source_validation_failed'] is True
        assert source['source_commit'] == row['recorded_commit']
        name = row['invalid_source_path']
        assert source['hashes'][name] == row['recorded_sha256']
        assert git_digest(tree(row['recorded_commit'])[name]) == row['git_sha256'] != row['recorded_sha256']
        snapshot = (root/row['snapshot_path']).resolve()
        assert root.resolve() in snapshot.parents and cached(snapshot) == row['recorded_sha256']
        manifest = read(snapshot.parent/'manifest.json')
        assert manifest['complete'] is True and manifest['scientifically_valid'] is False
        failure = snapshot.parent/'failure.json'
        assert read(failure) == row
        assert manifest['artifacts'][snapshot.name] == row['recorded_sha256']
        assert manifest['artifacts']['failure.json'] == cached(failure)
        failure_files.add(failure.relative_to(Path.cwd()).as_posix())
        key = (source_path.as_posix(), name)
        assert key not in invalid_sources
        invalid_sources[key] = row
        validation_failures.append(dict(row, authentication='snapshot_only_validation_failed'))
    actual_failures = {path.as_posix() for path in root.rglob('failure.json') if out.resolve() not in path.resolve().parents}
    assert actual_failures == failure_files, 'Unregistered failure record'
    return manifests, attempts, failures, validation_failures, invalid_sources


def authenticate_sources(root, out, cached, invalid_sources):
    pinned = {name: digest for name, digest in read('results/functional_sensitivity_v1/main/source.json')['hashes'].items()
              if name.startswith('outputs/tdc_v2/')}
    records, inventories, digests, consumed = [], {}, {}, set()
    registered = tree(PREREGISTRATION)
    for path in sorted(root.rglob('source.json')):
        if out.resolve() in path.resolve().parents:
            continue
        source = read(path)
        revision = source['source_commit']
        if revision not in inventories:
            inventories[revision] = tree(revision)
        inventory = inventories[revision]
        assert source['source_tree'] == git('rev-parse', revision+'^{tree}')
        assert source['tracked_changes'] is False
        assert source['protocol_commit'] == PREREGISTRATION
        assert source['every_tracked_hash_authenticated_at_capture'] is True
        assert source['parent_bank_frozen'] is True and source['new_trained_parameters'] == 0
        subprocess.run(['git', 'merge-base', '--is-ancestor', PREREGISTRATION, revision], check=True)
        for name in [CONFIG, PROTOCOL]:
            assert inventory[name] == registered[name]
        for name, expected in source['hashes'].items():
            key = (path.as_posix(), name)
            if name in inventory:
                blob = inventory[name]
                if blob not in digests:
                    digests[blob] = git_digest(blob)
                actual = digests[blob]
                if expected != actual:
                    assert key in invalid_sources, ('Unregistered captured source mismatch', path, name)
                    row = invalid_sources[key]
                    assert row['recorded_commit'] == revision and row['recorded_sha256'] == expected and row['git_sha256'] == actual
                    consumed.add(key)
                    authentication = 'snapshot_only_validation_failed'
                else:
                    assert key not in invalid_sources
                    authentication = 'recorded_git_blob'
            else:
                assert name in pinned and expected == pinned[name] == cached(name), (path, name)
                authentication = 'immutable_pinned_external_bytes'
            records.append(dict(record=path.as_posix(), path=name, sha256=expected,
                                source_commit=revision, authentication=authentication))
    assert records and consumed == set(invalid_sources)
    return records, pinned


def resource_check(folder, c):
    record = read(folder/'resources.json')
    assert record['seconds'] <= c['max_seconds']
    assert record['peak_sampled_rss_bytes'] <= c['max_rss_bytes']
    assert (record.get('os_process_peak_working_set_bytes') or 0) <= c['max_rss_bytes']
    for key in ['peak_aggregate_sampled_rss_bytes', 'peak_sampled_process_tree_rss_bytes']:
        if key in record:
            assert record[key] <= c['max_rss_bytes']
    return record


def document_check(out, current, cached):
    documents = [PROTOCOL, 'docs/counterfactual-tracking-results.md', *sorted(OVERVIEWS)]
    links = []
    for name in documents:
        assert Path(name).is_file()
        for target in re.findall(r'\]\(([^)]+)\)', Path(name).read_text(encoding='utf-8')):
            if '://' in target or target.startswith(('#', 'mailto:', 'app:')):
                continue
            target = target.strip('<>')
            destination = posixpath.normpath(posixpath.join(Path(name).parent.as_posix(), target.split('#')[0]))
            assert (destination in current or Path(destination).exists()
                    or Path(destination).resolve() == (out/'audit.json').resolve()), (name, target)
            links.append(dict(document=name, target=target, destination=destination))
    return links, {name: cached(name) for name in documents}


def scalar_check(actual, expected):
    if expected is None:
        assert actual is None or pd.isna(actual)
    elif isinstance(expected, (bool, str, int)):
        assert actual == expected
    else:
        close(actual, expected)


def fixed_config(c, smoke=False):
    registered = read(CONFIG)
    assert (registered['baseline_commit'], registered['namespace'], registered['protocol']) == (BASELINE, NAMESPACE.as_posix(), PROTOCOL)
    assert registered['levels'] == ['legacy5', 'brain5'] and registered['primary_level'] == 'legacy5'
    assert registered['circuit_seeds'] == [701]
    assert registered['blocks'] == [dict(seed=1600001+i, parent_seed=1500001+i,
        input_seed=1510001+i, stream_seed=1610001+i) for i in range(6)]
    assert registered['blocks_by_level'] == {'legacy5': list(range(1600001, 1600007)),
                                           'brain5': list(range(1600001, 1600004))}
    assert (registered['alphabet_size'], registered['input_fraction'], registered['input_amplitude'],
            registered['gain'], registered['leak'], registered['carry_multiplier'], registered['alpha']) == (10, .1, .5, .9, .6, 0., 1.)
    assert registered['normalization'] == 'incoming_l1' and registered['schedule'] == 'mbon_after_kc'
    assert registered['synaptic_history'] is True
    assert (registered['warmup'], registered['test_samples'], registered['probe_count'], registered['primary_lag']) == (200, 2000, 64, 2)
    assert registered['minimum_joint_accuracy'] == .9 and registered['performance_eligibility_gate'] is False
    assert registered['new_trained_parameters'] == 0 and registered['graph_searches'] == 0
    assert registered['biological_plasticity_performed'] is False
    assert (registered['bootstrap_seed'], registered['bootstrap_draws']) == (1620001, 10000)
    assert (registered['score_atol'], registered['score_rtol']) == (1e-10, 1e-10)
    assert (registered['neural_workers'], registered['max_seconds'], registered['max_rss_bytes']) == (4, 7200, 4294967296)
    assert registered['parent_namespace'] == 'results/functional_sensitivity_v1/main'
    assert registered['parent_source_commit'] == 'c541993484f6778a031d71ad4eb1798c748f6ce7'
    assert registered['parent_protocol_commit'] == '37212abeb6cfc1a4907e08348ff41030f3edfb8b'
    assert registered['parent_manifest_sha256'] == '98eb3b4c3c49934365caa30608fd50b677dd883b7e3ed722271db4aceeaf73af'
    expected = dict(registered)
    if smoke:
        expected['test_samples'] = registered['smoke']['test_samples']
        expected['blocks_by_level'] = {level: seeds[:1] for level, seeds in registered['blocks_by_level'].items()}
    assert c == expected, 'Saved settings differ from registration'
    inventory = tree(PREREGISTRATION)
    for name in [CONFIG, PROTOCOL]:
        assert sha(name) == git_digest(inventory[name]), 'Registration bytes changed'


def matrix_hash(weights):
    return hashlib.sha256(weights.data.tobytes()+weights.indices.tobytes()+weights.indptr.tobytes()).hexdigest()


def paired_counts(original, counterfactual, target, replacement):
    """Every registered anchor contributes once, including original errors."""
    assert original.shape == counterfactual.shape == target.shape == replacement.shape == (64,)
    assert all(value.dtype.kind in 'iu' for value in [original, counterfactual, target, replacement])
    assert all(np.all((value >= 0) & (value < 10)) for value in [original, counterfactual, target, replacement])
    assert np.all(target != replacement)
    first, second = original == target, counterfactual == replacement
    return dict(probes=64, original_correct=int(first.sum()), counterfactual_correct=int(second.sum()),
                joint_correct=int(np.count_nonzero(first & second)),
                predictions_changed=int(np.count_nonzero(original != counterfactual)),
                predictions_unchanged=int(np.count_nonzero(original == counterfactual)))


def exact_gate(joint, smoke=False):
    assert type(joint) is int and 0 <= joint <= 64
    return None if smoke else 10*joint >= 9*64


def score(states, arrays, lag=2):
    # Use only archived old M5 means, scales and selected affine coefficients.
    return ((states-arrays['mean'])/arrays['scale']) @ arrays[f'coefficients_lag{lag}']+arrays[f'intercept_lag{lag}']


def parent_inputs(c, level, block, cached):
    parent = Path(c['parent_namespace'])
    assert cached(parent/'manifest.json') == c['parent_manifest_sha256']
    source = read(parent/'source.json')
    assert source['source_commit'] == c['parent_source_commit'] and source['protocol_commit'] == c['parent_protocol_commit']
    folder = parent/f'{level}_s{block["parent_seed"]}'
    graph = read(folder/'graph.json')
    required = ['input_patterns', 'observed_indices', 'root_ids', 'roles', 'mean', 'scale',
                'coefficients', 'intercept', 'current_tables', 'majority']
    with np.load(folder/'case.npz', allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in required}
    for name, value in arrays.items():
        assert array_sha(value) == graph['array_hashes'][name], ('Frozen parent array changed', folder, name)
    manifest = read(parent/'manifest.json')
    case_manifest = read(folder/'manifest.json')
    assert manifest['complete'] is True and manifest['source_commit'] == c['parent_source_commit']
    assert case_manifest['complete'] is True and case_manifest['level'] == level
    assert case_manifest['block']['seed'] == block['parent_seed'] and case_manifest['block']['input_seed'] == block['input_seed']
    for name in ['graph.json', 'case.npz', 'manifest.json']:
        relative = folder.relative_to(parent).as_posix()+'/'+name
        assert manifest['artifacts'][relative] == cached(folder/name)
    assert arrays['mean'].shape == arrays['scale'].shape == (48,) and np.all(arrays['scale'] >= 1e-5)
    assert arrays['coefficients'].shape == (48, 210) and arrays['intercept'].shape == (210,)
    assert arrays['current_tables'].shape == (21, 10, 10) and arrays['majority'].shape == (21,)
    extracted = {name: arrays[name] for name in ['input_patterns', 'observed_indices', 'root_ids', 'roles', 'mean', 'scale']}
    extracted.update(coefficients_lag2=arrays['coefficients'][:, 20:30], intercept_lag2=arrays['intercept'][20:30],
        coefficients_lag0=arrays['coefficients'][:, :10], intercept_lag0=arrays['intercept'][:10],
        current_table_lag2=arrays['current_tables'][2], majority_lag2=arrays['majority'][2])
    majority, table = int(extracted['majority_lag2']), extracted['current_table_lag2']
    extracted['current_lookup_lag2'] = np.asarray([row.argmax() if row.sum() else majority for row in table])
    return extracted, graph, dict(parent_case=folder.as_posix(), parent_manifest_sha256=c['parent_manifest_sha256'],
        parent_case_manifest_sha256=cached(folder/'manifest.json'), parent_case_npz_sha256=cached(folder/'case.npz'),
        parent_source_commit=source['source_commit'], parent_protocol_commit=source['protocol_commit'],
        selected_array_hashes={name: array_sha(value) for name, value in extracted.items()},
        parent_replacement_outcome_arrays_read=False)


def same_tree(actual, expected, c):
    if isinstance(expected, dict):
        assert set(actual) == set(expected), (set(actual), set(expected))
        for name, value in expected.items():
            same_tree(actual[name], value, c)
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for first, second in zip(actual, expected):
            same_tree(first, second, c)
    elif isinstance(expected, float):
        close(actual, expected, c['score_atol'], c['score_rtol'])
    else:
        assert actual == expected, (actual, expected)


def case_replay(case, validation, c, block, level, raw, cached, pinned, smoke):
    arrays = load_archive(case/'case.npz')
    graph = read(case/'graph.json')
    hashed_arrays(graph, arrays)
    frozen, parent_graph, lineage = parent_inputs(c, level, block, cached)
    for name, value in frozen.items():
        np.testing.assert_array_equal(arrays[name], value)
    assert graph['frozen_array_hashes'] == lineage['selected_array_hashes']
    assert graph['parent_case_path'] == lineage['parent_case']
    assert graph['parent_case_sha256'] == lineage['parent_case_npz_sha256']
    assert graph['parent_case_manifest_sha256'] == lineage['parent_case_manifest_sha256']
    assert graph['parent_main_manifest_sha256'] == c['parent_manifest_sha256']
    assert graph['parent_source_commit'] == c['parent_source_commit']
    assert graph['parent_protocol_commit'] == c['parent_protocol_commit']
    assert graph['parent_graph_sha256'] == cached(Path(lineage['parent_case'])/'graph.json')
    parent = Path(lineage['parent_case'])
    with np.load(parent/'case.npz', allow_pickle=False) as archive:
        for name, digest in graph['parent_array_hashes'].items():
            assert name in ['input_patterns', 'observed_indices', 'root_ids', 'roles', 'mean', 'scale',
                            'coefficients', 'intercept', 'current_tables', 'majority']
            assert digest == parent_graph['array_hashes'][name] == array_sha(archive[name])
    assert len(graph['parent_array_hashes']) == 10
    weights, patterns, roles, observed, ids = independent_graph(c, level, block['input_seed'])
    for name, value in [('input_patterns', patterns), ('roles', roles), ('root_ids', ids), ('observed_indices', observed)]:
        np.testing.assert_array_equal(arrays[name], value)
    assert (len(ids), weights.nnz) == ((686, 3309) if level == 'legacy5' else (138639, 2700513))
    assert graph['weight_sha256'] == parent_graph['weight_sha256'] == matrix_hash(weights)
    assert (graph['direct_carry_multiplier'], graph['drive_multiplier'], graph['decoder_parameters_per_lag'], graph['new_trained_parameters']) == (0., .6, 490, 0)
    assert graph['synaptic_history'] is True and graph['same_step_kc_to_mbon_retained'] is True
    assert graph['input_root_ids'] == parent_graph['input_root_ids']
    assert graph['observation_root_ids'] == parent_graph['observation_root_ids'] == ids[observed].astype(str).tolist()
    _, small_ids, _ = partial_source()
    local = pd.Index(ids).get_indexer(small_ids)
    assert np.all(local >= 0)
    assert graph['input_mapping_sha256'] == hashlib.sha256(small_ids.tobytes()+patterns[:, local].tobytes()).hexdigest()
    assert graph['input_mapping_sha256'] == parent_graph['input_mapping_sha256']
    assert patterns.shape == (10, len(ids)) and set(np.unique(patterns)) == {0., .5}
    assert np.all(np.count_nonzero(patterns, axis=1) == 51)
    assert len({row.tobytes() for row in patterns}) == 10, 'Invalid replacement assay; never resample'
    assert np.all(roles[np.any(patterns != 0, axis=0)] == 'KC') and np.all(roles[observed] == 'MBON')
    for name, digest in graph['graph_cache_hashes'].items():
        assert digest == pinned['outputs/tdc_v2/whole_cache/'+name]
    for name, source in graph['source'].items():
        assert source['sha256'] == pinned['outputs/tdc_v2/raw/'+name]
    symbols = np.random.default_rng(block['stream_seed']).integers(0, 10, c['warmup']+c['test_samples'], dtype=np.int64)
    times = np.arange(c['warmup'], len(symbols), dtype=np.int64)
    anchors = c['warmup']+np.floor(np.linspace(0, c['test_samples']-1-2, 64)).astype(np.int64)
    assert len(np.unique(anchors)) == 64 and anchors[-1]+2 < len(symbols)
    for name, value in [('stream_symbols', symbols), ('times', times), ('y0', symbols[times]), ('y2', symbols[times-2]),
                         ('anchors', anchors), ('evaluation_times', anchors+2), ('original_symbols', symbols[anchors]),
                         ('alternative_symbols', (symbols[anchors]+1)%10), ('current_symbols', symbols[anchors+2])]:
        np.testing.assert_array_equal(arrays[name], value)
    assert arrays['features'].shape == (len(symbols), 48) and arrays['norms'].shape == (len(symbols),)
    assert arrays['final_state'].shape == (len(ids),)
    assert not any(value.shape in [(64, len(ids)), (64, 3, len(ids))] for value in arrays.values()), 'Full-state probe bank was not registered'
    assert graph['final_state_sha256'] == hashlib.sha256(arrays['final_state'].tobytes()).hexdigest()
    assert re.fullmatch('[0-9a-f]{64}', graph['baseline_full_state_trajectory_sha256'])
    stream_metrics = read(case/'metrics.json')
    assert [row['lag'] for row in stream_metrics] == [0, 2]
    for lag, row in zip([0, 2], stream_metrics):
        predicted_scores = score(arrays['features'][c['warmup']:], arrays, lag)
        assert arrays[f'scores_lag{lag}'].shape == (c['test_samples'], 10)
        close(arrays[f'scores_lag{lag}'], predicted_scores, c['score_atol'], c['score_rtol'])
        np.testing.assert_array_equal(arrays[f'predictions_lag{lag}'], predicted_scores.argmax(axis=1))
        correct = int(np.count_nonzero(predicted_scores.argmax(axis=1) == arrays[f'y{lag}']))
        for key, value in dict(level=level, seed=block['seed'], lag=lag, samples=c['test_samples'], correct=correct,
                               accuracy=float(Fraction(correct, c['test_samples'])), chance=.1).items():
            scalar_check(row[key], value)
    majority = int(arrays['majority_lag2'])
    np.testing.assert_array_equal(arrays['frequency_predictions'], np.full(c['test_samples'], majority, dtype=np.int64))
    np.testing.assert_array_equal(arrays['current_predictions'], arrays['current_lookup_lag2'][symbols[times]])
    for name in ['frequency', 'current']:
        count = int(np.count_nonzero(arrays[name+'_predictions'] == arrays['y2']))
        scalar_check(stream_metrics[1][name+'_correct'], count)
        scalar_check(stream_metrics[1][name+'_accuracy'], float(Fraction(count, c['test_samples'])))
    original_windows = np.asarray([symbols[a:a+3] for a in anchors])
    changed_windows = original_windows.copy()
    changed_windows[:, 0] = (changed_windows[:, 0]+1)%10
    np.testing.assert_array_equal(arrays['trial_original_symbols'], original_windows)
    np.testing.assert_array_equal(arrays['trial_counterfactual_symbols'], changed_windows)
    assert np.all(np.count_nonzero(original_windows != changed_windows, axis=1) == 1)
    np.testing.assert_array_equal(original_windows[:, 1:], changed_windows[:, 1:])
    worlds = ['original', 'counterfactual', 'instantaneous_original', 'instantaneous_counterfactual']
    for world in worlds:
        states = arrays[f'trial_{world}_features']
        assert states.shape == (64, 3, 48)
        values = score(states[:, -1], arrays)
        assert arrays['scores_'+world].shape == (64, 10)
        close(arrays['scores_'+world], values, c['score_atol'], c['score_rtol'])
        np.testing.assert_array_equal(arrays['predictions_'+world], values.argmax(axis=1))
    np.testing.assert_array_equal(arrays['trial_instantaneous_original_features'][:, 1:], arrays['trial_instantaneous_counterfactual_features'][:, 1:])
    np.testing.assert_array_equal(arrays['scores_instantaneous_original'], arrays['scores_instantaneous_counterfactual'])
    np.testing.assert_array_equal(arrays['predictions_instantaneous_original'], arrays['predictions_instantaneous_counterfactual'])
    for probe, anchor in enumerate(anchors):
        np.testing.assert_array_equal(arrays['trial_original_features'][probe], arrays['features'][anchor:anchor+3])
        close(arrays['scores_original'][probe], arrays['scores_lag2'][anchor+2-c['warmup']], c['score_atol'], c['score_rtol'])
        assert arrays['predictions_original'][probe] == arrays['predictions_lag2'][anchor+2-c['warmup']]
    original, replacement = arrays['original_symbols'], arrays['alternative_symbols']
    original_predictions, replacement_predictions = arrays['predictions_original'], arrays['predictions_counterfactual']
    counts = paired_counts(original_predictions, replacement_predictions, original, replacement)
    np.testing.assert_array_equal(arrays['frequency_trial_predictions'], np.full(64, majority, dtype=np.int64))
    np.testing.assert_array_equal(arrays['current_trial_predictions'], arrays['current_lookup_lag2'][arrays['current_symbols']])
    controls = {}
    for name in ['frequency', 'current']:
        predictions = arrays[name+'_trial_predictions']
        controls[name+'_joint_correct'] = paired_counts(predictions, predictions, original, replacement)['joint_correct']
    controls['instantaneous_joint_correct'] = paired_counts(arrays['predictions_instantaneous_original'],
        arrays['predictions_instantaneous_counterfactual'], original, replacement)['joint_correct']
    assert set(controls.values()) == {0}
    cell = dict(level=level, seed=block['seed'], parent_seed=block['parent_seed'], samples=64,
        original_correct=counts['original_correct'], replacement_correct=counts['counterfactual_correct'],
        joint_correct=counts['joint_correct'], original_accuracy=float(Fraction(counts['original_correct'], 64)),
        replacement_accuracy=float(Fraction(counts['counterfactual_correct'], 64)),
        joint_accuracy=float(Fraction(counts['joint_correct'], 64)), prediction_changes=counts['predictions_changed'],
        **controls, scientific_rule_pass=exact_gate(counts['joint_correct'], smoke), technical_checks_pass=True)
    same_tree(read(case/'primary-cell.json'), cell, c)
    trial_metrics = read(case/'trial-metrics.json')
    same_tree(trial_metrics['counts'], cell, c)
    indices = np.arange(64)
    score0, score1 = arrays['scores_original'], arrays['scores_counterfactual']
    masked0, masked1 = score0.copy(), score1.copy()
    masked0[indices, original] = -np.inf
    masked1[indices, replacement] = -np.inf
    contrast0 = score0[indices, replacement]-score0[indices, original]
    contrast1 = score1[indices, replacement]-score1[indices, original]
    delta = arrays['trial_counterfactual_features'][:, -1]-arrays['trial_original_features'][:, -1]
    beta = arrays['coefficients_lag2']
    projection = np.asarray([(delta[i]/arrays['scale']) @ (beta[:, replacement[i]]-beta[:, original[i]]) for i in indices])
    margins = dict(original_target_margin=score0[indices, original]-masked0.max(axis=1),
        replacement_target_margin=score1[indices, replacement]-masked1.max(axis=1),
        original_contrast_margin=contrast0, replacement_contrast_margin=contrast1,
        contrast_margin_change=contrast1-contrast0, projected_contrast_margin_change=projection,
        observed_delta_norm=np.linalg.norm(delta, axis=1))
    for name, values in margins.items():
        close(arrays[name], values, c['score_atol'], c['score_rtol'])
    close(contrast1-contrast0, projection, c['score_atol'], c['score_rtol'])
    rows = []
    for index, anchor in enumerate(anchors):
        row = dict(level=level, seed=block['seed'], probe_index=index, anchor=int(anchor), evaluation_time=int(anchor)+2,
            original_symbol=int(original[index]), replacement_symbol=int(replacement[index]), current_symbol=int(arrays['current_symbols'][index]),
            original_correct=bool(original_predictions[index] == original[index]),
            replacement_correct=bool(replacement_predictions[index] == replacement[index]),
            joint_correct=bool(original_predictions[index] == original[index] and replacement_predictions[index] == replacement[index]),
            prediction_changed=bool(original_predictions[index] != replacement_predictions[index]))
        row.update({'prediction_'+world: int(arrays['predictions_'+world][index]) for world in worlds})
        row.update({'prediction_'+name: int(arrays[name+'_trial_predictions'][index]) for name in ['frequency', 'current']})
        row.update({name: float(value[index]) for name, value in margins.items()})
        rows.append(row)
    same_tree(trial_metrics['rows'], rows, c)
    actual_rows = raw[(raw.level == level) & (raw.seed == block['seed'])].sort_values('probe_index')
    assert len(actual_rows) == 64 and list(actual_rows.probe_index) == list(range(64))
    same_tree(actual_rows.to_dict('records'), rows, c)
    info = read(case/'trial-info.json')
    assert read(validation/'recomputed-trial-info.json') == info
    for key in ['technical_checks_pass', 'every_fixed_anchor_counted', 'original_replay_equals_baseline',
                'instantaneous_full_endpoint_pair_equal', 'fixed_head_margin_projection_verified']:
        assert info[key] is True
    assert info['new_trained_parameters'] == 0 and len(info['trials']) == 64
    for index, record in enumerate(info['trials']):
        assert (record['probe_index'], record['anchor'], record['evaluation_time']) == (index, int(anchors[index]), int(anchors[index])+2)
        for key in ['pre_state_sha256', 'baseline_window_sha256', 'baseline_endpoint_full_state_sha256']:
            assert re.fullmatch('[0-9a-f]{64}', record[key])
        for world in worlds:
            for suffix in ['full_state_trajectory_sha256', 'final_state_sha256']:
                assert re.fullmatch('[0-9a-f]{64}', record[world+'_'+suffix])
        assert record['baseline_window_sha256'] == record['original_full_state_trajectory_sha256']
        assert record['baseline_endpoint_full_state_sha256'] == record['original_final_state_sha256']
        assert record['instantaneous_original_final_state_sha256'] == record['instantaneous_counterfactual_final_state_sha256']
    return dict(cell=cell, unconditional_joint_accuracy_exact=str(Fraction(counts['joint_correct'], 64)),
        predictions_unchanged=counts['predictions_unchanged'], parent_authentication=lineage,
        baseline_full_state_trajectory_sha256=graph['baseline_full_state_trajectory_sha256'],
        stream_metrics=stream_metrics, input_mapping_sha256=graph['input_mapping_sha256'],
        input_root_ids=graph['input_root_ids'], observation_root_ids=graph['observation_root_ids'],
        paired_array_hashes={name: array_sha(arrays[name]) for name in ['stream_symbols', 'times', 'y0', 'y2', 'anchors',
            'evaluation_times', 'original_symbols', 'alternative_symbols', 'current_symbols', 'trial_original_symbols', 'trial_counterfactual_symbols']})


def case_verifier_evidence(case, validation, replay, c, cached):
    checks = read(validation/'checks.json')
    cell = replay['cell']
    assert checks['all_checks_pass'] is True and (checks['level'], checks['seed']) == (cell['level'], cell['seed'])
    assert checks['exact_baseline_full_state_trace_hash'] == replay['baseline_full_state_trajectory_sha256']
    for key in ['independent_graph_mapping_and_normalization', 'exact_frozen_head_scaler_and_controls',
                'old_parent_lineage_authenticated', 'frequency_current_instantaneous_joint_counts_exact_zero',
                'exact_integer_primary_recalculated', 'official_predictions_use_exact_archived_parameters',
                'all_anchors_unconditionally_scored', 'one_actual_symbol_changed', 'prefix_and_future_identical',
                'frozen_head_projection_of_margin_change_verified']:
        assert checks[key] is True, (validation, key)
    assert checks['scientific_threshold_tolerance_applied'] is False
    assert checks['new_trained_parameters'] == 0 and checks['old_training_samples'] == 4000
    assert (checks['old_head_independent_svd_authentications'], checks['full_stream_replay_count'], checks['evaluation_lag']) == (1, 1, 2)
    assert (checks['exact_full_state_prestate_hashes'], checks['exact_four_world_window_trace_hashes'],
            checks['exact_four_world_final_state_hashes'], checks['instantaneous_final_state_pairs_exact']) == (64, 256, 256, 64)
    parent = replay['parent_authentication']
    assert checks['parent_case_npz_sha256'] == parent['parent_case_npz_sha256']
    assert checks['parent_case_manifest_sha256'] == parent['parent_case_manifest_sha256']
    assert checks['parent_graph_sha256'] == cached(Path(parent['parent_case'])/'graph.json')
    same_tree(checks['official_cell'], cell, c)
    same_tree(read(validation/'recomputed-cell.json'), cell, c)
    same_tree(read(validation/'recomputed-trials.json'), read(case/'trial-metrics.json')['rows'], c)
    assert read(validation/'manifest.json')['result_case_manifest_sha256'] == cached(case/'manifest.json')
    return checks


def stage_replay(stage, validation, c, smoke, cached, pinned):
    raw = pd.read_csv(stage/'raw-trials.csv')
    metrics = pd.read_csv(stage/'stream-metrics.csv')
    block_scores = pd.read_csv(stage/'block-scores.csv')
    summary = read(stage/'summary.json')
    expected_cases = 2 if smoke else 9
    assert len(raw) == expected_cases*64 and len(metrics) == expected_cases*2 and len(block_scores) == expected_cases
    for frame, keys in [(raw, ['level', 'seed', 'probe_index']), (metrics, ['level', 'seed', 'lag']), (block_scores, ['level', 'seed'])]:
        assert not frame[keys].duplicated().any()
        assert set(frame.level) == set(c['levels'])
        assert len(frame[['level', 'seed']].drop_duplicates()) == expected_cases
    assert set(metrics.samples) == {c['test_samples']} and set(metrics.lag) == {0, 2}
    for field in ['original_correct', 'replacement_correct', 'joint_correct']:
        assert block_scores[field].between(0, 64).all()
        assert np.array_equal(block_scores[field], block_scores[field].astype(np.int64))
    assert set(block_scores.samples) == {64}
    cells, details, levels, pairs, case_checks, resources = [], [], {}, {}, {}, {}
    for level in c['levels']:
        selected = []
        for seed in c['blocks_by_level'][level]:
            block = next(row for row in c['blocks'] if row['seed'] == seed)
            identity = f'{level}_s{seed}'
            case, verified = stage/identity, validation/identity
            replay = case_replay(case, verified, c, block, level, raw, cached, pinned, smoke)
            case_checks[identity] = case_verifier_evidence(case, verified, replay, c, cached)
            for folder in [case, verified]:
                resources[folder.as_posix()] = resource_check(folder, c)
            manifest = read(case/'manifest.json')
            assert manifest['complete'] is True and manifest['block'] == block and manifest['level'] == level
            assert manifest['smoke'] is smoke
            cell = replay['cell']
            selected.append(cell)
            cells.append(cell)
            details.append(replay)
            if seed in pairs:
                for key in ['input_mapping_sha256', 'input_root_ids', 'observation_root_ids', 'paired_array_hashes']:
                    assert replay[key] == pairs[seed][key], (seed, key)
            else:
                pairs[seed] = replay
            actual_metrics = metrics[(metrics.level == level) & (metrics.seed == seed)].sort_values('lag')
            expected_metrics = pd.DataFrame(replay['stream_metrics'])
            pd.testing.assert_frame_equal(actual_metrics.reset_index(drop=True)[expected_metrics.columns], expected_metrics,
                check_dtype=False, check_exact=False, atol=1e-12, rtol=1e-12)
            actual_cell = block_scores[(block_scores.level == level) & (block_scores.seed == seed)]
            assert len(actual_cell) == 1
            values = actual_cell.iloc[0].to_dict()
            if smoke:
                assert pd.isna(values['scientific_rule_pass'])
                values['scientific_rule_pass'] = None
            same_tree(values, cell, c)
        rule = None if smoke else all(row['scientific_rule_pass'] is True for row in selected)
        outcome = 'smoke' if smoke else 'PASS' if rule else 'FAIL'
        expected = dict(outcome=outcome, registered_rule_pass=rule, cells=selected,
            **{name: estimates([row[name] for row in selected], c) for name in
               ['joint_accuracy', 'original_accuracy', 'replacement_accuracy']})
        same_tree(summary['levels'][level], expected, c)
        levels[level] = dict(outcome=outcome, registered_rule_pass=rule,
            unconditional_joint_counts=[row['joint_correct'] for row in selected],
            unconditional_denominators=[64 for row in selected],
            unconditional_joint_accuracy_exact=[str(Fraction(row['joint_correct'], 64)) for row in selected],
            cells=selected)
    cells.sort(key=lambda row: (row['level'], row['seed']))
    same_tree(read(stage/'primary-cells.json'), cells, c)
    primary = levels[c['primary_level']]
    assert summary['smoke'] is smoke and summary['outcome'] == primary['outcome']
    assert summary['primary_pass'] is primary['registered_rule_pass']
    assert (summary['cases'], summary['probe_pairs'], summary['frozen_heads_used'], summary['new_trained_parameters'],
            summary['decoder_parameters_per_lag']) == (expected_cases, 64*expected_cases, expected_cases, 0, 490)
    assert summary['criterion_unchanged'] is True and summary['every_fixed_anchor_counted'] is True
    assert summary['performance_eligibility_gate'] is False
    assert summary['biological_plasticity_performed'] is False and summary['unique_cycle_effect_claim'] is False
    assert summary['m1_outcome_unchanged'] == 'assay-invalid' and summary['m4_outcome_unchanged'] == 'INFEASIBLE'
    paired_seeds = c['blocks_by_level']['brain5']
    assert read(stage/'pairing.json') == dict(paired_block_seeds=paired_seeds, inputs_targets_anchors_identical=True,
        computational_blocks_are_not_biological_replicates=True)
    return dict(outcome=primary['outcome'], primary_pass=primary['registered_rule_pass'], cases=expected_cases,
        new_trained_parameters=0, old_head_independent_svd_authentications=expected_cases,
        baseline_full_stream_hash_entries=expected_cases, full_prestate_hash_entries=64*expected_cases,
        four_world_window_hash_entries=256*expected_cases, four_world_final_state_hash_entries=256*expected_cases,
        instantaneous_equal_state_pairs=64*expected_cases, probe_pairs=64*expected_cases,
        levels=levels, case_evidence=details, case_verifier_checks=case_checks, case_resources=resources)


def stage_verifier_evidence(stage, validation, c, smoke, replay, cached, pinned):
    source, verifier_source = read(stage/'source.json'), read(validation/'source.json')
    manifest, verified_manifest = read(stage/'manifest.json'), read(validation/'manifest.json')
    assert manifest['complete'] is True and manifest['source_commit'] == source['source_commit']
    assert manifest['smoke'] is smoke and manifest['config'] == c
    assert verified_manifest['complete'] is True and verified_manifest['result_manifest_sha256'] == cached(stage/'manifest.json')
    checks = read(validation/'checks.json')
    assert checks['all_checks_pass'] is True and checks['result_manifest_sha256'] == cached(stage/'manifest.json')
    assert checks['source_commit'] == verifier_source['source_commit'] and checks['protocol_commit'] == PREREGISTRATION
    assert checks['outcome'] == replay['outcome'] and checks['primary_pass'] is replay['primary_pass']
    counts = dict(cases=replay['cases'], replayed_streams=replay['cases'], exact_full_state_hashes=replay['cases'],
        old_head_independent_svd_authentications=replay['cases'], frozen_heads_used=replay['cases'], new_trained_parameters=0,
        probe_pairs=replay['probe_pairs'], exact_full_state_prestate_hashes=replay['full_prestate_hash_entries'],
        exact_four_world_window_trace_hashes=replay['four_world_window_hash_entries'],
        exact_four_world_final_state_hashes=replay['four_world_final_state_hash_entries'],
        instantaneous_final_state_pairs_exact=replay['instantaneous_equal_state_pairs'])
    for key, value in counts.items():
        assert checks[key] == value, (validation, key)
    for key in ['all_anchors_unconditionally_scored', 'one_actual_symbol_changed', 'prefix_and_future_identical',
                'frequency_current_instantaneous_joint_counts_exact_zero', 'frozen_head_projection_of_margin_change_verified',
                'exact_frozen_head_scaler_and_controls', 'exact_integer_primary_recalculated',
                'independent_manual_dynamics_no_main_scorer_import']:
        assert checks[key] is True
    assert checks['evaluation_lag'] == 2
    for key in ['scientific_threshold_tolerance_applied', 'performance_eligibility_gate', 'biological_plasticity_performed', 'unique_cycle_effect_claim']:
        assert checks[key] is False
    assert checks['pairing'] == read(stage/'pairing.json')
    validation_source = read(validation/'source-validation.json')
    assert validation_source == dict(result_source_commit=source['source_commit'], verifier_source_commit=verifier_source['source_commit'],
        protocol_commit=PREREGISTRATION, recorded_source_hashes_authenticated=len(source['hashes']),
        independent_manual_dynamics_no_main_scorer_import=True)
    raw_source = read(validation/'source-graph-audit.json')
    assert checks['source_graph_reconstruction'] == raw_source
    assert (raw_source['neurons'], raw_source['edges']) == (138639, 2700513)
    for key in ['full_graph_rebuilt_from_pinned_parquet', 'roles_rebuilt_from_pinned_annotations', 'partial_induced_graph_exact']:
        assert raw_source[key] is True
    for name, digest in raw_source['raw_source_sha256'].items():
        assert digest == pinned['outputs/tdc_v2/raw/'+name]
    required = {'scripts/counterfactual_support.py', 'scripts/counterfactual_tracking.py',
        'scripts/verify_counterfactual_tracking.py', 'scripts/audit_counterfactual_tracking.py',
        'scripts/run_counterfactual_guards.py', 'tests/test_counterfactual_tracking.py',
        'tests/test_counterfactual_tracking_verifier.py', CONFIG, PROTOCOL}
    assert required <= set(source['hashes']) and required <= set(verifier_source['hashes'])
    assert all(source['hashes'][name] == verifier_source['hashes'][name] == cached(name) for name in required), 'M6 source freeze changed'
    for name in ['raw-trials', 'stream-metrics', 'block-scores']:
        first, second = pd.read_csv(stage/(name+'.csv')), pd.read_csv(validation/('recomputed-'+name+'.csv'))
        pd.testing.assert_frame_equal(first, second, check_dtype=False, check_exact=False, atol=c['score_atol'], rtol=c['score_rtol'])
    same_tree(read(stage/'summary.json'), read(validation/'recomputed-summary.json'), c)
    saved_case_checks = read(validation/'case-checks.json')
    assert len(saved_case_checks) == replay['cases']
    assert {f'{row["level"]}_s{row["seed"]}' for row in saved_case_checks} == set(replay['case_verifier_checks'])
    for row in saved_case_checks:
        assert row == replay['case_verifier_checks'][f'{row["level"]}_s{row["seed"]}']
    return checks


def audit(out, main_validation='main_validation', smoke_validation='smoke_validation', guard_stage='guard_tests'):
    root = NAMESPACE
    c = read(root/'main/config.json')
    fixed_config(c)
    assert root.resolve() in out.resolve().parents and not out.exists()
    with attempt(out, c, 'm6-counterfactual-tracking-closeout'):
        preservation = history_preserved(c)
        assert preservation['prior_result_identities_preserved'] == 44571
        assert preservation['protected_baseline_paths'] == 45219
        old = read(root/'baseline_audit/baseline-git-blobs.json')
        assert old == tree(BASELINE), 'Baseline inventory must equal its exact historical Git tree'
        protected = {name: blob for name, blob in old.items() if name not in OVERVIEWS}
        current = tree('HEAD')
        assert all(current.get(name) == blob for name, blob in protected.items())
        names = sorted(name for name in protected if Path(name).is_file())
        actual = subprocess.run(['git', 'hash-object', '--stdin-paths'], input='\n'.join(names)+'\n',
            text=True, capture_output=True, check=True).stdout.splitlines()
        assert len(actual) == len(names) and all(value == protected[name] for name, value in zip(names, actual))
        digests = {}
        def cached(path):
            path = Path(path).resolve()
            if path not in digests:
                digests[path] = sha(path)
            return digests[path]
        manifests, attempts, failures, validation_failures, invalid_sources = manifest_and_failures(root, out, cached)
        recorded_sources, pinned = authenticate_sources(root, out, cached, invalid_sources)
        replay, verifier_checks, usage = {}, {}, {}
        for stage, validation_name in [('smoke', smoke_validation), ('main', main_validation)]:
            assert Path(validation_name).name == validation_name
            folder, validation = root/stage, root/validation_name
            assert not any(key[0] == (folder/'source.json').as_posix() or key[0] == (validation/'source.json').as_posix() for key in invalid_sources)
            smoke = stage == 'smoke'
            saved_config = read(folder/'config.json')
            fixed_config(saved_config, smoke)
            fixed_config(read(validation/'config.json'), smoke)
            replay[stage] = stage_replay(folder, validation, saved_config, smoke, cached, pinned)
            verifier_checks[stage] = stage_verifier_evidence(folder, validation, saved_config, smoke, replay[stage], cached, pinned)
            for path in [folder, validation]:
                usage[path.as_posix()] = resource_check(path, c)
                pool = read(path/'process-tree-resources.json')
                assert pool['seconds'] <= 7200 and pool['peak_sampled_process_tree_rss_bytes'] <= 4294967296
                assert pool['sample_seconds'] == .05
                jobs = read(path/'job-resources.json')
                assert len(jobs) == replay[stage]['cases']
                assert {row['identity'] for row in jobs} == {f'{row["cell"]["level"]}/s{row["cell"]["seed"]}' for row in replay[stage]['case_evidence']}
                for row in jobs:
                    level, seed = row['identity'].split('/s')
                    assert row['resources'] == read(path/f'{level}_s{seed}'/'resources.json')
                usage[path.as_posix()]['process_tree'] = pool
        assert Path(guard_stage).name == guard_stage
        assert guard_stage not in {row['stage'] for row in validation_failures}
        guards = read(root/guard_stage/'checks.json')
        assert guards['all_checks_pass'] is True and guards['parameters_changed'] is False
        assert guards['source_graph_outcomes_executed'] is False and guards['source_freeze_rechecked'] is True
        assert type(guards['tests_passed']) is int and guards['tests_passed'] > 0
        pytest_counts = re.findall(r'(\d+) passed', (root/guard_stage/'pytest.txt').read_text(encoding='utf-8'))
        assert pytest_counts and int(pytest_counts[-1]) == guards['tests_passed']
        guard_source = read(root/guard_stage/'source.json')
        assert guards['source_commit'] == guard_source['source_commit']
        main_source = read(root/'main/source.json')
        assert all(guard_source['hashes'][name] == main_source['hashes'][name] for name in main_source['hashes']), 'Guard/source freeze changed before main'
        required_tests = {'tests/test_counterfactual_tracking.py', 'tests/test_counterfactual_tracking_verifier.py',
                          'tests/test_temporal_mechanism.py', 'tests/test_temporal_cycles.py'}
        assert set(guards['test_hashes']) == required_tests
        guard_tree = tree(guards['source_commit'])
        for name, digest in guards['test_hashes'].items():
            assert digest == git_digest(guard_tree[name])
        archives = arrays = 0
        for path in sorted(root.rglob('*.npz')):
            if out.resolve() in path.resolve().parents:
                continue
            with np.load(path, allow_pickle=False) as archive:
                archives += 1
                for name in archive.files:
                    value = archive[name]
                    arrays += 1
                    assert value.dtype.kind != 'O'
                    if value.dtype.kind in 'fci':
                        assert np.isfinite(value).all(), (path, name)
        old_failure = read('results/cycle_attribution_v1/validation-failures.json')
        assert len(old_failure) == 1 and old_failure[0]['source_validation_failed'] is True
        old_snapshot = Path('results/cycle_attribution_v1')/old_failure[0]['snapshot_path']
        assert cached(old_snapshot) == old_failure[0]['recorded_sha256']
        links, documents = document_check(out, current, cached)
        write(out/'manifest-checks.json', manifests)
        write(out/'attempt-checks.json', attempts)
        write(out/'recorded-source-checks.json', recorded_sources)
        write(out/'retained-validation-failures.json', validation_failures)
        write(out/'exact-paired-tracking-replay.json', replay)
        write(out/'stage-verifier-checks.json', verifier_checks)
        write(out/'stage-resources.json', usage)
        write(out/'document-links.json', links)
        write(out/'document-sha256.json', documents)
        write(out/'audit.json', {**preservation, 'all_checks_pass': True,
            'source_commit': git('rev-parse', 'HEAD'), 'audit_source_sha256': sha(__file__),
            'protocol_commit': PREREGISTRATION, 'protocol_precedes_recorded_main_source': True,
            'recorded_source_entries': len(recorded_sources), 'manifests_checked': len(manifests),
            'checksum_entries': sum(row['artifacts'] for row in manifests), 'attempts_checked': len(attempts),
            'retained_incomplete_attempts': failures, 'retained_validation_failures': validation_failures,
            'failure_registry_sha256': {path.as_posix(): cached(path) for path in
                [root/'known-incomplete-attempts.json', root/'validation-failures.json'] if path.exists()},
            'old_m4_source_validation_failure_and_snapshot_preserved': True,
            'guard_stage': guard_stage, 'guards': guards,
            'archives_inspected': archives, 'arrays_inspected': arrays, 'nonfinite_arrays': 0,
            'exact_integer_primary_recalculated': True, 'scientific_threshold_tolerance_applied': False,
            'primary_rule': '10*joint_correct >= 9*64 in all six fixed primary blocks',
            'minimum_joint_correct_per_block': 58, 'every_primary_anchor_counted': True,
            'performance_eligibility_gate': False,
            'outcome': replay['main']['outcome'], 'primary_pass': replay['main']['primary_pass'],
            'cases': replay['main']['cases'], 'baseline_full_stream_hash_entries': replay['main']['baseline_full_stream_hash_entries'],
            'full_prestate_hash_entries': replay['main']['full_prestate_hash_entries'],
            'four_world_window_hash_entries': replay['main']['four_world_window_hash_entries'],
            'four_world_final_state_hash_entries': replay['main']['four_world_final_state_hash_entries'],
            'hash_counts_are_entries_not_unique_values': True,
            'instantaneous_equal_state_pairs': replay['main']['instantaneous_equal_state_pairs'],
            'old_head_independent_svd_authentications': replay['main']['old_head_independent_svd_authentications'],
            'new_trained_parameters': 0, 'frozen_decoder_parameter_budget': 490,
            'probe_pairs': replay['main']['probe_pairs'], 'smoke_scientific_endpoint': None,
            'probe_replicates_are_independent_animals': False,
            'scientific_settings_and_outcomes_unchanged': True,
            'biological_plasticity_performed': False, 'unique_cycle_effect_claim': False,
            'graph_searches': 0})
    seal(out, dict(complete=True, kind='m6-counterfactual-tracking-closeout', source_commit=git('rev-parse', 'HEAD')))
    print(read(out/'audit.json'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--main-validation', default='main_validation')
    parser.add_argument('--smoke-validation', default='smoke_validation')
    parser.add_argument('--guard-stage', default='guard_tests')
    args = parser.parse_args()
    audit(args.out, args.main_validation, args.smoke_validation, args.guard_stage)
