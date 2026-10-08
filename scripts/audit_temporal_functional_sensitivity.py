"""M5 closeout: authenticated artifacts and exact saved-float endpoint replay.

The auditor imports neither the experiment's dynamics nor its scientific
summary. It independently checks labels, saved predictions, pulse directions,
integer counts and rational endpoint arithmetic. Separate verifier evidence
authenticates full trajectories, ridge refits and complex/finite derivatives.
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

from sensitivity_support import read, write, sha, array_sha, attempt, seal, git, history_preserved
from verify_temporal_memory_curve import independent_graph


BASELINE = '4a72db553f80317c282f48561e33b5edbfc3a765'
PREREGISTRATION = '37212abeb6cfc1a4907e08348ff41030f3edfb8b'
NAMESPACE = Path('results/functional_sensitivity_v1')
CONFIG = 'configs/temporal_functional_sensitivity.json'
PROTOCOL = 'docs/temporal-functional-sensitivity-protocol.md'
OVERVIEWS = {'README.md', 'docs/research-status.md',
             'docs/research-roadmap.md', 'docs/next-work.md'}
MINIMUM_POWER = Fraction(1, 10**16)


def tree(revision):
    return {line.split('\t', 1)[1]: line.split('\t', 1)[0].split()[2]
            for line in git('ls-tree', '-r', revision).splitlines()}


def git_digest(blob):
    return hashlib.sha256(subprocess.check_output(['git', 'cat-file', 'blob', blob])).hexdigest()


def fixed_config(c, smoke=False):
    registered = read(CONFIG)
    assert (registered['baseline_commit'], registered['namespace'], registered['protocol']) == (
        BASELINE, NAMESPACE.as_posix(), PROTOCOL)
    assert registered['levels'] == ['legacy5', 'brain5'] and registered['primary_level'] == 'legacy5'
    assert registered['circuit_seeds'] == [701]
    assert registered['blocks'] == [dict(seed=1500001+i, input_seed=1510001+i,
                                       train_seed=1520001+i, test_seed=1530001+i) for i in range(6)]
    assert registered['blocks_by_level'] == {'legacy5': list(range(1500001, 1500007)),
                                           'brain5': list(range(1500001, 1500004))}
    assert (registered['alphabet_size'], registered['input_fraction'], registered['input_amplitude'],
            registered['gain'], registered['leak'], registered['carry_multiplier'], registered['alpha']) == (10, .1, .5, .9, .6, 0., 1.)
    assert registered['normalization'] == 'incoming_l1' and registered['schedule'] == 'mbon_after_kc'
    assert registered['synaptic_history'] is True
    assert (registered['warmup'], registered['train_samples'], registered['test_samples'], registered['probe_count']) == (200, 4000, 2000, 32)
    assert registered['lags'] == list(range(21))
    assert registered['finite_difference_probe_indices'] == [0, 10, 21, 31]
    assert registered['finite_difference_epsilons'] == [1e-4, 1e-5]
    assert (registered['finite_difference_atol'], registered['finite_difference_rtol']) == (2e-9, 2e-5)
    assert (registered['complex_step_size'], registered['complex_step_atol'], registered['complex_step_rtol']) == (1e-20, 1e-12, 1e-9)
    assert (registered['reduction_atol'], registered['reduction_rtol']) == (1e-12, 1e-12)
    assert (registered['minimum_current_gain'], registered['primary_lag'], registered['maximum_primary_relative_gain']) == (1e-8, 5, .1)
    assert (registered['bootstrap_seed'], registered['bootstrap_draws'], registered['replacement_eta']) == (1540001, 10000, 1.)
    assert (registered['neural_workers'], registered['max_seconds'], registered['max_rss_bytes']) == (4, 7200, 4294967296)
    assert registered['graph_searches'] == 0 and registered['biological_plasticity_performed'] is False
    expected = dict(registered)
    if smoke:
        expected.update({name: registered['smoke'][name] for name in ['train_samples', 'test_samples']})
        expected['blocks_by_level'] = {level: seeds[:1] for level, seeds in registered['blocks_by_level'].items()}
    assert c == expected, 'Saved settings differ from registration'
    for name in [CONFIG, PROTOCOL]:
        assert sha(name) == git_digest(tree(PREREGISTRATION)[name]), 'Registration bytes changed'


def matrix_hash(weights):
    return hashlib.sha256(weights.data.tobytes()+weights.indices.tobytes()+weights.indptr.tobytes()).hexdigest()


def exact_primary(observed, directions, technical_valid=True):
    """Square float64 coordinates as rationals before any reduction or ratio."""
    assert observed.shape == (32, 21, 48) and directions.ndim == 2 and len(directions) == 32
    assert observed.dtype == directions.dtype == np.dtype('float64')
    assert np.isfinite(observed).all() and np.isfinite(directions).all()
    powers = {lag: Fraction(0) for lag in [0, 5]}
    denominators = []
    for tangent, direction in zip(observed, directions):
        denominator = sum((Fraction.from_float(float(value))**2 for value in direction), Fraction(0))
        denominators.append(denominator)
        if denominator:
            for lag in powers:
                numerator = sum((Fraction.from_float(float(value))**2 for value in tangent[lag]), Fraction(0))
                powers[lag] += numerator/denominator/32
    nonzero = all(value > 0 for value in denominators)
    eligible = bool(technical_valid and nonzero and powers[0] >= MINIMUM_POWER)
    passed = bool(powers[5] <= powers[0]/100) if eligible else None
    return dict(eligible=eligible, scientific_rule_pass=passed,
                s0_exact=str(powers[0]), s5_exact=str(powers[5]),
                s0_numerator=powers[0].numerator, s0_denominator=powers[0].denominator,
                s5_numerator=powers[5].numerator, s5_denominator=powers[5].denominator,
                all_contrasts_nonzero=nonzero,
                minimum_current_power_exact=str(MINIMUM_POWER)), denominators


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
    pinned = {name: digest for name, digest in read('results/cycle_attribution_v1/main/source.json')['hashes'].items()
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
    documents = [PROTOCOL, 'docs/temporal-functional-sensitivity-results.md', *sorted(OVERVIEWS)]
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


def cell_check(actual, expected, level, seed):
    assert (actual['level'], actual['seed']) == (level, seed)
    for name in ['eligible', 'scientific_rule_pass', 's0_numerator', 's0_denominator',
                 's5_numerator', 's5_denominator']:
        assert actual[name] == expected[name], (level, seed, name)
    s0 = Fraction(expected['s0_numerator'], expected['s0_denominator'])
    s5 = Fraction(expected['s5_numerator'], expected['s5_denominator'])
    close(actual['current_gain'], np.sqrt(float(s0)))
    if s0:
        close(actual['relative_gain'], np.sqrt(float(s5/s0)))
    else:
        assert actual['relative_gain'] is None
    assert actual['invalid_reason'] is None if expected['eligible'] else bool(actual['invalid_reason'])


def scalar_check(actual, expected):
    if expected is None:
        assert actual is None or pd.isna(actual)
    elif isinstance(expected, (bool, str, int)):
        assert actual == expected
    else:
        close(actual, expected)


def case_replay(case, verifier, c, block, level, tdc, raw, curves, pinned):
    """Audit exact inputs/labels/counts and saved pulse arrays independently."""
    graph = read(case/'graph.json')
    arrays = load_archive(case/'case.npz')
    hashed_arrays(graph, arrays)
    meta = read(case/'probe-info.json')
    hashed_arrays(meta, arrays)
    weights, patterns, roles, observed, ids = independent_graph(c, level, block['input_seed'])
    np.testing.assert_array_equal(arrays['input_patterns'], patterns)
    np.testing.assert_array_equal(arrays['observed_indices'], observed)
    np.testing.assert_array_equal(arrays['root_ids'], ids)
    np.testing.assert_array_equal(arrays['roles'], roles)
    n = len(ids)
    assert (n, weights.nnz) == ((686, 3309) if level == 'legacy5' else (138639, 2700513))
    assert graph['weight_sha256'] == matrix_hash(weights)
    assert graph['decoder_parameters_per_lag'] == 490
    assert graph['direct_carry_multiplier'] == 0. and graph['drive_multiplier'] == .6
    assert graph['same_step_kc_to_mbon_retained'] is True and graph['synaptic_history'] is True
    assert np.max(np.asarray(abs(weights).sum(axis=1))) <= .9+1e-12
    assert observed.shape == (48,) and np.all(roles[observed] == 'MBON')
    assert [set(ids[value != 0].astype(str)) for value in patterns] == [set(value) for value in graph['input_root_ids']]
    assert ids[observed].astype(str).tolist() == graph['observation_root_ids']
    assert patterns.shape == (10, n) and set(np.unique(patterns)) == {0., .5}
    assert np.all(np.count_nonzero(patterns, axis=1) == 51)
    assert np.all(roles[np.any(patterns != 0, axis=0)] == 'KC')
    for name, digest in graph['graph_cache_hashes'].items():
        assert digest == pinned['outputs/tdc_v2/whole_cache/'+name]
    for name, source in graph['source'].items():
        assert source['sha256'] == pinned['outputs/tdc_v2/raw/'+name]
    anchors = c['warmup']+np.floor(np.linspace(0, c['test_samples']-1-20, 32)).astype(np.int64)
    np.testing.assert_array_equal(arrays['anchors'], anchors)
    assert len(np.unique(anchors)) == 32 and np.all(anchors+20 < c['warmup']+c['test_samples'])
    assert arrays['pre_states'].shape == arrays['replacement_final_states'].shape == (32, n)
    for split in ['train', 'test']:
        count = c[split+'_samples']
        symbols = np.random.default_rng(block[split+'_seed']).integers(0, 10, c['warmup']+count, dtype=np.int64)
        times = np.arange(c['warmup'], len(symbols))
        labels = symbols[times[:, None]-np.arange(21)[None, :]]
        for name, value in [('symbols', symbols), ('times', times)]:
            np.testing.assert_array_equal(arrays[split+'_'+name], value)
        np.testing.assert_array_equal(arrays['y'+split], labels)
        assert arrays[split+'_features'].shape == (len(symbols), 48)
        assert arrays[split+'_final_state'].shape == (n,)
        assert arrays[split+'_norms'].shape == (len(symbols),)
        assert re.fullmatch('[0-9a-f]{64}', graph[split+'_full_state_trajectory_sha256'])
    training = arrays['train_features'][c['warmup']:]
    close(arrays['mean'], training.mean(axis=0))
    close(arrays['scale'], np.maximum(training.std(axis=0), 1e-5))
    close(arrays['intercept'], np.eye(10)[arrays['ytrain']].reshape(c['train_samples'], -1).mean(axis=0))
    assert arrays['coefficients'].shape == (48, 210) and arrays['intercept'].shape == (210,)
    scores = (((arrays['test_features'][c['warmup']:]-arrays['mean'])/arrays['scale']) @ arrays['coefficients']+arrays['intercept']).reshape(c['test_samples'], 21, 10)
    close(arrays['scores'], scores)
    np.testing.assert_array_equal(arrays['predictions'], scores.argmax(axis=2))
    case_rows = tdc[(tdc.level == level) & (tdc.seed == block['seed'])].sort_values('lag')
    assert len(case_rows) == 21 and list(case_rows.lag) == list(range(21))
    assert read(case/'metrics.json') == case_rows.to_dict('records') or all(
        all(np.isclose(row[key], actual[key], atol=1e-12, rtol=1e-12) if isinstance(row[key], float)
            else row[key] == actual[key] for key in row)
        for row, actual in zip(read(case/'metrics.json'), case_rows.to_dict('records')))
    for lag, row in enumerate(case_rows.to_dict('records')):
        truth = arrays['ytest'][:, lag]
        train_truth = arrays['ytrain'][:, lag]
        majority = int(np.bincount(train_truth, minlength=10).argmax())
        table = np.zeros((10, 10), dtype=np.int64)
        np.add.at(table, (arrays['train_symbols'][c['warmup']:], train_truth), 1)
        lookup = np.asarray([value.argmax() if value.sum() else majority for value in table])
        np.testing.assert_array_equal(arrays['current_tables'][lag], table)
        np.testing.assert_array_equal(arrays['current_predictions'][:, lag], lookup[arrays['test_symbols'][c['warmup']:]])
        np.testing.assert_array_equal(arrays['frequency_predictions'][:, lag], np.full(c['test_samples'], majority))
        assert arrays['majority'][lag] == majority
        counts = {}
        for name, field in [('predictions', 'correct'), ('frequency_predictions', 'frequency_correct'), ('current_predictions', 'current_correct')]:
            counts[field] = int(np.count_nonzero(arrays[name][:, lag] == truth))
            assert row[field] == counts[field]
        count = c['test_samples']
        acc = Fraction(counts['correct'], count)
        freq = Fraction(counts['frequency_correct'], count)
        now = Fraction(counts['current_correct'], count)
        baseline = max(Fraction(1, 10), freq, now)
        for name, value in dict(samples=count, chance=.1, accuracy=float(acc),
                frequency_accuracy=float(freq), current_accuracy=float(now), baseline=float(baseline),
                baseline_excess=float(acc-baseline), chance_adjusted=float((acc-Fraction(1, 10))/Fraction(9, 10))).items():
            scalar_check(row[name], value)
    pulse = arrays['test_symbols'][anchors]
    alternative = (pulse+1) % 10
    np.testing.assert_array_equal(arrays['pulse_symbols'], pulse)
    np.testing.assert_array_equal(arrays['alternative_symbols'], alternative)
    directions = patterns[alternative]-patterns[pulse]
    for name in ['probe_observed_tangents', 'probe_observed_envelopes', 'probe_observed_replacement_differences',
                 'instantaneous_observed_tangents', 'instantaneous_observed_replacement_differences', 'contrast_support']:
        assert arrays[name].shape == (32, 21, 48)
    assert arrays['contrast_support'].dtype == np.dtype('bool')
    assert meta['technical_checks_pass'] is True and meta['finite_difference_checks_pass'] is True
    primary, denominators = exact_primary(arrays['probe_observed_tangents'], directions, meta['technical_checks_pass'])
    np.testing.assert_array_equal(arrays['contrast_input_squared_norm'], np.asarray(list(map(float, denominators))))
    for value, saved in zip(denominators, arrays['contrast_input_squared_norm']):
        assert value == Fraction.from_float(float(saved)), 'Saved denominator must be the exact contrast norm'
    cell_check(read(case/'primary-cell.json'), primary, level, block['seed'])
    bounds = .3*.54**np.arange(21)
    tolerance = 1e-12+1e-12*arrays['probe_observed_envelopes']
    assert np.all(arrays['probe_observed_envelopes'] >= 0)
    assert np.all(np.abs(arrays['probe_observed_tangents']) <= arrays['probe_observed_envelopes']+tolerance)
    for name in ['tangent_inf_norm', 'envelope_inf_norm', 'replacement_inf_norm']:
        assert arrays[name].shape == (32, 21) and np.all(arrays[name] >= 0)
    for name in ['tangent_inf_norm', 'replacement_inf_norm']:
        assert np.all(arrays[name] <= bounds+1e-12+1e-12*bounds)
    assert np.count_nonzero(arrays['instantaneous_observed_tangents'][:, 1:]) == 0
    assert np.count_nonzero(arrays['instantaneous_observed_replacement_differences'][:, 1:]) == 0
    assert np.count_nonzero(arrays['probe_observed_tangents'][~arrays['contrast_support']]) == 0
    assert np.count_nonzero(arrays['probe_observed_envelopes'][~arrays['contrast_support']]) == 0
    assert arrays['instantaneous_state_prototypes'].shape == (10, n)
    assert len(meta['probes']) == 32
    assert read(verifier/'probe-checks.json') == meta['probes']
    for index, record in enumerate(meta['probes']):
        assert (record['probe_index'], record['anchor'], record['symbol'], record['alternative_symbol']) == (index, int(anchors[index]), int(pulse[index]), int(alternative[index]))
        for name in ['envelope_bound_pass', 'contraction_bound_pass', 'instantaneous_zero_past_exact', 'instantaneous_prototype_exact']:
            assert record[name] is True
        for name in ['baseline_full_state_trajectory_sha256', 'replacement_full_state_trajectory_sha256',
                     'instantaneous_baseline_full_state_trajectory_sha256', 'instantaneous_replacement_full_state_trajectory_sha256']:
            assert re.fullmatch('[0-9a-f]{64}', record[name])
    independent = load_archive(verifier/'independent-probes.npz')
    np.testing.assert_array_equal(independent['anchors'], anchors)
    np.testing.assert_array_equal(independent['contrast_input_squared_norm'], arrays['contrast_input_squared_norm'])
    for name, source in [('complex_observed_tangents', 'probe_observed_tangents'),
                         ('independent_observed_envelopes', 'probe_observed_envelopes')]:
        close(independent[name], arrays[source], 1e-12, 1e-9 if name == 'complex_observed_tangents' else 1e-12)
    for name, source in [('independent_observed_replacement_differences', 'probe_observed_replacement_differences'),
                         ('independent_instantaneous_observed_replacement_differences', 'instantaneous_observed_replacement_differences')]:
        np.testing.assert_array_equal(independent[name], arrays[source])
    close(independent['independent_instantaneous_observed_tangents'], arrays['instantaneous_observed_tangents'], 1e-12, 1e-9)
    independent_cell, _ = exact_primary(independent['complex_observed_tangents'], directions)
    cell_check(read(verifier/'official-saved-float-cell.json'), primary, level, block['seed'])
    cell_check(read(verifier/'independent-complex-cell.json'), independent_cell, level, block['seed'])
    assert independent_cell['eligible'] is primary['eligible']
    assert independent_cell['scientific_rule_pass'] is primary['scientific_rule_pass'], 'No tolerance can reconcile exact decision disagreement'
    fd_checks = read(verifier/'finite-difference-checks.json')
    assert [value['probe_index'] for value in fd_checks] == [0, 10, 21, 31]
    for index, verification in zip([0, 10, 21, 31], fd_checks):
        fd = load_archive(case/f'fd-p{index:02d}.npz')
        fdmeta = read(case/f'fd-p{index:02d}.json')
        hashed_arrays(fdmeta, fd)
        assert fdmeta['all_checks_pass'] is True and fdmeta['probe_index'] == index and fdmeta['anchor'] == int(anchors[index])
        assert fd['full_tangent'].shape == fd['full_envelope'].shape == (21, n)
        assert fd['fd_observed'].shape == (2, 21, 48)
        assert fd['last_plus_states'].shape == fd['last_minus_states'].shape == (2, n)
        np.testing.assert_array_equal(fd['epsilon_values'], [1e-4, 1e-5])
        np.testing.assert_array_equal(fd['full_tangent'][:, observed], arrays['probe_observed_tangents'][index])
        np.testing.assert_array_equal(fd['full_envelope'][:, observed], arrays['probe_observed_envelopes'][index])
        close(np.abs(fd['full_tangent']).max(axis=1), arrays['tangent_inf_norm'][index])
        close(fd['full_envelope'].max(axis=1), arrays['envelope_inf_norm'][index])
        assert np.all(np.abs(fd['full_tangent']) <= fd['full_envelope']+1e-12+1e-12*fd['full_envelope'])
        assert fd['max_abs_error'].shape == fd['max_tolerance_excess'].shape == (2, 21)
        assert np.all(fd['max_abs_error'] >= 0) and np.all(fd['max_tolerance_excess'] <= 0)
        observed_full = fd['full_tangent'][:, observed][None, :, :]
        error = np.abs(fd['fd_observed']-observed_full)
        assert np.all(error <= 2e-9+2e-5*np.abs(observed_full))
        for key in ['plus_full_state_trajectory_sha256', 'minus_full_state_trajectory_sha256']:
            assert len(fdmeta[key]) == 2 and all(re.fullmatch('[0-9a-f]{64}', value) for value in fdmeta[key])
        assert (verification['full_coordinates'], verification['lags'], verification['epsilons']) == (n, 21, [1e-4, 1e-5])
        assert verification['full_complex_step_agreement'] is True and verification['full_fd_gate'] is True
        assert verification['exact_plus_trace_hashes'] == fdmeta['plus_full_state_trajectory_sha256']
        assert verification['exact_minus_trace_hashes'] == fdmeta['minus_full_state_trajectory_sha256']
        complex_full = load_archive(verifier/f'complex-p{index:02d}.npz')
        close(complex_full['full_complex_tangent'], fd['full_tangent'], 1e-12, 1e-9)
        close(complex_full['full_independent_envelope'], fd['full_envelope'])
        np.testing.assert_array_equal(complex_full['independently_recomputed_fd_observed'], fd['fd_observed'])
        close(complex_full['max_abs_error'], fd['max_abs_error'])
        close(complex_full['max_tolerance_excess'], fd['max_tolerance_excess'])
    case_raw = raw[(raw.level == level) & (raw.seed == block['seed'])].sort_values(['probe_index', 'lag'])
    assert len(case_raw) == 32*21 and not case_raw[['probe_index', 'lag']].duplicated().any()
    gains = np.linalg.norm(arrays['probe_observed_tangents'], axis=2)/np.sqrt(arrays['contrast_input_squared_norm'])[:, None]
    unsigned = np.linalg.norm(arrays['probe_observed_envelopes'], axis=2)/np.sqrt(arrays['contrast_input_squared_norm'])[:, None]
    finite = np.linalg.norm(arrays['probe_observed_replacement_differences'], axis=2)/np.sqrt(arrays['contrast_input_squared_norm'])[:, None]
    for row in case_raw.to_dict('records'):
        p, lag = int(row['probe_index']), int(row['lag'])
        g, u, f = gains[p, lag], unsigned[p, lag], finite[p, lag]
        expected = dict(anchor=int(anchors[p]), symbol=int(pulse[p]), alternative_symbol=int(alternative[p]),
            contrast_squared_norm=float(denominators[p]), gain=g, unsigned_gain=u, replacement_gain=f,
            signed_to_envelope=g/u if u else None, replacement_to_tangent=f/g if g else None,
            tangent_inf_norm=arrays['tangent_inf_norm'][p, lag], envelope_inf_norm=arrays['envelope_inf_norm'][p, lag],
            replacement_inf_norm=arrays['replacement_inf_norm'][p, lag], global_inf_bound=bounds[lag],
            structural_observed_count=int(arrays['contrast_support'][p, lag].sum()))
        for name, value in expected.items():
            scalar_check(row[name], value)
    pooled = np.sqrt(np.mean(gains**2, axis=0))
    pooled_unsigned = np.sqrt(np.mean(unsigned**2, axis=0))
    pooled_finite = np.sqrt(np.mean(finite**2, axis=0))
    case_curves = curves[(curves.level == level) & (curves.seed == block['seed'])]
    assert len(case_curves) == 21 and sorted(case_curves.lag) == list(range(21))
    for row in case_curves.to_dict('records'):
        lag = int(row['lag'])
        g, u, f = pooled[lag], pooled_unsigned[lag], pooled_finite[lag]
        for name, value in dict(gain=g, unsigned_gain=u, replacement_gain=f,
                relative_gain=g/pooled[0] if pooled[0] else None,
                relative_replacement_gain=f/pooled[0] if pooled[0] else None,
                signed_to_envelope=g/u if u else None, replacement_to_tangent=f/g if g else None,
                structural_observed_fraction=float(arrays['contrast_support'][:, lag].mean())).items():
            scalar_check(row[name], value)
    checks = read(verifier/'checks.json')
    assert checks['all_checks_pass'] is True
    assert checks['exact_full_state_trace_hash'] == {split: graph[split+'_full_state_trajectory_sha256'] for split in ['train', 'test']}
    assert checks['independently_refit_heads'] == 21 and checks['exact_prediction_arrays'] is True
    for key in ['independent_complex_cell_decision_equal', 'observed_complex_step_agreement',
                'full_coordinate_complex_step_agreement', 'full_coordinate_fd_gate',
                'unsigned_envelope_bound', 'contraction_bound', 'instantaneous_zero_history_exact']:
        assert checks[key] is True
    assert (checks['probes'], checks['lags'], checks['selected_full_state_complex_probes'], checks['full_state_fd_epsilons']) == (32, 21, 4, 2)
    assert checks['exact_replacement_and_instantaneous_full_state_hashes'] == 128
    assert checks['exact_centered_fd_full_state_hashes'] == 16
    return dict(level=level, seed=block['seed'], primary=primary, independently_reconstructed_primary=independent_cell,
        exact_full_state_trace_hash=checks['exact_full_state_trace_hash'], decoder_heads=21,
        observed_probe_vectors=32*21, selected_full_probe_vectors=4*21,
        full_fd_comparisons=4*2*21, input_mapping_sha256=graph['input_mapping_sha256'],
        input_root_ids=graph['input_root_ids'], observation_root_ids=graph['observation_root_ids'],
        paired_arrays={name: array_sha(arrays[name]) for name in ['train_symbols', 'test_symbols', 'ytrain', 'ytest', 'anchors']})


def statistics_check(frame, saved, metrics, c):
    assert len(saved) == 42 and not saved[['level', 'lag']].duplicated().any()
    for (level, lag), part in frame.groupby(['level', 'lag'], sort=True):
        part = part.sort_values('seed')
        row = saved[(saved.level == level) & (saved.lag == lag)]
        assert len(row) == 1
        actual = row.iloc[0]
        assert actual['n_blocks'] == len(part)
        for metric in metrics:
            values = part[metric].to_numpy(dtype=float)
            expected = estimates(values, c) if np.isfinite(values).all() else {
                key: None for key in ['mean', 'median', 'sd', 'bootstrap_lo', 'bootstrap_hi']}
            for key, value in expected.items():
                scalar_check(actual[metric+'_'+key], value)


def stage_replay(stage, validation, c, smoke, pinned):
    tdc = pd.read_csv(stage/'raw-lags.csv')
    raw = pd.read_csv(stage/'raw-sensitivity.csv')
    curves = pd.read_csv(stage/'block-sensitivity.csv')
    summary = read(stage/'summary.json')
    expected_cases = 2 if smoke else 9
    assert len(tdc) == len(curves) == expected_cases*21
    assert len(raw) == expected_cases*32*21
    for frame, keys in [(tdc, ['level', 'seed', 'lag']), (curves, ['level', 'seed', 'lag']),
                        (raw, ['level', 'seed', 'probe_index', 'lag'])]:
        assert not frame[keys].duplicated().any()
        assert set(frame.level) == set(c['levels'])
        assert len(frame[['level', 'seed']].drop_duplicates()) == expected_cases
    assert set(tdc.samples) == {c['test_samples']}
    for field in ['correct', 'frequency_correct', 'current_correct']:
        assert tdc[field].between(0, c['test_samples']).all()
        assert np.array_equal(tdc[field], tdc[field].astype(np.int64))
    cases, levels, paired, resources = [], {}, {}, {}
    saved_cells = read(stage/'primary-cells.json')
    assert len(saved_cells) == expected_cases
    assert [(cell['level'], cell['seed']) for cell in saved_cells] == sorted(
        (level, seed) for level in c['levels'] for seed in c['blocks_by_level'][level])
    for level in c['levels']:
        cells = []
        for seed in c['blocks_by_level'][level]:
            block = next(row for row in c['blocks'] if row['seed'] == seed)
            identity = f'{level}_s{seed}'
            case, verifier = stage/identity, validation/identity
            result = case_replay(case, verifier, c, block, level, tdc, raw, curves, pinned)
            for folder in [case, verifier]:
                resources[folder.as_posix()] = resource_check(folder, c)
            manifest = read(case/'manifest.json')
            assert manifest['complete'] is True and manifest['block'] == block and manifest['level'] == level
            assert read(verifier/'manifest.json')['result_case_manifest_sha256'] == sha(case/'manifest.json')
            primary = result['primary']
            saved_cell = next(row for row in saved_cells if row['level'] == level and row['seed'] == seed)
            cell_check(saved_cell, primary, level, seed)
            cells.append(primary)
            if seed in paired:
                for key in ['input_mapping_sha256', 'input_root_ids', 'observation_root_ids', 'paired_arrays']:
                    assert result[key] == paired[seed][key], (seed, key)
            else:
                paired[seed] = result
            cases.append(result)
        eligible = all(row['eligible'] for row in cells)
        rule = None if smoke or not eligible else all(row['scientific_rule_pass'] is True for row in cells)
        outcome = 'smoke' if smoke else 'assay-invalid' if not eligible else 'PASS' if rule else 'FAIL'
        actual = summary['levels'][level]
        assert actual['eligible'] is eligible and actual['registered_rule_pass'] is rule
        assert actual['outcome'] == outcome
        assert [row['seed'] for row in actual['cells']] == c['blocks_by_level'][level]
        for row, primary in zip(actual['cells'], cells):
            cell_check(row, primary, level, row['seed'])
        relative = [float(np.sqrt(float(Fraction(row['s5_numerator'], row['s5_denominator']) /
                                       Fraction(row['s0_numerator'], row['s0_denominator']))))
                    if row['s0_numerator'] else None for row in cells]
        if all(value is not None for value in relative):
            estimate_check(actual['relative_lag5_gain'], estimates(relative, c))
        else:
            assert actual['relative_lag5_gain'] is None
        levels[level] = dict(eligible=eligible, registered_rule_pass=rule, outcome=outcome,
                            seedwise_exact_cells=cells)
    primary = levels[c['primary_level']]
    assert summary['smoke'] is smoke
    assert summary['outcome'] == primary['outcome'] and summary['primary_pass'] is primary['registered_rule_pass']
    assert (summary['cases'], summary['lag_heads'], summary['probe_pairs'], summary['selected_full_fd_probes']) == (
        expected_cases, expected_cases*21, expected_cases*32, expected_cases*4)
    assert summary['criterion_unchanged'] is True
    for key in ['biological_plasticity_performed', 'unique_cycle_effect_claim', 'memory_loss_from_attenuation_claim']:
        assert summary[key] is False
    assert summary['m1_outcome_unchanged'] == 'assay-invalid' and summary['m4_outcome_unchanged'] == 'INFEASIBLE'
    statistics_check(tdc, pd.read_csv(stage/'tdc-summary.csv'),
        ['accuracy', 'chance_adjusted', 'baseline_excess', 'frequency_accuracy', 'current_accuracy'], c)
    statistics_check(curves, pd.read_csv(stage/'sensitivity-summary.csv'),
        ['gain', 'relative_gain', 'unsigned_gain', 'replacement_gain', 'signed_to_envelope',
         'replacement_to_tangent', 'relative_replacement_gain', 'structural_observed_fraction'], c)
    return dict(outcome=primary['outcome'], primary_pass=primary['registered_rule_pass'],
        cases=expected_cases, lag_heads=expected_cases*21, baseline_streams=expected_cases*2,
        observed_complex_probes=expected_cases*32, observed_complex_vectors=expected_cases*32*21,
        selected_full_probes=expected_cases*4, full_coordinate_fd_comparisons=expected_cases*4*2*21,
        levels=levels, case_evidence=cases, case_resources=resources)


def verifier_evidence(stage, validation, c, smoke, replay, cached, pinned):
    source = read(stage/'source.json')
    verifier_source = read(validation/'source.json')
    manifest = read(stage/'manifest.json')
    assert manifest['complete'] is True and manifest['source_commit'] == source['source_commit']
    assert manifest['smoke'] is smoke and manifest['config'] == c
    assert read(validation/'manifest.json')['complete'] is True
    checks = read(validation/'checks.json')
    assert checks['all_checks_pass'] is True
    assert checks['result_manifest_sha256'] == cached(stage/'manifest.json')
    assert checks['source_commit'] == verifier_source['source_commit']
    assert checks['protocol_commit'] == PREREGISTRATION
    assert checks['outcome'] == replay['outcome'] and checks['primary_pass'] is replay['primary_pass']
    counts = dict(cases=replay['cases'], replayed_streams=replay['baseline_streams'],
        exact_full_state_hashes=replay['baseline_streams'], independently_refit_lag_heads=replay['lag_heads'],
        observed_complex_step_probes=replay['observed_complex_probes'],
        observed_complex_step_vectors=replay['observed_complex_vectors'],
        selected_full_complex_step_probes=replay['selected_full_probes'],
        full_coordinate_fd_probe_epsilon_lag_checks=replay['full_coordinate_fd_comparisons'],
        exact_replacement_and_instantaneous_probe_trace_hashes=replay['cases']*128,
        exact_centered_fd_trace_hashes=replay['cases']*16)
    for key, value in counts.items():
        assert checks[key] == value, (validation, key)
    for key in ['exact_saved_float_primary_recalculated', 'independent_complex_primary_decisions_same',
                'independent_complex_step_no_main_jacobian_import', 'full_coordinate_finite_differences_both_epsilons',
                'instantaneous_past_influence_exact_zero', 'unsigned_envelope_structural_support_and_contraction_verified']:
        assert checks[key] is True
    for key in ['scientific_threshold_tolerance_applied', 'biological_plasticity_performed',
                'unique_cycle_effect_claim', 'memory_loss_from_attenuation_claim']:
        assert checks[key] is False
    assert checks['pairing'] == dict(paired_partial_whole_blocks=1 if smoke else 3,
        streams_mapping_observed_ids_and_budget_paired=True)
    validation_source = read(validation/'source-validation.json')
    assert validation_source['result_source_commit'] == source['source_commit']
    assert validation_source['verifier_source_commit'] == verifier_source['source_commit']
    assert validation_source['protocol_commit'] == PREREGISTRATION
    assert validation_source['recorded_source_hashes_authenticated'] == len(source['hashes'])
    assert validation_source['independent_complex_step_no_main_jacobian_import'] is True
    raw_source = read(validation/'source-graph-audit.json')
    assert checks['source_graph_reconstruction'] == raw_source
    assert (raw_source['neurons'], raw_source['edges']) == (138639, 2700513)
    for key in ['full_graph_rebuilt_from_pinned_parquet', 'roles_rebuilt_from_pinned_annotations', 'partial_induced_graph_exact']:
        assert raw_source[key] is True
    for name, digest in raw_source['raw_source_sha256'].items():
        assert digest == pinned['outputs/tdc_v2/raw/'+name]
    required = {'scripts/sensitivity_support.py', 'scripts/sensitivity_dynamics.py',
        'scripts/temporal_functional_sensitivity.py', 'scripts/verify_temporal_functional_sensitivity.py',
        'scripts/audit_temporal_functional_sensitivity.py', 'scripts/run_sensitivity_guards.py',
        'tests/test_temporal_functional_sensitivity.py', 'tests/test_functional_sensitivity_verifier.py', CONFIG, PROTOCOL}
    assert required <= set(source['hashes']) and required <= set(verifier_source['hashes'])
    assert all(source['hashes'][name] == verifier_source['hashes'][name] for name in required), 'Main/verifier scientific source differs'
    assert all(source['hashes'][name] == cached(name) for name in required), 'Frozen M5 source changed after execution'
    pd.testing.assert_frame_equal(pd.read_csv(stage/'raw-lags.csv'),
        pd.read_csv(validation/'recomputed-raw-lags.csv'), check_dtype=False, check_exact=False, atol=1e-12, rtol=1e-12)
    for name in ['raw-sensitivity', 'block-sensitivity', 'tdc-summary', 'sensitivity-summary']:
        pd.testing.assert_frame_equal(pd.read_csv(stage/(name+'.csv')),
            pd.read_csv(validation/('recomputed-'+name+'.csv')), check_dtype=False, check_exact=False, atol=1e-12, rtol=1e-12)
    actual_summary, rebuilt_summary = read(stage/'summary.json'), read(validation/'recomputed-summary.json')
    assert actual_summary == rebuilt_summary
    return checks


def audit(out, main_validation='main_validation', smoke_validation='smoke_validation', guard_stage='guard_tests'):
    root = NAMESPACE
    c = read(root/'main/config.json')
    fixed_config(c)
    assert root.resolve() in out.resolve().parents and not out.exists()
    with attempt(out, c, 'm5-functional-sensitivity-closeout'):
        preservation = history_preserved(c)
        assert preservation['prior_result_identities_preserved'] == 44147
        assert preservation['protected_baseline_paths'] == 44784
        old = read(root/'baseline_audit/baseline-git-blobs.json')
        assert old == tree(BASELINE), 'Baseline inventory must equal the exact historical Git tree'
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
            replay[stage] = stage_replay(folder, validation, saved_config, smoke, pinned)
            verifier_checks[stage] = verifier_evidence(folder, validation, saved_config, smoke, replay[stage], cached, pinned)
            for path in [folder, validation]:
                usage[path.as_posix()] = resource_check(path, c)
                pool = read(path/'process-tree-resources.json')
                assert pool['seconds'] <= 7200 and pool['peak_sampled_process_tree_rss_bytes'] <= 4294967296
                assert pool['sample_seconds'] == .05
                jobs = read(path/'job-resources.json')
                assert len(jobs) == replay[stage]['cases']
                assert {row['identity'] for row in jobs} == {f'{case["level"]}/s{case["seed"]}' for case in replay[stage]['case_evidence']}
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
        assert all(guard_source['hashes'][name] == main_source['hashes'][name]
                   for name in main_source['hashes']), 'Guard/source freeze changed before main'
        required_tests = {'tests/test_temporal_functional_sensitivity.py', 'tests/test_functional_sensitivity_verifier.py',
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
        write(out/'exact-primary-replay.json', replay)
        write(out/'stage-verifier-checks.json', verifier_checks)
        write(out/'stage-resources.json', usage)
        write(out/'document-links.json', links)
        write(out/'document-sha256.json', documents)
        write(out/'audit.json', {**preservation, 'all_checks_pass': True,
            'source_commit': git('rev-parse', 'HEAD'), 'audit_source_sha256': sha(__file__),
            'protocol_commit': PREREGISTRATION, 'protocol_precedes_recorded_main_source': True,
            'recorded_source_entries': len(recorded_sources), 'manifests_checked': len(manifests),
            'checksum_entries': sum(row['artifacts'] for row in manifests), 'attempts_checked': len(attempts),
            'failure_registry_sha256': {path.as_posix(): cached(path) for path in
                [root/'known-incomplete-attempts.json', root/'validation-failures.json'] if path.exists()},
            'retained_incomplete_attempts': failures, 'retained_validation_failures': validation_failures,
            'old_m4_source_validation_failure_and_snapshot_preserved': True,
            'guard_stage': guard_stage, 'guards': guards,
            'archives_inspected': archives, 'arrays_inspected': arrays, 'nonfinite_arrays': 0,
            'exact_saved_float_primary_recalculated': True, 'scientific_threshold_tolerance_applied': False,
            'normalization_resolution_power_threshold_exact': str(MINIMUM_POWER),
            'primary_power_ratio_threshold_exact': str(Fraction(1, 100)),
            'every_primary_block_required': True,
            'outcome': replay['main']['outcome'], 'primary_pass': replay['main']['primary_pass'],
            'cases': replay['main']['cases'], 'exact_baseline_full_state_hashes': replay['main']['baseline_streams'],
            'independently_refit_lag_heads': replay['main']['lag_heads'],
            'observed_complex_step_probes': replay['main']['observed_complex_probes'],
            'observed_complex_step_vectors': replay['main']['observed_complex_vectors'],
            'selected_full_complex_step_probes': replay['main']['selected_full_probes'],
            'full_coordinate_fd_probe_epsilon_lag_checks': replay['main']['full_coordinate_fd_comparisons'],
            'smoke_scientific_endpoint': None,
            'probe_replicates_are_independent_animals': False,
            'scientific_settings_and_outcomes_unchanged': True,
            'biological_plasticity_performed': False, 'unique_cycle_effect_claim': False,
            'memory_loss_from_attenuation_claim': False})
    seal(out, dict(complete=True, kind='m5-functional-sensitivity-closeout', source_commit=git('rev-parse', 'HEAD')))
    print(read(out/'audit.json'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--main-validation', default='main_validation')
    parser.add_argument('--smoke-validation', default='smoke_validation')
    parser.add_argument('--guard-stage', default='guard_tests')
    args = parser.parse_args()
    audit(args.out, args.main_validation, args.smoke_validation, args.guard_stage)
