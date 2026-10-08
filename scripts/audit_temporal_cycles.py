"""Independent final artifact, rational endpoint and provenance audit for M3.

This closeout does not import experimental dynamics, graph construction,
decoder fitting or summary code. Neural replay belongs to the separate verifier;
this audit authenticates its evidence and independently rechecks saved outcomes.
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
from scipy import sparse

from cycle_support import read, write, sha, array_sha, attempt, seal, git, history_preserved


BASELINE = 'eb571bd83107e2b108bf270003a918f8eb21e797'
PREREGISTRATION = '438cae437e110dbeb9cb2a02d0a58cfe61e6d233'
NAMESPACE = Path('results/tdc_cycles_v1')
CONFIG = 'configs/temporal_cycles.json'
OVERVIEWS = {'README.md', 'docs/research-status.md',
             'docs/research-roadmap.md', 'docs/next-work.md'}


def tree(revision):
    inventory = {}
    for line in git('ls-tree', '-r', revision).splitlines():
        meta, name = line.split('\t', 1)
        inventory[name] = meta.split()[2]
    return inventory


def fixed_config(c, smoke=False):
    """Literal registered values, independently asserted without runner imports."""
    assert c['baseline_commit'] == BASELINE
    assert c['namespace'] == NAMESPACE.as_posix()
    assert c['protocol'] == 'docs/temporal-cycles-protocol.md'
    assert c['levels'] == ['legacy5', 'brain5'] and c['circuit_seeds'] == [701]
    assert c['arms'] == {
        'intact_synaptic': {'carry': False, 'synaptic_history': True, 'dag': False},
        'dag_synaptic': {'carry': False, 'synaptic_history': True, 'dag': True},
        'instantaneous': {'carry': False, 'synaptic_history': False, 'dag': False}}
    for name, start in [('seed', 1300001), ('input_seed', 1310001),
                        ('train_seed', 1320001), ('test_seed', 1330001),
                        ('initial_a_seed', 1350001), ('initial_b_seed', 1350101),
                        ('prefix_a_seed', 1360001), ('prefix_b_seed', 1360101),
                        ('suffix_seed', 1370001)]:
        assert [block[name] for block in c['blocks']] == list(range(start, start + 6))
    expected_blocks = {'legacy5': list(range(1300001, 1300007)),
                       'brain5': list(range(1300001, 1300004))}
    if smoke:
        expected_blocks = {name: seeds[:1] for name, seeds in expected_blocks.items()}
    assert c['blocks_by_level'] == expected_blocks
    assert c['lags'] == list(range(21)) and c['primary_level'] == 'legacy5'
    assert (c['primary_arm'], c['primary_lag'], c['minimum_excess']) == ('dag_synaptic', 2, .1)
    assert (c['alphabet_size'], c['warmup'], c['train_samples'], c['test_samples']) == (
        10, 200, 300 if smoke else 4000, 200 if smoke else 2000)
    assert (c['gain'], c['leak'], c['alpha'], c['input_fraction'],
            c['input_amplitude'], c['carry_multiplier']) == (.9, .6, 1., .1, .5, 0.)
    assert c['normalization'] == 'incoming_l1' and c['schedule'] == 'mbon_after_kc'
    assert c['reference_current_min'] == .99
    assert c['mask_order'] == 'KC/other/MBON tiers, ascending numeric root ID'
    assert c['expected_depth'] == {'legacy5': 11, 'brain5': 153}
    assert c['certificate_prefix_steps'] == 64
    assert c['bootstrap_seed'] == 1340001 and c['bootstrap_draws'] == 10000
    assert c['neural_workers'] == 4
    assert c['max_seconds'] == 7200 and c['max_rss_bytes'] == 4294967296
    registered = read(CONFIG)
    expected = dict(registered)
    if smoke:
        expected.update(train_samples=300, test_samples=200, blocks_by_level=expected_blocks)
    assert c == expected, 'All saved configuration values must match registration'


def frac_accuracy(row, column='correct'):
    return Fraction(int(row[column]), int(row['samples']))


def baseline_excess(row):
    return frac_accuracy(row) - max(Fraction(1, 10),
                                    frac_accuracy(row, 'frequency_correct'),
                                    frac_accuracy(row, 'current_correct'))


def estimates(values, c):
    """Independent descriptive block bootstrap; never an inferential animal CI."""
    values = np.asarray(values, dtype=float)
    assert values.ndim == 1 and len(values)
    resamples = np.random.default_rng(c['bootstrap_seed']).integers(
        0, len(values), size=(c['bootstrap_draws'], len(values)))
    bounds = np.percentile(values[resamples].mean(axis=1), [2.5, 97.5])
    return {'mean': float(values.mean()), 'median': float(np.median(values)),
            'sd': float(values.std(ddof=1)) if len(values) > 1 else 0.,
            'bootstrap_lo': float(bounds[0]), 'bootstrap_hi': float(bounds[1])}


def assert_estimate(actual, expected):
    for name, value in expected.items():
        assert abs(actual[name] - value) <= 1e-12, (name, actual[name], value)


def criterion_replay(frame, c, summary, smoke=False):
    """Recompute lag2 excess/gates from integer held-out counts, no float gate."""
    replay = {}
    for level, seeds in c['blocks_by_level'].items():
        rows = {}
        long_scores = {arm: [] for arm in c['arms']}
        for seed in seeds:
            rows[seed] = {}
            for arm in c['arms']:
                part = frame[(frame.level == level) & (frame.seed == seed) & (frame.arm == arm)]
                assert len(part) == 21 and sorted(part.lag) == list(range(21))
                rows[seed][arm] = {int(row['lag']): row for row in part.to_dict('records')}
                mean_accuracy = sum((frac_accuracy(rows[seed][arm][lag])
                                     for lag in range(1, 21)), Fraction(0)) / 20
                long_scores[arm].append(float((mean_accuracy - Fraction(1, 10)) / Fraction(9, 10)))
        excess = [baseline_excess(rows[seed]['dag_synaptic'][2]) for seed in seeds]
        current = [frac_accuracy(rows[seed]['instantaneous'][0]) for seed in seeds]
        reference_valid = all(value >= Fraction(99, 100) for value in current)
        all_meet = all(value >= Fraction(1, 10) for value in excess)
        registered_rule_pass = None if smoke or not reference_valid else all_meet
        actual = summary['levels'][level]
        assert actual['reference_valid'] is reference_valid
        primary = actual['primary']
        assert (primary['lag'], primary['arm']) == (2, 'dag_synaptic')
        assert_estimate(primary['excess'], estimates(list(map(float, excess)), c))
        mean_excess = sum(excess, Fraction(0)) / len(excess)
        for name in ['excess_mean_exact', 'mean_exact']:
            if name in primary:
                assert primary[name] == str(mean_excess)
        assert primary['minimum_excess_exact'] == str(min(excess))
        assert abs(primary['minimum_excess'] - float(min(excess))) <= 1e-12
        assert primary['all_blocks_meet_margin'] is all_meet
        assert primary['registered_rule_pass'] is registered_rule_pass
        for arm in c['arms']:
            assert_estimate(actual['arm_long_scores'][arm], estimates(long_scores[arm], c))
        replay[level] = {'reference_valid': reference_valid,
                         'minimum_excess_exact': str(min(excess)),
                         'mean_excess_exact': str(mean_excess),
                         'all_blocks_meet_margin': all_meet,
                         'registered_rule_pass': registered_rule_pass,
                         'paired_values': [{'seed': seed, 'excess_exact': str(value),
                                            'instantaneous_current_exact': str(cur)}
                                           for seed, value, cur in zip(seeds, excess, current)]}
    primary = replay[c['primary_level']]
    expected_outcome = ('smoke' if smoke else 'assay-invalid' if not primary['reference_valid']
                        else 'PASS' if primary['registered_rule_pass'] else 'FAIL')
    assert summary['smoke'] is smoke
    assert summary['primary_pass'] is primary['registered_rule_pass']
    assert summary['outcome'] == expected_outcome
    assert summary['criterion_unchanged'] is True
    assert summary['m1_outcome_unchanged'] == 'assay-invalid'
    assert summary['m2_outcome_unchanged'] == 'PASS'
    assert summary['cycle_effect_isolation_claim'] is False
    return replay


def structural_checks(root, cached):
    """Check fixed DAG ranks, signed CSR hashes, degree arrays and delay bounds."""
    folder = root / 'structural_audit'
    checks = read(folder / 'checks.json')
    assert checks['all_checks_pass'] is True
    assert checks['new_neural_outcomes_generated'] is False
    assert checks['source_commit'] == BASELINE
    assert cached(folder / 'cycle-structure-source.py.txt') == checks['helper_sha256']
    # The prospective commit must contain exactly this structure-only helper.
    inventory = tree(PREREGISTRATION)
    helper_blob = subprocess.check_output(['git', 'cat-file', 'blob', inventory['scripts/cycle_structure.py']])
    assert hashlib.sha256(helper_blob).hexdigest() == checks['helper_sha256']
    results = {}
    for level, counts in {'legacy5': (686, 3309, 2500, 1474, 11),
                          'brain5': (138639, 2700513, 1353524, 21438, 153)}.items():
        record = read(folder / (level + '.json'))
        assert tuple(record[name] for name in ['nodes', 'original_edges', 'dag_edges',
                                              'protected_kc_to_mbon_edges', 'global_dependency_depth']) == counts
        assert record['dag_cycle_edges'] == 0
        assert record['strict_rank_increase_on_every_retained_edge'] is True
        assert record['protected_kc_to_mbon_all_kept_exactly'] is True
        assert record['dag_scc']['components'] == counts[0]
        assert record['dag_scc']['nontrivial_components'] == record['dag_scc']['self_edges'] == 0
        assert record['normalization'] == 'original incoming-L1 normalized weights masked without rescaling'
        with np.load(folder / (level + '-dag.npz'), allow_pickle=False) as stored:
            dag = sparse.csr_matrix((stored['data'], stored['indices'], stored['indptr']),
                                    shape=tuple(stored['shape']))
        assert dag.shape == (counts[0], counts[0]) and dag.nnz == counts[2]
        assert dag.has_sorted_indices and dag.has_canonical_format
        dag_hash = hashlib.sha256(dag.data.tobytes() + dag.indices.tobytes() + dag.indptr.tobytes()).hexdigest()
        assert dag_hash == record['dag_weight_sha256']
        with np.load(folder / (level + '-per-neuron.npz'), allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        ids, roles, rank = arrays['root_ids'], arrays['roles'], arrays['rank']
        assert ids.shape == roles.shape == rank.shape == (counts[0],)
        assert len(np.unique(ids)) == len(ids)
        groups = np.where(roles == 'KC', 0, np.where(roles == 'MBON', 2, 1))
        order = np.lexsort((ids, groups))
        expected_rank = np.empty(len(ids), dtype=np.int64)
        expected_rank[order] = np.arange(len(ids))
        np.testing.assert_array_equal(rank, expected_rank)
        target = np.repeat(np.arange(len(ids)), np.diff(dag.indptr))
        assert np.all(rank[dag.indices] < rank[target])
        protected = (roles[dag.indices] == 'KC') & (roles[target] == 'MBON')
        assert int(protected.sum()) == counts[3]
        np.testing.assert_array_equal(arrays['dag_indegree'], np.diff(dag.indptr))
        np.testing.assert_array_equal(arrays['dag_outdegree'], np.bincount(dag.indices, minlength=len(ids)))
        np.testing.assert_allclose(arrays['dag_incoming_abs_weight'],
                                   np.bincount(target, weights=np.abs(dag.data), minlength=len(ids)), atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(arrays['dag_outgoing_abs_weight'],
                                   np.bincount(dag.indices, weights=np.abs(dag.data), minlength=len(ids)), atol=1e-12, rtol=1e-12)
        assert int(arrays['original_indegree'].sum()) == counts[1]
        assert int(arrays['original_outdegree'].sum()) == counts[1]
        observed = arrays['observed_indices']
        assert observed.shape == (48,) and len(np.unique(observed)) == 48
        assert np.all(roles[observed] == 'MBON')
        outgoing = dag.T.tocsr()
        depth = np.zeros(len(ids), dtype=np.int64)
        input_depth = np.full(len(ids), -1, dtype=np.int64)
        input_depth[roles == 'KC'] = 0
        for pre in order:
            post = outgoing.indices[outgoing.indptr[pre]:outgoing.indptr[pre + 1]]
            delays = np.where((roles[pre] == 'KC') & (roles[post] == 'MBON'), 0, 1)
            depth[post] = np.maximum(depth[post], depth[pre] + delays)
            if input_depth[pre] >= 0:
                input_depth[post] = np.maximum(input_depth[post], input_depth[pre] + delays)
        np.testing.assert_array_equal(depth, arrays['global_dependency_depth'])
        np.testing.assert_array_equal(input_depth, arrays['input_pool_dependency_depth'])
        assert int(depth.max()) == counts[4]
        results[level] = dict(record, rank_sha256=array_sha(rank),
                             structure_only_manifest_sha256=cached(folder / 'manifest.json'))
    return results


def predictions_replay(stage, c, frame, structures, smoke=False):
    """Recheck raw held-out counts, split labels, mapping and arm graph linkage."""
    cases = streams = heads = certificates = 0
    pairing = {}
    cross_level = {}
    for level, seeds in c['blocks_by_level'].items():
        for seed in seeds:
            block = next(block for block in c['blocks'] if block['seed'] == seed)
            for arm in c['arms']:
                case = stage / f'{level}_{arm}_s{seed}'
                graph = read(case / 'graph.json')
                assert graph['decoder_parameters_per_lag'] == 490
                selected_hash = structures[level]['dag_weight_sha256' if arm == 'dag_synaptic' else 'original_weight_sha256']
                assert graph['weight_sha256'] == selected_hash
                assert graph['original_weight_sha256'] == structures[level]['original_weight_sha256']
                assert graph['direct_carry_multiplier'] == 0. and graph['drive_multiplier'] == .6
                assert graph['same_step_kc_to_mbon_retained'] is True
                assert graph['switches'] == c['arms'][arm]
                assert graph['zero_state_all_arm_response_equal'] is True
                pinned = read(NAMESPACE / 'baseline_audit/checks.json')['pinned_hashes']
                for name, expected in graph['graph_cache_hashes'].items():
                    assert expected == pinned['outputs/tdc_v2/whole_cache/' + name]
                for name, source in graph['source'].items():
                    assert source['sha256'] == pinned['outputs/tdc_v2/raw/' + name]
                with np.load(case / 'case.npz', allow_pickle=False) as data:
                    cases += 1
                    assert set(data.files) == set(graph['array_hashes'])
                    for name in data.files:
                        assert array_sha(data[name]) == graph['array_hashes'][name], (case, name)
                    observed = data['observed_indices']
                    assert observed.shape == (48,)
                    common = {name: array_sha(data[name]) for name in [
                        'input_patterns', 'observed_indices', 'zero_state_observed_prototypes']}
                    for split in ['train', 'test']:
                        count = c[split + '_samples']
                        symbols = np.random.default_rng(block[split + '_seed']).integers(
                            0, 10, c['warmup'] + count, dtype=np.int64)
                        np.testing.assert_array_equal(data[split + '_symbols'], symbols)
                        times = np.arange(c['warmup'], len(symbols))
                        np.testing.assert_array_equal(data[split + '_times'], times)
                        labels = np.column_stack([symbols[times - lag] for lag in range(21)])
                        np.testing.assert_array_equal(data['y' + split], labels)
                        assert data[split + '_features'].shape == (len(symbols), 48)
                        common[split + '_symbols'] = array_sha(symbols)
                        streams += 1
                    key = (level, seed)
                    if key in pairing:
                        assert common == pairing[key], (key, arm)
                    else:
                        pairing[key] = common
                    count = c['test_samples']
                    assert data['predictions'].shape == data['ytest'].shape == (count, 21)
                    assert data['scores'].shape == (count, 21, 10)
                    np.testing.assert_array_equal(data['predictions'], data['scores'].argmax(axis=2))
                    assert ((data['predictions'] >= 0) & (data['predictions'] < 10)).all()
                    for lag in range(21):
                        part = frame[(frame.level == level) & (frame.arm == arm)
                                     & (frame.seed == seed) & (frame.lag == lag)]
                        assert len(part) == 1
                        row = part.iloc[0]
                        truth = data['ytest'][:, lag]
                        for column, array in [('correct', 'predictions'),
                                              ('frequency_correct', 'frequency_predictions'),
                                              ('current_correct', 'current_predictions')]:
                            assert int(row[column]) == int(np.count_nonzero(data[array][:, lag] == truth))
                        for column, expected in [('accuracy', frac_accuracy(row)),
                                                 ('frequency_accuracy', frac_accuracy(row, 'frequency_correct')),
                                                 ('current_accuracy', frac_accuracy(row, 'current_correct')),
                                                 ('baseline_excess', baseline_excess(row)),
                                                 ('chance_adjusted', (frac_accuracy(row) - Fraction(1, 10)) / Fraction(9, 10))]:
                            assert abs(float(row[column]) - float(expected)) <= 1e-12, (case, lag, column)
                        heads += 1
                    if arm == 'dag_synaptic':
                        assert array_sha(data['dag_rank']) == structures[level]['rank_sha256']
                roots = (graph['input_root_ids'], graph['observation_root_ids'])
                if seed in cross_level:
                    assert roots == cross_level[seed], (seed, level, arm)
                else:
                    cross_level[seed] = roots
                if arm == 'dag_synaptic':
                    certificate = read(case / 'certificate.json')
                    assert certificate['all_checks_pass'] is True
                    horizon = structures[level]['global_dependency_depth']
                    assert certificate['global_dependency_depth'] == horizon
                    assert certificate['common_suffix_steps'] == horizon + 1
                    assert certificate['prefix_steps'] == 64
                    assert certificate['distinct_prefix_states'] is True
                    assert certificate['exact_terminal_equality'] is True
                    for branch in ['a', 'b']:
                        assert re.fullmatch(r'[0-9a-f]{64}', certificate[branch + '_full_state_trajectory_sha256'])
                    with np.load(case / 'certificate.npz', allow_pickle=False) as data:
                        assert set(data.files) == set(certificate['array_hashes'])
                        np.testing.assert_array_equal(data['final_a_state'], data['final_b_state'])
                        assert data['final_a_state'].shape == (structures[level]['nodes'],)
                        assert not np.array_equal(data['initial_a'], data['initial_b'])
                        assert not np.array_equal(data['prefix_a_final_state'], data['prefix_b_final_state'])
                        for branch in ['a', 'b']:
                            np.testing.assert_array_equal(data['initial_' + branch],
                                np.random.default_rng(block['initial_' + branch + '_seed']).standard_normal(structures[level]['nodes']))
                            np.testing.assert_array_equal(data['prefix_' + branch + '_symbols'],
                                np.random.default_rng(block['prefix_' + branch + '_seed']).integers(0, 10, 64, dtype=np.int64))
                        np.testing.assert_array_equal(data['suffix_symbols'],
                            np.random.default_rng(block['suffix_seed']).integers(0, 10, horizon + 1, dtype=np.int64))
                        # Every explicitly reported array identity must match the preserved data.
                        for name, expected in certificate['array_hashes'].items():
                            assert array_sha(data[name]) == expected, (case, name)
                    certificates += 1
    expected = (6, 12, 126, 2) if smoke else (27, 54, 567, 9)
    assert (cases, streams, heads, certificates) == expected
    return {'cases': cases, 'saved_streams_checked': streams,
            'prediction_columns_checked': heads, 'certificate_pairs_checked': certificates,
            'split_safe_labels': True, 'paired_arm_streams_and_inputs': True,
            'fixed_decoder_parameter_budget': 490}


def frame_checks(frame, c, smoke):
    expected_cases, expected_rows = (6, 126) if smoke else (27, 567)
    assert len(frame) == expected_rows
    assert len(frame[['level', 'arm', 'seed']].drop_duplicates()) == expected_cases
    assert not frame[['level', 'arm', 'seed', 'lag']].duplicated().any()
    assert set(frame.samples) == {c['test_samples']}
    assert set(frame.level) == set(c['levels']) and set(frame.arm) == set(c['arms'])
    for name in ['correct', 'frequency_correct', 'current_correct']:
        assert frame[name].between(0, c['test_samples']).all()
        assert (frame[name].astype(np.int64) == frame[name]).all()


def audit(out, main_validation='main_validation', smoke_validation='smoke_validation'):
    root = NAMESPACE
    c = read(root / 'main/config.json')
    fixed_config(c)
    with attempt(out, c, 'cycles-closeout-audit'):
        preservation = history_preserved(c)
        baseline = read(root / 'baseline_audit/baseline-git-blobs.json')
        assert baseline == tree(BASELINE), 'Baseline inventory must authenticate its full Git tree'
        inventory = tree('HEAD')
        protected = {name: blob for name, blob in baseline.items()
                     if name.startswith(('results/', 'src/', 'data/', 'configs/',
                                         'scripts/', 'tests/', 'docs/')) and name not in OVERVIEWS}
        assert all(inventory.get(name) == blob for name, blob in protected.items())
        assert sum(name.startswith('results/') for name in protected) == 43495
        assert not set(git('diff', '--name-only', 'HEAD').splitlines()).intersection(protected)
        digests = {}

        def cached(path):
            path = Path(path).resolve()
            if path not in digests:
                digests[path] = sha(path)
            return digests[path]

        manifests, incomplete = [], []
        for path in sorted(root.rglob('manifest.json')):
            if out.resolve() in path.resolve().parents:
                continue
            manifest = read(path)
            assert isinstance(manifest['artifacts'], dict)
            for name, expected in manifest['artifacts'].items():
                target = (path.parent / name).resolve()
                assert path.parent.resolve() in target.parents, (path, name)
                assert cached(target) == expected, (path, name)
            complete = manifest.get('complete', True)
            assert isinstance(complete, bool)
            manifests.append({'path': path.as_posix(), 'sha256': cached(path),
                              'artifacts': len(manifest['artifacts']), 'complete': complete})
            if not complete:
                incomplete.append(path.as_posix())
        registry_path = root / 'known-incomplete-attempts.json'
        registry = read(registry_path) if registry_path.exists() else []
        assert isinstance(registry, list)
        assert sorted(entry['manifest_path'] for entry in registry) == sorted(incomplete), 'Unregistered failed attempt'
        authenticated_failures = []
        for entry in registry:
            assert isinstance(entry['reason'], str) and entry['reason'].strip()
            failure_path = Path(entry['failure_path'])
            assert root.resolve() in failure_path.resolve().parents and failure_path.is_file()
            failure = read(failure_path)
            assert failure.get('complete') is False and failure.get('message')
            authenticated_failures.append(dict(entry, failure_sha256=cached(failure_path)))
        recorded, trees, blobs = [], {}, {}
        pinned = read(root / 'baseline_audit/checks.json')['pinned_hashes']
        for path in sorted(root.rglob('source.json')):
            if out.resolve() in path.resolve().parents:
                continue
            source = read(path)
            revision = source['source_commit']
            if revision not in trees:
                trees[revision] = tree(revision)
            assert source['source_tree'] == git('rev-parse', revision + '^{tree}')
            assert source['tracked_changes'] is False
            for name, expected in source['hashes'].items():
                if name in trees[revision]:
                    blob = trees[revision][name]
                    if blob not in blobs:
                        blobs[blob] = hashlib.sha256(subprocess.check_output(
                            ['git', 'cat-file', 'blob', blob])).hexdigest()
                    assert blobs[blob] == expected, (path, name, revision)
                    mode = 'recorded_git_blob'
                else:
                    assert name in pinned and name.startswith('outputs/tdc_v2/'), (path, name)
                    assert cached(name) == expected == pinned[name], (path, name)
                    mode = 'authenticated_pinned_external_bytes'
                recorded.append({'record': path.as_posix(), 'source_commit': revision,
                                 'path': name, 'sha256': expected, 'authentication': mode})
        assert recorded, 'Missing authenticated numerical input sources'
        structures = structural_checks(root, cached)
        criterion, predictions, validations = {}, {}, {}
        for stage, validation, counts in [('smoke', smoke_validation, (6, 12, 126, 2)),
                                          ('main', main_validation, (27, 54, 567, 9))]:
            smoke = stage == 'smoke'
            stage_root = root / stage
            saved_config = read(stage_root / 'config.json')
            fixed_config(saved_config, smoke)
            summary = read(stage_root / 'summary.json')
            checks = read(root / validation / 'checks.json')
            assert checks['all_checks_pass'] is True
            head_count = checks.get('independent_refit_lag_heads', checks.get('independently_refit_lag_heads'))
            pair_count = checks.get('certificate_pairs', checks.get('finite_window_certificate_pairs'))
            certificate_hashes = checks.get('certificate_full_state_hashes', checks.get('exact_certificate_trace_hashes'))
            assert (checks['cases'], checks['replayed_streams'], head_count, pair_count) == counts
            assert certificate_hashes == counts[3] * 2
            assert checks['instantaneous_certificates'] == counts[3]
            assert checks['zero_state_input_access_certificates'] == counts[0]
            assert checks['exact_full_state_trace_hashes'] == counts[1]
            assert checks['result_manifest_sha256'] == cached(stage_root / 'manifest.json')
            assert checks['outcome'] == summary['outcome']
            assert checks['primary_pass'] is summary['primary_pass']
            assert (summary['cases'], summary['lag_heads'], summary['certificate_pairs']) == (counts[0], counts[2], counts[3])
            source = read(stage_root / 'source.json')
            assert source['source_commit'] == read(stage_root / 'manifest.json')['source_commit']
            required_sources = ['scripts/temporal_cycles.py', 'scripts/verify_temporal_cycles.py',
                                'scripts/cycle_support.py', 'scripts/cycle_structure.py',
                                'scripts/temporal_mechanism.py', 'scripts/verify_temporal_mechanism.py',
                                'scripts/temporal_memory_curve.py', 'scripts/verify_temporal_memory_curve.py',
                                'scripts/tdc_support.py', 'scripts/tdc_pool.py', CONFIG, saved_config['protocol']]
            assert all(name in source['hashes'] for name in required_sources)
            assert subprocess.run(['git', 'merge-base', '--is-ancestor', PREREGISTRATION,
                                   source['source_commit']], check=False).returncode == 0
            for name in [saved_config['protocol'], CONFIG]:
                assert tree(PREREGISTRATION)[name] == tree(source['source_commit'])[name]
            frame = pd.read_csv(stage_root / 'raw-lags.csv')
            frame_checks(frame, saved_config, smoke)
            predictions[stage] = predictions_replay(stage_root, saved_config, frame, structures, smoke)
            criterion[stage] = criterion_replay(frame, saved_config, summary, smoke)
            topology_checks = read(root / validation / 'registered-structure-checks.json')
            assert topology_checks['all_checks_pass'] is True
            assert topology_checks['source_manifest_sha256'] == cached(root / 'structural_audit/manifest.json')
            assert set(topology_checks['levels']) == set(c['levels'])
            for level in c['levels']:
                assert topology_checks['levels'][level]['all_checks_pass'] is True
                for name in ['nodes', 'original_edges', 'dag_edges', 'global_dependency_depth']:
                    assert topology_checks['levels'][level][name] == structures[level][name]
                for seed in saved_config['blocks_by_level'][level]:
                    case_name = f'{level}_dag_synaptic_s{seed}'
                    finite = read(root / validation / case_name / 'finite-window-checks.json')
                    certificate = read(stage_root / case_name / 'certificate.json')
                    assert finite['all_checks_pass'] is True and finite['exact_terminal_equality'] is True
                    assert finite['global_dependency_depth'] == structures[level]['global_dependency_depth']
                    assert finite['source_certificate_sha256'] == cached(stage_root / case_name / 'certificate.json')
                    for branch in ['a', 'b']:
                        assert finite['exact_certificate_full_state_trace_hashes'][branch] == certificate[branch + '_full_state_trajectory_sha256']
            source_graph = read(root / validation / 'source-graph-audit.json')
            for name in ['full_graph_rebuilt_from_pinned_parquet',
                         'roles_rebuilt_from_pinned_annotations', 'partial_induced_graph_exact']:
                assert source_graph[name] is True
            assert (source_graph['neurons'], source_graph['edges']) == (138639, 2700513)
            for name, digest in source_graph['raw_source_sha256'].items():
                assert digest == pinned['outputs/tdc_v2/raw/' + name]
            validations[stage] = checks
        arrays = archives = 0
        for path in sorted(root.rglob('*.npz')):
            with np.load(path, allow_pickle=False) as archive:
                archives += 1
                for name in archive.files:
                    value = archive[name]
                    arrays += 1
                    if value.dtype.kind in 'fci':
                        assert np.isfinite(value).all(), (path, name)
        usage = {}
        for stage in ['smoke', smoke_validation, 'main', main_validation]:
            parent = read(root / stage / 'resources.json')
            pool = read(root / stage / 'process-tree-resources.json')
            assert parent['seconds'] <= c['max_seconds'] and pool['seconds'] <= c['max_seconds']
            assert pool['peak_sampled_process_tree_rss_bytes'] <= c['max_rss_bytes']
            assert parent['peak_sampled_rss_bytes'] <= c['max_rss_bytes']
            os_peak = parent.get('os_process_peak_working_set_bytes')
            assert os_peak is None or os_peak <= c['max_rss_bytes']
            usage[stage] = {'parent': parent, 'process_tree': pool}
        documents = [c['protocol'], 'docs/temporal-cycles-results.md', *sorted(OVERVIEWS)]
        links = []
        for name in documents:
            assert Path(name).is_file()
            for target in re.findall(r'\]\(([^)]+)\)', Path(name).read_text(encoding='utf-8')):
                if '://' in target or target.startswith(('#', 'mailto:', 'app:')):
                    continue
                target = target.strip('<>')
                destination = posixpath.normpath(posixpath.join(
                    Path(name).parent.as_posix(), target.split('#')[0]))
                assert (destination in inventory or Path(destination).exists()
                        or Path(destination).resolve() == (out / 'audit.json').resolve()), (name, target)
                links.append({'document': name, 'target': target, 'destination': destination})
        guard_path = root / 'guard_tests/checks.json'
        guards = read(guard_path)
        assert guards['all_checks_pass'] is True and guards['tests_passed'] == 11
        main = validations['main']
        summary = read(root / 'main/summary.json')
        write(out / 'manifest-checks.json', manifests)
        write(out / 'recorded-source-checks.json', recorded)
        write(out / 'structure-checks.json', structures)
        write(out / 'stage-resources.json', usage)
        write(out / 'document-sha256.json', {name: cached(name) for name in documents})
        write(out / 'document-links.json', links)
        write(out / 'criterion-replay.json', criterion)
        write(out / 'prediction-checks.json', predictions)
        write(out / 'audit.json', {
            **preservation, 'all_checks_pass': True,
            'baseline_commit': BASELINE, 'prior_result_blobs_unchanged': 43495,
            'protected_baseline_paths': len(protected),
            'historical_documents_and_tests_unchanged': True,
            'source_commit': git('rev-parse', 'HEAD'), 'verifier_sha256': sha(__file__),
            'manifests_checked': len(manifests),
            'checksum_entries': sum(record['artifacts'] for record in manifests),
            'recorded_source_entries': len(recorded),
            'retained_incomplete_attempts': authenticated_failures,
            'archives_inspected': archives, 'arrays_inspected': arrays, 'nonfinite_arrays': 0,
            'independent_main_streams': 54, 'independent_main_heads': 567,
            'certificate_pairs': 9, 'certificate_full_state_hashes': 18,
            'instantaneous_certificates': 9, 'zero_state_input_access_certificates': 27,
            'exact_main_state_hashes': main['exact_full_state_trace_hashes'],
            'outcome': summary['outcome'], 'primary_pass': summary['primary_pass'],
            'protocol_precedes_source': True, 'criterion_replayed_with_rational_counts': True,
            'cycle_effect_isolation_claim': False, 'biological_plasticity_performed': False,
            'guard_tests': guards})
    seal(out, {'complete': True, 'source_commit': git('rev-parse', 'HEAD')})
    print(read(out / 'audit.json'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--main-validation', default='main_validation')
    parser.add_argument('--smoke-validation', default='smoke_validation')
    arguments = parser.parse_args()
    audit(arguments.out, arguments.main_validation, arguments.smoke_validation)
