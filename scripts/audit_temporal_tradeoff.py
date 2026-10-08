"""Independent artifact, rational endpoint and historical provenance closeout.

No import of the experiment's dynamics, decoder, diagnostics or summaries.
This auditor reads sealed outcomes only after their independent verification.
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

from tradeoff_support import read, write, sha, attempt, seal, git, history_preserved


BASELINE = 'e52d1ff959d4426ce61315ad17788aa7c66fcb88'
PREREGISTRATION = '7552c234'
NAMESPACE = Path('results/tdc_tradeoff_v1')
OVERVIEWS = {'README.md', 'docs/research-status.md',
             'docs/research-roadmap.md', 'docs/next-work.md'}


def tree(revision):
    inventory = {}
    for line in git('ls-tree', '-r', revision).splitlines():
        meta, name = line.split('\t', 1)
        inventory[name] = meta.split()[2]
    return inventory


def fixed_config(c):
    """Check the literal registered settings independently of runner assertions."""
    assert c['baseline_commit'] == BASELINE
    assert c['namespace'] == NAMESPACE.as_posix()
    assert c['protocol'] == 'docs/temporal-tradeoff-protocol.md'
    assert c['levels'] == ['legacy5', 'brain5'] and c['circuit_seeds'] == [701]
    assert c['arms'] == {
        'full': {'carry': True, 'synaptic_history': True},
        'carry_only': {'carry': True, 'synaptic_history': False},
        'synaptic_only': {'carry': False, 'synaptic_history': True},
        'instantaneous': {'carry': False, 'synaptic_history': False}}
    for name, start in [('seed', 1200001), ('input_seed', 1210001),
                        ('train_seed', 1220001), ('test_seed', 1230001)]:
        assert [block[name] for block in c['blocks']] == list(range(start, start + 6))
    assert c['blocks_by_level'] == {'legacy5': list(range(1200001, 1200007)),
                                  'brain5': list(range(1200001, 1200004))}
    assert c['lags'] == list(range(21)) and c['score_lags'] == list(range(1, 6))
    assert c['primary_level'] == 'legacy5'
    assert c['primary_contrast'] == ['carry_only', 'instantaneous']
    assert (c['alphabet_size'], c['warmup'], c['train_samples'],
            c['test_samples']) == (10, 200, 4000, 2000)
    assert (c['gain'], c['leak'], c['alpha'], c['input_fraction'],
            c['input_amplitude']) == (.9, .6, 1., .1, .5)
    assert c['normalization'] == 'incoming_l1' and c['schedule'] == 'mbon_after_kc'
    assert (c['reference_current_min'], c['historical_gain_min'],
            c['current_difference_max']) == (.99, .03, -.05)
    assert c['bootstrap_seed'] == 1240001 and c['bootstrap_draws'] == 10000
    assert c['neural_workers'] == 4
    assert c['max_seconds'] == 7200 and c['max_rss_bytes'] == 4294967296


def frac_accuracy(row):
    return Fraction(int(row['correct']), int(row['samples']))


def history_score(rows, lags):
    values = [frac_accuracy(row) for row in rows if int(row['lag']) in lags]
    assert len(values) == len(lags)
    return (sum(values, Fraction(0)) / len(values) - Fraction(1, 10)) / Fraction(9, 10)


def criterion_replay(frame, c, summary):
    """Recompute both primary dimensions from integer held-out correct counts."""
    replay = {}
    for level, seeds in c['blocks_by_level'].items():
        values = {}
        for seed in seeds:
            values[seed] = {}
            for arm in c['arms']:
                rows = frame[(frame.level == level) & (frame.seed == seed)
                             & (frame.arm == arm)].to_dict('records')
                assert len(rows) == 21 and sorted(int(row['lag']) for row in rows) == list(range(21))
                current = frac_accuracy(next(row for row in rows if int(row['lag']) == 0))
                values[seed][arm] = {'current': current,
                                     'history': history_score(rows, c['score_lags']),
                                     'long_history': history_score(rows, list(range(1, 21)))}
        d_history = [values[seed]['carry_only']['history']
                     - values[seed]['instantaneous']['history'] for seed in seeds]
        d_current = [values[seed]['carry_only']['current']
                     - values[seed]['instantaneous']['current'] for seed in seeds]
        mean_history = sum(d_history, Fraction(0)) / len(seeds)
        mean_current = sum(d_current, Fraction(0)) / len(seeds)
        reference_valid = all(values[seed]['instantaneous']['current'] >= Fraction(99, 100)
                              for seed in seeds)
        historical_mean_pass = mean_history >= Fraction(3, 100)
        current_mean_pass = mean_current <= -Fraction(1, 20)
        all_history_positive = all(value > 0 for value in d_history)
        all_current_negative = all(value < 0 for value in d_current)
        joint = (historical_mean_pass and current_mean_pass and all_history_positive
                 and all_current_negative) if reference_valid else None
        actual = summary['levels'][level]
        assert actual['reference_valid'] is reference_valid
        for arm in c['arms']:
            for metric in ['current', 'history', 'long_history']:
                mean = sum((values[seed][arm][metric] for seed in seeds), Fraction(0)) / len(seeds)
                assert abs(actual['arm_metrics'][arm][metric]['mean'] - float(mean)) <= 1e-12
        primary = actual['primary']
        assert primary['historical_gain']['mean_exact'] == str(mean_history)
        assert primary['current_difference']['mean_exact'] == str(mean_current)
        assert abs(primary['historical_gain']['mean'] - float(mean_history)) <= 1e-12
        assert abs(primary['current_difference']['mean'] - float(mean_current)) <= 1e-12
        for name, expected in [('historical_mean_pass', historical_mean_pass),
                               ('current_mean_pass', current_mean_pass),
                               ('all_history_positive', all_history_positive),
                               ('all_current_negative', all_current_negative),
                               ('registered_joint_pass', joint)]:
            assert primary[name] is expected, (level, name)
        replay[level] = {'reference_valid': reference_valid,
                         'historical_mean_exact': str(mean_history),
                         'current_mean_exact': str(mean_current),
                         'historical_mean_pass': historical_mean_pass,
                         'current_mean_pass': current_mean_pass,
                         'all_history_positive': all_history_positive,
                         'all_current_negative': all_current_negative,
                         'registered_joint_pass': joint,
                         'paired_values': [{'seed': seed, 'd_history_exact': str(dh),
                                            'd_current_exact': str(dc)}
                                           for seed, dh, dc in zip(seeds, d_history, d_current)]}
    primary = replay[c['primary_level']]
    expected_outcome = ('assay-invalid' if not primary['reference_valid']
                        else 'PASS' if primary['registered_joint_pass'] else 'FAIL')
    assert summary['primary_pass'] is primary['registered_joint_pass']
    assert summary['outcome'] == expected_outcome and summary['smoke'] is False
    assert summary['criterion_unchanged'] is True
    return replay


def predictions_replay(root, c, frame):
    """Confirm raw counts and split-safe labels directly from saved predictions."""
    cases = streams = heads = 0
    pairing = {}
    for level, seeds in c['blocks_by_level'].items():
        for seed in seeds:
            block = next(block for block in c['blocks'] if block['seed'] == seed)
            for arm in c['arms']:
                case = root / 'main' / f'{level}_{arm}_s{seed}'
                with np.load(case / 'case.npz', allow_pickle=False) as data:
                    cases += 1
                    observed = data['observed_indices']
                    assert observed.shape == (48,)
                    graph = read(case / 'graph.json')
                    assert graph['decoder_parameters_per_lag'] == 490
                    common = {'input_patterns': hashlib.sha256(data['input_patterns'].tobytes()).hexdigest(),
                              'observed_indices': hashlib.sha256(observed.tobytes()).hexdigest(),
                              'weight_sha256': graph['weight_sha256']}
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
                        common[split + '_symbols'] = hashlib.sha256(symbols.tobytes()).hexdigest()
                        streams += 1
                    key = (level, seed)
                    if key in pairing:
                        assert common == pairing[key], (key, arm)
                    else:
                        pairing[key] = common
                    assert data['predictions'].shape == data['ytest'].shape == (2000, 21)
                    assert data['scores'].shape == (2000, 21, 10)
                    np.testing.assert_array_equal(data['predictions'], data['scores'].argmax(axis=2))
                    assert ((data['predictions'] >= 0) & (data['predictions'] < 10)).all()
                    for lag in range(21):
                        row = frame[(frame.level == level) & (frame.arm == arm)
                                    & (frame.seed == seed) & (frame.lag == lag)]
                        assert len(row) == 1
                        row = row.iloc[0]
                        truth = data['ytest'][:, lag]
                        for column, array in [('correct', 'predictions'),
                                              ('frequency_correct', 'frequency_predictions'),
                                              ('current_correct', 'current_predictions')]:
                            assert int(row[column]) == int(np.count_nonzero(data[array][:, lag] == truth))
                        assert abs(float(row['accuracy']) - int(row['correct']) / 2000) <= 1e-12
                        heads += 1
    assert (cases, streams, heads) == (36, 72, 756)
    return {'cases': cases, 'saved_streams_checked': streams,
            'prediction_columns_checked': heads, 'split_safe_labels': True,
            'paired_arm_streams_and_inputs': True}


def audit(out, main_validation='main_validation', smoke_validation='smoke_validation'):
    root = NAMESPACE
    c = read(root / 'main/config.json')
    fixed_config(c)
    assert c == read('configs/temporal_tradeoff.json')
    with attempt(out, c, 'tradeoff-closeout-audit'):
        preservation = history_preserved(c)
        baseline = read(root / 'baseline_audit/baseline-git-blobs.json')
        assert baseline == tree(BASELINE), 'Baseline inventory must authenticate the full recorded Git tree'
        inventory = tree('HEAD')
        protected = {name: blob for name, blob in baseline.items()
                     if name.startswith(('results/', 'src/', 'data/', 'configs/',
                                         'scripts/', 'tests/', 'docs/')) and name not in OVERVIEWS}
        assert all(inventory.get(name) == blob for name, blob in protected.items())
        assert sum(name.startswith('results/') for name in protected) == 42717
        changed = set(git('diff', '--name-only', 'HEAD').splitlines())
        assert not changed.intersection(protected), 'Uncommitted protected edits'
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
        assert sorted(entry['manifest_path'] for entry in registry) == incomplete, 'Unregistered failed attempt'
        authenticated_failures = []
        for entry in registry:
            assert isinstance(entry['reason'], str) and entry['reason'].strip()
            failure_path = Path(entry['failure_path'])
            assert root.resolve() in failure_path.resolve().parents and failure_path.is_file()
            failure = read(failure_path)
            assert failure.get('complete') is False and failure.get('message')
            authenticated_failures.append(dict(entry, failure_sha256=cached(failure_path)))
        recorded, trees, blobs = [], {}, {}
        for path in sorted(root.rglob('source.json')):
            if out.resolve() in path.resolve().parents:
                continue
            source = read(path)
            revision = source['source_commit']
            if revision not in trees:
                trees[revision] = tree(revision)
            for name, expected in source['hashes'].items():
                if name in trees[revision]:
                    blob = trees[revision][name]
                    if blob not in blobs:
                        blobs[blob] = hashlib.sha256(subprocess.check_output(
                            ['git', 'cat-file', 'blob', blob])).hexdigest()
                    assert blobs[blob] == expected, (path, name, revision)
                    mode = 'recorded_git_blob'
                else:
                    assert name.startswith('outputs/tdc_v2/'), (path, name)
                    assert cached(name) == expected, (path, name)
                    mode = 'current_pinned_external_bytes'
                recorded.append({'record': path.as_posix(), 'source_commit': revision,
                                 'path': name, 'sha256': expected, 'authentication': mode})
        summary = read(root / 'main/summary.json')
        smoke = read(root / smoke_validation / 'checks.json')
        main = read(root / main_validation / 'checks.json')
        assert smoke['all_checks_pass'] is True and main['all_checks_pass'] is True
        for checks, expected, stage in [(smoke, (8, 16, 168), 'smoke'),
                                        (main, (36, 72, 756), 'main')]:
            assert (checks['cases'], checks['replayed_streams'],
                    checks['independently_refit_lag_heads']) == expected
            assert checks['result_manifest_sha256'] == cached(root / stage / 'manifest.json')
        assert main['instantaneous_certificates'] == 9
        assert main['zero_state_input_access_certificates'] == 36
        assert 0 <= main['exact_full_state_trace_hashes'] <= 72
        assert summary['cases'] == 36 and summary['lag_heads'] == 756
        assert main['outcome'] == summary['outcome']
        assert main['primary_pass'] is summary['primary_pass']
        main_source = read(root / 'main/source.json')
        assert main_source['hashes'], 'Missing recorded numerical input hashes'
        assert recorded, 'Missing authenticated recorded sources'
        assert subprocess.run(['git', 'merge-base', '--is-ancestor', PREREGISTRATION,
                               main_source['source_commit']], check=False).returncode == 0
        assert tree(PREREGISTRATION)[c['protocol']] == tree(main_source['source_commit'])[c['protocol']]
        assert tree(PREREGISTRATION)['configs/temporal_tradeoff.json'] == tree(main_source['source_commit'])['configs/temporal_tradeoff.json']
        frame = pd.read_csv(root / 'main/raw-lags.csv')
        assert len(frame) == 756 and len(frame[['level', 'arm', 'seed']].drop_duplicates()) == 36
        assert not frame[['level', 'arm', 'seed', 'lag']].duplicated().any()
        assert set(frame.samples) == {2000}
        assert set(frame.level) == set(c['levels']) and set(frame.arm) == set(c['arms'])
        assert (frame.correct.between(0, 2000)).all()
        assert (frame.correct.astype(np.int64) == frame.correct).all()
        prediction_checks = predictions_replay(root, c, frame)
        criterion = criterion_replay(frame, c, summary)
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
        documents = [c['protocol'], 'docs/temporal-tradeoff-results.md',
                     'README.md', 'docs/research-status.md',
                     'docs/research-roadmap.md', 'docs/next-work.md']
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
        write(out / 'manifest-checks.json', manifests)
        write(out / 'recorded-source-checks.json', recorded)
        write(out / 'stage-resources.json', usage)
        write(out / 'document-sha256.json', {name: cached(name) for name in documents})
        write(out / 'document-links.json', links)
        write(out / 'criterion-replay.json', criterion)
        write(out / 'prediction-checks.json', prediction_checks)
        write(out / 'audit.json', {
            **preservation, 'all_checks_pass': True,
            'baseline_commit': BASELINE, 'prior_result_blobs_unchanged': 42717,
            'protected_baseline_paths': len(protected), 'historical_documents_and_tests_unchanged': True,
            'source_commit': git('rev-parse', 'HEAD'), 'verifier_sha256': sha(__file__),
            'manifests_checked': len(manifests),
            'checksum_entries': sum(record['artifacts'] for record in manifests),
            'recorded_source_entries': len(recorded),
            'retained_incomplete_attempts': authenticated_failures,
            'archives_inspected': archives, 'arrays_inspected': arrays, 'nonfinite_arrays': 0,
            'independent_main_streams': 72, 'independent_main_heads': 756,
            'instantaneous_certificates': 9, 'zero_state_input_access_certificates': 36,
            'exact_main_state_hashes': main['exact_full_state_trace_hashes'],
            'outcome': summary['outcome'], 'primary_pass': summary['primary_pass'],
            'protocol_precedes_source': True, 'criterion_replayed_with_rational_counts': True,
            'biological_plasticity_performed': False})
    seal(out, {'complete': True, 'source_commit': git('rev-parse', 'HEAD')})
    print(read(out / 'audit.json'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--main-validation', default='main_validation')
    parser.add_argument('--smoke-validation', default='smoke_validation')
    arguments = parser.parse_args()
    audit(arguments.out, arguments.main_validation, arguments.smoke_validation)
