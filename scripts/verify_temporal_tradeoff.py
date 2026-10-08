"""Independent M2 reconstruction, readout refits, diagnostics and joint endpoint.

Frozen M1 verification helpers independently implement source graphs, dynamics,
split labels and SVD ridge fits. No numerical or summary function is imported
from either main experiment runner.
"""
import argparse
from fractions import Fraction
from itertools import combinations
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from verify_temporal_mechanism import case_job as independent_state_refit
from verify_temporal_memory_curve import validate_whole_source
from tradeoff_support import (config, read, write, sha, attempt, seal,
                              check_manifest, history_preserved)
from tdc_support import environment, git
from tdc_pool import pooled


ESTIMATES = ['mean', 'median', 'sd', 'bootstrap_lo', 'bootstrap_hi']
CURVE_METRICS = ['accuracy', 'baseline_excess', 'chance_adjusted',
                 'frequency_accuracy', 'current_accuracy']


def descriptive(values, c):
    x = np.asarray([float(v) for v in values], dtype=float)
    assert len(x) and np.isfinite(x).all()
    draws = np.random.default_rng(c['bootstrap_seed']).integers(
        0, len(x), size=(c['bootstrap_draws'], len(x)))
    lo, hi = np.quantile(np.mean(x[draws], axis=1), [.025, .975])
    return dict(mean=float(np.mean(x)), median=float(np.median(x)),
                sd=float(np.std(x, ddof=1)) if len(x) > 1 else 0.,
                bootstrap_lo=float(lo), bootstrap_hi=float(hi))


def compare_estimates(actual, expected):
    assert set(actual) == set(ESTIMATES)
    for key in ESTIMATES:
        assert abs(actual[key]-expected[key]) <= 1e-12, (key, actual, expected)


def adjusted_fraction(part):
    accuracy = sum((Fraction(int(r.correct), int(r.samples))
                    for r in part.itertuples()), Fraction(0))/len(part)
    return (accuracy-Fraction(1, 10))/Fraction(9, 10)


def current_diagnostics(saved, c):
    """Recalculate descriptive held-out probes after predictions were refitted."""
    w = c['warmup']
    symbols = np.asarray(saved['test_symbols'])
    actual = symbols[w:]
    previous = symbols[w-1:-1]
    prediction = np.asarray(saved['predictions'])[:, c['lags'].index(0)]
    np.testing.assert_array_equal(saved['ytest'][:, c['lags'].index(0)], actual)
    assert len(actual) == len(previous) == len(prediction) == c['test_samples']
    correct = prediction == actual
    same = actual == previous

    def stratum(mask):
        n = int(np.count_nonzero(mask))
        hit = int(np.count_nonzero(correct & mask))
        return dict(samples=n, correct=hit, accuracy=hit/n if n else None)

    errors = ~correct
    error_count = int(np.count_nonzero(errors))
    equal_previous = int(np.count_nonzero(errors & (prediction == previous)))
    k = c['alphabet_size']
    confusion = np.bincount(actual*k+prediction, minlength=k*k).reshape(k, k)
    assert int(confusion.sum()) == len(actual)
    assert int(np.trace(confusion)) == int(np.count_nonzero(correct))
    return dict(all=stratum(np.ones(len(actual), dtype=bool)),
                repeat=stratum(same), switch=stratum(~same),
                errors_total=error_count,
                errors_equal_previous_symbol=equal_previous,
                error_equals_previous_fraction=equal_previous/error_count if error_count else None,
                current_confusion=confusion.tolist())


def compare_diagnostics(actual, expected):
    assert set(actual) == set(expected)
    for name in ['all', 'repeat', 'switch']:
        assert set(actual[name]) == {'samples', 'correct', 'accuracy'}
        for count in ['samples', 'correct']:
            assert actual[name][count] == expected[name][count]
        if expected[name]['accuracy'] is None:
            assert actual[name]['accuracy'] is None
        else:
            assert abs(actual[name]['accuracy']-expected[name]['accuracy']) <= 1e-12
    for name in ['errors_total', 'errors_equal_previous_symbol', 'current_confusion']:
        assert actual[name] == expected[name], name
    fraction = expected['error_equals_previous_fraction']
    if fraction is None:
        assert actual['error_equals_previous_fraction'] is None
    else:
        assert abs(actual['error_equals_previous_fraction']-fraction) <= 1e-12


def diagnostic_row(level, arm, seed, diag):
    row = dict(level=level, arm=arm, seed=seed)
    for source, prefix in [('all', 'current'), ('repeat', 'repeat'), ('switch', 'switch')]:
        for metric in ['samples', 'correct', 'accuracy']:
            row[prefix+'_'+metric] = diag[source][metric]
    for name in ['errors_total', 'errors_equal_previous_symbol', 'error_equals_previous_fraction']:
        row[name] = diag[name]
    return row


def case_job(job):
    c, out = job['config'], Path(job['out'])
    root = Path(job['result'])/f'{job["level"]}_{job["arm"]}_s{job["block"]["seed"]}'
    with attempt(out, c, 'tradeoff-independent-case'):
        # Preserve the frozen helper's complete manifest, then seal M2 diagnostics
        # in a containing directory. No sealed evidence is appended or rewritten.
        nested = dict(job, out=str(out/'state_refit'))
        replay = independent_state_refit(nested)
        assert all(replay['exact'].values()), 'Complete state trace hash mismatch'
        with np.load(root/'case.npz', allow_pickle=False) as saved:
            diag = current_diagnostics(saved, c)
        compare_diagnostics(read(root/'diagnostics.json'), diag)
        write(out/'recomputed-diagnostics.json', diag)
        write(out/'checks.json', dict(all_checks_pass=True,
                                     diagnostics_independently_recomputed=True,
                                     exact_full_state_trace_hash=replay['exact'],
                                     source_case_manifest_sha256=sha(root/'manifest.json')))
    seal(out, dict(complete=True, result_case_manifest_sha256=sha(root/'manifest.json')))
    return dict(identity=replay['identity'], rows=replay['rows'], exact=replay['exact'],
                diagnostics=diagnostic_row(job['level'], job['arm'], job['block']['seed'], diag),
                resources=read(out/'resources.json'))


def compare_csv(path, expected, keys):
    actual = pd.read_csv(path)
    assert set(actual.columns) == set(expected.columns), (path, actual.columns, expected.columns)
    assert not actual.duplicated(keys).any(), path
    assert not expected.duplicated(keys).any(), path
    a = actual.sort_values(keys).reset_index(drop=True)
    e = expected[actual.columns].sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, e, check_exact=False, atol=1e-12, rtol=0,
                                  check_dtype=False)
    # Integer correct counts, rational gates and label coordinates remain exact.
    for n in a.columns:
        if n in ['seed', 'lag', 'samples', 'correct', 'frequency_correct', 'current_correct',
                 'n_blocks', 'errors_total', 'errors_equal_previous_symbol'] or n.endswith(
                     ('_numerator', '_denominator', '_samples', '_correct')):
            np.testing.assert_array_equal(a[n].to_numpy(), e[n].to_numpy())


def verify_pairing(result, c):
    comparisons = 0
    keys = ['input_patterns', 'observed_indices', 'train_symbols', 'test_symbols',
            'ytrain', 'ytest', 'train_times', 'test_times', 'zero_state_observed_prototypes']
    for block in c['blocks']:
        anchor_info = None
        anchor_streams = None
        for level in c['levels']:
            if block['seed'] not in c['blocks_by_level'][level]:
                continue
            full = result/f'{level}_full_s{block["seed"]}'
            info = read(full/'graph.json')
            with np.load(full/'case.npz', allow_pickle=False) as a:
                stream_values = {n: a[n] for n in ['train_symbols', 'test_symbols', 'ytrain', 'ytest']}
                if anchor_info is None:
                    anchor_info, anchor_streams = info, stream_values
                else:
                    for n in ['input_root_ids', 'observation_root_ids', 'input_mapping_sha256']:
                        assert info[n] == anchor_info[n], (level, n)
                    for n in stream_values:
                        np.testing.assert_array_equal(stream_values[n], anchor_streams[n])
                for arm in c['arms']:
                    other = result/f'{level}_{arm}_s{block["seed"]}'
                    paired = read(other/'graph.json')
                    for n in ['weight_sha256', 'input_root_ids', 'observation_root_ids',
                              'input_mapping_sha256', 'decoder_parameters_per_lag']:
                        assert paired[n] == info[n], (level, arm, n)
                    assert paired['decoder_parameters_per_lag'] == 490
                    with np.load(other/'case.npz', allow_pickle=False) as b:
                        for n in keys:
                            np.testing.assert_array_equal(a[n], b[n])
                    comparisons += 1
    return dict(all_checks_pass=True, cases_checked=comparisons,
                input_stream_label_coordinate_weight_pairing=True,
                cross_level_source_ids_and_stream_pairing=True)


def verify_summaries(f, result, c, smoke):
    compare_csv(result/'raw-lags.csv', f, ['level', 'arm', 'seed', 'lag'])
    expected_case_count = 8 if smoke else 36
    expected_head_count = expected_case_count*len(c['lags'])
    assert len(f) == expected_head_count
    assert len(f[['level', 'arm', 'seed']].drop_duplicates()) == expected_case_count
    exact = {}
    scores = []
    for (level, arm, seed), part in f.groupby(['level', 'arm', 'seed']):
        assert part.lag.tolist() == c['lags']
        lag0 = part[part.lag == 0].iloc[0]
        values = dict(current=Fraction(int(lag0.correct), int(lag0.samples)),
                      history=adjusted_fraction(part[part.lag.isin(c['score_lags'])]),
                      long_history=adjusted_fraction(part[part.lag.between(1, 20)]))
        exact[(level, arm, int(seed))] = values
        row = dict(level=level, arm=arm, seed=int(seed))
        for dimension, value in values.items():
            row.update({dimension: float(value), dimension+'_numerator': value.numerator,
                        dimension+'_denominator': value.denominator})
        scores.append(row)
    compare_csv(result/'arm-scores.csv', pd.DataFrame(scores), ['level', 'arm', 'seed'])
    summary = read(result/'summary.json')
    assert summary['criterion_unchanged'] is True
    assert summary['smoke'] is smoke
    endpoints, paired, pairwise = {}, [], []
    assert set(summary['levels']) == set(c['levels'])
    for level in c['levels']:
        seeds = c['blocks_by_level'][level]
        reported = summary['levels'][level]
        reference = [exact[(level, 'instantaneous', seed)]['current'] for seed in seeds]
        valid = bool(all(value >= Fraction(99, 100) for value in reference))
        assert reported['reference_valid'] is valid
        assert set(reported['arm_metrics']) == set(c['arms'])
        for arm in c['arms']:
            assert set(reported['arm_metrics'][arm]) == {'current', 'history', 'long_history'}
            for dimension in ['current', 'history', 'long_history']:
                compare_estimates(reported['arm_metrics'][arm][dimension],
                                  descriptive([exact[(level, arm, seed)][dimension] for seed in seeds], c))
        dh = [exact[(level, 'carry_only', s)]['history']-
              exact[(level, 'instantaneous', s)]['history'] for s in seeds]
        dc = [exact[(level, 'carry_only', s)]['current']-
              exact[(level, 'instantaneous', s)]['current'] for s in seeds]
        mean_h = sum(dh, Fraction(0))/len(dh)
        mean_c = sum(dc, Fraction(0))/len(dc)
        components = dict(historical_mean_pass=bool(mean_h >= Fraction(3, 100)),
                          current_mean_pass=bool(mean_c <= -Fraction(1, 20)),
                          all_history_positive=bool(min(dh) > 0),
                          all_current_negative=bool(max(dc) < 0))
        gate = None if smoke or not valid else bool(all(components.values()))
        primary = reported['primary']
        for name, values, mean in [('historical_gain', dh, mean_h), ('current_difference', dc, mean_c)]:
            assert primary[name]['mean_exact'] == str(mean)
            compare_estimates({n: primary[name][n] for n in ESTIMATES}, descriptive(values, c))
        for name, value in components.items():
            assert primary[name] is value, (level, name)
        assert primary['registered_joint_pass'] is gate
        endpoints[level] = dict(reference_valid=valid, registered_joint_pass=gate,
                                historical_gain_exact=str(mean_h), current_difference_exact=str(mean_c),
                                **components)
        for seed, h, current in zip(seeds, dh, dc):
            paired.append(dict(level=level, seed=seed, historical_gain=float(h),
                               current_difference=float(current), history_numerator=h.numerator,
                               history_denominator=h.denominator, current_numerator=current.numerator,
                               current_denominator=current.denominator))
        for first, second in combinations(c['arms'], 2):
            for dimension in ['current', 'history', 'long_history']:
                values = [exact[(level, first, s)][dimension]-exact[(level, second, s)][dimension]
                          for s in seeds]
                pairwise.append(dict(level=level, first=first, second=second,
                                     dimension=dimension, **descriptive(values, c)))
    compare_csv(result/'paired-tradeoff.csv', pd.DataFrame(paired), ['level', 'seed'])
    compare_csv(result/'descriptive-pairwise.csv', pd.DataFrame(pairwise),
                ['level', 'first', 'second', 'dimension'])
    curves = []
    for (level, arm, lag), part in f.groupby(['level', 'arm', 'lag']):
        part = part.sort_values('seed')
        row = dict(level=level, arm=arm, lag=int(lag))
        for metric in CURVE_METRICS:
            row.update({metric+'_'+n: v for n, v in descriptive(part[metric], c).items()})
        curves.append(row)
    compare_csv(result/'curve-summary.csv', pd.DataFrame(curves), ['level', 'arm', 'lag'])
    primary = endpoints[c['primary_level']]['registered_joint_pass']
    outcome = ('smoke' if smoke else 'assay-invalid'
               if not endpoints[c['primary_level']]['reference_valid'] else 'PASS' if primary else 'FAIL')
    assert summary['primary_pass'] is primary
    assert summary['outcome'] == outcome
    assert summary['cases'] == expected_case_count and summary['lag_heads'] == expected_head_count
    return endpoints, outcome, primary


def verify(result, out):
    m = check_manifest(result)
    assert isinstance(m['smoke'], bool)
    c = read(result/'config.json')
    assert c == m['config'] == config(m['smoke'])
    # A verifier may not silently run against edited source after generation.
    assert subprocess.run(['git', 'diff', '--quiet', 'HEAD']).returncode == 0, 'Tracked edits before validation'
    recorded = read(result/'source.json')
    assert recorded['tracked_changes'] is False
    assert recorded['source_commit'] == m['source_commit']
    assert recorded['source_tree'] == git('rev-parse', recorded['source_commit']+'^{tree}')
    required = ['scripts/verify_temporal_tradeoff.py', 'scripts/tradeoff_support.py',
                'scripts/verify_temporal_mechanism.py', 'scripts/verify_temporal_memory_curve.py',
                'configs/temporal_tradeoff.json', c['protocol']]
    assert all(n in recorded['hashes'] for n in required)
    for name, digest in recorded['hashes'].items():
        assert sha(name) == digest, name
    preservation = history_preserved(c)
    with attempt(out, c, 'tradeoff-independent-validation') as resources:
        write(out/'environment.json', environment())
        write(out/'source-graph-audit.json', validate_whole_source(c, resources))
        jobs = [dict(config=c, level=level, arm=arm, block=block, result=str(result),
                     out=str(out/f'{level}_{arm}_s{block["seed"]}'))
                for level in c['levels'] for block in c['blocks']
                if block['seed'] in c['blocks_by_level'][level] for arm in c['arms']]
        results, usage = pooled(case_job, jobs, c['neural_workers'], c,
                                'Tradeoff independent verification')
        f = pd.DataFrame([row for r in results for row in r['rows']]).sort_values(
            ['level', 'arm', 'seed', 'lag']).reset_index(drop=True)
        endpoints, outcome, primary = verify_summaries(f, result, c, m['smoke'])
        diag = pd.DataFrame([r['diagnostics'] for r in results])
        compare_csv(result/'diagnostics.csv', diag, ['level', 'arm', 'seed'])
        pairing = verify_pairing(result, c)
        f.to_csv(out/'recomputed-lags.csv', index=False)
        diag.sort_values(['level', 'arm', 'seed']).to_csv(out/'recomputed-diagnostics.csv', index=False)
        write(out/'pairing-checks.json', pairing)
        write(out/'process-tree-resources.json', usage)
        write(out/'job-resources.json', [dict(identity=r['identity'], resources=r['resources']) for r in results])
        assert history_preserved(c) == preservation
        for name, digest in recorded['hashes'].items():
            assert sha(name) == digest, name
        write(out/'checks.json', dict(all_checks_pass=True, cases=len(results),
                                     replayed_streams=2*len(results), independently_refit_lag_heads=len(f),
                                     exact_full_state_trace_hashes=sum(sum(r['exact'].values()) for r in results),
                                     instantaneous_certificates=sum('/instantaneous/' in r['identity'] for r in results),
                                     zero_state_input_access_certificates=len(results),
                                     diagnostics_independently_recomputed=len(results),
                                     outcome=outcome, primary_pass=primary, endpoints=endpoints,
                                     result_manifest_sha256=sha(result/'manifest.json'),
                                     verifier_commit=git('rev-parse', 'HEAD'), verifier_sha256=sha(__file__),
                                     historical_preservation=preservation))
    seal(out, dict(complete=True, result_manifest_sha256=sha(result/'manifest.json')))
    print(dict(all_checks_pass=True, outcome=outcome, primary_pass=primary, cases=len(results)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('result', type=Path)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    with threadpool_limits(1):
        verify(a.result, a.out)
