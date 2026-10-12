"""Independent M11 raw-source/manual-state/SVD fit and target-free replay."""
import argparse
from fractions import Fraction
import hashlib
from pathlib import Path
import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits
from verify_temporal_memory_curve import partial_source
from verify_structural_specificity_design import independent_swaps, invariants
from verify_rule_transfer_design import reconstruct
from tdc_pool import pooled
from rule_transfer_support import (config, source, unchanged, history, parents, coverage, authenticate_source,
                                   attempt, seal, read, write, array_sha, sha, case_key, artifact_limit)


def matrix_hash(a):
    return hashlib.sha256(a.data.tobytes() + a.indices.tobytes() + a.indptr.tobytes()).hexdigest()


def rebuild_bank(c, resources, smoke):
    original, ids, roles = partial_source()
    original.sort_indices()
    roles = np.asarray(roles, dtype='U5')
    old_config = read('configs/structural_specificity_design.json')
    selected = [g for g in c['models'] if g.startswith('g')]
    reports = []
    for graph in selected:
        seed = int(graph[1:])
        rebuilt, swap_log = independent_swaps(original, roles, seed, 5, 200, resources)
        root = Path(c['graph_parent']) / 'graphs' / ('legacy5_' + graph)
        saved = sparse.load_npz(root / 'raw.npz').tocsr()
        for attr in ['data', 'indices', 'indptr']:
            np.testing.assert_array_equal(getattr(rebuilt, attr), getattr(saved, attr))
        report = invariants(original, rebuilt, roles, swap_log, old_config)
        report.update(level='legacy5', seed=seed)
        assert report == read(root / 'audit.json')
        reports.append(dict(model=graph, independent_seeded_raw_regeneration_exact=True, raw_sha256=matrix_hash(rebuilt)))
    return dict(partial_source_neurons=len(ids), partial_source_edges=original.nnz,
                members=reports, all_checks_pass=True, whole_brain_replayed=False)


def construct(c, graph, seed):
    original, ids, roles = partial_source()
    roles = np.asarray(roles, dtype='U5')
    pool, obs = np.flatnonzero(roles == 'KC'), np.flatnonzero(roles == 'MBON')
    if graph == 'shift':
        return dict(kind='shift', n=len(ids), observed=np.arange(48), weights=None, patterns=None)
    raw = (original if graph == 'real' else sparse.load_npz(Path(c['graph_parent']) / 'graphs' / ('legacy5_' + graph) / 'raw.npz').tocsr())
    weights = raw.copy().tocsr()
    strength = np.asarray(abs(raw).sum(axis=1)).ravel()
    factor = np.zeros(len(ids))
    factor[strength != 0] = .9 / strength[strength != 0]
    weights.data *= np.repeat(factor, np.diff(weights.indptr))
    weights.sort_indices()
    patterns = np.zeros((11, len(ids)))
    rng = np.random.Generator(np.random.PCG64(seed))
    for symbol in range(11):
        locations = rng.choice(len(pool), 51, replace=False)
        patterns[symbol, pool[locations]] = .5
    return dict(kind='source', n=len(ids), weights=weights, patterns=patterns, observed=obs,
                kc=pool, mb=obs, mb_weights=weights[obs], ids=ids, roles=roles, raw=raw)


def step(m, previous, symbol):
    if m['kind'] == 'shift':
        state = np.zeros(m['n'])
        state[int(symbol)] = 1.
        for coordinate in range(22):
            state[coordinate + 11] = previous[coordinate]
        return state
    stimulus = m['patterns'][int(symbol)]
    recurrent = m['weights'].dot(previous)
    state = np.zeros_like(previous) + .6 * np.tanh(recurrent + stimulus)
    mixed = previous.copy()
    mixed[m['kc']] = state[m['kc']]
    state[m['mb']] = np.zeros(len(m['mb'])) + .6 * np.tanh(m['mb_weights'].dot(mixed) + stimulus[m['mb']])
    return state


def teacher(m, rows, resources):
    x = np.zeros((len(rows), 8, 48))
    hashes = []
    for i, values in enumerate(rows):
        previous = np.zeros(m['n'])
        full = hashlib.sha256()
        for input_index in range(10):
            previous = step(m, previous, values[input_index])
            full.update(previous.tobytes())
            if input_index >= 2:
                x[i, input_index - 2] = previous[m['observed']]
        hashes.append(full.hexdigest())
        if i % 50 == 0:
            resources.check()
    return x, hashes


def fit_svd(x, labels):
    mean = np.mean(x, axis=0)
    scale = np.maximum(np.std(x, axis=0), .00001)
    standardized = (x - mean) / scale
    target = np.zeros((len(labels), 11))
    target[np.arange(len(labels)), labels] = 1.
    intercept = target.mean(axis=0)
    augmented_x = np.concatenate([standardized, np.eye(48)])
    augmented_target = np.concatenate([target - intercept, np.zeros((48, 11))])
    beta = np.linalg.lstsq(augmented_x, augmented_target, rcond=None)[0]
    return dict(mean=mean, scale=scale, coefficients=beta, intercept=intercept)


def scores(x, head):
    return ((np.atleast_2d(x) - head['mean']) / head['scale']) @ head['coefficients'] + head['intercept']


def classify(values):
    flat = values.reshape(-1, 11)
    output = []
    for row in flat:
        maximum = max(row)
        output.append(next(i for i, value in enumerate(row) if value >= maximum - 1e-10))
    return np.array(output, dtype=np.int64).reshape(values.shape[:-1])


def rollout(m, heads, prefixes, resources):
    independently_fitted, recorded = heads
    predicted = np.empty((len(prefixes), 8), dtype=np.int64)
    recorded_scores = np.empty((len(prefixes), 8, 11))
    digests, feature_digest = [], hashlib.sha256()
    for i, prefix in enumerate(prefixes):
        state, full = np.zeros(m['n']), hashlib.sha256()
        for symbol in prefix:
            state = step(m, state, symbol)
            full.update(state.tobytes())
        for j in range(8):
            observed = state[m['observed']]
            feature_digest.update(observed.tobytes())
            svd_scores, original_scores = scores(observed, independently_fitted)[0], scores(observed, recorded)[0]
            np.testing.assert_allclose(svd_scores, original_scores, atol=1e-9, rtol=1e-9)
            next_value = int(classify(svd_scores))
            assert next_value == int(classify(original_scores))
            predicted[i, j] = next_value
            recorded_scores[i, j] = original_scores
            state = step(m, state, next_value)
            full.update(state.tobytes())
        digests.append(full.hexdigest())
        if i % 50 == 0:
            resources.check()
    return predicted, recorded_scores, digests, feature_digest.hexdigest()


def count_metrics(teacher_predictions, generated, targets, indices):
    first, teacher_total, prefix_sum, complete = 0, 0, 0, 0
    for i in indices:
        first += int(teacher_predictions[i, 0] == targets[i, 0])
        teacher_total += sum(int(x == y) for x, y in zip(teacher_predictions[i], targets[i]))
        prefix = 0
        for x, y in zip(generated[i], targets[i]):
            if x != y:
                break
            prefix += 1
        prefix_sum += prefix
        complete += int(prefix == 8)
    return dict(episodes=len(indices), first_correct=first, teacher_correct=teacher_total,
                teacher_predictions=len(indices) * 8, prefix_sum=prefix_sum, completed=complete)


def metrics_for_case(predictions, datasets, metadata, identity, c):
    rows = []
    for split in c['splits']:
        meta = metadata[split][:len(datasets[split])]
        groups = [('all', list(range(len(meta))))]
        if split == 'parameter_test':
            groups.append(('nonperiod2', [i for i, v in enumerate(meta) if v['a'] in (7, 8, 9)]))
        for a in sorted({v['a'] for v in meta}):
            groups.append((f'a{a}', [i for i, v in enumerate(meta) if v['a'] == a]))
            for b in sorted({v['b'] for v in meta if v['a'] == a}):
                groups.append((f'a{a}_b{b}', [i for i, v in enumerate(meta) if (v['a'], v['b']) == (a, b)]))
        for group, indices in groups:
            if indices:
                rows.append(dict(**identity, split=split, group=group,
                                 **count_metrics(predictions[split + '_teacher_predictions'], predictions[split + '_rollout_predictions'], datasets[split][:, 3:], indices)))
    return rows


def independently_summarize(rows, c, smoke):
    if smoke:
        return dict(outcome='smoke', primary_pass=None, secondary_pass=None, fits=3, affine_parameters=1617)
    cohort_results = {}
    for cohort, seeds in c['cohorts'].items():
        def value(model, split, group, count):
            selected = [r for r in rows if r['family'] == 'affine' and r['model'] == model and r['input_seed'] in seeds and r['split'] == split and r['group'] == group]
            assert len(selected) == 3
            denominator = 'teacher_predictions' if count == 'teacher_correct' else 'episodes'
            return sum(Fraction(r[count], r[denominator]) for r in selected) / 3
        completions = {g: value(g, 'parameter_test', 'nonperiod2', 'completed') for g in c['models']}
        train = value('real', 'train', 'all', 'teacher_correct')
        first = value('real', 'parameter_test', 'nonperiod2', 'first_correct')
        by_multiplier = {str(a): value('real', 'parameter_test', f'a{a}', 'completed') for a in (7, 8, 9, 10)}
        nulls = [completions[g] for g in c['models'] if g.startswith('g')]
        mean_null = sum(nulls) / 10
        gates = dict(train_acquisition=train >= Fraction(1, 2), first_next=first >= Fraction(1, 4),
                     completion=completions['real'] >= Fraction(1, 10),
                     every_nonperiod2_multiplier=all(by_multiplier[str(a)] >= Fraction(1, 20) for a in (7, 8, 9)),
                     shift_gap=completions['real'] - completions['shift'] >= Fraction(1, 20))
        cohort_results[cohort] = dict(primary_gates=gates, primary_pass=all(gates.values()),
                                     real_train_teacher=str(train), real_nonperiod2_first=str(first),
                                     nonperiod2_completions={g: str(v) for g, v in completions.items()},
                                     real_completions_by_multiplier={a: str(v) for a, v in by_multiplier.items()},
                                     real_minus_shift_completion=str(completions['real'] - completions['shift']),
                                     null_mean_completion=str(mean_null), real_minus_null_completion=str(completions['real'] - mean_null),
                                     real_strictly_above_every_null=all(completions['real'] > x for x in nulls),
                                     secondary_local_pass=all(completions['real'] > x for x in nulls) and completions['real'] - mean_null >= Fraction(1, 20))
    passed = all(v['primary_pass'] for v in cohort_results.values())
    return dict(outcome='PASS' if passed else 'FAIL', primary_pass=passed,
                secondary_pass=passed and all(v['secondary_local_pass'] for v in cohort_results.values()),
                cohorts=cohort_results, fits=216, affine_parameters=116424, scaler_moments=20736,
                recurrent_learning=False, iid_shuffled_secondary_diagnostics=True,
                effective_independent_shift_interfaces=1, task_instances_shared_across_input_maps=True)


def case_job(job):
    c, graph, seed, family = job['config'], job['model'], job['seed'], job['family']
    root, out, smoke = Path(job['result']), Path(job['out']), job['smoke']
    with threadpool_limits(1), attempt(out, c, 'm11-independent-case') as resources:
        coverage(root)
        m = construct(c, graph, seed)
        complete_data, metadata = reconstruct(read('configs/rule_transfer_design.json'))
        datasets = {s: complete_data[s][family][:11] if smoke else complete_data[s][family] for s in c['splits']}
        info = read(root / 'graph.json')
        assert info['frozen_before_evaluation'] and info['head_parameters'] == 539 and info['scaler_moments'] == 96
        assert info['train_rows'] == c['training_rows'] and not info['recurrent_learning']
        assert info['observed_indices'] == m['observed'].tolist()
        if graph != 'shift':
            assert info['normalized_weight_sha256'] == matrix_hash(m['weights'])
            assert info['raw_weight_sha256'] == matrix_hash(m['raw'])
            assert info['input_patterns_sha256'] == array_sha(m['patterns'])
            assert info['source_ids_sha256'] == array_sha(m['ids']) and info['roles_sha256'] == array_sha(m['roles'])
            assert info['pool_sha256'] == array_sha(m['kc'])
        features, hashes = {}, {}
        for split in c['splits']:
            features[split], hashes[split] = teacher(m, datasets[split], resources)
            assert info['trace'][split]['input_rows_sha256'] == array_sha(datasets[split])
            assert hashes[split] == info['trace'][split]['teacher_full_state_hashes']
            assert array_sha(features[split]) == info['trace'][split]['teacher_observed_features_sha256']
        labels = datasets['train'][:, 3:].reshape(-1)
        assert array_sha(labels) == info['trained_label_sha256']
        fitted = fit_svd(features['train'].reshape(-1, 48), labels)
        predictions = {}
        with np.load(root / 'case.npz', allow_pickle=False) as saved:
            assert set(saved.files) == set(info['arrays'])
            assert all(array_sha(saved[n]) == h for n, h in info['arrays'].items())
            assert all(np.isfinite(saved[n]).all() for n in saved.files)
            recorded = {n: saved[n].copy() for n in fitted}
            for n in fitted:
                np.testing.assert_allclose(fitted[n], recorded[n], atol=1e-9, rtol=1e-9)
            for split in c['splits']:
                x = features[split].reshape(-1, 48)
                svd_scores = scores(x, fitted).reshape(len(datasets[split]), 8, 11)
                original_scores = scores(x, recorded).reshape(svd_scores.shape)
                np.testing.assert_allclose(svd_scores, saved[split + '_teacher_scores'], atol=1e-9, rtol=1e-9)
                np.testing.assert_allclose(original_scores, saved[split + '_teacher_scores'], atol=1e-9, rtol=1e-9)
                predicted = classify(svd_scores)
                np.testing.assert_array_equal(predicted, saved[split + '_teacher_predictions'])
                np.testing.assert_array_equal(classify(original_scores), predicted)
                generated, generated_scores, digest, observed_digest = rollout(m, (fitted, recorded), datasets[split][:, :3], resources)
                np.testing.assert_array_equal(generated, saved[split + '_rollout_predictions'])
                np.testing.assert_allclose(generated_scores, saved[split + '_rollout_scores'], atol=1e-9, rtol=1e-9)
                assert digest == info['trace'][split]['rollout_full_state_hashes']
                assert observed_digest == info['trace'][split]['rollout_observed_bytes_sha256']
                predictions[split + '_teacher_predictions'] = predicted
                predictions[split + '_rollout_predictions'] = generated
        identity = dict(family=family, model=graph, input_seed=seed)
        metrics = metrics_for_case(predictions, datasets, metadata, identity, c)
        assert metrics == read(root / 'metrics.json')
        write(out / 'metrics.json', metrics)
        write(out / 'checks.json', dict(all_checks_pass=True, independent_svd_refits=1,
                                        full_state_trajectories_exact=2 * sum(len(r) for r in datasets.values()),
                                        observed_features_exact=True, all_predictions_and_metrics_exact=True,
                                        result_manifest_sha256=sha(root / 'manifest.json')))
    seal(out, dict(complete=True))
    return dict(identity=case_key(family, graph, seed), rows=metrics)


def independent_lags(c):
    all_data, metadata = reconstruct(read('configs/rule_transfer_design.json'))
    output = []
    for family in ['affine', 'iid', 'shuffled']:
        datasets = {s: all_data[s][family] for s in c['splits']}
        for lag in (2, 3):
            predictions = {}
            for split, rows in datasets.items():
                tp, rp = np.zeros((len(rows), 8), dtype=np.int64), np.zeros((len(rows), 8), dtype=np.int64)
                for i, row in enumerate(rows):
                    memory = list(row[:3])
                    for j in range(8):
                        tp[i, j] = row[j + 3 - lag]
                        rp[i, j] = memory[len(memory) - lag]
                        memory.append(rp[i, j])
                predictions[split + '_teacher_predictions'] = tp
                predictions[split + '_rollout_predictions'] = rp
            output += metrics_for_case(predictions, datasets, metadata, dict(family=family, model='fixed_lag' + str(lag), input_seed=None), c)
    return output


def independent_descriptions(rows, c):
    import statistics
    result = {}
    for cohort, seeds in c['cohorts'].items():
        gap1, gap2 = [], []
        for seed in seeds:
            selected = [r for r in rows if r['family'] == 'affine' and r['input_seed'] == seed and r['split'] == 'parameter_test' and r['group'] == 'nonperiod2']
            values = {r['model']: Fraction(r['completed'], r['episodes']) for r in selected}
            gap1.append(values['real'] - values['shift'])
            gap2.append(values['real'] - sum(v for g, v in values.items() if g.startswith('g')) / 10)
        result[cohort] = {}
        for name, values in [('real_minus_shift', gap1), ('real_minus_null_mean', gap2)]:
            mean = sum(values) / 3
            sd = statistics.stdev([float(v) for v in values])
            result[cohort][name] = dict(input_seeds=seeds, paired_differences_exact=[str(v) for v in values],
                                        mean_exact=str(mean), median_exact=str(statistics.median(values)), sample_sd=sd,
                                        standardized_paired_effect=float(mean) / sd if sd else None)
    return result


def compare_description(a, b):
    if isinstance(a, dict):
        assert set(a) == set(b)
        for key in a:
            compare_description(a[key], b[key])
    elif isinstance(a, float):
        np.testing.assert_allclose(a, b, atol=1e-9, rtol=1e-9)
    else:
        assert a == b


def verify(result, out, smoke=False):
    c, record = config(smoke), source(config(smoke))
    with threadpool_limits(1), attempt(out, c, 'm11-independent-smoke' if smoke else 'm11-independent-main') as resources:
        coverage(result)
        producer_source = read(result / 'source.json')
        authenticate_source(producer_source)
        assert read(result / 'config.json') == c
        write(out / 'source.json', record)
        write(out / 'config.json', c)
        write(out / 'history.json', history())
        write(out / 'parent-checks.json', parents(c))
        write(out / 'independent-bank.json', rebuild_bank(c, resources, smoke))
        jobs = [dict(config=c, model=g, seed=s, family=f, result=str(result / 'cases' / case_key(f, g, s)),
                     out=str(out / 'cases' / case_key(f, g, s)), smoke=smoke)
                for f in c['families'] for g in c['models'] for s in c['input_seeds']]
        assert {p.name for p in (result / 'cases').iterdir()} == {case_key(j['family'], j['model'], j['seed']) for j in jobs}
        results, usage = pooled(case_job, jobs, c['workers'], c, 'M11 independent')
        rows = [r for v in sorted(results, key=lambda v: v['identity']) for r in v['rows']]
        assert rows == read(result / 'metrics.json')
        summary = independently_summarize(rows, c, smoke)
        assert summary == read(result / 'summary.json')
        if not smoke:
            reference = independent_lags(c)
            assert reference == read(result / 'fixed-lag-reference-metrics.json')
            assert read(result / 'm10-prior-control-metrics.json') == read(Path(c['task_parent']) / 'metrics.json')
            write(out / 'recomputed-fixed-lag-metrics.json', reference)
            descriptions = independent_descriptions(rows, c)
            compare_description(descriptions, read(result / 'paired-descriptions.json'))
            write(out / 'recomputed-paired-descriptions.json', descriptions)
        write(out / 'recomputed-metrics.json', rows)
        write(out / 'recomputed-summary.json', summary)
        write(out / 'process-tree-resources.json', usage)
        write(out / 'checks.json', dict(all_checks_pass=True, fits=len(jobs), independent_svd_refits=len(jobs),
                                        full_state_trajectories_exact=len(jobs) * (66 if smoke else 1980),
                                        all_predictions_scores_metrics_and_exact_gates_replayed=True,
                                        no_m11_producer_scientific_algorithms_imported=True,
                                        result_manifest_sha256=sha(result / 'manifest.json'), outcome=summary['outcome']))
        write(out / 'artifact-bytes.json', dict(bytes_before_final_seal=artifact_limit(out, c)))
        unchanged(record)
    seal(out, dict(complete=True, study='M11 independent', smoke=smoke))
    print(read(out / 'checks.json'), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--result', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    verify(a.result, a.out, a.smoke)
