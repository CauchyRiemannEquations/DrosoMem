"""M11 fixed source/null/shift reservoirs; fitted then frozen K11 affine heads."""
import argparse
from fractions import Fraction
import hashlib
from pathlib import Path
import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits
from flying.data.connectome import load_connectome
from flying.data.mushroom_body import load_roles
from flying.brain.diagnostics import normalize_condition
from flying.brain.timed_reservoir import TimedReservoir
from flying.training.whole_brain_memory import MappedEncoder, weight_hash
from flying.models.ridge import RidgeDecoder
from temporal_mechanism import FactorReservoir
from tdc_pool import pooled
from rule_transfer_support import (config, source, unchanged, history, parents, attempt, seal,
                                   read, write, array_sha, case_key, artifact_limit)


class Shift:
    def __init__(self, n=686):
        self.n = n
        self.reset()
    def reset(self):
        self.state = np.zeros(self.n)
    def step(self, symbol):
        following = np.zeros(self.n)
        following[int(symbol)] = 1.
        following[11:33] = self.state[:22]
        self.state = following
        return following.copy()


def setup(c, graph, seed):
    raw, ids, _ = load_connectome(Path(c['source_data']))
    roles, _ = load_roles(Path(c['source_data']), ids)
    roles, ids = np.asarray(roles, dtype='U5'), np.asarray(ids, dtype=np.int64)
    pool, observed = np.flatnonzero(roles == 'KC'), np.flatnonzero(roles == 'MBON')
    assert len(pool) == 512 and len(observed) == 48 and len(ids) == 686
    patterns = np.zeros((11, len(ids)))
    rng = np.random.default_rng(seed)
    for symbol in range(11):
        patterns[symbol, pool[rng.choice(512, 51, replace=False)]] = .5
    patterns.flags.writeable = False
    if graph == 'shift':
        model, observed = Shift(len(ids)), np.arange(48)
        info = dict(dynamics='three_slot_one_hot', scalar_states=686, active_states=33,
                    seed_independent_reference=True, input_patterns_sha256=None, normalized_weight_sha256=None)
    else:
        if graph != 'real':
            raw = sparse.load_npz(Path(c['graph_parent']) / 'graphs' / ('legacy5_' + graph) / 'raw.npz').tocsr()
        weights = normalize_condition(raw, 'incoming_l1', .9)
        weights.sort_indices()
        base = TimedReservoir(weights, MappedEncoder(patterns), roles, .6, 'mbon_after_kc')
        model = FactorReservoir(base, False, True)
        expected = (read(Path(c['graph_parent']) / 'definitions/legacy5/definition.json')['source_normalized_sha256']
                    if graph == 'real' else read(Path(c['graph_parent']) / 'graphs' / ('legacy5_' + graph) / 'audit.json')['null_normalized_sha256'])
        assert weight_hash(model.weights) == expected
        info = dict(dynamics='zero_carry_source', scalar_states=686, raw_weight_sha256=weight_hash(raw),
                    normalized_weight_sha256=expected, input_patterns_sha256=array_sha(patterns),
                    source_ids_sha256=array_sha(ids), roles_sha256=array_sha(roles), pool_sha256=array_sha(pool))
    info.update(observed_indices=observed.tolist(), recurrent_learning=False, input_seed=seed, model=graph)
    return model, observed, info


def data(c, family, smoke=False):
    result = {}
    for split in c['splits']:
        with np.load(Path(c['task_parent']) / f'{family}-{split}-learner.npz', allow_pickle=False) as a:
            rows = np.concatenate([a['prefixes'], a['targets']], axis=1)
        if smoke:
            rows = rows[:11]
        result[split] = rows
    return result


def decisions(scores, atol=1e-10):
    return (scores >= np.max(scores, axis=-1, keepdims=True) - atol).argmax(axis=-1)


def teacher_features(model, observed, rows, resources):
    features = np.empty((len(rows), 8, 48))
    full_hashes = []
    for i, row in enumerate(rows):
        model.reset()
        assert not np.any(model.state)
        digest = hashlib.sha256()
        for t, symbol in enumerate(row[:10]):
            state = model.step(int(symbol))
            digest.update(state.tobytes())
            if t >= 2:
                features[i, t - 2] = state[observed]
        full_hashes.append(digest.hexdigest())
        if i % 50 == 0:
            resources.check()
    assert np.isfinite(features).all()
    return features, full_hashes


def target_free_rollout(model, observed, head, prefixes, horizon, resources):
    predictions = np.empty((len(prefixes), horizon), dtype=np.int64)
    scores = np.empty((len(prefixes), horizon, 11))
    full_hashes, feature_digest = [], hashlib.sha256()
    for i, prefix in enumerate(prefixes):
        model.reset()
        assert not np.any(model.state)
        digest = hashlib.sha256()
        for symbol in prefix:
            state = model.step(int(symbol))
            digest.update(state.tobytes())
        for j in range(horizon):
            observed_state = state[observed]
            feature_digest.update(observed_state.tobytes())
            scores[i, j] = head.scores(observed_state)[0]
            symbol = int(decisions(scores[i, j]))
            predictions[i, j] = symbol
            state = model.step(symbol)
            digest.update(state.tobytes())
        full_hashes.append(digest.hexdigest())
        if i % 50 == 0:
            resources.check()
    assert np.isfinite(scores).all()
    return predictions, scores, full_hashes, feature_digest.hexdigest()


def recount(teacher, generated, targets, mask):
    t, p, y = teacher[mask], generated[mask], targets[mask]
    same = p == y
    lengths = np.where(same.all(axis=1), 8, (~same).argmax(axis=1))
    return dict(episodes=len(y), first_correct=int((t[:, 0] == y[:, 0]).sum()),
                teacher_correct=int((t == y).sum()), teacher_predictions=len(y) * 8,
                prefix_sum=int(lengths.sum()), completed=int(same.all(axis=1).sum()))


def metric_rows(arrays, rows, c, identity, smoke=False):
    metadata = read(Path(c['task_parent']) / 'audit-metadata.json')
    output = []
    for split in c['splits']:
        meta = metadata[split][:len(rows[split])]
        aa, bb = np.array([r['a'] for r in meta]), np.array([r['b'] for r in meta])
        groups = {'all': np.ones(len(aa), dtype=bool)}
        if split == 'parameter_test':
            groups['nonperiod2'] = np.isin(aa, [7, 8, 9])
        for a in np.unique(aa):
            groups['a' + str(a)] = aa == a
            for b in np.unique(bb[aa == a]):
                groups[f'a{a}_b{b}'] = (aa == a) & (bb == b)
        for group, mask in groups.items():
            if not mask.any():
                continue
            output.append(dict(**identity, split=split, group=group,
                               **recount(arrays[split + '_teacher_predictions'], arrays[split + '_rollout_predictions'], rows[split][:, 3:], mask)))
    return output


def execute(job):
    c, graph, seed, family = job['config'], job['model'], job['seed'], job['family']
    out, smoke = Path(job['out']), job['smoke']
    with threadpool_limits(1), attempt(out, c, 'm11-model-case') as resources:
        model, observed, info = setup(c, graph, seed)
        rows = data(c, family, smoke)
        features, hashes = {}, {}
        for split in c['splits']:
            features[split], hashes[split] = teacher_features(model, observed, rows[split], resources)
        x = features['train'].reshape(-1, 48)
        labels = rows['train'][:, 3:].reshape(-1)
        assert len(x) == c['training_rows']
        head = RidgeDecoder(1.).fit(x, np.eye(11)[labels])
        arrays = dict(mean=head.mean.copy(), scale=head.scale.copy(), coefficients=head.weights.copy(), intercept=head.target_mean.copy())
        for value in arrays.values():
            assert np.isfinite(value).all()
            value.flags.writeable = False
        for attr in ['mean', 'scale', 'weights', 'target_mean']:
            getattr(head, attr).flags.writeable = False
        before = {n: array_sha(a) for n, a in arrays.items()}
        trace = {}
        for split in c['splits']:
            scores = head.scores(features[split].reshape(-1, 48)).reshape(len(rows[split]), 8, 11)
            predictions, rollout_scores, rollout_hashes, rollout_features_hash = target_free_rollout(
                model, observed, head, rows[split][:, :3], 8, resources)
            arrays.update({split + '_teacher_scores': scores, split + '_teacher_predictions': decisions(scores),
                           split + '_rollout_scores': rollout_scores, split + '_rollout_predictions': predictions})
            trace[split] = dict(input_rows_sha256=array_sha(rows[split]), teacher_full_state_hashes=hashes[split],
                                teacher_observed_features_sha256=array_sha(features[split]),
                                rollout_full_state_hashes=rollout_hashes, rollout_observed_bytes_sha256=rollout_features_hash)
        assert before == {n: array_sha(arrays[n]) for n in before}
        if graph != 'shift':
            assert weight_hash(model.weights) == info['normalized_weight_sha256']
            assert array_sha(model.encoder.patterns) == info['input_patterns_sha256']
        info.update(frozen_before_evaluation=True, head_parameters=539, scaler_moments=96,
                    train_rows=len(x), trained_label_sha256=array_sha(labels), trace=trace,
                    arrays={n: array_sha(a) for n, a in arrays.items()})
        identity = dict(family=family, model=graph, input_seed=seed)
        metrics = metric_rows(arrays, rows, c, identity, smoke)
        np.savez_compressed(out / 'case.npz', **arrays)
        write(out / 'graph.json', info)
        write(out / 'metrics.json', metrics)
    seal(out, dict(complete=True))
    return dict(identity=case_key(family, graph, seed), rows=metrics)


def rate(row, kind):
    return Fraction(row[kind], row['teacher_predictions'] if kind == 'teacher_correct' else row['episodes'])


def summarize(rows, c, smoke=False):
    if smoke:
        return dict(outcome='smoke', primary_pass=None, secondary_pass=None, fits=3, affine_parameters=1617)
    cells = {(r['family'], r['model'], r['input_seed'], r['split'], r['group']): r for r in rows}
    cohorts = {}
    for name, seeds in c['cohorts'].items():
        def avg(graph, split, group, metric):
            return sum((rate(cells['affine', graph, seed, split, group], metric) for seed in seeds), Fraction(0)) / len(seeds)
        source_train = avg('real', 'train', 'all', 'teacher_correct')
        source_first = avg('real', 'parameter_test', 'nonperiod2', 'first_correct')
        score = {g: avg(g, 'parameter_test', 'nonperiod2', 'completed') for g in c['models']}
        multipliers = {str(a): avg('real', 'parameter_test', 'a' + str(a), 'completed') for a in [7, 8, 9, 10]}
        null = [score[g] for g in c['models'] if g not in ['real', 'shift']]
        null_mean = sum(null, Fraction(0)) / len(null)
        gates = dict(train_acquisition=source_train >= Fraction(c['minimum_source_train_teacher']),
                     first_next=source_first >= Fraction(c['minimum_source_first']),
                     completion=score['real'] >= Fraction(c['minimum_source_completion']),
                     every_nonperiod2_multiplier=all(multipliers[str(a)] >= Fraction(c['minimum_each_rule_completion']) for a in [7, 8, 9]),
                     shift_gap=score['real'] - score['shift'] >= Fraction(c['minimum_source_shift_completion_gap']))
        cohorts[name] = dict(primary_gates=gates, primary_pass=all(gates.values()),
                             real_train_teacher=str(source_train), real_nonperiod2_first=str(source_first),
                             nonperiod2_completions={g: str(s) for g, s in score.items()},
                             real_completions_by_multiplier={a: str(s) for a, s in multipliers.items()},
                             real_minus_shift_completion=str(score['real'] - score['shift']),
                             null_mean_completion=str(null_mean), real_minus_null_completion=str(score['real'] - null_mean),
                             real_strictly_above_every_null=all(score['real'] > n for n in null),
                             secondary_local_pass=all(score['real'] > n for n in null) and score['real'] - null_mean >= Fraction(c['minimum_source_null_completion_gap']))
    primary = all(v['primary_pass'] for v in cohorts.values())
    secondary = primary and all(v['secondary_local_pass'] for v in cohorts.values())
    return dict(outcome='PASS' if primary else 'FAIL', primary_pass=primary, secondary_pass=secondary,
                cohorts=cohorts, fits=216, affine_parameters=116424, scaler_moments=20736,
                recurrent_learning=False, iid_shuffled_secondary_diagnostics=True,
                effective_independent_shift_interfaces=1, task_instances_shared_across_input_maps=True)


def static_lags(c):
    output = []
    for family in ['affine', 'iid', 'shuffled']:
        rows = data(config(), family)
        for lag in c['fixed_lag_copies']:
            arrays = {}
            for split, values in rows.items():
                teacher, rollout = [], []
                for row in values.tolist():
                    teacher.append([row[3 + j - lag] for j in range(8)])
                    generated = row[:3]
                    result = []
                    for j in range(8):
                        value = generated[-lag]
                        result.append(value)
                        generated.append(value)
                    rollout.append(result)
                arrays[split + '_teacher_predictions'] = np.array(teacher)
                arrays[split + '_rollout_predictions'] = np.array(rollout)
            output += metric_rows(arrays, rows, c, dict(family=family, model='fixed_lag' + str(lag), input_seed=None))
    return output


def paired_descriptions(rows, c):
    lookup = {(r['family'], r['model'], r['input_seed'], r['split'], r['group']): r for r in rows}
    output = {}
    for cohort, seeds in c['cohorts'].items():
        differences = {'real_minus_shift': [], 'real_minus_null_mean': []}
        for seed in seeds:
            values = {g: rate(lookup['affine', g, seed, 'parameter_test', 'nonperiod2'], 'completed') for g in c['models']}
            null_mean = sum((v for g, v in values.items() if g.startswith('g')), Fraction(0)) / 10
            differences['real_minus_shift'].append(values['real'] - values['shift'])
            differences['real_minus_null_mean'].append(values['real'] - null_mean)
        output[cohort] = {}
        for name, values in differences.items():
            mean = sum(values, Fraction(0)) / len(values)
            median = sorted(values)[len(values) // 2]
            sd = float(np.asarray([float(v) for v in values]).std(ddof=1))
            output[cohort][name] = dict(input_seeds=seeds, paired_differences_exact=list(map(str, values)),
                                        mean_exact=str(mean), median_exact=str(median), sample_sd=sd,
                                        standardized_paired_effect=float(mean) / sd if sd else None)
    return output


def run(out, smoke=False):
    c = config(smoke)
    record = source(c)
    with threadpool_limits(1), attempt(out, c, 'm11-smoke' if smoke else 'm11-main'):
        write(out / 'source.json', record)
        write(out / 'config.json', c)
        write(out / 'parent-checks.json', parents(c))
        write(out / 'history.json', history())
        jobs = [dict(config=c, model=g, seed=s, family=f, out=str(out / 'cases' / case_key(f, g, s)), smoke=smoke)
                for f in c['families'] for g in c['models'] for s in c['input_seeds']]
        results, usage = pooled(execute, jobs, c['workers'], c, 'M11')
        rows = [row for result in sorted(results, key=lambda v: v['identity']) for row in result['rows']]
        write(out / 'metrics.json', rows)
        write(out / 'summary.json', summarize(rows, c, smoke))
        write(out / 'process-tree-resources.json', usage)
        if not smoke:
            write(out / 'fixed-lag-reference-metrics.json', static_lags(c))
            write(out / 'm10-prior-control-metrics.json', read(Path(c['task_parent']) / 'metrics.json'))
            write(out / 'paired-descriptions.json', paired_descriptions(rows, c))
        write(out / 'artifact-bytes.json', dict(bytes_before_final_seal=artifact_limit(out, c)))
        unchanged(record)
    seal(out, dict(complete=True, study='M11', smoke=smoke))
    print(read(out / 'summary.json'), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    run(args.out, args.smoke)
