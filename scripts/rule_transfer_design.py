"""Finite M10 affine-family split and predictor validity preflight, no neurons."""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
import numpy as np
from rule_transfer_design_support import config, source, unchanged, history, budget, attempt, write, read, seal

FAMILIES = ('affine', 'iid', 'shuffled')


def affine_episode(a, b, y, modulus, length):
    p = (b * pow((1 - a) % modulus, -1, modulus)) % modulus
    values = [(p + y) % modulus]
    while len(values) < length:
        values.append((a * values[-1] + b) % modulus)
    return values, p


def datasets(c):
    data, metadata = {}, {}
    k, size = c['modulus'], c['prefix_length'] + c['horizon']
    for split in c['splits']:
        multipliers = c['parameter_test_multipliers'] if split == 'parameter_test' else c['train_multipliers']
        offsets = (c['train_offsets_from_fixed_point'] + c['instance_test_offsets_from_fixed_point']
                   if split == 'parameter_test' else c['train_offsets_from_fixed_point'] if split == 'train'
                   else c['instance_test_offsets_from_fixed_point'])
        rows, meta = [], []
        for a in multipliers:
            for b in range(k):
                for y in offsets:
                    values, p = affine_episode(a, b, y, k, size)
                    rows.append(values)
                    meta.append(dict(a=a, b=b, y=y, fixed_point=p))
        structured = np.asarray(rows, dtype=np.int64)
        iid = np.random.default_rng(c['iid_seeds'][split]).integers(0, k, structured.shape, dtype=np.int64)
        rng = np.random.default_rng(c['shuffle_seeds'][split])
        shuffled = np.asarray([rng.permutation(row) for row in structured], dtype=np.int64)
        data[split] = dict(affine=structured, iid=iid, shuffled=shuffled)
        metadata[split] = meta
    return data, metadata


def fit_tables(train, c):
    counts = Counter()
    tables = {order: defaultdict(Counter) for order in (1, 2, 3)}
    for row in train:
        for t in range(c['prefix_length'] - 1, len(row) - 1):
            counts[int(row[t + 1])] += 1
            for order in tables:
                key = tuple(map(int, row[t - order + 1:t + 1]))
                tables[order][key][int(row[t + 1])] += 1
    return counts, tables


def winner(counts):
    return min(counts, key=lambda s: (-counts[s], s))


def infer(prefix, k):
    d = int(prefix[1] - prefix[0]) % k
    if not d:
        return None
    a = (int(prefix[2] - prefix[1]) * pow(d, -1, k)) % k
    return a, (int(prefix[1]) - a * int(prefix[0])) % k


def predict(past, name, fitted, k, rule=None):
    counts, tables = fitted
    if name == 'frequency':
        return winner(counts)
    if name == 'current':
        return int(past[-1])
    if name == 'repeat_difference':
        return int(2 * past[-1] - past[-2]) % k
    if name == 'known_form_oracle':
        return int(past[-1]) if rule is None else (rule[0] * int(past[-1]) + rule[1]) % k
    for order in range(int(name[-1]), 0, -1):
        key = tuple(map(int, past[-order:]))
        if key in tables[order]:
            return winner(tables[order][key])
    return winner(counts)


def predictions(prefixes, targets, name, fitted, c):
    teacher, rollout = [], []
    for prefix, target in zip(prefixes, targets):
        rule = infer(prefix, c['modulus'])
        real, generated = list(map(int, prefix)), list(map(int, prefix))
        tp, rp = [], []
        for true in target:
            tp.append(predict(real, name, fitted, c['modulus'], rule))
            next_value = predict(generated, name, fitted, c['modulus'], rule)
            rp.append(next_value)
            real.append(int(true))
            generated.append(next_value)
        teacher.append(tp)
        rollout.append(rp)
    return np.asarray(teacher, dtype=np.int64), np.asarray(rollout, dtype=np.int64)


def metrics(teacher, rollout, targets):
    match = rollout == targets
    lengths = np.where(match.all(axis=1), match.shape[1], (~match).argmax(axis=1))
    n, h = targets.shape
    first = int((teacher[:, 0] == targets[:, 0]).sum())
    correct = int((teacher == targets).sum())
    completed = int(match.all(axis=1).sum())
    return dict(episodes=n, horizon=h, first_correct=first, first_accuracy=str(Fraction(first, n)),
                teacher_correct=correct, teacher_predictions=n * h, teacher_accuracy=str(Fraction(correct, n * h)),
                autonomous_prefix_sum=int(lengths.sum()), autonomous_prefix_mean=str(Fraction(int(lengths.sum()), n)),
                autonomous_completed=completed, autonomous_completion_rate=str(Fraction(completed, n)))


def table_record(fitted):
    counts, tables = fitted
    return dict(frequency={str(s): int(n) for s, n in sorted(counts.items())},
                orders={str(o): {','.join(map(str, key)): {str(s): int(n) for s, n in sorted(cell.items())}
                                for key, cell in sorted(table.items())} for o, table in tables.items()})


def cycle_signature(a, b, start, k):
    cycle, x = [], start
    while x not in cycle:
        cycle.append(x)
        x = (a * x + b) % k
    assert x == start
    return min(tuple(cycle[i:] + cycle[:i]) for i in range(len(cycle)))


def design_checks(data, meta, reports, c):
    rows = {s: data[s]['affine'] for s in c['splits']}
    prefixes = {s: {tuple(map(int, r[:3])) for r in a} for s, a in rows.items()}
    windows = {s: {tuple(map(int, r[t - 2:t + 1])) for r in a
                   for t in range(2, len(r) - 1)} for s, a in rows.items()}
    cycles = {s: {cycle_signature(m['a'], m['b'], int(r[0]), c['modulus']) for r, m in zip(a, meta[s])}
              for s, a in rows.items()}
    check = dict(exact_split_sizes=all(len(rows[s]) == c['expected_episodes'][s] for s in rows),
                 all_prefixes_identifiable=all(infer(r[:3], c['modulus']) == (m['a'], m['b'])
                                              for s in rows for r, m in zip(rows[s], meta[s])),
                 unique_structured_prefixes=len(set().union(*prefixes.values())) == sum(map(len, rows.values())),
                 parameter_prefix_overlap=len(prefixes['train'] & prefixes['parameter_test']),
                 parameter_teacher_window_overlap=len(windows['train'] & windows['parameter_test']),
                 parameter_cycle_overlap=len(cycles['train'] & cycles['parameter_test']),
                 instance_cycle_overlap=len(cycles['train'] & cycles['instance_test']),
                 distinct_cycles={s: len(a) for s, a in cycles.items()},
                 oracle_structured_complete=all(reports['affine'][s]['known_form_oracle']['autonomous_completed'] == len(rows[s])
                                               and reports['affine'][s]['known_form_oracle']['teacher_correct'] == len(rows[s]) * c['horizon'] for s in rows),
                 trivial_parameter_first_max=str(max(Fraction(reports['affine']['parameter_test'][n]['first_accuracy'])
                                                     for n in c['controls'] if n != 'known_form_oracle')),
                 negative_oracle_first={f: reports[f]['parameter_test']['known_form_oracle']['first_accuracy']
                                        for f in ('iid', 'shuffled')})
    gates = dict(split_sizes=check['exact_split_sizes'], identifiable=check['all_prefixes_identifiable'],
                 unique_prefixes=check['unique_structured_prefixes'], no_parameter_prefix_leak=check['parameter_prefix_overlap'] == 0,
                 no_parameter_teacher_window_leak=check['parameter_teacher_window_overlap'] == 0,
                 no_parameter_cycle_leak=check['parameter_cycle_overlap'] == 0, oracle_complete=check['oracle_structured_complete'],
                 trivial_separation=Fraction(check['trivial_parameter_first_max']) <= Fraction(c['maximum_trivial_parameter_first_accuracy']),
                 iid_separation=Fraction(check['negative_oracle_first']['iid']) <= Fraction(c['maximum_iid_oracle_first_accuracy']),
                 shuffled_separation=Fraction(check['negative_oracle_first']['shuffled']) <= Fraction(c['maximum_shuffled_oracle_first_accuracy']))
    return dict(**check, gates=gates, all_task_gates_pass=all(gates.values()))


def run(out):
    c, record = config(), source()
    with attempt(out, budget(c), 'm10-design-preflight') as resources:
        write(out / 'source.json', record)
        write(out / 'config.json', c)
        write(out / 'history.json', history(check_bytes=True))
        data, meta = datasets(c)
        write(out / 'audit-metadata.json', meta)
        reports, storage = {}, {}
        for family in FAMILIES:
            reports[family] = {}
            fitted = fit_tables(data['train'][family], c)
            tables = table_record(fitted)
            write(out / (family + '-training-tables.json'), tables)
            storage[family] = dict(training_transitions=sum(fitted[0].values()),
                                   suffix_cells={str(o): len(t) for o, t in fitted[1].items()},
                                   suffix_count_entries={str(o): sum(len(v) for v in t.values()) for o, t in fitted[1].items()})
            saved = {}
            for split in c['splits']:
                a = data[split][family]
                prefix, target = a[:, :3], a[:, 3:]
                np.savez_compressed(out / (family + '-' + split + '-learner.npz'), prefixes=prefix, targets=target)
                saved[split + '_oracle_identifiable'] = np.asarray([infer(r, c['modulus']) is not None for r in prefix])
                reports[family][split] = {}
                for name in c['controls']:
                    teacher, rollout = predictions(prefix, target, name, fitted, c)
                    saved[split + '_' + name + '_teacher'] = teacher
                    saved[split + '_' + name + '_rollout'] = rollout
                    reports[family][split][name] = metrics(teacher, rollout, target)
                resources.check()
            np.savez_compressed(out / (family + '-predictions.npz'), **saved)
        checks = design_checks(data, meta, reports, c)
        write(out / 'metrics.json', reports)
        write(out / 'storage-budgets.json', storage)
        write(out / 'design-checks.json', checks)
        summary = dict(outcome='READY_PENDING_INDEPENDENT_VERIFICATION' if checks['all_task_gates_pass'] else 'INVALID',
                       all_task_gates_pass=checks['all_task_gates_pass'], neural_model_fits=0,
                       structured_episodes=sum(c['expected_episodes'].values()), families=list(FAMILIES),
                       control_evaluations=len(FAMILIES) * len(c['splits']) * len(c['controls']),
                       future_performance_preregistered=False)
        write(out / 'summary.json', summary)
        unchanged(record)
    seal(out, dict(complete=True, study='M10 task design'))
    print(summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args().out)
