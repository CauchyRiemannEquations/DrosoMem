"""Independent M10 reconstruction; never imports the task/predictor producer."""
import argparse
from fractions import Fraction
from pathlib import Path
import numpy as np
from rule_transfer_design_support import config, source, unchanged, history, budget, attempt, write, read, seal, sha, authenticate


def reconstruct(c):
    data, metadata = {}, {}
    for split in c['splits']:
        aa = c['parameter_test_multipliers'] if split == 'parameter_test' else c['train_multipliers']
        ys = list(range(1, 11)) if split == 'parameter_test' else list(range(1, 6)) if split == 'train' else list(range(6, 11))
        symbols, meta = [], []
        for a in aa:
            for b in range(11):
                p, = [x for x in range(11) if (a * x + b) % 11 == x]
                for y in ys:
                    symbols.append([(p + y * pow(a, t, 11)) % 11 for t in range(11)])
                    meta.append({'a': a, 'b': b, 'y': y, 'fixed_point': p})
        affine = np.array(symbols, dtype=np.int64)
        iid_rng = np.random.Generator(np.random.PCG64(c['iid_seeds'][split]))
        shuffle_rng = np.random.Generator(np.random.PCG64(c['shuffle_seeds'][split]))
        data[split] = {'affine': affine, 'iid': iid_rng.integers(11, size=affine.shape, dtype=np.int64),
                       'shuffled': np.stack([shuffle_rng.permutation(r) for r in affine])}
        metadata[split] = meta
    return data, metadata


def count_training(rows):
    frequency = [0] * 11
    tables = {str(o): {} for o in (1, 2, 3)}
    for row in rows.tolist():
        for target_index in range(3, 11):
            target = row[target_index]
            frequency[target] += 1
            for o in (1, 2, 3):
                key = ','.join(str(v) for v in row[target_index - o:target_index])
                table = tables[str(o)]
                if key not in table:
                    table[key] = [0] * 11
                table[key][target] += 1
    record = {'frequency': {str(v): n for v, n in enumerate(frequency) if n},
              'orders': {o: {key: {str(v): n for v, n in enumerate(cell) if n} for key, cell in sorted(table.items())}
                         for o, table in tables.items()}}
    return frequency, tables, record


def brute_rule(prefix):
    x0, x1, x2 = map(int, prefix)
    if x0 == x1:
        return None
    matches = [(a, b) for a in range(11) for b in range(11)
               if (a * x0 + b) % 11 == x1 and (a * x1 + b) % 11 == x2]
    assert len(matches) == 1
    return matches[0]


def next_symbol(inputs, name, frequency, tables, inferred):
    if name == 'current':
        return inputs[-1]
    if name == 'repeat_difference':
        return (inputs[-1] + (inputs[-1] - inputs[-2])) % 11
    if name == 'known_form_oracle':
        if inferred is None:
            return inputs[-1]
        return (inferred[0] * inputs[-1] + inferred[1]) % 11
    if name.startswith('history'):
        maximum = int(name.removeprefix('history'))
        candidates = [','.join(map(str, inputs[-o:])) for o in range(maximum, 0, -1)]
        for o, key in zip(range(maximum, 0, -1), candidates):
            if key in tables[str(o)]:
                return int(np.argmax(tables[str(o)][key]))
    return int(np.argmax(frequency))


def replay(rows, name, frequency, tables):
    teacher, rollout = np.empty((len(rows), 8), dtype=np.int64), np.empty((len(rows), 8), dtype=np.int64)
    identifiable = []
    for i, row in enumerate(rows.tolist()):
        inferred = brute_rule(row[:3])
        identifiable.append(inferred is not None)
        generated = row[:3]
        for j in range(8):
            teacher[i, j] = next_symbol(row[:3 + j], name, frequency, tables, inferred)
            rollout[i, j] = next_symbol(generated, name, frequency, tables, inferred)
            generated.append(int(rollout[i, j]))
    return teacher, rollout, np.array(identifiable, dtype=bool)


def recount(teacher, rollout, target):
    n, h = target.shape
    first = sum(int(t[0] == y[0]) for t, y in zip(teacher, target))
    correct = sum(int(x == y) for t, ys in zip(teacher, target) for x, y in zip(t, ys))
    prefix_sum, completed = 0, 0
    for generated, expected in zip(rollout.tolist(), target.tolist()):
        prefix = 0
        for x, y in zip(generated, expected):
            if x != y:
                break
            prefix += 1
        prefix_sum += prefix
        completed += int(prefix == h)
    return {'episodes': n, 'horizon': h, 'first_correct': first, 'first_accuracy': str(Fraction(first, n)),
            'teacher_correct': correct, 'teacher_predictions': n * h, 'teacher_accuracy': str(Fraction(correct, n * h)),
            'autonomous_prefix_sum': prefix_sum, 'autonomous_prefix_mean': str(Fraction(prefix_sum, n)),
            'autonomous_completed': completed, 'autonomous_completion_rate': str(Fraction(completed, n))}


def orbit(row):
    # Nonfixed multiplicative orbits have length <=10; use observed repeat,
    # instead of the producer's transition walk and rule coefficients.
    length = next(j for j in range(1, 11) if row[j] == row[0])
    observed = list(map(int, row[:length]))
    rotated = [tuple(observed[j:] + observed[:j]) for j in range(length)]
    return sorted(rotated)[0]


def recheck(data, meta, scores, c):
    a = {s: data[s]['affine'] for s in c['splits']}
    prefixes = {s: set(tuple(map(int, r[:3])) for r in v) for s, v in a.items()}
    windows = {s: set(tuple(map(int, r[j:j + 3])) for r in v for j in range(8)) for s, v in a.items()}
    cycles = {s: set(orbit(r) for r in v) for s, v in a.items()}
    sizes = all(len(a[s]) == c['expected_episodes'][s] for s in a)
    identified = all(brute_rule(r[:3]) == (m['a'], m['b']) for s in a for r, m in zip(a[s], meta[s]))
    unique = len(set().union(*prefixes.values())) == sum(len(v) for v in a.values())
    po = len(prefixes['train'].intersection(prefixes['parameter_test']))
    wo = len(windows['train'].intersection(windows['parameter_test']))
    co = len(cycles['train'].intersection(cycles['parameter_test']))
    maximum = max(Fraction(scores['affine']['parameter_test'][n]['first_accuracy']) for n in c['controls'] if n != 'known_form_oracle')
    oracle_complete = all(scores['affine'][s]['known_form_oracle']['autonomous_completed'] == len(a[s])
                          and scores['affine'][s]['known_form_oracle']['teacher_correct'] == 8 * len(a[s]) for s in a)
    negative = {f: scores[f]['parameter_test']['known_form_oracle']['first_accuracy'] for f in ('iid', 'shuffled')}
    gates = dict(split_sizes=sizes, identifiable=identified, unique_prefixes=unique,
                 no_parameter_prefix_leak=po == 0, no_parameter_teacher_window_leak=wo == 0,
                 no_parameter_cycle_leak=co == 0, oracle_complete=oracle_complete,
                 trivial_separation=maximum <= Fraction(c['maximum_trivial_parameter_first_accuracy']),
                 iid_separation=Fraction(negative['iid']) <= Fraction(c['maximum_iid_oracle_first_accuracy']),
                 shuffled_separation=Fraction(negative['shuffled']) <= Fraction(c['maximum_shuffled_oracle_first_accuracy']))
    return dict(exact_split_sizes=sizes, all_prefixes_identifiable=identified, unique_structured_prefixes=unique,
                parameter_prefix_overlap=po, parameter_teacher_window_overlap=wo, parameter_cycle_overlap=co,
                instance_cycle_overlap=len(cycles['train'].intersection(cycles['instance_test'])),
                distinct_cycles={s: len(v) for s, v in cycles.items()}, oracle_structured_complete=oracle_complete,
                trivial_parameter_first_max=str(maximum), negative_oracle_first=negative,
                gates=gates, all_task_gates_pass=all(gates.values()))


def verify(result, out):
    c, verifier_source = config(), source()
    with attempt(out, budget(c), 'm10-independent-task-verification') as resources:
        producer_source = authenticate(result)
        assert read(result / 'config.json') == c
        write(out / 'source.json', verifier_source)
        write(out / 'history.json', history(check_bytes=True))
        data, meta = reconstruct(c)
        assert meta == read(result / 'audit-metadata.json')
        scores, storage, arrays = {}, {}, 0
        for family in ('affine', 'iid', 'shuffled'):
            frequency, tables, serialized = count_training(data['train'][family])
            assert serialized == read(result / (family + '-training-tables.json'))
            storage[family] = dict(training_transitions=sum(frequency),
                                   suffix_cells={o: len(v) for o, v in tables.items()},
                                   suffix_count_entries={o: sum(sum(n > 0 for n in row) for row in v.values()) for o, v in tables.items()})
            scores[family] = {}
            with np.load(result / (family + '-predictions.npz'), allow_pickle=False) as predictions:
                expected_names = set()
                for split in c['splits']:
                    rows = data[split][family]
                    with np.load(result / (family + '-' + split + '-learner.npz'), allow_pickle=False) as saved:
                        assert set(saved.files) == {'prefixes', 'targets'}, 'Rule metadata leaked into learner arrays'
                        for name, expected in [('prefixes', rows[:, :3]), ('targets', rows[:, 3:])]:
                            np.testing.assert_array_equal(saved[name], expected)
                            arrays += 1
                    scores[family][split] = {}
                    for control in c['controls']:
                        tp, rp, identified = replay(rows, control, frequency, tables)
                        for mode, array in [('teacher', tp), ('rollout', rp)]:
                            name = split + '_' + control + '_' + mode
                            np.testing.assert_array_equal(predictions[name], array)
                            expected_names.add(name)
                            arrays += 1
                        identified_name = split + '_oracle_identifiable'
                        np.testing.assert_array_equal(predictions[identified_name], identified)
                        expected_names.add(identified_name)
                        scores[family][split][control] = recount(tp, rp, rows[:, 3:])
                    resources.check()
                assert set(predictions.files) == expected_names
        assert scores == read(result / 'metrics.json')
        assert storage == read(result / 'storage-budgets.json')
        design = recheck(data, meta, scores, c)
        assert design == read(result / 'design-checks.json')
        expected_summary = dict(outcome='READY_PENDING_INDEPENDENT_VERIFICATION' if design['all_task_gates_pass'] else 'INVALID',
                                all_task_gates_pass=design['all_task_gates_pass'], neural_model_fits=0,
                                structured_episodes=990, families=['affine', 'iid', 'shuffled'], control_evaluations=63,
                                future_performance_preregistered=False)
        assert expected_summary == read(result / 'summary.json')
        checks = dict(all_checks_pass=True, outcome='READY' if design['all_task_gates_pass'] else 'INVALID',
                      all_task_gates_pass=design['all_task_gates_pass'], independently_reconstructed_episodes=2970,
                      predictor_evaluations=63, exact_arrays_checked=arrays, oracle_identifiability_arrays=9,
                      all_predictions_metrics_tables_and_split_checks_exact=True,
                      result_manifest_sha256=sha(result / 'manifest.json'), producer_source_commit=producer_source['source_commit'],
                      no_producer_scientific_algorithms_imported=True, neural_model_fits=0)
        write(out / 'recomputed-metrics.json', scores)
        write(out / 'recomputed-design-checks.json', design)
        write(out / 'checks.json', checks)
        unchanged(verifier_source)
    seal(out, dict(complete=True, study='M10 independent task design'))
    print(checks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    verify(args.result, args.out)
