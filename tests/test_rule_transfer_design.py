"""Small safeguards for withheld rules, learner input and target-free rollout."""
import numpy as np
import rule_transfer_design as main
import verify_rule_transfer_design as independent
import ast
from pathlib import Path


def settings():
    return dict(modulus=11, prefix_length=3, horizon=8)


def test_nonconstant_prefix_identifies_held_out_rule_and_constant_is_flagged():
    row, _ = main.affine_episode(9, 4, 6, 11, 11)
    assert main.infer(row[:3], 11) == (9, 4)
    assert main.infer([2, 2, 2], 11) is None


def test_rule_assignment_and_cycle_rotation_are_distinct():
    row, _ = main.affine_episode(2, 4, 1, 11, 11)
    a = main.cycle_signature(2, 4, row[0], 11)
    assert a == main.cycle_signature(2, 4, row[1], 11)
    assert a != main.cycle_signature(9, 4, row[0], 11)


def test_targets_change_teacher_inputs_but_cannot_change_rollout():
    c = settings()
    train = np.asarray([main.affine_episode(2, 3, y, 11, 11)[0] for y in (1, 2)])
    tables = main.fit_tables(train, c)
    prefix = np.array([[2, 3, 4]])
    first = np.zeros((1, 8), dtype=np.int64)
    second = np.full((1, 8), 7, dtype=np.int64)
    for name in ('frequency', 'current', 'repeat_difference', 'history1', 'history2', 'history3', 'known_form_oracle'):
        t1, r1 = main.predictions(prefix, first, name, tables, c)
        t2, r2 = main.predictions(prefix, second, name, tables, c)
        np.testing.assert_array_equal(r1, r2)
        assert t1[0, 0] == t2[0, 0]
    assert not np.array_equal(t1[:, 1:], t2[:, 1:])


def test_no_transitions_cross_episodes_and_ties_take_smallest_symbol():
    c = settings()
    train = np.array([[0, 1, 2, 3], [9, 8, 7, 4]])
    counts, tables = main.fit_tables(train, c)
    assert sum(counts.values()) == 2
    assert (3, 9) not in tables[2]
    assert main.winner({7: 3, 2: 3}) == 2


def test_exact_prefix_completion_is_separate_from_teacher_accuracy():
    target = np.array([[1, 2, 3], [4, 5, 6]])
    teacher = target.copy()
    rollout = np.array([[0, 2, 3], [4, 5, 0]])
    report = main.metrics(teacher, rollout, target)
    assert report['teacher_accuracy'] == '1'
    assert report['autonomous_prefix_mean'] == '1'
    assert report['autonomous_completed'] == 0


def test_independent_bruteforce_and_replay_agree_on_small_negative_example():
    c = settings()
    train = np.array([main.affine_episode(2, 3, y, 11, 11)[0] for y in (1, 2)])
    fitted = main.fit_tables(train, c)
    freq, tables, record = independent.count_training(train)
    assert record == main.table_record(fitted)
    test = np.array([[2, 2, 4, 8, 7, 5, 4, 1, 3, 0, 9]])
    for name in ('frequency', 'current', 'repeat_difference', 'history1', 'history2', 'history3', 'known_form_oracle'):
        teacher, rollout = main.predictions(test[:, :3], test[:, 3:], name, fitted, c)
        tp, rp, identified = independent.replay(test, name, freq, tables)
        np.testing.assert_array_equal(tp, teacher)
        np.testing.assert_array_equal(rp, rollout)
        assert not identified[0]
        assert independent.recount(tp, rp, test[:, 3:]) == main.metrics(teacher, rollout, test[:, 3:])


def test_verifier_imports_no_producer_scientific_algorithms():
    code = ast.parse(Path(independent.__file__).read_text(encoding='utf-8'))
    imports = [node.module for node in ast.walk(code) if isinstance(node, ast.ImportFrom)]
    imports += [alias.name for node in ast.walk(code) if isinstance(node, ast.Import) for alias in node.names]
    assert 'rule_transfer_design' not in imports
