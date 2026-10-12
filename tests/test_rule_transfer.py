"""M11 synthetic dynamics, independent fitting, no-target rollout and exact gates."""
import ast
import inspect
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from scipy import sparse
import rule_transfer as main
import verify_rule_transfer as independent


class Resources:
    def check(self):
        pass


def test_shift_encodes_exactly_current_and_two_previous_symbols_with_padding():
    m = main.Shift()
    state = np.zeros(686)
    definition = dict(kind='shift', n=686, observed=np.arange(48))
    for symbol in [2, 4, 8]:
        state = independent.step(definition, state, symbol)
        np.testing.assert_array_equal(m.step(symbol), state)
    assert np.argmax(state[:11]) == 8 and np.argmax(state[11:22]) == 4 and np.argmax(state[22:33]) == 2
    assert not state[33:].any()
    m.reset()
    assert not m.state.any()


def test_source_zero_carry_and_same_step_kc_mbon_match_independent_manual_update():
    rng = np.random.default_rng(83)
    w = sparse.csr_matrix(rng.normal(size=(7, 7)) * .03)
    patterns = rng.normal(size=(11, 7)) * .1
    kc, mb = np.array([0, 1]), np.array([5, 6])
    base = SimpleNamespace(weights=w, encoder=main.MappedEncoder(patterns), kc=kc, mbon=mb, mbon_weights=w[mb], leak=.6)
    model = main.FactorReservoir(base, False, True)
    model.state = rng.normal(size=7)
    definition = dict(kind='source', weights=w, patterns=patterns, kc=kc, mb=mb, mb_weights=w[mb])
    before = main.weight_hash(w)
    expected = independent.step(definition, model.state, 3)
    np.testing.assert_array_equal(model.step(3), expected)
    assert before == main.weight_hash(w)


def test_episode_reset_prevents_previous_episode_carryover():
    model = main.Shift()
    rows = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6]])
    first, hashes = main.teacher_features(model, np.arange(48), rows, Resources())
    second, h2 = main.teacher_features(model, np.arange(48), rows[1:], Resources())
    np.testing.assert_array_equal(first[1:], second)
    assert hashes[1:] == h2


def test_equal_539_parameter_ridge_and_independent_svd_with_train_only_scaler():
    rng = np.random.default_rng(84)
    x = rng.normal(size=(200, 48))
    x[:, 35:] = 0
    y = rng.integers(11, size=200)
    head = main.RidgeDecoder(1.).fit(x, np.eye(11)[y])
    fitted = independent.fit_svd(x, y)
    np.testing.assert_allclose(head.weights, fitted['coefficients'], atol=1e-9, rtol=1e-9)
    np.testing.assert_array_equal(head.mean, fitted['mean'])
    assert head.weights.size + head.target_mean.size == 539
    assert (head.scale[35:] == 1e-5).all()
    snapshot = head.mean.copy()
    head.scores(np.ones((4, 48)) * 999)
    np.testing.assert_array_equal(head.mean, snapshot)
    head.weights.flags.writeable = False
    with pytest.raises(ValueError):
        head.weights[0, 0] = 0


def test_target_free_rollout_signature_and_independent_full_state_hashes():
    assert all('target' not in p for p in inspect.signature(main.target_free_rollout).parameters)
    rng = np.random.default_rng(85)
    h = main.RidgeDecoder(1.).fit(rng.normal(size=(70, 48)), np.eye(11)[rng.integers(11, size=70)])
    head = dict(mean=h.mean, scale=h.scale, coefficients=h.weights, intercept=h.target_mean)
    prefixes = np.array([[0, 1, 2], [4, 3, 5]])
    generated, scores, full, observed = main.target_free_rollout(main.Shift(), np.arange(48), h, prefixes, 8, Resources())
    expected, logits, digest, obs = independent.rollout(dict(kind='shift', n=686, observed=np.arange(48)), (head, head), prefixes, Resources())
    np.testing.assert_array_equal(generated, expected)
    np.testing.assert_allclose(scores, logits, atol=1e-9, rtol=1e-9)
    assert full == digest and observed == obs


def test_prospective_stable_ties_choose_lowest_label_in_both_scorers():
    row = np.zeros(11)
    row[2], row[6] = 1., 1. + .5e-10
    assert int(main.decisions(row)) == int(independent.classify(row)) == 2
    row[6] = 1. + 2e-10
    assert int(main.decisions(row)) == int(independent.classify(row)) == 6


def synthetic_cells():
    c = main.config()
    rows = []
    for seed in c['input_seeds']:
        for graph in c['models']:
            rows.append(dict(family='affine', model=graph, input_seed=seed, split='parameter_test', group='nonperiod2', episodes=100,
                             first_correct=25, teacher_predictions=800, teacher_correct=200, completed=10 if graph == 'real' else 5 if graph == 'shift' else 4))
        rows.append(dict(family='affine', model='real', input_seed=seed, split='train', group='all', episodes=25, teacher_predictions=200, teacher_correct=100))
        for a in (7, 8, 9, 10):
            rows.append(dict(family='affine', model='real', input_seed=seed, split='parameter_test', group=f'a{a}', episodes=100, completed=5))
    return c, rows


def test_exact_primary_boundary_and_failure_cannot_be_rescued_by_secondary():
    c, rows = synthetic_cells()
    result = main.summarize(rows, c)
    assert result == independent.independently_summarize(rows, c, False)
    assert result['primary_pass'] and result['secondary_pass']
    assert result['cohorts']['discovery']['real_minus_shift_completion'] == '1/20'
    next(r for r in rows if r['model'] == 'real' and r['group'] == 'nonperiod2')['first_correct'] = 24
    result = main.summarize(rows, c)
    assert result == independent.independently_summarize(rows, c, False)
    assert not result['primary_pass'] and not result['secondary_pass']


def test_prefix_and_completion_do_not_equal_teacher_accuracy():
    target = np.tile(np.arange(8), (2, 1))
    generated = target.copy()
    generated[0, 0], generated[1, 3] = 9, 9
    result = main.recount(target, generated, target, np.ones(2, dtype=bool))
    assert result['teacher_correct'] == 16 and result['completed'] == 0 and result['prefix_sum'] == 3
    assert result == independent.count_metrics(target, generated, target, [0, 1])


def test_verifier_has_no_m11_model_fitter_or_scorer_import():
    tree = ast.parse(Path(independent.__file__).read_text(encoding='utf-8'))
    imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    imports += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    assert 'rule_transfer' not in imports and not any(n.startswith('flying') for n in imports)
