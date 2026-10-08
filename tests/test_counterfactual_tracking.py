"""Prospective synthetic M6 safeguards; no source-graph outcomes."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from counterfactual_tracking import (scoring, extract_head, joint_pass, trial_counts,
    endpoint, anchor_times, replacement_window, replay_window, score_trials)


def test_one_actual_symbol_replacement_preserves_every_other_input_and_labels():
    symbols = np.asarray([8, 1, 4, 7, 2, 9, 3])
    before = symbols.copy()
    original, replacement = replacement_window(symbols, 2)
    np.testing.assert_array_equal(original, [4, 7, 2])
    np.testing.assert_array_equal(replacement, [5, 7, 2])
    np.testing.assert_array_equal(symbols, before)
    assert symbols[2+2] == 2 and original[0] == 4 and replacement[0] == 5
    assert original[0] != original[-1]  # lag2 target is not current symbol.
    original, replacement = replacement_window(np.asarray([9, 4, 8]), 0)
    np.testing.assert_array_equal(replacement, [0, 4, 8])


def test_anchor_endpoints_and_fixed_denominator_are_in_bounds_in_both_stages():
    for samples in [200, 2000]:
        anchors = anchor_times(dict(warmup=200, test_samples=samples, primary_lag=2, probe_count=64))
        assert len(anchors) == len(np.unique(anchors)) == 64
        assert anchors[0] == 200 and anchors[-1]+2 == 200+samples-1


def test_parent_lag_major_head_has_exact_columns_and_does_not_mutate_parent():
    coefficients = np.arange(48*210, dtype=np.float64).reshape(48, 210)
    intercept = np.arange(210, dtype=np.float64)
    before = coefficients.copy()
    head, bias = extract_head(coefficients, intercept, 2)
    np.testing.assert_array_equal(head, coefficients[:, 20:30])
    np.testing.assert_array_equal(bias, np.arange(20, 30))
    head[0, 0] = -1
    np.testing.assert_array_equal(coefficients, before)
    np.testing.assert_array_equal(extract_head(coefficients, intercept, 0)[1], np.arange(10))
    assert head.size+bias.size == 490


def test_scoring_uses_frozen_scaler_and_bias_and_lowest_tie_index():
    features = np.asarray([[103., 12.], [105., 16.]])
    mean, scale = np.asarray([100., 10.]), np.asarray([2., 4.])
    head = np.asarray([[1., 1., 0.], [0., 0., 1.]])
    intercept = np.asarray([.25, .25, -.5])
    snapshots = [value.copy() for value in [mean, scale, head, intercept]]
    scores = scoring(features, mean, scale, head, intercept)
    np.testing.assert_array_equal(scores, [[1.75, 1.75, 0.], [2.75, 2.75, 1.]])
    np.testing.assert_array_equal(scores.argmax(axis=1), [0, 0])
    for value, before in zip([mean, scale, head, intercept], snapshots):
        np.testing.assert_array_equal(value, before)
    # New-feature means differ from archived means and never replace them.
    assert not np.array_equal(features.mean(axis=0), mean)


def test_joint_credit_requires_both_worlds_at_same_unfiltered_anchor():
    s, r = np.zeros(64, dtype=np.int64), np.ones(64, dtype=np.int64)
    original = np.r_[np.zeros(32, dtype=np.int64), np.full(32, 2)]
    replacement = np.r_[np.full(32, 2), np.ones(32, dtype=np.int64)]
    counts = trial_counts(original, replacement, s, r)
    assert counts['samples'] == 64 and counts['original_correct'] == counts['replacement_correct'] == 32
    assert counts['joint_correct'] == 0 and counts['joint_accuracy'] == 0
    # Unchanged pair predictions cannot jointly match distinct targets.
    assert trial_counts(s, s, s, r)['joint_correct'] == 0
    with pytest.raises(AssertionError):
        trial_counts(s, s, s, s)


def test_exact_registered_boundary_and_all_block_conjunction():
    assert not joint_pass(57, 64) and joint_pass(58, 64)
    assert joint_pass(90, 100) and not joint_pass(89, 100)
    cells = [dict(samples=64, joint_correct=64, technical_checks_pass=True) for _ in range(6)]
    assert endpoint(cells)['registered_rule_pass'] is True
    cells[-1]['joint_correct'] = 57
    assert sum(row['joint_correct'] for row in cells)/(6*64) > .9
    assert endpoint(cells)['registered_rule_pass'] is False
    assert endpoint(cells)['outcome'] == 'FAIL'


def test_poor_original_accuracy_remains_scientific_failure_without_filtering():
    s, r = np.zeros(64, dtype=np.int64), np.ones(64, dtype=np.int64)
    original = np.full(64, 2, dtype=np.int64)
    original[:5] = 0
    counts = trial_counts(original, r, s, r)
    assert counts['original_correct'] == counts['joint_correct'] == 5
    assert counts['replacement_correct'] == 64 and counts['samples'] == 64
    cell = dict(**counts, technical_checks_pass=True)
    result = endpoint([dict(cell) for _ in range(6)])
    assert result == dict(outcome='FAIL', registered_rule_pass=False)


def test_smoke_endpoint_is_null_even_if_every_pair_succeeds():
    cells = [dict(samples=64, joint_correct=64, technical_checks_pass=True)]
    assert endpoint(cells, smoke=True) == dict(outcome='smoke', registered_rule_pass=None)


class ToyEncoder:
    def __init__(self):
        self.patterns = np.zeros((10, 3))
        self.patterns[:, 0] = np.arange(10)/20

    def __call__(self, symbol):
        return self.patterns[int(symbol)].copy()


def toy_base():
    weights = sparse.csr_matrix([[0., 0., 0.], [.5, 0., 0.], [.1, .5, 0.]])
    return SimpleNamespace(weights=weights, encoder=ToyEncoder(), kc=np.asarray([0]),
        mbon=np.asarray([2]), mbon_weights=weights[[2]], leak=.6)


def test_direct_discrete_pulse_has_identical_future_and_instant_zero_history():
    base = toy_base()
    initial = np.asarray([.12, -.02, .25])
    snapshot = initial.copy()
    original = np.asarray([1, 4, 6])
    replacement = np.asarray([2, 4, 6])
    features, final, trace = replay_window(base, initial, original, np.arange(3), True)
    changed, changed_final, changed_trace = replay_window(base, initial, replacement, np.arange(3), True)
    assert len(trace) == len(changed_trace) == 64 and trace != changed_trace
    assert not np.array_equal(features, changed)
    np.testing.assert_array_equal(initial, snapshot)
    first, endpoint_original, _ = replay_window(base, initial, original, np.arange(3), False)
    second, endpoint_changed, _ = replay_window(base, initial, replacement, np.arange(3), False)
    assert not np.array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1:], second[1:])
    np.testing.assert_array_equal(endpoint_original, endpoint_changed)


def test_trial_panels_have_unconditional_joint_credit_zero_controls_and_linear_margin_identity():
    s = np.arange(64, dtype=np.int64)%10
    r = (s+1)%10
    head = np.zeros((48, 10)); head[:10] = np.eye(10)
    original, replacement = np.zeros((64, 3, 48)), np.zeros((64, 3, 48))
    original[np.arange(64), 2, s] = 1
    replacement[np.arange(64), 2, r] = 1
    reference = np.zeros_like(original)
    a = dict(mean=np.zeros(48), scale=np.ones(48), coefficients_lag2=head,
        intercept_lag2=np.zeros(10), original_symbols=s, alternative_symbols=r,
        current_symbols=np.full(64, 2, dtype=np.int64), majority_lag2=np.asarray(3),
        current_lookup_lag2=np.arange(10), anchors=np.arange(64)+200,
        evaluation_times=np.arange(64)+202,
        trial_original_features=original, trial_counterfactual_features=replacement,
        trial_instantaneous_original_features=reference,
        trial_instantaneous_counterfactual_features=reference.copy())
    c = dict(score_atol=1e-10, score_rtol=1e-10)
    cell, rows = score_trials(a, c, dict(seed=1600001, parent_seed=1500001), 'legacy5')
    assert cell['joint_correct'] == 64 and cell['scientific_rule_pass'] is True
    assert cell['frequency_joint_correct'] == cell['current_joint_correct'] == cell['instantaneous_joint_correct'] == 0
    np.testing.assert_array_equal(a['contrast_margin_change'], a['projected_contrast_margin_change'])
    assert len(rows) == 64 and all(row['joint_correct'] for row in rows)
    assert score_trials(a, c, dict(seed=1600001, parent_seed=1500001), 'legacy5', smoke=True)[0]['scientific_rule_pass'] is None
