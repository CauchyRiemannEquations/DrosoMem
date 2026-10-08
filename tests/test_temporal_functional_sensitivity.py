"""Synthetic pre-outcome safeguards for the registered M5 assay."""
from fractions import Fraction
import numpy as np
import pytest
from scipy import sparse

from flying.brain.timed_reservoir import TimedReservoir
from flying.models.ridge import RidgeDecoder
from flying.training.whole_brain_memory import MappedEncoder
from temporal_mechanism import FactorReservoir
from sensitivity_dynamics import (PulseDynamics, replay, fd_gate,
    squared_gain_exact, primary_cell, endpoint)
from temporal_functional_sensitivity import anchors, decoding_rows


def fixture():
    roles = np.asarray(['KC', 'DAN', 'APL', 'MBON'])
    w = sparse.csr_matrix(([.9, .9, .45, .45], ([1, 2, 3, 3], [0, 1, 0, 2])), shape=(4, 4))
    return PulseDynamics(w, roles), roles


def test_same_step_chain_and_frozen_explicit_zero_state():
    w = sparse.csr_matrix(([.9], ([1], [0])), shape=(2, 2))
    roles = np.asarray(['KC', 'MBON'])
    dynamics = PulseDynamics(w, roles)
    patterns = np.zeros((10, 2)); patterns[:, 0] = np.arange(10)/20
    base = TimedReservoir(w, MappedEncoder(patterns), roles, .6, 'mbon_after_kc')
    model = FactorReservoir(base, False, True)
    previous = np.asarray([-.2, .3]); model.state = previous.copy()
    np.testing.assert_array_equal(dynamics.step(previous, patterns[9]), model.step(9))
    state, v, e = dynamics.linearized_step(np.zeros(2), np.asarray([.5, 0.]),
        np.zeros(2), np.zeros(2), np.asarray([.5, 0.]))
    expected = .6*(1-np.tanh(.9*.6*np.tanh(.5))**2)*.9*.6*(1-np.tanh(.5)**2)*.5
    assert v[1] == pytest.approx(expected, abs=1e-16)
    assert np.array_equal(v, e)
    np.testing.assert_array_equal(state, dynamics.step(np.zeros(2), np.asarray([.5, 0.])))


def test_jvp_complex_and_both_centered_epsilons():
    dynamics, _ = fixture()
    pre = np.asarray([.11, -.09, .05, -.02])
    inputs = np.asarray([[.4, 0., 0., 0.], [.1, 0., 0., 0.], [.3, 0., 0., 0.]])
    direction = np.asarray([-.5, 0., 0., 0.])
    state, v, e = pre.copy(), np.zeros(4), np.zeros(4)
    tangent = []
    for lag, u in enumerate(inputs):
        state, v, e = dynamics.linearized_step(state, u, v, e, direction if not lag else np.zeros(4))
        tangent.append(v.copy())
    tangent = np.asarray(tangent)
    complex_states, _ = replay(dynamics, pre.astype(complex), inputs, direction, 1e-20j)
    np.testing.assert_allclose(tangent, complex_states.imag/1e-20, atol=1e-12, rtol=1e-9)
    centered = []
    for epsilon in [1e-4, 1e-5]:
        plus, _ = replay(dynamics, pre, inputs, direction, epsilon)
        minus, _ = replay(dynamics, pre, inputs, direction, -epsilon)
        centered.append((plus-minus)/(2*epsilon))
    assert fd_gate(tangent, np.asarray(centered), 2e-9, 2e-5)['all_checks_pass']
    wrong = np.asarray(centered); wrong[1, 2, 1] += 1e-4
    assert not fd_gate(tangent, wrong, 2e-9, 2e-5)['all_checks_pass']


def test_signed_cancellation_despite_walk_support_and_envelope():
    roles = np.asarray(['KC', 'KC', 'DAN', 'APL', 'MBON'])
    w = sparse.csr_matrix(([.4, .4, .4, -.4], ([2, 3, 4, 4], [0, 0, 2, 3])), shape=(5, 5))
    dynamics = PulseDynamics(w, roles)
    direction = np.asarray([.5, 0., 0., 0., 0.])
    x = v = e = np.zeros(5)
    for lag in range(3):
        x, v, e = dynamics.linearized_step(x, np.zeros(5), v, e, direction if lag == 0 else np.zeros(5))
    assert v[4] == 0. and e[4] > 0.
    assert dynamics.support(direction[None, :], np.asarray([4]), list(range(3)))[0, 2, 0]


def test_hidden_delayed_signal_can_reemerge_in_projection():
    roles = np.asarray(['KC', 'DAN', 'MBON'])
    w = sparse.csr_matrix(([.9, .9], ([1, 2], [0, 1])), shape=(3, 3))
    dynamics = PulseDynamics(w, roles)
    x = v = e = np.zeros(3); observed = []
    for lag in range(3):
        x, v, e = dynamics.linearized_step(x, np.zeros(3), v, e,
            np.asarray([.5, 0., 0.]) if not lag else np.zeros(3))
        observed.append(v[2])
    assert observed[:2] == [0., 0.] and observed[2] > 0.


def test_instantaneous_zero_past_and_valid_single_symbol_replacement():
    dynamics, roles = fixture()
    instant = PulseDynamics(dynamics.weights, roles, synaptic_history=False)
    inputs = np.asarray([[.1, 0., 0., 0.], [.3, 0., 0., 0.], [.2, 0., 0., 0.]])
    direction = np.asarray([.4, 0., 0., 0.])
    pre = np.ones(4)*.2
    baseline, _ = replay(instant, pre, inputs)
    replaced, _ = replay(instant, pre, inputs, direction, 1.)
    np.testing.assert_array_equal(replaced[0], instant.step(pre, np.asarray([.5, 0., 0., 0.])))
    np.testing.assert_array_equal(replaced[1:], baseline[1:])
    x, v, e = pre.copy(), np.zeros(4), np.zeros(4)
    for lag, u in enumerate(inputs):
        x, v, e = instant.linearized_step(x, u, v, e, direction if not lag else np.zeros(4))
        if lag:
            np.testing.assert_array_equal(v, np.zeros(4))
            np.testing.assert_array_equal(e, np.zeros(4))
    unchanged = inputs.copy()
    replay(dynamics, pre, inputs, direction, 1.)
    np.testing.assert_array_equal(inputs, unchanged)


def test_all_coordinate_envelope_and_contraction_bounds():
    dynamics, _ = fixture()
    inputs = np.zeros((21, 4)); inputs[:, 0] = .3
    direction = np.asarray([-.5, 0., 0., 0.])
    pre = np.asarray([.1, -.2, .3, -.4])
    baseline, _ = replay(dynamics, pre, inputs)
    replaced, _ = replay(dynamics, pre, inputs, direction, 1.)
    x, v, e = pre.copy(), np.zeros(4), np.zeros(4)
    for lag, u in enumerate(inputs):
        x, v, e = dynamics.linearized_step(x, u, v, e, direction if not lag else np.zeros(4))
        assert np.all(np.abs(v) <= e+1e-12)
        bound = .3*.54**lag
        assert np.max(np.abs(v)) <= bound+1e-12
        assert np.max(e) <= bound+1e-12
        assert np.max(np.abs(replaced[lag]-baseline[lag])) <= bound+1e-12


def test_exact_float_squares_and_per_probe_normalization():
    vectors = np.asarray([[.1, .2], [.3, .4]])
    denominators = np.asarray([.25, 1.])
    expected = sum((sum((Fraction.from_float(float(x))**2 for x in row), Fraction(0)) /
        Fraction.from_float(float(d)) for row, d in zip(vectors, denominators)), Fraction(0))/2
    assert squared_gain_exact(vectors, denominators) == expected
    assert float(expected) != pytest.approx(float(np.sum(vectors*vectors)/np.sum(denominators)), abs=1e-6)
    with pytest.raises(ValueError):
        squared_gain_exact(vectors, np.asarray([0., 1.]))


def test_exact_equality_boundary_all_seed_conjunction_and_smoke_null():
    tangent = np.zeros((1, 21, 101))
    tangent[0, 0, :100] = 1.
    tangent[0, 5, 0] = 1.
    good = primary_cell(tangent, np.asarray([1.]), 1, 'legacy5')
    assert good['scientific_rule_pass'] is True
    assert Fraction(good['s5_numerator'], good['s5_denominator']) == Fraction(good['s0_numerator'], good['s0_denominator'])/100
    failed = dict(good, seed=6, scientific_rule_pass=False)
    assert endpoint([good]*5+[failed])['outcome'] == 'FAIL'
    assert endpoint([good]*6, smoke=True)['registered_rule_pass'] is None
    assert endpoint([good]*6)['outcome'] == 'PASS'
    bad = primary_cell(np.zeros_like(tangent), np.asarray([1.]), 1, 'legacy5')
    assert bad['eligible'] is False
    assert endpoint([bad])['registered_rule_pass'] is None
    zero = primary_cell(tangent, np.asarray([0.]), 1, 'legacy5')
    assert zero['eligible'] is False


def test_exact_decimal_current_resolution_gate():
    tangent = np.zeros((1, 21, 1))
    tangent[0, 0, 0] = np.nextafter(1e-8, 0.)
    assert primary_cell(tangent, np.ones(1), 1, 'legacy5')['eligible'] is False
    tangent[0, 0, 0] = np.nextafter(1e-8, np.inf)
    assert primary_cell(tangent, np.ones(1), 1, 'legacy5')['eligible'] is True


def test_registered_anchor_and_lag_label_alignment():
    c = dict(warmup=200, test_samples=2000, lags=list(range(21)), probe_count=32)
    chosen = anchors(c)
    assert len(chosen) == 32 and chosen[0] == 200 and chosen[-1] == 2179
    assert len(np.unique(chosen)) == 32 and chosen[-1]+20 == 2199
    symbols = np.arange(2200) % 10
    times = np.arange(200, len(symbols))
    labels = symbols[times[:, None]-np.asarray(c['lags'])[None, :]]
    np.testing.assert_array_equal(labels[:, 0], symbols[times])
    np.testing.assert_array_equal(labels[:, 5], symbols[times-5])


def test_ridge_normalization_uses_only_training_features():
    train = np.asarray([[1., 0.], [2., 0.], [3., 0.], [4., 0.]])
    head = RidgeDecoder(1.).fit(train, np.eye(2)[[0, 1, 0, 1]])
    np.testing.assert_array_equal(head.mean, train.mean(axis=0))
    np.testing.assert_array_equal(head.scale, np.maximum(train.std(axis=0), 1e-5))
    original_mean, original_scale = head.mean.copy(), head.scale.copy()
    head.scores(np.asarray([[1e9, -1e9]]))
    np.testing.assert_array_equal(head.mean, original_mean)
    np.testing.assert_array_equal(head.scale, original_scale)
