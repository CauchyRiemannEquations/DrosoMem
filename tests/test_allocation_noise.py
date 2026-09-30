import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import allocation_noise as study
from verify_allocation_noise import independent_noise, independent_head
from flying.models.nonlinear_readout import NonlinearReadout


def test_expected_energy_heterogeneity_and_zero_variance():
    training = np.array([[-1., -2., 0.], [1., 2., 0.]])
    sd, q, a = study.allocation(training)
    np.testing.assert_allclose(sd, [1, 2, 0]); assert q == 1
    assert a[2] == 0 and a[1] == 2*a[0]
    np.testing.assert_allclose(np.sum((.3*q*a)**2), 3*(.3*q)**2, rtol=1e-14)
    _, _, uniform = study.allocation(np.array([[-1., 1.], [1., -1.]]))
    np.testing.assert_array_equal(uniform, np.ones(2))


@pytest.mark.parametrize('bad', [np.zeros((5, 3)), np.ones((1, 3)), np.array([[np.nan, 0.], [1., 1.]])])
def test_nonidentifiable_training_stops(bad):
    with pytest.raises(ValueError):
        study.allocation(bad)


def test_no_evaluation_fit_and_scale_invariance():
    training = np.array([[-.01, -.3], [.01, .3], [.02, -.1]])
    _, q, a = study.allocation(training); _, q2, a2 = study.allocation(7*training)
    np.testing.assert_allclose(a, a2); np.testing.assert_allclose(q2, 7*q)
    assert study.perturb(np.zeros((2, 2)), np.ones((2, 2)), a*q).shape == (2, 2)


def test_zero_clipping_and_immutability():
    clean = np.array([[.9, -.9], [.1, .2]]); noise = np.array([[2., -3.], [-1., 2.]])
    before = clean.copy(); z_before = noise.copy()
    np.testing.assert_array_equal(study.perturb(clean, noise, np.zeros(2)), clean)
    np.testing.assert_allclose(study.perturb(clean, noise, np.array([.2, .1])), [[1., -1.], [-.1, .4]])
    np.testing.assert_array_equal(clean, before); np.testing.assert_array_equal(noise, z_before)
    with pytest.raises(ValueError):
        study.perturb(clean, noise, np.array([-.1, .1]))


def test_canonical_pairing_reordering_and_distinct_fresh_streams():
    ids = np.array([44, 22, 39]); a = study.fresh_bank(ids, 7, 701, 408001, 11)
    b = study.fresh_bank(ids[::-1], 7, 701, 408001, 11)
    np.testing.assert_array_equal(a[:, ::-1], b)
    np.testing.assert_array_equal(a, independent_noise(ids, 7, 701, 408001, 11))
    assert not np.array_equal(a, study.fresh_bank(ids, 7, 701, 408002, 11))


def test_expected_energy_is_not_realized_energy_or_clipped_energy():
    noise = np.array([[2., 0.]]); weights = np.array([np.sqrt(2), 0.])
    assert np.sum(weights**2) == pytest.approx(2)
    assert np.sum((noise*weights)**2) != np.sum(noise**2)
    clean = np.array([[.99, 0.]])
    x = study.perturb(clean, noise, weights)
    assert np.sum((x-clean)**2) < np.sum((noise*weights)**2)


def test_exact_count_ties_and_four_cell_confirmation():
    assert study.gain(17, 17, 197) == 0
    assert study.gain(17, 18, 197) == 1/197
    c = study.config(); gates = {f'{co}/{stage}/{g}': True for co in c['cohorts'] for stage in ['archived', 'fresh'] for g in c['conditions']}
    assert study.confirmed(gates, c) == c['conditions']
    gates['confirmation/fresh/brain1'] = False
    assert 'brain1' not in study.confirmed(gates, c)
    assert not study.gate([.04]*3, 'confirmation', c)
    assert not study.gate([.2, .2, 0], 'confirmation', c)


def test_independent_frozen_head():
    x = np.array([[-.01, .4], [.02, -.3], [.003, .1]])
    head = NonlinearReadout([0, 1], 8, 10); head.initialize(x); before = head.digest()
    cp = dict(mean=head.mean, scale=head.scale, **head.parameters)
    p = independent_head(x, cp); logits = head.logits(x)
    exp = np.exp(logits-logits.max(1, keepdims=True))
    np.testing.assert_allclose(p, exp/exp.sum(1, keepdims=True), atol=1e-14)
    np.testing.assert_array_equal(p.argmax(1), head.predict(x)); assert head.digest() == before
