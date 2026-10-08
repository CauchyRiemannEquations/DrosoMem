"""Synthetic independent-derivative guards; no source-connectome outcomes."""
from fractions import Fraction
import unittest

import numpy as np
from scipy import sparse

from verify_temporal_functional_sensitivity import (
    manual_step, complex_window, envelope_window, real_window,
    structural_window, finite_difference_gate, exact_power,
    exact_block_endpoint, endpoint, independent_ridge, cell_from_vectors,
    independent_summary)


class FunctionalSensitivityVerifierTests(unittest.TestCase):
    def test_current_kc_to_mbon_complex_chain_rule(self):
        w = sparse.csr_matrix([[0., .3], [.8, 0.]])
        kc, mb = np.array([0]), np.array([1])
        initial, inputs, direction = np.array([.2, -.1]), np.array([[.5, 0.]]), np.array([.5, 0.])
        result = complex_window(w, kc, mb, initial, inputs, direction)[0]
        first = .6*np.tanh(.3*initial[1]+.5)
        vk = .6*(1-np.tanh(.3*initial[1]+.5)**2)*.5
        vm = .6*(1-np.tanh(.8*first)**2)*.8*vk
        np.testing.assert_allclose(result, [vk, vm], atol=1e-14, rtol=1e-14)
        state = manual_step(w, kc, mb, initial, inputs[0])
        np.testing.assert_allclose(state, [first, .6*np.tanh(.8*first)], atol=1e-14)

    def test_both_centered_epsilon_gates_full_coordinates(self):
        w = sparse.csr_matrix([[0., -.4, 0.], [.3, 0., 0.], [.2, .4, 0.]])
        kc, mb = np.array([0]), np.array([2])
        initial = np.array([.2, -.1, .05])
        inputs = np.array([[.5, 0., 0.], [0., 0., 0.], [.5, 0., 0.], [0., 0., 0.]])
        direction = np.array([-.5, 0., 0.])
        tangent = complex_window(w, kc, mb, initial, inputs, direction)
        for epsilon in [1e-4, 1e-5]:
            plus, _ = real_window(w, kc, mb, initial, inputs, direction, epsilon)
            minus, _ = real_window(w, kc, mb, initial, inputs, direction, -epsilon)
            fd, _, excess, passed = finite_difference_gate(plus, minus, tangent, epsilon)
            self.assertTrue(passed)
            self.assertTrue(np.all(excess <= 0))
            np.testing.assert_allclose(fd, tangent, atol=2e-9, rtol=2e-5)
        corrupted = tangent.copy()
        corrupted[1, 1] += .01
        self.assertFalse(finite_difference_gate(plus, minus, corrupted, epsilon)[-1])

    def test_signed_cancellation_despite_structural_and_unsigned_support(self):
        w = sparse.csr_matrix([[0., 0., 0., 0.], [.4, 0., 0., 0.],
                               [-.4, 0., 0., 0.], [0., .4, .4, 0.]])
        kc, mb = np.array([0]), np.array([3])
        initial, inputs, direction = np.zeros(4), np.zeros((4, 4)), np.array([.5, 0., 0., 0.])
        tangent = complex_window(w, kc, mb, initial, inputs, direction)
        envelope = envelope_window(w, kc, mb, initial, inputs, direction)
        support = structural_window(w, kc, mb, direction != 0, 4)
        self.assertTrue(support[2, 3])
        self.assertGreater(envelope[2, 3], 0)
        self.assertEqual(tangent[2, 3], 0)
        self.assertTrue(np.all(np.abs(tangent) <= envelope+1e-14))

    def test_observed_gain_can_reemerge_after_zero(self):
        w = sparse.csr_matrix([[0., 0., 0.], [.4, 0., 0.], [.001, .4, 0.]])
        kc, mb = np.array([0]), np.array([2])
        initial, inputs, direction = np.zeros(3), np.zeros((4, 3)), np.array([.5, 0., 0.])
        tangent = complex_window(w, kc, mb, initial, inputs, direction)
        self.assertGreater(tangent[0, 2], 0)
        self.assertEqual(tangent[1, 2], 0)
        self.assertGreater(tangent[2, 2], tangent[0, 2])
        bound = .3*.54**np.arange(4)
        self.assertTrue(np.all(np.max(np.abs(tangent), axis=1) <= bound))

    def test_instantaneous_history_is_exact_zero_at_every_coordinate(self):
        w = sparse.csr_matrix([[0., -.4], [.8, 0.]])
        kc, mb = np.array([0]), np.array([1])
        initial = np.array([.2, -.1])
        inputs = np.array([[.5, 0.], [0., 0.], [.5, 0.], [0., 0.]])
        direction = np.array([-.5, 0.])
        tangent = complex_window(w, kc, mb, initial, inputs, direction, history=False)
        baseline, _ = real_window(w, kc, mb, initial, inputs, history=False)
        altered, _ = real_window(w, kc, mb, initial, inputs, direction, 1., history=False)
        self.assertEqual(np.count_nonzero(tangent[1:]), 0)
        np.testing.assert_array_equal(altered[1:], baseline[1:])
        direct = inputs.copy(); direct[0] += direction
        actual, _ = real_window(w, kc, mb, initial, direct, history=False)
        np.testing.assert_array_equal(actual, altered)
        self.assertGreater(abs(tangent[0, 1]), 0)

    def test_decimal_eligibility_and_exact_primary_equality(self):
        # Binary-exact components ensure S5 == S0/100 exactly using norm weight.
        power = [Fraction(1)]*6
        power[5] = Fraction(1, 100)
        result = exact_block_endpoint(power, True)
        self.assertTrue(result['eligible'])
        self.assertTrue(result['attenuation_pass'])
        power[5] += Fraction(1, 10**30)
        self.assertFalse(exact_block_endpoint(power, True)['attenuation_pass'])
        power[0] = Fraction(1, 10**16)
        self.assertTrue(exact_block_endpoint(power, True)['eligible'])
        power[0] -= Fraction(1, 10**30)
        self.assertFalse(exact_block_endpoint(power, True)['eligible'])

    def test_all_blocks_conjunction_smoke_null_and_zero_direction(self):
        cells = [dict(eligible=True, attenuation_pass=True) for _ in range(6)]
        self.assertTrue(endpoint(cells)['primary_pass'])
        cells[-1]['attenuation_pass'] = False
        self.assertFalse(endpoint(cells)['primary_pass'])
        self.assertIsNone(endpoint(cells, smoke=True)['primary_pass'])
        cells[-1]['eligible'] = False
        self.assertEqual(endpoint(cells)['outcome'], 'assay-invalid')
        self.assertIsNone(endpoint(cells)['primary_pass'])
        _, valid = exact_power(np.ones((1, 6, 2)), np.zeros((1, 2)))
        self.assertFalse(valid)
        c = dict(primary_lag=5)
        cell = cell_from_vectors(np.ones((1, 6, 2)), np.zeros(1), 'legacy5', 1, c)
        self.assertFalse(cell['eligible'])
        self.assertIsNone(cell['scientific_rule_pass'])
        cell = cell_from_vectors(np.zeros((1, 6, 2)), np.ones(1), 'legacy5', 1, c)
        self.assertFalse(cell['eligible'])
        self.assertIsNone(cell['scientific_rule_pass'])

    def test_actual_summary_strict_conjunction_and_smoke_null(self):
        c = dict(primary_lag=5, levels=['legacy5'], primary_level='legacy5',
                 blocks_by_level={'legacy5': list(range(1, 7))}, lags=list(range(6)),
                 probe_count=1, finite_difference_probe_indices=[0],
                 bootstrap_seed=10, bootstrap_draws=100)
        vectors = np.zeros((1, 6, 1));vectors[0, 0, 0] = 1.;vectors[0, 5, 0] = .0625
        cells = [cell_from_vectors(vectors, np.ones(1), 'legacy5', seed, c) for seed in range(1, 7)]
        self.assertTrue(independent_summary(cells, c, False)['primary_pass'])
        vectors[0, 5, 0] = .125
        cells[-1] = cell_from_vectors(vectors, np.ones(1), 'legacy5', 6, c)
        self.assertFalse(independent_summary(cells, c, False)['primary_pass'])
        self.assertIsNone(independent_summary(cells, c, True)['primary_pass'])
        self.assertIsNone(independent_summary(cells, c, True)['levels']['legacy5']['registered_rule_pass'])

    def test_ridge_normalization_uses_training_only(self):
        c = dict(alphabet_size=2, lags=[0, 1], alpha=1.)
        train = np.array([[0., 1.], [1., 1.], [2., 1.], [3., 1.]])
        labels = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])
        first = independent_ridge(train, np.array([[100., 8.]]), labels, c)
        second = independent_ridge(train, np.array([[-200., -50.]]), labels, c)
        np.testing.assert_array_equal(first['mean'], train.mean(axis=0))
        np.testing.assert_array_equal(first['mean'], second['mean'])
        np.testing.assert_array_equal(first['scale'], second['scale'])
        self.assertEqual(first['scale'][1], 1e-5)
        np.testing.assert_array_equal(first['coefficients'], second['coefficients'])


if __name__ == '__main__':
    unittest.main()
