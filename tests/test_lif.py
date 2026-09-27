from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

b = pytest.importorskip('brian2')
from flying.brain.lif import LIFParameters, LIFReservoir
from flying.brain.lif_reference import reference_trace


def make(weights=None, p=None, trace=True):
    weights = np.zeros((3, 3)) if weights is None else weights
    encoder = SimpleNamespace(patterns=np.tile([1., 0., 0.], (10, 1)))
    return LIFReservoir(weights, encoder, np.array(['KC', 'MBON', 'MBON']), p, trace)


def compare(r, digits):
    ids, ticks = r._events(digits)
    p = r.parameters
    expected = reference_trace(r.weights, r.roles, ids, ticks,
                               round(len(digits)*p.digit_ms/p.dt_ms), p,
                               r.neurons.v[:]/b.mV, r.neurons.g[:]/b.mV)
    r.advance(digits)
    np.testing.assert_allclose(r.trace.v.T/b.mV, expected['v_mv'], atol=2e-10, rtol=0)
    np.testing.assert_allclose(r.trace.g.T/b.mV, expected['g_mv'], atol=2e-10, rtol=0)
    for key, values in r.spike_arrays().items():
        np.testing.assert_array_equal(values, expected[key])
    return expected


def test_passive_analytic_decay():
    r = make(p=LIFParameters(input_mv=.01, digit_ms=10.))
    r.patterns[:] = False
    r.neurons.v = [-50., -52., -54.]*b.mV
    r.neurons.g = [.2, -.3, 0.]*b.mV
    initial_v = np.asarray(r.neurons.v[:]/b.mV)
    initial_g = np.asarray(r.neurons.g[:]/b.mV)
    result = compare(r, [0])
    t = (np.arange(len(result['v_mv']))+1)*.1
    em, es = np.exp(-t/20.), np.exp(-t/5.)
    expected = -52+(initial_v+52)*em[:, None]+initial_g*(em-es)[:, None]/3
    np.testing.assert_allclose(result['v_mv'], expected, atol=2e-11, rtol=0)


@pytest.mark.parametrize('dt', [.1, .05, .025])
def test_signed_delayed_recurrent_reference(dt):
    w = np.array([[0, -30, 0], [160, 0, 15], [-90, 50, 0.]])
    result = compare(make(w, LIFParameters(dt_ms=dt, digit_ms=20.)), [1, 4])
    assert np.any(result['spike_indices'] == 1)
    assert np.min(result['g_mv'][:, 2]) < 0


def test_direction_delay_and_postsynaptic_sign():
    r = make(np.array([[0, 0, 0], [10, 0, 0], [-10, 0, 0]]),
             LIFParameters(digit_ms=10.))
    result = compare(r, [3])
    assert result['spike_ticks'][0] == 1  # voltage kick follows threshold slot
    assert np.flatnonzero(result['g_mv'][:, 1])[0] == 19  # +1.8 ms
    assert result['g_mv'][19, 1] == 2.75
    assert result['g_mv'][19, 2] == -2.75


def test_refractory_freezes_drive_and_ignores_synapses():
    r = make(np.array([[0, 0, 0], [160, 0, 0], [0, 0, 0]]),
             LIFParameters(digit_ms=10., pulse_ms=.5))
    result = compare(r, [0])
    fired = result['spike_ticks'][result['spike_indices'] == 1]
    assert len(fired) >= 2 and np.all(np.diff(fired) >= 22)
    first = fired[0]
    np.testing.assert_array_equal(result['g_mv'][first:first+22, 1], 0.)
    np.testing.assert_array_equal(result['v_mv'][first:first+22, 1], -52.)


def test_step_batch_and_reset_identical():
    r = make(np.array([[0, 0, 0], [160, 0, 0], [0, -20, 0.]]), trace=False)
    batch = r.states([3, 1, 4]); events = r.spike_arrays()
    r.reset()
    individual = np.asarray([r.step(d) for d in [3, 1, 4]])
    np.testing.assert_array_equal(batch, individual)
    for name in events:
        np.testing.assert_array_equal(events[name], r.spike_arrays()[name])


def test_zero_edges_no_postsynaptic_spikes():
    r = make(trace=False)
    states = r.states([3, 1, 4])
    np.testing.assert_array_equal(states[:, 1:], 0.)
    np.testing.assert_array_equal(states[:, 0], 5.)


def test_dimension_mismatch_rejected():
    from brian2.units.fundamentalunits import DimensionMismatchError
    r = make(trace=False)
    with pytest.raises(DimensionMismatchError):
        r.neurons.v = 20*b.ms


def test_threshold_is_strict_and_reset_clears_drive():
    r = make(p=LIFParameters(rest_mv=-45., digit_ms=10.))
    r.patterns[:] = False
    r.neurons.v = [-45., -44.999, -52.]*b.mV
    r.advance([0])
    events = r.spike_arrays()
    assert not np.any(events['spike_indices'] == 0)
    assert events['spike_ticks'][0] == 0 and events['spike_indices'][0] == 1
    assert r.trace.v[1, 0]/b.mV == -52. and r.trace.g[1, 0]/b.mV == 0.


@pytest.mark.parametrize('kwargs', [{'dt_ms': .3}, {'synapse_ms': 0}, {'contact_mv': np.nan},
                                    {'reset_mv': -40}, {'pulse_ms': 60}])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        LIFParameters(**kwargs)


def test_invalid_digit_and_no_non_kc_input():
    r = make(trace=False)
    for value in [-1, 10, 3.5]:
        with pytest.raises(ValueError):
            r.step(value)
    with pytest.raises(ValueError):
        LIFReservoir(sparse.eye(3), SimpleNamespace(patterns=np.ones((10, 3))),
                     np.array(['KC', 'MBON', 'DAN']))
