from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from flying.brain.lif import LIFParameters, LIFReservoir
from flying.brain.sparse_lif import SparseLIFReservoir


def models(p):
    w = sparse.csc_matrix([[0, -30, 0], [160, 0, 15], [-90, 50, 0.]])
    enc = SimpleNamespace(patterns=np.tile([1., 0., 0.], (10, 1)))
    roles = np.array(['KC', 'MBON', 'MBON'])
    return SparseLIFReservoir(w, enc, roles, p), LIFReservoir(w, enc, roles, p)


@pytest.mark.parametrize('kwargs', [dict(), dict(dt_ms=.05), dict(delay_ms=0),
                                  dict(synapse_ms=10), dict(pulse_ms=.5)])
def test_sparse_brian_schedule_and_final_states(kwargs):
    b = pytest.importorskip('brian2')
    fast, brian = models(replace(LIFParameters(digit_ms=20.), **kwargs))
    np.testing.assert_array_equal(fast.states([3, 1]), brian.states([3, 1]))
    for key, value in brian.spike_arrays().items():
        np.testing.assert_array_equal(fast.spike_arrays()[key], value)
    np.testing.assert_allclose(fast.v, brian.neurons.v[:]/b.mV, atol=1e-8, rtol=0)
    np.testing.assert_allclose(fast.g, brian.neurons.g[:]/b.mV, atol=1e-8, rtol=0)
    assert np.any(fast.spike_arrays()['spike_indices'] == 1)


def test_reset_chunks_invalid_input_and_delayed_events():
    w = sparse.csc_matrix([[0., 0.], [160, 0]])
    enc = SimpleNamespace(patterns=np.tile([1., 0.], (10, 1)))
    model = SparseLIFReservoir(w, enc, np.array(['KC', 'MBON']))
    expected = model.states([3, 1, 4])
    events = model.spike_arrays()
    v, g = model.v.copy(), model.g.copy()
    model.reset()
    np.testing.assert_array_equal(expected, np.array([model.step(d) for d in [3, 1, 4]]))
    for key in events:
        np.testing.assert_array_equal(events[key], model.spike_arrays()[key])
    np.testing.assert_array_equal(v, model.v)
    np.testing.assert_array_equal(g, model.g)
    tick = model.tick
    with pytest.raises(ValueError):
        model.advance([3, 10])
    assert model.tick == tick
    assert model.advance([]).shape == (0, 2)
    with pytest.raises(ValueError):
        SparseLIFReservoir(w, SimpleNamespace(patterns=np.ones((10, 2))), ['KC', 'MBON'])


def test_passive_decay_and_strict_threshold():
    p = LIFParameters(digit_ms=10.)
    enc = SimpleNamespace(patterns=np.zeros((10, 2)))
    model = SparseLIFReservoir(sparse.csc_matrix((2, 2)), enc, ['KC', 'MBON'], p)
    model.v[:] = [-50., -54.]
    model.g[:] = [.2, -.3]
    v, g = model.v.copy(), model.g.copy()
    model.advance([0])
    em, es = np.exp(-10/20), np.exp(-10/5)
    np.testing.assert_allclose(model.v, -52+(v+52)*em+g*(em-es)/3, atol=2e-11, rtol=0)
    np.testing.assert_allclose(model.g, g*es, atol=2e-11, rtol=0)
    assert not len(model.spike_arrays()['spike_ticks'])
    model = SparseLIFReservoir(sparse.csc_matrix((2, 2)), enc, ['KC', 'MBON'],
                               replace(p, rest_mv=-45.))
    model.v[:] = [-45., -44.999]
    model.advance([0])
    assert not np.any(model.spike_arrays()['spike_indices'] == 0)
    assert model.spike_arrays()['spike_indices'][0] == 1


def test_equal_time_constants_analytic_limit():
    # Brian2's generic exact integrator divides by tau_m-tau_s for this case;
    # check the analytic limit directly without changing the locked defaults.
    p = LIFParameters(synapse_ms=20., digit_ms=10.)
    model = SparseLIFReservoir(sparse.csc_matrix((1, 1)),
                               SimpleNamespace(patterns=np.zeros((10, 1))), ['KC'], p)
    model.g[:] = .2
    model.advance([0])
    np.testing.assert_allclose(model.v, -52+.2*(10/20)*np.exp(-10/20), atol=2e-11, rtol=0)
