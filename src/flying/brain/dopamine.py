"""Compartment-local heterosynaptic LTD approximation, driven by actual spikes.

The causal eligibility kernel and learning constants are engineering choices;
this is not a receptor-kinetic model or a fitted reproduction of Hige et al.
"""
import numpy as np

from flying.brain.lif import LIFReservoir


class CausalDepression:
    def __init__(self, weights, roles, mbon_indices, dan_indices, dt_ms=.1,
                 eligibility_ms=1000., learning_rate=.1, floor_fraction=.1):
        if not (np.isfinite([dt_ms, eligibility_ms, learning_rate, floor_fraction]).all()
                and dt_ms > 0 and eligibility_ms > 0 and learning_rate >= 0
                and 0 < floor_fraction <= 1):
            raise ValueError('Invalid plasticity constants')
        self.roles = np.asarray(roles)
        self.weights = weights.copy().tocsr(); self.weights.sort_indices()
        self.initial = self.weights.data.copy()
        self.pre = self.weights.indices.copy()
        self.post = np.repeat(np.arange(len(roles)), np.diff(self.weights.indptr))
        self.mask = (self.roles[self.pre] == 'KC') & np.isin(self.post, mbon_indices)
        self.positions = np.flatnonzero(self.mask)
        self.dan_indices = np.asarray(dan_indices, dtype=int)
        if (not len(self.positions) or not len(self.dan_indices)
                or np.any(self.roles[self.dan_indices] != 'DAN')
                or np.any(self.roles[np.asarray(mbon_indices)] != 'MBON')
                or np.any(self.initial[self.mask] <= 0)):
            raise ValueError('Need excitatory existing KC-to-MBON edges and mapped DANs')
        self.dt_ms = dt_ms; self.eligibility_ms = eligibility_ms
        self.learning_rate = learning_rate; self.floor_fraction = floor_fraction
        self.reset()

    def reset(self):
        self.weights.data[:] = self.initial
        self.last_kc_tick = np.full(len(self.roles), -np.inf)
        self.last_tick = -1
        self.events = []

    def observe(self, tick, spikes):
        if type(tick) is not int or tick <= self.last_tick:
            raise ValueError('Spike batches must have strictly increasing integer ticks')
        spikes = np.asarray(spikes, dtype=int)
        if np.any(spikes < 0) or np.any(spikes >= len(self.roles)) or len(np.unique(spikes)) != len(spikes):
            raise ValueError('Invalid spike indices')
        self.last_tick = tick
        self.last_kc_tick[spikes[self.roles[spikes] == 'KC']] = tick
        gates = spikes[np.isin(spikes, self.dan_indices)]
        if not len(gates):
            return False
        eligibility = np.exp(-(tick-self.last_kc_tick[self.pre[self.mask]])*self.dt_ms/self.eligibility_ms)
        old = self.weights.data[self.mask].copy()
        updated = np.maximum(self.initial[self.mask]*self.floor_fraction,
                             old*np.exp(-self.learning_rate*len(gates)*eligibility))
        self.weights.data[self.mask] = updated
        self.events.append(dict(tick=tick, dan_indices=gates.tolist(),
                                eligible_edges=int(np.count_nonzero(eligibility)),
                                changed_edges=int(np.count_nonzero(updated != old))))
        return bool(np.any(updated != old))


class DopamineLIF(LIFReservoir):
    """Observe endogenous/evoked DAN spikes; update only mapped KC->MBON weights.

    Learning executes in Brian2's end slot, after this tick's transmissions.
    Weight changes apply to subsequent transmissions. Direct DAN stimulation is
    prescribed externally; neither targets nor classifier gradients enter LTD.
    """
    def __init__(self, weights, encoder, roles, mapping, parameters=None,
                 eligibility_ms=1000., learning_rate=.1, floor_fraction=.1):
        super().__init__(weights, encoder, roles, parameters)
        b = self.b
        self.mapping = mapping
        self.rule = CausalDepression(self.weights, roles, mapping['gamma1_pedc']['MBON'],
                                     mapping['gamma1_pedc']['DAN'], self.parameters.dt_ms,
                                     eligibility_ms, learning_rate, floor_fraction)
        # Connect(i,j) preserves our CSR/COO order; fail rather than update an
        # incorrectly aligned synapse array after an upstream behavior change.
        np.testing.assert_array_equal(self.synapses.i[:], self.rule.pre)
        np.testing.assert_array_equal(self.synapses.j[:], self.rule.post)
        self.dan_input = b.SpikeGeneratorGroup(len(roles), [], []*b.ms, clock=self.clock,
                                              codeobj_class=b.codegen.runtime.numpy_rt.NumpyCodeObject)
        self.dan_stimulus = b.Synapses(self.dan_input, self.neurons,
            on_pre='v_post += kick', namespace={'kick': self.parameters.input_mv*b.mV},
            clock=self.clock, codeobj_class=b.codegen.runtime.numpy_rt.NumpyCodeObject)
        self.dan_stimulus.connect(j='i')
        self.learning_enabled = False
        self.observer = b.NetworkOperation(self._observe_spikes, clock=self.clock, when='end', order=1)
        self.network.add(self.dan_input, self.dan_stimulus, self.observer)
        self.network.store('plastic_initial')

    def _observe_spikes(self):
        if not self.learning_enabled:
            return
        spikes = self.neurons.spikes
        if not len(spikes):
            return
        tick = round(float(self.clock.t/self.b.ms)/self.parameters.dt_ms)
        if self.rule.observe(tick, spikes):
            values = self.rule.weights.data[self.rule.mask]*self.parameters.contact_mv
            self.synapses.w[self.rule.positions] = values*self.b.mV
            self.weights.data[:] = self.rule.weights.data

    def reset(self):
        self.network.restore('plastic_initial')
        self.rule.reset()
        self.weights.data[:] = self.rule.weights.data
        self.learning_enabled = False

    def condition(self, digit, cs_start_ms, cs_duration_ms, dan_times_ms,
                  dan_compartment='gamma1_pedc', duration_ms=3000.):
        p, b = self.parameters, self.b
        if digit is not None and (type(digit) is not int or not 0 <= digit <= 9):
            raise ValueError('Invalid conditioned digit')
        if dan_compartment not in self.mapping or cs_start_ms < 0 or cs_duration_ms < 0:
            raise ValueError('Invalid conditioning schedule')
        dan_times_ms = np.asarray(dan_times_ms, dtype=float)
        if (not np.isfinite(dan_times_ms).all() or np.any(dan_times_ms < 0)
                or np.any(dan_times_ms >= duration_ms) or cs_start_ms+cs_duration_ms > duration_ms):
            raise ValueError('Conditioning events outside duration')
        for value in [duration_ms, cs_start_ms, cs_duration_ms, *dan_times_ms]:
            if not np.isclose(value/p.dt_ms, round(value/p.dt_ms), atol=1e-9, rtol=0):
                raise ValueError('Off-grid conditioning event')
        self.reset()
        cells = np.flatnonzero(self.patterns[digit]) if digit is not None else np.array([], dtype=int)
        times = np.arange(round(cs_start_ms/p.dt_ms),
                          round((cs_start_ms+cs_duration_ms)/p.dt_ms), round(p.pulse_ms/p.dt_ms))
        self.input.set_spikes(np.tile(cells, len(times)), np.repeat(times, len(cells))*p.dt_ms*b.ms, sorted=True)
        dan_cells = self.mapping[dan_compartment]['DAN']
        self.dan_input.set_spikes(np.tile(dan_cells, len(dan_times_ms)),
                                  np.repeat(dan_times_ms, len(dan_cells))*b.ms)
        self.learning_enabled = True
        self.network.run(duration_ms*b.ms, namespace={})
        self.learning_enabled = False
        np.testing.assert_allclose(self.synapses.w[:]/b.mV,
                                   self.weights.data*p.contact_mv, atol=0, rtol=0)
        return self.weights.copy(), self.spike_arrays(), self.rule.events.copy()


def replay_depression(initial, roles, mbon_indices, dan_indices, spikes, **parameters):
    """Closed-form product per edge, independent of the online rule's state."""
    result = initial.copy().tocsr(); result.sort_indices()
    pre = result.indices
    post = np.repeat(np.arange(len(roles)), np.diff(result.indptr))
    gates = spikes['spike_ticks'][np.isin(spikes['spike_indices'], dan_indices)]
    selected = np.flatnonzero((np.asarray(roles)[pre] == 'KC') & np.isin(post, mbon_indices))
    for position in selected:
        kc_ticks = spikes['spike_ticks'][spikes['spike_indices'] == pre[position]]
        slots = np.searchsorted(kc_ticks, gates, side='right')-1
        valid = slots >= 0
        eligibility_sum = np.exp(-(gates[valid]-kc_ticks[slots[valid]])*parameters['dt_ms']/parameters['eligibility_ms']).sum()
        factor = max(parameters['floor_fraction'], np.exp(-parameters['learning_rate']*eligibility_sum))
        result.data[position] *= factor
    return result
