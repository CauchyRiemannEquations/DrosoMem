"""Dimensionful Brian2 LIF adapter; raw signed contacts are W[post, pre].

Uniform model parameters follow Shiu's pinned model.py, not cell-specific fits.
The digit pulse code below is an explicit engineering adaptation.
"""
from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class LIFParameters:
    rest_mv: float = -52.
    reset_mv: float = -52.
    threshold_mv: float = -45.
    membrane_ms: float = 20.
    synapse_ms: float = 5.
    refractory_ms: float = 2.2
    delay_ms: float = 1.8
    contact_mv: float = .275
    input_mv: float = 68.75
    dt_ms: float = .1
    digit_ms: float = 50.
    pulse_ms: float = 10.

    def __post_init__(self):
        if not all(np.isfinite(v) for v in vars(self).values()):
            raise ValueError('Nonfinite LIF parameter')
        if min(self.membrane_ms, self.synapse_ms, self.contact_mv, self.dt_ms,
               self.digit_ms, self.pulse_ms, self.input_mv) <= 0:
            raise ValueError('Time constants and scales must be positive')
        if min(self.delay_ms, self.refractory_ms) < 0 or self.reset_mv >= self.threshold_mv:
            raise ValueError('Invalid refractory/delay/reset')
        for name in ['delay_ms', 'refractory_ms', 'digit_ms', 'pulse_ms']:
            steps = getattr(self, name) / self.dt_ms
            if not np.isclose(steps, round(steps), rtol=0, atol=1e-9):
                raise ValueError(f'{name} must be an integer number of timesteps')
        if self.pulse_ms > self.digit_ms:
            raise ValueError('Pulse interval exceeds digit duration')


class LIFReservoir:
    """Frozen spiking network; readout sees per-digit spike counts only.

    Source-style direct voltage stimulation has zero refractory time for all
    input-eligible cells. Other cells retain 2.2 ms. No learning occurs here.
    Imports Brian2 lazily so the rate-model installation remains independent.
    """
    def __init__(self, weights, encoder, roles, parameters=None, trace=False):
        import brian2 as b

        self.b = b
        self.parameters = p = parameters or LIFParameters()
        self.weights = sparse.csr_matrix(weights, dtype=float).copy()
        self.weights.sum_duplicates(); self.weights.eliminate_zeros(); self.weights.sort_indices()
        n = self.weights.shape[0]
        self.roles = np.asarray(roles)
        self.patterns = np.asarray(encoder.patterns != 0)
        if self.weights.shape != (n, n) or self.roles.shape != (n,) or self.patterns.shape != (10, n):
            raise ValueError('LIF dimensions do not match')
        if not np.all(np.isfinite(self.weights.data)):
            raise ValueError('Nonfinite contacts')
        if np.any(self.patterns[:, self.roles != 'KC']):
            raise ValueError('Digit input must target KC only')
        # Explicit per-object clocks and numpy target avoid global clock state
        # and a platform-dependent C++ toolchain requirement.
        self.clock = b.Clock(dt=p.dt_ms*b.ms)
        namespace = dict(v_rest=p.rest_mv*b.mV, v_reset=p.reset_mv*b.mV,
                         v_threshold=p.threshold_mv*b.mV, tau_m=p.membrane_ms*b.ms,
                         tau_s=p.synapse_ms*b.ms, input_weight=p.input_mv*b.mV)
        self.neurons = b.NeuronGroup(n, '''
            dv/dt = (v_rest - v + g) / tau_m : volt (unless refractory)
            dg/dt = -g / tau_s : volt (unless refractory)
            rfc : second
            ''', threshold='v > v_threshold', reset='v = v_reset; g = 0*mV',
            refractory='rfc', method='exact', namespace=namespace,
            clock=self.clock, codeobj_class=b.codegen.runtime.numpy_rt.NumpyCodeObject)
        self.neurons.v = p.rest_mv*b.mV
        self.neurons.g = 0*b.mV
        self.neurons.rfc = np.where(self.roles == 'KC', 0., p.refractory_ms)*b.ms
        self.synapses = b.Synapses(self.neurons, self.neurons, 'w : volt',
                                  on_pre='g_post += w', delay=p.delay_ms*b.ms,
                                  clock=self.clock, codeobj_class=b.codegen.runtime.numpy_rt.NumpyCodeObject)
        coo = self.weights.tocoo()
        if coo.nnz:
            self.synapses.connect(i=coo.col, j=coo.row)
            self.synapses.w = coo.data*p.contact_mv*b.mV
        else:
            self.synapses.active = False
        self.input = b.SpikeGeneratorGroup(n, [], []*b.ms, clock=self.clock,
                                          codeobj_class=b.codegen.runtime.numpy_rt.NumpyCodeObject)
        self.stimulus = b.Synapses(self.input, self.neurons, on_pre='v_post += input_weight',
                                  namespace=namespace, clock=self.clock,
                                  codeobj_class=b.codegen.runtime.numpy_rt.NumpyCodeObject)
        self.stimulus.connect(j='i')
        self.spikes = b.SpikeMonitor(self.neurons, codeobj_class=b.codegen.runtime.numpy_rt.NumpyCodeObject)
        objects = [self.neurons, self.synapses, self.input, self.stimulus, self.spikes]
        self.trace = None
        if trace:
            self.trace = b.StateMonitor(self.neurons, ['v', 'g'], record=True, when='end',
                                        codeobj_class=b.codegen.runtime.numpy_rt.NumpyCodeObject)
            objects.append(self.trace)
        self.network = b.Network(objects)
        self.network.store('initial')

    def reset(self):
        self.network.restore('initial')

    def _events(self, digits):
        p = self.parameters
        window = round(p.digit_ms/p.dt_ms)
        interval = round(p.pulse_ms/p.dt_ms)
        start = round(float(self.network.t/self.b.ms)/p.dt_ms)
        indices, ticks = [], []
        for k, digit in enumerate(digits):
            if not isinstance(digit, (int, np.integer)) or not 0 <= digit <= 9:
                raise ValueError('Expected integer digit 0..9')
            cells = np.flatnonzero(self.patterns[digit])
            for tick in range(0, window, interval):
                indices.extend(cells)
                ticks.extend([start+k*window+tick]*len(cells))
        return np.asarray(indices, dtype=int), np.asarray(ticks, dtype=int)

    def advance(self, digits):
        digits = list(digits)
        if not digits:
            return np.empty((0, len(self.roles)))
        b, p = self.b, self.parameters
        indices, ticks = self._events(digits)
        before = len(self.spikes.i)
        start = round(float(self.network.t/b.ms)/p.dt_ms)
        self.input.set_spikes(indices, ticks*p.dt_ms*b.ms, sorted=True)
        self.network.run(len(digits)*p.digit_ms*b.ms, namespace={})
        counts = np.zeros((len(digits), len(self.roles)))
        event_ticks = np.rint(np.asarray(self.spikes.t[before:]/b.ms)/p.dt_ms).astype(int)
        windows = (event_ticks-start)//round(p.digit_ms/p.dt_ms)
        np.add.at(counts, (windows, np.asarray(self.spikes.i[before:], dtype=int)), 1)
        if not np.all(np.isfinite(self.neurons.v[:]/b.mV)) or not np.all(np.isfinite(self.neurons.g[:]/b.mV)):
            raise FloatingPointError('Nonfinite LIF state')
        return counts

    def states(self, digits):
        self.reset()
        return self.advance(digits)

    def step(self, digit):
        return self.advance([digit])[0]

    def spike_arrays(self):
        ticks = np.rint(np.asarray(self.spikes.t/self.b.ms)/self.parameters.dt_ms).astype(np.int64)
        return dict(spike_ticks=ticks, spike_indices=np.asarray(self.spikes.i, dtype=np.int64))
