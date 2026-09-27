"""Bounded-memory, frozen-weight LIF backend in mV/ms.

CSC fan-out and a fixed delay ring avoid per-synapse event queues and full state
traces. Scheduling follows the validated Brian2 groups/thresholds/synapses/reset
order. This backend is separately tested against Brian2, not a replacement for
the independent reference oracle used in Stage B/C.
"""
import numpy as np
from scipy import sparse

from flying.brain.lif import LIFParameters


class SparseLIFReservoir:
    def __init__(self, weights, encoder, roles, parameters=None):
        self.parameters = p = parameters or LIFParameters()
        self.weights = sparse.csc_matrix(weights, dtype=float, copy=True)
        self.weights.sum_duplicates()
        self.weights.eliminate_zeros()
        self.weights.sort_indices()
        n = self.weights.shape[0]
        self.roles = np.asarray(roles)
        self.patterns = np.asarray(encoder.patterns != 0).copy()
        if (self.weights.shape != (n, n) or self.roles.shape != (n,)
                or self.patterns.shape != (10, n)):
            raise ValueError('LIF dimensions do not match')
        if not np.all(np.isfinite(self.weights.data)):
            raise ValueError('Nonfinite contacts')
        if np.any(self.patterns[:, self.roles != 'KC']):
            raise ValueError('Digit input must target KC only')
        self.inputs = [np.flatnonzero(row) for row in self.patterns]
        self.rfc = np.where(self.roles == 'KC', 0, round(p.refractory_ms/p.dt_ms))
        self.delay = round(p.delay_ms/p.dt_ms)
        self.window = round(p.digit_ms/p.dt_ms)
        self.interval = round(p.pulse_ms/p.dt_ms)
        self.em = np.exp(-p.dt_ms/p.membrane_ms)
        self.es = np.exp(-p.dt_ms/p.synapse_ms)
        self.coupling = (p.dt_ms/p.membrane_ms*self.em
                         if p.synapse_ms == p.membrane_ms else
                         p.synapse_ms/(p.membrane_ms-p.synapse_ms)*(self.em-self.es))
        self.v = np.empty(n)
        self.g = np.empty(n)
        self.last = np.empty(n, dtype=np.int64)
        self.pending = np.zeros((self.delay+1, n))
        self.reset()

    def reset(self):
        self.v.fill(self.parameters.rest_mv)
        self.g.fill(0)
        self.last.fill(-10**12)
        self.pending.fill(0)
        self.tick = 0
        self._ticks, self._indices = [], []

    def advance(self, digits):
        digits = list(digits)
        if any(not isinstance(d, (int, np.integer)) or not 0 <= d <= 9 for d in digits):
            raise ValueError('Expected integer digit 0..9')
        p, w = self.parameters, self.weights
        counts = np.zeros((len(digits), len(self.roles)), dtype=np.int32)
        for window, digit in enumerate(digits):
            cells = self.inputs[digit]
            for within in range(self.window):
                tick = self.tick
                active = tick-self.last >= self.rfc
                np.copyto(self.v, p.rest_mv+(self.v-p.rest_mv)*self.em
                          +self.coupling*self.g, where=active)
                np.multiply(self.g, self.es, out=self.g, where=active)
                fired = np.flatnonzero((self.v > p.threshold_mv) & active)
                destination = self.pending[(tick+self.delay) % len(self.pending)]
                for pre in fired:
                    a, z = w.indptr[pre:pre+2]
                    # Canonical CSC has unique destinations in each column.
                    destination[w.indices[a:z]] += w.data[a:z]*p.contact_mv
                current = self.pending[tick % len(self.pending)]
                np.add(self.g, current, out=self.g, where=active)
                current.fill(0)
                if within % self.interval == 0:
                    self.v[cells[active[cells]]] += p.input_mv
                self.v[fired] = p.reset_mv
                self.g[fired] = 0
                self.last[fired] = tick
                counts[window, fired] += 1
                if len(fired):
                    self._ticks.append(np.full(len(fired), tick, dtype=np.int64))
                    self._indices.append(fired)
                self.tick += 1
        if not np.isfinite(self.v).all() or not np.isfinite(self.g).all():
            raise FloatingPointError('Nonfinite LIF state')
        return counts

    def states(self, digits):
        self.reset()
        return self.advance(digits)

    def step(self, digit):
        return self.advance([digit])[0]

    def spike_arrays(self):
        return dict(spike_ticks=np.concatenate(self._ticks) if self._ticks else np.empty(0, dtype=np.int64),
                    spike_indices=np.concatenate(self._indices) if self._indices else np.empty(0, dtype=np.int64))
