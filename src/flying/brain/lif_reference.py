"""Independent discrete-clock reference for validating the Brian2 adapter.

Numbers are explicitly mV and ms. This is a test oracle, not the experiment
backend. Implements groups -> thresholds -> synapses -> resets -> end.
"""
import numpy as np
from scipy import sparse


def reference_trace(weights, roles, event_indices, event_ticks, steps, parameters,
                    initial_v=None, initial_g=None):
    p = parameters
    matrix = sparse.csc_matrix(weights)*p.contact_mv
    n = matrix.shape[0]
    v = np.full(n, p.rest_mv) if initial_v is None else np.array(initial_v, dtype=float)
    g = np.zeros(n) if initial_g is None else np.array(initial_g, dtype=float)
    rfc = np.where(np.asarray(roles) == 'KC', 0, round(p.refractory_ms/p.dt_ms))
    last = np.full(n, -10**12, dtype=np.int64)
    delay = round(p.delay_ms/p.dt_ms)
    pending = np.zeros((delay+1, n))
    events = {}
    for cell, tick in zip(event_indices, event_ticks):
        events.setdefault(int(tick), []).append(int(cell))
    em, es = np.exp(-p.dt_ms/p.membrane_ms), np.exp(-p.dt_ms/p.synapse_ms)
    coupling = (p.dt_ms/p.membrane_ms*em if p.synapse_ms == p.membrane_ms else
                p.synapse_ms/(p.membrane_ms-p.synapse_ms)*(em-es))
    voltages, drives = np.empty((steps, n)), np.empty((steps, n))
    spike_ticks, spike_indices = [], []
    for tick in range(steps):
        active = tick-last >= rfc
        v[active] = p.rest_mv+(v[active]-p.rest_mv)*em+coupling*g[active]
        g[active] *= es
        fired = np.flatnonzero((v > p.threshold_mv) & active)
        for pre in fired:
            a, z = matrix.indptr[pre:pre+2]
            np.add.at(pending[(tick+delay) % len(pending)], matrix.indices[a:z], matrix.data[a:z])
        g[active] += pending[tick % len(pending), active]
        pending[tick % len(pending)] = 0
        cells = np.asarray(events.get(tick, []), dtype=int)
        np.add.at(v, cells[active[cells]], p.input_mv)
        v[fired] = p.reset_mv; g[fired] = 0.; last[fired] = tick
        voltages[tick] = v; drives[tick] = g
        spike_ticks.extend([tick]*len(fired)); spike_indices.extend(fired)
    return dict(v_mv=voltages, g_mv=drives,
                spike_ticks=np.asarray(spike_ticks, dtype=np.int64),
                spike_indices=np.asarray(spike_indices, dtype=np.int64))
