"""Zero-carry staged dynamics and directional derivatives for registered M5.

This helper handles continuous encoder inputs. It never changes encoder patterns.
The state step keeps the explicit zero additions of frozen FactorReservoir.
"""
from fractions import Fraction
import hashlib

import numpy as np
from scipy import sparse


class PulseDynamics:
    def __init__(self, weights, roles, drive=.6, synaptic_history=True):
        self.weights = weights.tocsr(copy=False)
        self.roles = np.asarray(roles)
        self.kc = np.flatnonzero(self.roles == 'KC')
        self.mbon = np.flatnonzero(self.roles == 'MBON')
        self.mbon_weights = self.weights[self.mbon].copy()
        self.absolute = abs(self.weights)
        self.absolute_mbon = self.absolute[self.mbon].copy()
        self.drive = float(drive)
        self.synaptic_history = bool(synaptic_history)

    def step(self, previous, stimulation):
        previous = np.asarray(previous)
        zero = np.zeros_like(previous)
        synaptic = self.weights @ previous if self.synaptic_history else zero
        following = zero + self.drive * np.tanh(synaptic + stimulation)
        mixed = previous.copy() if self.synaptic_history else zero.copy()
        mixed[self.kc] = following[self.kc]
        following[self.mbon] = zero[self.mbon] + self.drive * np.tanh(
            self.mbon_weights @ mixed + stimulation[self.mbon])
        return following

    def linearized_step(self, previous, stimulation, tangent, envelope, direction):
        """Return the unperturbed state, JVP and unsigned path envelope."""
        zero = np.zeros_like(previous)
        synaptic = self.weights @ previous if self.synaptic_history else zero
        first_tanh = np.tanh(synaptic + stimulation)
        following = zero + self.drive * first_tanh
        factors = self.drive * (1. - first_tanh * first_tanh)
        history_v = self.weights @ tangent if self.synaptic_history else zero
        history_e = self.absolute @ envelope if self.synaptic_history else zero
        next_v = factors * (history_v + direction)
        next_e = factors * (history_e + np.abs(direction))
        mixed = previous.copy() if self.synaptic_history else zero.copy()
        mixed_v = tangent.copy() if self.synaptic_history else zero.copy()
        mixed_e = envelope.copy() if self.synaptic_history else zero.copy()
        mixed[self.kc] = following[self.kc]
        mixed_v[self.kc] = next_v[self.kc]
        mixed_e[self.kc] = next_e[self.kc]
        final_tanh = np.tanh(self.mbon_weights @ mixed + stimulation[self.mbon])
        following[self.mbon] = zero[self.mbon] + self.drive * final_tanh
        mbon_factors = self.drive * (1. - final_tanh * final_tanh)
        next_v[self.mbon] = mbon_factors * (
            self.mbon_weights @ mixed_v + direction[self.mbon])
        next_e[self.mbon] = mbon_factors * (
            self.absolute_mbon @ mixed_e + np.abs(direction[self.mbon]))
        return following, next_v, next_e

    def support(self, directions, observed, lags):
        """Exact-delay boolean walk support for each nonzero contrast source set."""
        target = np.repeat(np.arange(len(self.roles)), np.diff(self.weights.indptr))
        source = self.weights.indices
        immediate = (self.roles[source] == 'KC') & (self.roles[target] == 'MBON')
        def matrix(mask):
            return sparse.csr_matrix((np.ones(int(mask.sum()), dtype=bool),
                (target[mask], source[mask])), shape=self.weights.shape)
        delayed, current = matrix(~immediate), matrix(immediate)
        state = np.asarray(directions != 0, dtype=bool).T.copy()
        state |= current @ state
        result = np.zeros((len(directions), len(lags), len(observed)), dtype=bool)
        wanted = {int(lag): i for i, lag in enumerate(lags)}
        for lag in range(max(lags)+1):
            if lag in wanted:
                result[:, wanted[lag]] = state[observed].T
            if lag < max(lags):
                state = delayed @ state
                state |= current @ state
        return result


def replay(dynamics, pre_state, inputs, direction=None, eta=0.):
    """One pulse at first input, then identical ordinary inputs; full digest."""
    previous = np.asarray(pre_state).copy()
    digest = hashlib.sha256()
    states = []
    for lag, stimulation in enumerate(inputs):
        pulse = stimulation if lag or direction is None else stimulation + eta * direction
        previous = dynamics.step(previous, pulse)
        digest.update(previous.tobytes())
        states.append(previous.copy())
    return np.asarray(states), digest.hexdigest()


def fd_gate(tangent, differences, atol, rtol):
    error = np.abs(differences-tangent[None, :, :])
    excess = error - (atol + rtol*np.abs(tangent))[None, :, :]
    return dict(all_checks_pass=bool(np.all(excess <= 0)),
                max_abs_error=error.max(axis=2),
                max_tolerance_excess=excess.max(axis=2))


def squared_gain_exact(observed_tangents, contrast_squared_norm):
    """Exact arithmetic on saved float64 values; no floating sum before squaring."""
    vectors = np.asarray(observed_tangents, dtype=np.float64)
    denominators = np.asarray(contrast_squared_norm, dtype=np.float64)
    if vectors.ndim != 2 or len(vectors) != len(denominators) or not len(vectors):
        raise ValueError('Expected aligned nonempty probe vectors and denominators')
    if not np.isfinite(vectors).all() or not np.isfinite(denominators).all() or np.any(denominators <= 0):
        raise ValueError('Invalid or zero contrast; probes cannot be replaced')
    return sum((sum((Fraction.from_float(float(x))**2 for x in v), Fraction(0)) /
                Fraction.from_float(float(d)) for v, d in zip(vectors, denominators)),
               Fraction(0))/len(vectors)


def primary_cell(observed_tangents, contrast_squared_norm, seed, level,
                 minimum_gain=1e-8, primary_lag=5, technical_valid=True):
    """One fixed block; scientific rule remains distinct from assay eligibility."""
    try:
        s0 = squared_gain_exact(observed_tangents[:, 0], contrast_squared_norm)
        s5 = squared_gain_exact(observed_tangents[:, primary_lag], contrast_squared_norm)
    except ValueError:
        return dict(level=level, seed=int(seed), eligible=False, scientific_rule_pass=None,
                    s0_numerator=None, s0_denominator=None, s5_numerator=None,
                    s5_denominator=None, current_gain=None, relative_gain=None,
                    invalid_reason='zero or invalid contrast')
    current = float(np.sqrt(float(s0)))
    eligible = bool(technical_valid and s0 >= Fraction(str(minimum_gain))**2)
    return dict(level=level, seed=int(seed), eligible=eligible,
                scientific_rule_pass=bool(s5 <= s0/Fraction(100)) if eligible else None,
                s0_numerator=s0.numerator, s0_denominator=s0.denominator,
                s5_numerator=s5.numerator, s5_denominator=s5.denominator,
                current_gain=current,
                relative_gain=float(np.sqrt(float(s5/s0))) if s0 else None,
                invalid_reason=None if eligible else 'technical check or unresolved immediate gain')


def endpoint(cells, smoke=False):
    valid = bool(cells and all(cell['eligible'] for cell in cells))
    passed = bool(valid and all(cell['scientific_rule_pass'] for cell in cells))
    return dict(eligible=valid, registered_rule_pass=None if smoke or not valid else passed,
                outcome='smoke' if smoke else 'assay-invalid' if not valid else 'PASS' if passed else 'FAIL')
