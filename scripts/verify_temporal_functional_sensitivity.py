"""Independent M5 raw-source, complex-step, finite-pulse and ridge replay.

The verifier imports no experiment runner, tangent implementation, reservoir
class, decoder, label builder or scientific summary. The only scientific
reuse is the frozen independent raw-source reconstruction from P1.
"""
import argparse
from fractions import Fraction
import hashlib
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits

from verify_temporal_memory_curve import independent_graph, validate_whole_source, partial_source
from sensitivity_support import (config, read, write, sha, array_sha, attempt, seal,
                                 check_manifest, history_preserved, source_record,
                                 assert_source_unchanged)
from tdc_support import environment, git
from tdc_pool import pooled


PROTOCOL_COMMIT = '37212abeb6cfc1a4907e08348ff41030f3edfb8b'
ESTIMATES = ('mean', 'median', 'sd', 'bootstrap_lo', 'bootstrap_hi')


def manual_step(weights, kc, mb, previous, stimulus, b=.6, history=True):
    """Continuous, complex-compatible scheduled map with explicit zero carry."""
    old = np.asarray(previous).copy()
    synaptic = weights.dot(old) if history else np.zeros_like(old)
    following = np.zeros_like(old) + b*np.tanh(synaptic + stimulus)
    mixed = old.copy() if history else np.zeros_like(old)
    mixed[kc] = following[kc]
    following[mb] = np.zeros(len(mb)) + b*np.tanh(weights[mb].dot(mixed) + stimulus[mb])
    return following


def complex_window(weights, kc, mb, previous, inputs, direction, step=1e-20,
                   b=.6, history=True):
    """One imaginary pulse; later real inputs preserve the original schedule."""
    state = np.asarray(previous, dtype=np.complex128).copy()
    rows = []
    for lag, raw in enumerate(inputs):
        stimulus = np.asarray(raw, dtype=np.complex128).copy()
        if lag == 0:
            stimulus += 1j*step*np.asarray(direction)
        state = manual_step(weights, kc, mb, state, stimulus, b, history)
        rows.append(state.imag.copy()/step)
    return np.asarray(rows)


def envelope_window(weights, kc, mb, previous, inputs, direction, b=.6,
                    history=True):
    """Own unsigned chain recurrence at unperturbed signed preactivations."""
    unsigned = abs(weights).tocsr()
    old, envelope = np.asarray(previous).copy(), np.zeros(weights.shape[0])
    rows = []
    for lag, stimulus in enumerate(inputs):
        external = np.abs(direction) if lag == 0 else np.zeros(weights.shape[0])
        argument = weights.dot(old) if history else np.zeros_like(old)
        first_tanh = np.tanh(argument + stimulus)
        first = b*first_tanh
        recurrent = unsigned.dot(envelope) if history else np.zeros_like(envelope)
        following = b*(1-first_tanh*first_tanh)*(recurrent+external)
        mixed = old.copy() if history else np.zeros_like(old)
        mixed[kc] = first[kc]
        factor = 1-np.tanh(weights[mb].dot(mixed)+stimulus[mb])**2
        mixed_envelope = envelope.copy() if history else np.zeros_like(envelope)
        mixed_envelope[kc] = following[kc]
        following[mb] = b*factor*(unsigned[mb].dot(mixed_envelope)+external[mb])
        envelope = following
        old = manual_step(weights, kc, mb, old, stimulus, b, history)
        rows.append(envelope.copy())
    return np.asarray(rows)


def real_window(weights, kc, mb, previous, inputs, direction=None, eta=0.,
                b=.6, history=True):
    state, states = np.asarray(previous).copy(), []
    digest = hashlib.sha256()
    for lag, raw in enumerate(inputs):
        stimulus = np.asarray(raw).copy()
        if lag == 0 and direction is not None:
            stimulus = stimulus + eta*direction
        state = manual_step(weights, kc, mb, state, stimulus, b, history)
        states.append(state.copy())
        digest.update(state.tobytes())
    return np.asarray(states), digest.hexdigest()


def structural_window(weights, kc, mb, source, steps):
    """Boolean dependency support; positive tanh factors do not resolve cancellation."""
    graph = weights.astype(bool)
    present = np.zeros(weights.shape[0], dtype=bool)
    rows = []
    for lag in range(steps):
        stimulus = source if lag == 0 else np.zeros(weights.shape[0], dtype=bool)
        first = np.asarray(graph.dot(present), dtype=bool) | stimulus
        mixed = present.copy()
        mixed[kc] = first[kc]
        first[mb] = np.asarray(graph[mb].dot(mixed), dtype=bool) | stimulus[mb]
        present = first
        rows.append(present.copy())
    return np.asarray(rows)


def finite_difference_gate(plus, minus, tangent, epsilon, atol=2e-9, rtol=2e-5):
    derivative = (np.asarray(plus)-np.asarray(minus))/(2*epsilon)
    error = np.abs(derivative-tangent)
    excess = error-atol-rtol*np.abs(tangent)
    return derivative, error.max(axis=-1), excess.max(axis=-1), bool(np.all(excess <= 0))


def exact_power(observed, contrasts):
    """Exact rational metric on float64 coordinates; no tolerance changes the gate."""
    observed = np.asarray(observed)
    contrasts = np.asarray(contrasts)
    assert observed.ndim == 3 and contrasts.ndim == 2 and len(observed) == len(contrasts)
    power = [Fraction(0) for _ in range(observed.shape[1])]
    valid = True
    for values, direction in zip(observed, contrasts):
        denominator = sum((Fraction.from_float(float(value))**2 for value in direction), Fraction(0))
        if denominator == 0:
            valid = False
            continue
        for lag, row in enumerate(values):
            power[lag] += sum((Fraction.from_float(float(value))**2 for value in row), Fraction(0))/denominator
    return [value/len(observed) for value in power], valid


def exact_block_endpoint(power, nonzero, minimum=1e-8, lag=5):
    minimum_square = Fraction(str(minimum))**2
    eligible = bool(nonzero and power[0] >= minimum_square)
    passed = bool(power[lag] <= power[0]/100) if eligible else None
    return dict(eligible=eligible, attenuation_pass=passed,
                lag0_power_exact=str(power[0]), lag5_power_exact=str(power[lag]))


def endpoint(blocks, smoke=False):
    assert blocks
    eligible = all(row['eligible'] for row in blocks)
    passed = None if smoke or not eligible else all(row['attenuation_pass'] is True for row in blocks)
    return dict(eligible=eligible, primary_pass=passed,
                outcome='smoke' if smoke else 'assay-invalid' if not eligible else 'PASS' if passed else 'FAIL')


def estimates(values, c):
    values = np.asarray(values, dtype=float)
    assert len(values) and np.isfinite(values).all()
    draws = np.random.default_rng(c['bootstrap_seed']).integers(
        0, len(values), (c['bootstrap_draws'], len(values)))
    lower, upper = np.quantile(np.mean(values[draws], axis=1), [.025, .975])
    return dict(mean=float(values.mean()), median=float(np.median(values)),
                sd=float(values.std(ddof=1)) if len(values)>1 else 0.,
                bootstrap_lo=float(lower), bootstrap_hi=float(upper))


def close(actual, expected, c, complex_values=False):
    atol = c['complex_step_atol'] if complex_values else c['reduction_atol']
    rtol = c['complex_step_rtol'] if complex_values else c['reduction_rtol']
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


def compare_tree(actual, expected, c):
    if isinstance(expected, dict):
        assert set(actual) == set(expected), (set(actual), set(expected))
        for key in expected:
            compare_tree(actual[key], expected[key], c)
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            compare_tree(a, e, c)
    elif isinstance(expected, float):
        close(actual, expected, c)
    else:
        assert actual == expected, (actual, expected)


def full_trajectory(weights, patterns, roles, observed, symbols, anchors, c, resources):
    kc, mb = np.flatnonzero(roles == 'KC'), np.flatnonzero(roles == 'MBON')
    state = np.zeros(weights.shape[0])
    features, norms = np.empty((len(symbols), len(observed))), np.empty(len(symbols))
    pre_states = {}
    digest = hashlib.sha256()
    anchor_set = set(map(int, anchors))
    for time, symbol in enumerate(symbols):
        if time in anchor_set:
            pre_states[time] = state.copy()
        state = manual_step(weights, kc, mb, state, patterns[symbol], c['leak'])
        features[time], norms[time] = state[observed], np.linalg.norm(state)
        digest.update(state.tobytes())
        if time % 100 == 0:
            resources.check()
    return features, norms, state, digest.hexdigest(), pre_states


def independent_ridge(train, test, train_labels, c):
    k, lags = c['alphabet_size'], c['lags']
    mean = train.mean(axis=0)
    scale = np.maximum(train.std(axis=0), 1e-5)
    normalized = (train-mean)/scale
    targets = np.eye(k)[train_labels].reshape(len(train), -1)
    intercept = targets.mean(axis=0)
    design = np.vstack([normalized, np.sqrt(c['alpha'])*np.eye(train.shape[1])])
    rhs = np.vstack([targets-intercept, np.zeros((train.shape[1], targets.shape[1]))])
    coefficients = np.linalg.lstsq(design, rhs, rcond=None)[0]
    scores = (((test-mean)/scale)@coefficients+intercept).reshape(len(test), len(lags), k)
    return dict(mean=mean, scale=scale, coefficients=coefficients, intercept=intercept,
                scores=scores, predictions=scores.argmax(axis=2))


def independent_tdc(saved, weights, patterns, roles, obs, c, block, level, resources):
    warmup, k = c['warmup'], c['alphabet_size']
    anchors = warmup + np.floor(np.linspace(0, c['test_samples']-1-max(c['lags']),
                                             c['probe_count'])).astype(np.int64)
    rebuilt, labels, streams, exact, pre = {}, {}, {}, {}, {}
    for split in ['train', 'test']:
        stream = np.random.default_rng(block[split+'_seed']).integers(
            0, k, warmup+c[split+'_samples'], dtype=np.int64)
        streams[split] = stream
        np.testing.assert_array_equal(stream, saved[split+'_symbols'])
        x, norms, final, digest, pre_states = full_trajectory(
            weights, patterns, roles, obs, stream, anchors if split == 'test' else [], c, resources)
        for suffix, value in [('features', x), ('norms', norms), ('final_state', final)]:
            np.testing.assert_array_equal(saved[split+'_'+suffix], value)
        exact[split] = digest
        if split == 'test':
            pre = pre_states
        times = np.arange(warmup, len(stream))
        np.testing.assert_array_equal(times, saved[split+'_times'])
        labels[split] = np.column_stack([stream[warmup-lag:len(stream)-lag]
                                         if lag else stream[warmup:] for lag in c['lags']])
        np.testing.assert_array_equal(labels[split], saved['y'+split])
        rebuilt[split] = x[warmup:]
    head = independent_ridge(rebuilt['train'], rebuilt['test'], labels['train'], c)
    for name in ['mean', 'scale', 'coefficients', 'intercept', 'scores']:
        np.testing.assert_allclose(saved[name], head[name], atol=1e-9, rtol=1e-9)
    np.testing.assert_array_equal(saved['predictions'], head['predictions'])
    rows = []
    for column, lag in enumerate(c['lags']):
        training, truth = labels['train'][:, column], labels['test'][:, column]
        majority = int(np.bincount(training, minlength=k).argmax())
        table = np.zeros((k, k), dtype=np.int64)
        for present, past in zip(streams['train'][warmup:], training):
            table[present, past] += 1
        lookup = np.array([row.argmax() if row.sum() else majority for row in table])
        current = lookup[streams['test'][warmup:]]
        frequency = np.full(len(truth), majority)
        np.testing.assert_array_equal(saved['current_tables'][column], table)
        np.testing.assert_array_equal(saved['current_predictions'][:, column], current)
        np.testing.assert_array_equal(saved['frequency_predictions'][:, column], frequency)
        assert saved['majority'][column] == majority
        count = int(np.count_nonzero(head['predictions'][:, column] == truth))
        freq = int(np.count_nonzero(frequency == truth))
        now = int(np.count_nonzero(current == truth))
        n = len(truth)
        baseline = max(1/k, freq/n, now/n)
        rows.append(dict(level=level, seed=block['seed'], lag=lag, samples=n, correct=count,
                         frequency_correct=freq, current_correct=now, accuracy=count/n,
                         frequency_accuracy=freq/n, current_accuracy=now/n, chance=1/k,
                         chance_adjusted=(count/n-1/k)/(1-1/k), baseline=baseline,
                         baseline_excess=count/n-baseline))
    return rows, exact, anchors, pre, streams['test']


def matrix_digest(matrix):
    h = hashlib.sha256()
    for vector in [matrix.data, matrix.indices, matrix.indptr]:
        h.update(vector.tobytes())
    return h.hexdigest()


def compare_frame(actual, expected, keys, c):
    assert set(actual.columns) == set(expected.columns)
    assert not actual.duplicated(keys).any() and not expected.duplicated(keys).any()
    actual = actual.sort_values(keys).reset_index(drop=True)
    expected = expected[actual.columns].sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected, check_exact=False,
        atol=c['reduction_atol'], rtol=c['reduction_rtol'], check_dtype=False)


def normalized_power(vectors, squared_norms):
    """Recompute a block power from saved doubles, not rounded curve CSVs."""
    values = []
    for row, denominator in zip(vectors, squared_norms):
        assert np.isfinite(row).all() and denominator > 0
        values.append(sum((Fraction.from_float(float(v))**2 for v in row), Fraction(0)) /
                      Fraction.from_float(float(denominator)))
    return sum(values, Fraction(0))/len(values)


def cell_from_vectors(vectors, denominators, level, seed, c, technical=True):
    if (not np.isfinite(vectors).all() or not np.isfinite(denominators).all()
            or np.any(np.asarray(denominators) <= 0)):
        return dict(level=level, seed=int(seed), eligible=False, scientific_rule_pass=None,
                    s0_numerator=None, s0_denominator=None, s5_numerator=None,
                    s5_denominator=None, current_gain=None, relative_gain=None,
                    invalid_reason='zero or invalid contrast')
    s0 = normalized_power(vectors[:, 0], denominators)
    s5 = normalized_power(vectors[:, c['primary_lag']], denominators)
    eligible = bool(technical and s0 >= Fraction(1, 10**16))
    return dict(level=level, seed=int(seed), eligible=eligible,
                scientific_rule_pass=bool(s5 <= s0/100) if eligible else None,
                s0_numerator=s0.numerator, s0_denominator=s0.denominator,
                s5_numerator=s5.numerator, s5_denominator=s5.denominator,
                current_gain=float(np.sqrt(float(s0))),
                relative_gain=float(np.sqrt(float(s5/s0))) if s0 else None,
                invalid_reason=None if eligible else 'technical check or unresolved immediate gain')


def verify_probes(root, out, saved, weights, patterns, roles, observed,
                  anchors, pre, test_symbols, c, resources):
    kc, mb = np.flatnonzero(roles == 'KC'), np.flatnonzero(roles == 'MBON')
    n, b, count = weights.shape[0], c['leak'], c['probe_count']
    np.testing.assert_array_equal(saved['anchors'], anchors)
    np.testing.assert_array_equal(saved['pre_states'], np.asarray([pre[int(t)] for t in anchors]))
    current = test_symbols[anchors]
    alternative = (current+1) % c['alphabet_size']
    np.testing.assert_array_equal(saved['pulse_symbols'], current)
    np.testing.assert_array_equal(saved['alternative_symbols'], alternative)
    directions = patterns[alternative]-patterns[current]
    squared = np.sum(directions*directions, axis=1)
    assert np.all(squared >= 0)
    np.testing.assert_array_equal(saved['contrast_input_squared_norm'], squared)
    row_mass = np.asarray(abs(weights).sum(axis=1)).ravel()
    assert row_mass.max() <= c['gain']+c['reduction_atol']
    prototype = np.asarray([manual_step(weights, kc, mb, np.zeros(n), p, b, False)
                            for p in patterns])
    np.testing.assert_array_equal(saved['instantaneous_state_prototypes'], prototype)
    metadata = read(root/'probe-info.json')
    assert metadata['technical_checks_pass'] is True and metadata['finite_difference_checks_pass'] is True
    for name, digest in metadata['array_hashes'].items():
        assert name in saved.files and array_sha(saved[name]) == digest, name
    assert len(metadata['probes']) == count
    complex_observed, envelope_observed, replacement_observed = [], [], []
    independent_instant, independent_instant_replacement = [], []
    all_probe_checks, selected_checks = [], []
    for index, anchor in enumerate(anchors):
        inputs = patterns[test_symbols[anchor:anchor+len(c['lags'])]]
        direction, initial = directions[index], pre[int(anchor)]
        tangent = complex_window(weights, kc, mb, initial, inputs, direction,
                                 c['complex_step_size'], b)
        envelope = envelope_window(weights, kc, mb, initial, inputs, direction, b)
        baseline, baseline_digest = real_window(weights, kc, mb, initial, inputs, b=b)
        replacement, replacement_digest = real_window(weights, kc, mb, initial, inputs,
                                                       direction, c['replacement_eta'], b)
        difference = replacement-baseline
        instant_tangent = complex_window(weights, kc, mb, initial, inputs, direction,
                                         c['complex_step_size'], b, False)
        instant_base, instant_base_digest = real_window(weights, kc, mb, initial, inputs,
                                                       b=b, history=False)
        instant_replacement, instant_replacement_digest = real_window(
            weights, kc, mb, initial, inputs, direction, c['replacement_eta'], b, False)
        instant_difference = instant_replacement-instant_base
        assert np.count_nonzero(instant_tangent[1:]) == 0
        assert np.count_nonzero(instant_difference[1:]) == 0
        np.testing.assert_array_equal(instant_base, prototype[test_symbols[anchor:anchor+len(c['lags'])]])
        replaced_symbols = test_symbols[anchor:anchor+len(c['lags'])].copy()
        replaced_symbols[0] = alternative[index]
        np.testing.assert_array_equal(instant_replacement, prototype[replaced_symbols])
        direct_replacement, direct_digest = real_window(weights, kc, mb, initial,
                                                       patterns[replaced_symbols], b=b)
        np.testing.assert_array_equal(direct_replacement, replacement)
        assert direct_digest == replacement_digest
        close(saved['probe_observed_tangents'][index], tangent[:, observed], c, True)
        close(saved['probe_observed_envelopes'][index], envelope[:, observed], c)
        np.testing.assert_array_equal(saved['probe_observed_replacement_differences'][index], difference[:, observed])
        close(saved['instantaneous_observed_tangents'][index], instant_tangent[:, observed], c, True)
        np.testing.assert_array_equal(saved['instantaneous_observed_replacement_differences'][index], instant_difference[:, observed])
        np.testing.assert_array_equal(saved['replacement_final_states'][index], replacement[-1])
        for name, values in [('tangent_inf_norm', tangent), ('envelope_inf_norm', envelope),
                             ('replacement_inf_norm', difference)]:
            close(saved[name][index], np.max(np.abs(values), axis=1), c, name=='tangent_inf_norm')
        tolerance = c['reduction_atol']+c['reduction_rtol']*np.abs(envelope)
        assert np.all(np.abs(tangent) <= envelope+tolerance)
        bound = b*np.max(np.abs(direction))*(b*c['gain'])**np.asarray(c['lags'])
        for values in [tangent, envelope, difference]:
            assert np.all(np.max(np.abs(values), axis=1) <= bound+c['reduction_atol']+c['reduction_rtol']*bound)
        support = structural_window(weights, kc, mb, direction != 0, len(c['lags']))
        np.testing.assert_array_equal(saved['contrast_support'][index], support[:, observed])
        assert np.count_nonzero(tangent[~support]) == 0
        assert np.count_nonzero(envelope[~support]) == 0
        record = metadata['probes'][index]
        expected = dict(probe_index=index, anchor=int(anchor), symbol=int(current[index]),
            alternative_symbol=int(alternative[index]), baseline_full_state_trajectory_sha256=baseline_digest,
            replacement_full_state_trajectory_sha256=replacement_digest,
            instantaneous_baseline_full_state_trajectory_sha256=instant_base_digest,
            instantaneous_replacement_full_state_trajectory_sha256=instant_replacement_digest,
            envelope_bound_pass=True, contraction_bound_pass=True,
            instantaneous_zero_past_exact=True, instantaneous_prototype_exact=True)
        compare_tree(record, expected, c)
        if index in c['finite_difference_probe_indices']:
            fdroot = root/f'fd-p{index:02d}.npz'
            fdmeta = read(root/f'fd-p{index:02d}.json')
            assert fdmeta['probe_index'] == index and fdmeta['anchor'] == int(anchor)
            assert fdmeta['all_checks_pass'] is True
            with np.load(fdroot, allow_pickle=False) as full:
                assert set(full.files) == set(fdmeta['array_hashes'])
                for name in full.files:
                    assert array_sha(full[name]) == fdmeta['array_hashes'][name], name
                close(full['full_tangent'], tangent, c, True)
                close(full['full_envelope'], envelope, c)
                np.testing.assert_array_equal(full['epsilon_values'], c['finite_difference_epsilons'])
                fds, error, excess, plus_final, minus_final = [], [], [], [], []
                plus_hashes, minus_hashes = [], []
                for epsilon in c['finite_difference_epsilons']:
                    plus, plus_digest = real_window(weights, kc, mb, initial, inputs, direction, epsilon, b)
                    minus, minus_digest = real_window(weights, kc, mb, initial, inputs, direction, -epsilon, b)
                    fd, e, ex, passed = finite_difference_gate(plus, minus, tangent, epsilon,
                        c['finite_difference_atol'], c['finite_difference_rtol'])
                    assert passed, 'Independent full-coordinate FD gate failed'
                    # The published gate is also recomputed against its exact saved JVP.
                    _, saved_error, saved_excess, saved_pass = finite_difference_gate(
                        plus, minus, full['full_tangent'], epsilon,
                        c['finite_difference_atol'], c['finite_difference_rtol'])
                    assert saved_pass
                    fds.append(fd[:, observed]);error.append(saved_error);excess.append(saved_excess)
                    plus_final.append(plus[-1]);minus_final.append(minus[-1])
                    plus_hashes.append(plus_digest);minus_hashes.append(minus_digest)
                np.testing.assert_array_equal(full['fd_observed'], np.asarray(fds))
                close(full['max_abs_error'], np.asarray(error), c)
                close(full['max_tolerance_excess'], np.asarray(excess), c)
                np.testing.assert_array_equal(full['last_plus_states'], np.asarray(plus_final))
                np.testing.assert_array_equal(full['last_minus_states'], np.asarray(minus_final))
                assert fdmeta['plus_full_state_trajectory_sha256'] == plus_hashes
                assert fdmeta['minus_full_state_trajectory_sha256'] == minus_hashes
                np.savez_compressed(out/f'complex-p{index:02d}.npz', full_complex_tangent=tangent,
                    full_independent_envelope=envelope, independently_recomputed_fd_observed=np.asarray(fds),
                    max_abs_error=np.asarray(error), max_tolerance_excess=np.asarray(excess))
                selected_checks.append(dict(probe_index=index, full_coordinates=n,
                    lags=len(c['lags']), epsilons=c['finite_difference_epsilons'],
                    full_complex_step_agreement=True, full_fd_gate=True,
                    exact_plus_trace_hashes=plus_hashes, exact_minus_trace_hashes=minus_hashes))
        complex_observed.append(tangent[:, observed]);envelope_observed.append(envelope[:, observed])
        replacement_observed.append(difference[:, observed])
        independent_instant.append(instant_tangent[:, observed])
        independent_instant_replacement.append(instant_difference[:, observed])
        all_probe_checks.append(expected)
        resources.check()
    independent = dict(complex_observed_tangents=np.asarray(complex_observed),
        independent_observed_envelopes=np.asarray(envelope_observed),
        independent_observed_replacement_differences=np.asarray(replacement_observed),
        independent_instantaneous_observed_tangents=np.asarray(independent_instant),
        independent_instantaneous_observed_replacement_differences=np.asarray(independent_instant_replacement),
        contrast_input_squared_norm=squared, anchors=anchors)
    np.savez_compressed(out/'independent-probes.npz', **independent)
    write(out/'probe-checks.json', all_probe_checks)
    write(out/'finite-difference-checks.json', selected_checks)
    return independent, dict(probes=count, lags=len(c['lags']),
        selected_full_state_complex_probes=len(selected_checks), full_state_fd_epsilons=2,
        observed_complex_step_agreement=True, full_coordinate_complex_step_agreement=True,
        full_coordinate_fd_gate=True, unsigned_envelope_bound=True,
        contraction_bound=True, instantaneous_zero_history_exact=True,
        exact_replacement_and_instantaneous_full_state_hashes=4*count,
        exact_centered_fd_full_state_hashes=4*len(selected_checks))


def case_job(job):
    c, block, level = job['config'], job['block'], job['level']
    root, out = Path(job['result'])/f'{level}_s{block["seed"]}', Path(job['out'])
    manifest = check_manifest(root)
    assert manifest['block'] == block and manifest['level'] == level
    graph = read(root/'graph.json')
    with threadpool_limits(1):
        with attempt(out, c, 'm5-independent-case') as resources:
            weights, patterns, roles, obs, ids = independent_graph(c, level, block['input_seed'])
            assert matrix_digest(weights) == graph['weight_sha256']
            assert graph['direct_carry_multiplier'] == 0 and graph['drive_multiplier'] == .6
            assert graph['decoder_parameters_per_lag'] == 490
            assert ids[obs].astype(str).tolist() == graph['observation_root_ids']
            actual_inputs = [set(ids[pattern != 0].astype(str)) for pattern in patterns]
            assert actual_inputs == [set(row) for row in graph['input_root_ids']]
            _, small_ids, _ = partial_source()
            local = pd.Index(ids).get_indexer(small_ids)
            assert (local >= 0).all()
            assert hashlib.sha256(small_ids.tobytes()+patterns[:, local].tobytes()).hexdigest() == graph['input_mapping_sha256']
            with np.load(root/'case.npz', allow_pickle=False) as saved:
                assert set(saved.files) == set(graph['array_hashes'])
                for name in saved.files:
                    assert array_sha(saved[name]) == graph['array_hashes'][name], name
                    if saved[name].dtype.kind in 'fc':
                        assert np.isfinite(saved[name]).all(), name
                np.testing.assert_array_equal(saved['input_patterns'], patterns)
                np.testing.assert_array_equal(saved['observed_indices'], obs)
                np.testing.assert_array_equal(saved['root_ids'], ids)
                np.testing.assert_array_equal(saved['roles'], roles)
                rows, traces, anchors, pre, symbols = independent_tdc(
                    saved, weights, patterns, roles, obs, c, block, level, resources)
                for split in ['train', 'test']:
                    assert traces[split] == graph[split+'_full_state_trajectory_sha256'], split
                compare_frame(pd.DataFrame(read(root/'metrics.json')), pd.DataFrame(rows), ['lag'], c)
                independent, probes = verify_probes(root, out, saved, weights, patterns, roles,
                                                    obs, anchors, pre, symbols, c, resources)
                official = cell_from_vectors(saved['probe_observed_tangents'],
                    saved['contrast_input_squared_norm'], level, block['seed'], c)
                reconstructed = cell_from_vectors(independent['complex_observed_tangents'],
                    independent['contrast_input_squared_norm'], level, block['seed'], c)
                assert official['eligible'] == reconstructed['eligible']
                assert official['scientific_rule_pass'] == reconstructed['scientific_rule_pass'], \
                    'Independent boundary decision disagreement: no tolerance rescues the criterion'
                compare_tree(read(root/'primary-cell.json'), official, c)
                sensitivity_rows = probe_metric_rows(saved, level, block['seed'], c)
                compare_frame(pd.DataFrame(read(root/'sensitivity-metrics.json')),
                              pd.DataFrame(sensitivity_rows), ['probe_index', 'lag'], c)
            write(out/'official-saved-float-cell.json', official)
            write(out/'independent-complex-cell.json', reconstructed)
            checks = dict(all_checks_pass=True, level=level, seed=block['seed'],
                exact_full_state_trace_hash=traces, independently_refit_heads=len(c['lags']),
                exact_prediction_arrays=True, official_cell=official,
                independent_complex_cell_decision_equal=True, **probes)
            write(out/'checks.json', checks)
        seal(out, dict(complete=True, result_case_manifest_sha256=sha(root/'manifest.json')))
    return dict(identity=f'{level}/s{block["seed"]}', rows=rows, sensitivity_rows=sensitivity_rows, cell=official,
                checks=checks, resources=read(out/'resources.json'))


def probe_metric_rows(saved, level, seed, c):
    result = []
    bounds = .3*.54**np.asarray(c['lags'])
    for probe, anchor in enumerate(saved['anchors']):
        denominator = saved['contrast_input_squared_norm'][probe]
        for index, lag in enumerate(c['lags']):
            gain = float(np.linalg.norm(saved['probe_observed_tangents'][probe, index])/np.sqrt(denominator)) if denominator else None
            unsigned = float(np.linalg.norm(saved['probe_observed_envelopes'][probe, index])/np.sqrt(denominator)) if denominator else None
            finite = float(np.linalg.norm(saved['probe_observed_replacement_differences'][probe, index])/np.sqrt(denominator)) if denominator else None
            result.append(dict(level=level, seed=int(seed), probe_index=probe, anchor=int(anchor),
                lag=int(lag), symbol=int(saved['pulse_symbols'][probe]),
                alternative_symbol=int(saved['alternative_symbols'][probe]),
                contrast_squared_norm=float(denominator), gain=gain, unsigned_gain=unsigned,
                replacement_gain=finite, signed_to_envelope=gain/unsigned if unsigned else None,
                replacement_to_tangent=finite/gain if gain else None,
                tangent_inf_norm=float(saved['tangent_inf_norm'][probe, index]),
                envelope_inf_norm=float(saved['envelope_inf_norm'][probe, index]),
                replacement_inf_norm=float(saved['replacement_inf_norm'][probe, index]),
                global_inf_bound=float(bounds[index]),
                structural_observed_count=int(saved['contrast_support'][probe, index].sum())))
    return result


def independent_block_curves(raw):
    records = []
    for (level, seed), part in raw.groupby(['level', 'seed'], sort=True):
        now = part[part.lag == 0]
        immediate = float(np.sqrt(np.mean(now.gain.to_numpy(dtype=float)**2)))
        for lag, probes in part.groupby('lag', sort=True):
            gain = float(np.sqrt(np.mean(probes.gain.to_numpy(dtype=float)**2)))
            unsigned = float(np.sqrt(np.mean(probes.unsigned_gain.to_numpy(dtype=float)**2)))
            finite = float(np.sqrt(np.mean(probes.replacement_gain.to_numpy(dtype=float)**2)))
            records.append(dict(level=level, seed=int(seed), lag=int(lag), gain=gain,
                unsigned_gain=unsigned, replacement_gain=finite,
                signed_to_envelope=gain/unsigned if unsigned else None,
                replacement_to_tangent=finite/gain if gain else None,
                structural_observed_fraction=float(probes.structural_observed_count.mean()/48),
                relative_gain=gain/immediate if immediate else None,
                relative_replacement_gain=finite/immediate if immediate else None))
    return pd.DataFrame(records)


def independent_statistics(frame, metrics, c):
    records = []
    for (level, lag), block in frame.groupby(['level', 'lag'], sort=True):
        record = dict(level=level, lag=int(lag), n_blocks=len(block))
        block = block.sort_values('seed')
        for metric in metrics:
            values = block[metric].to_numpy(dtype=float)
            summary = estimates(values, c) if np.isfinite(values).all() else {name: None for name in ESTIMATES}
            record.update({metric+'_'+name: value for name, value in summary.items()})
        records.append(record)
    return pd.DataFrame(records)


def independent_summary(cells, c, smoke):
    levels = {}
    for level in c['levels']:
        blocks = sorted([cell for cell in cells if cell['level'] == level], key=lambda row: row['seed'])
        assert [cell['seed'] for cell in blocks] == c['blocks_by_level'][level]
        eligible = all(cell['eligible'] for cell in blocks)
        rule = None if smoke or not eligible else all(cell['scientific_rule_pass'] is True for cell in blocks)
        outcome = 'smoke' if smoke else 'assay-invalid' if not eligible else 'PASS' if rule else 'FAIL'
        levels[level] = dict(eligible=eligible, registered_rule_pass=rule, outcome=outcome,
            cells=blocks, relative_lag5_gain=estimates([cell['relative_gain'] for cell in blocks], c)
                if all(cell['relative_gain'] is not None for cell in blocks) else None)
    primary = levels[c['primary_level']]
    return dict(smoke=bool(smoke), outcome=primary['outcome'], primary_pass=primary['registered_rule_pass'],
        levels=levels, cases=len(cells), lag_heads=len(cells)*len(c['lags']),
        probe_pairs=len(cells)*c['probe_count'],
        selected_full_fd_probes=len(cells)*len(c['finite_difference_probe_indices']),
        criterion_unchanged=True, biological_plasticity_performed=False,
        unique_cycle_effect_claim=False, memory_loss_from_attenuation_claim=False,
        m1_outcome_unchanged='assay-invalid', m4_outcome_unchanged='INFEASIBLE')


def verify_pairing(result, c):
    pairs = 0
    for block in c['blocks']:
        cases = [(level, result/f'{level}_s{block["seed"]}') for level in c['levels']
                 if block['seed'] in c['blocks_by_level'][level]]
        if len(cases) < 2:
            continue
        graph_a, graph_b = [read(root/'graph.json') for _, root in cases]
        for name in ['input_root_ids', 'observation_root_ids', 'input_mapping_sha256',
                     'decoder_parameters_per_lag']:
            assert graph_a[name] == graph_b[name], name
        with np.load(cases[0][1]/'case.npz', allow_pickle=False) as a, np.load(
                cases[1][1]/'case.npz', allow_pickle=False) as b:
            for name in ['train_symbols', 'test_symbols', 'ytrain', 'ytest', 'anchors',
                         'pulse_symbols', 'alternative_symbols', 'contrast_input_squared_norm']:
                np.testing.assert_array_equal(a[name], b[name])
        pairs += 1
    return dict(paired_partial_whole_blocks=pairs, streams_mapping_observed_ids_and_budget_paired=True)


def authenticate_source(source):
    assert source['protocol_commit'] == PROTOCOL_COMMIT
    assert source['source_tree'] == git('rev-parse', source['source_commit']+'^{tree}')
    assert source['tracked_changes'] is False and source['every_tracked_hash_authenticated_at_capture'] is True
    subprocess.run(['git', 'merge-base', '--is-ancestor', PROTOCOL_COMMIT, source['source_commit']], check=True)
    for name, digest in source['hashes'].items():
        assert sha(name) == digest, name
        if not name.startswith('outputs/'):
            blob = subprocess.check_output(['git', 'show', source['source_commit']+':'+name])
            assert hashlib.sha256(blob).hexdigest() == digest, name
    for name in ['configs/temporal_functional_sensitivity.json', 'docs/temporal-functional-sensitivity-protocol.md']:
        blob = subprocess.check_output(['git', 'show', PROTOCOL_COMMIT+':'+name])
        assert hashlib.sha256(blob).hexdigest() == source['hashes'][name]


def verify(result, out):
    manifest = check_manifest(result)
    smoke = manifest['smoke']
    assert type(smoke) is bool
    files = {path.relative_to(result).as_posix() for path in result.rglob('*') if path.is_file()}
    assert files == set(manifest['artifacts']) | {'manifest.json'}, 'Unmanifested result files'
    c = read(result/'config.json')
    assert c == config(smoke) == manifest['config']
    original_source = read(result/'source.json')
    assert original_source['source_commit'] == manifest['source_commit']
    authenticate_source(original_source)
    verifier_source = source_record(c)
    with attempt(out, c, 'm5-independent-validation') as resources:
        write(out/'config.json', c)
        write(out/'source.json', verifier_source)
        write(out/'environment.json', environment())
        preservation = history_preserved(c)
        compare_tree(read(result/'historical-preservation.json'), preservation, c)
        write(out/'source-validation.json', dict(result_source_commit=original_source['source_commit'],
            verifier_source_commit=verifier_source['source_commit'], protocol_commit=PROTOCOL_COMMIT,
            recorded_source_hashes_authenticated=len(original_source['hashes']),
            independent_complex_step_no_main_jacobian_import=True))
        whole = validate_whole_source(c, resources)
        write(out/'source-graph-audit.json', whole)
        jobs = [dict(config=c, block=block, level=level, result=str(result),
                     out=str(out/f'{level}_s{block["seed"]}'))
                for level in c['levels'] for block in c['blocks']
                if block['seed'] in c['blocks_by_level'][level]]
        results, usage = pooled(case_job, jobs, c['neural_workers'], c, 'M5 independent verification')
        tdc = pd.DataFrame([row for value in results for row in value['rows']])
        raw = pd.DataFrame([row for value in results for row in value['sensitivity_rows']])
        cells = [value['cell'] for value in results]
        curves = independent_block_curves(raw)
        for filename, frame, keys in [('raw-lags.csv', tdc, ['level', 'seed', 'lag']),
              ('raw-sensitivity.csv', raw, ['level', 'seed', 'probe_index', 'lag']),
              ('block-sensitivity.csv', curves, ['level', 'seed', 'lag']),
              ('tdc-summary.csv', independent_statistics(tdc,
                ['accuracy', 'chance_adjusted', 'baseline_excess', 'frequency_accuracy', 'current_accuracy'], c),
                ['level', 'lag']),
              ('sensitivity-summary.csv', independent_statistics(curves,
                ['gain', 'relative_gain', 'unsigned_gain', 'replacement_gain', 'signed_to_envelope',
                 'replacement_to_tangent', 'relative_replacement_gain', 'structural_observed_fraction'], c),
                ['level', 'lag'])]:
            compare_frame(pd.read_csv(result/filename), frame, keys, c)
            frame.sort_values(keys).to_csv(out/('recomputed-'+filename), index=False)
        compare_tree(read(result/'primary-cells.json'),
                     sorted(cells, key=lambda row: (row['level'], row['seed'])), c)
        summary = independent_summary(cells, c, smoke)
        compare_tree(read(result/'summary.json'), summary, c)
        write(out/'recomputed-summary.json', summary)
        checks = [value['checks'] for value in results]
        pairing = verify_pairing(result, c)
        write(out/'case-checks.json', checks)
        write(out/'process-tree-resources.json', usage)
        write(out/'job-resources.json', [dict(identity=value['identity'], resources=value['resources']) for value in results])
        write(out/'checks.json', dict(all_checks_pass=True, outcome=summary['outcome'],
            primary_pass=summary['primary_pass'], cases=len(checks),
            replayed_streams=2*len(checks), exact_full_state_hashes=2*len(checks),
            independently_refit_lag_heads=len(tdc),
            observed_complex_step_probes=c['probe_count']*len(checks),
            observed_complex_step_vectors=c['probe_count']*len(c['lags'])*len(checks),
            selected_full_complex_step_probes=len(c['finite_difference_probe_indices'])*len(checks),
            full_coordinate_fd_probe_epsilon_lag_checks=len(c['finite_difference_probe_indices'])*2*len(c['lags'])*len(checks),
            exact_replacement_and_instantaneous_probe_trace_hashes=4*c['probe_count']*len(checks),
            exact_centered_fd_trace_hashes=4*len(c['finite_difference_probe_indices'])*len(checks),
            exact_saved_float_primary_recalculated=True,
            independent_complex_primary_decisions_same=True,
            independent_complex_step_no_main_jacobian_import=True,
            full_coordinate_finite_differences_both_epsilons=True,
            instantaneous_past_influence_exact_zero=True,
            unsigned_envelope_structural_support_and_contraction_verified=True,
            source_graph_reconstruction=whole, pairing=pairing,
            scientific_threshold_tolerance_applied=False,
            biological_plasticity_performed=False, unique_cycle_effect_claim=False,
            memory_loss_from_attenuation_claim=False,
            result_manifest_sha256=sha(result/'manifest.json'),
            source_commit=verifier_source['source_commit'], protocol_commit=PROTOCOL_COMMIT,
            historical_preservation=preservation))
        assert history_preserved(c) == preservation
        assert_source_unchanged(verifier_source)
        resources.check()
    seal(out, dict(complete=True, result_manifest_sha256=sha(result/'manifest.json'),
                   source_commit=verifier_source['source_commit']))
    print(dict(all_checks_pass=True, outcome=summary['outcome'], cases=len(cells),
               exact_full_state_hashes=2*len(cells), independently_refit_heads=len(tdc)), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(1):
        verify(args.result, args.out)


if __name__ == '__main__':
    main()
