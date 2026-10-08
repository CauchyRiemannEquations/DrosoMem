"""Registered M5 fixed-graph finite-lag sensitivity with fresh iid TDC context."""
import argparse
from fractions import Fraction
import hashlib
from pathlib import Path
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from flying.data.mushroom_body import load_roles
from flying.models.ridge import RidgeDecoder
from flying.training import whole_brain_memory as core
from temporal_memory_curve import build, controls, statistics
from temporal_mechanism import FactorReservoir, estimates
from sensitivity_dynamics import PulseDynamics, replay, fd_gate, primary_cell, endpoint
from sensitivity_support import (config, read, write, array_sha, attempt, seal,
    source_record, assert_source_unchanged, history_preserved)
from tdc_pool import pooled


def anchors(c):
    return c['warmup'] + np.floor(np.linspace(
        0, c['test_samples']-1-max(c['lags']), c['probe_count'])).astype(np.int64)


def identifiers(c, level):
    if level == 'brain5':
        with np.load(Path(c['cache'])/'nodes.npz', allow_pickle=False) as a:
            return a['ids'].copy(), np.asarray(a['roles'], dtype='U5')
    folder = Path('data/flywire_783_mb_left_kc512_s701')
    _, ids, _ = core.load_connectome(folder)
    roles, _ = load_roles(folder, ids)
    return np.asarray(ids, dtype=np.int64), np.asarray(roles, dtype='U5')


def collect_with_pre_states(model, observed, symbols, requested, resources):
    model.reset()
    features = np.empty((len(symbols), len(observed)))
    norms = np.empty(len(symbols))
    pre = np.empty((len(requested), len(model.state)))
    lookup = {int(t): i for i, t in enumerate(requested)}
    digest = hashlib.sha256()
    for t, symbol in enumerate(symbols):
        if t in lookup:
            pre[lookup[t]] = model.state
        state = model.step(int(symbol))
        features[t] = state[observed]
        norms[t] = np.linalg.norm(state)
        digest.update(state.tobytes())
        if t % 100 == 0:
            resources.check()
    assert np.isfinite(features).all() and np.isfinite(pre).all()
    return features, norms, model.state.copy(), digest.hexdigest(), pre


def decoding_rows(a, c, block, level):
    w, k = c['warmup'], c['alphabet_size']
    targets = np.eye(k)[a['ytrain']].reshape(c['train_samples'], -1)
    decoder = RidgeDecoder(c['alpha']).fit(a['train_features'][w:], targets)
    scores = decoder.scores(a['test_features'][w:]).reshape(c['test_samples'], len(c['lags']), k)
    a.update(mean=decoder.mean, scale=decoder.scale, coefficients=decoder.weights,
             intercept=decoder.target_mean, scores=scores, predictions=scores.argmax(axis=2))
    a['frequency_predictions'], a['current_predictions'], a['current_tables'], a['majority'] = controls(
        a['train_symbols'], a['test_symbols'], a['ytrain'], a['ytest'], w, k)
    rows = []
    for j, lag in enumerate(c['lags']):
        truth = a['ytest'][:, j]
        n = len(truth)
        correct = int(np.count_nonzero(a['predictions'][:, j] == truth))
        freq = int(np.count_nonzero(a['frequency_predictions'][:, j] == truth))
        current = int(np.count_nonzero(a['current_predictions'][:, j] == truth))
        accuracy = correct/n
        baseline = max(1/k, freq/n, current/n)
        rows.append(dict(level=level, seed=block['seed'], lag=int(lag), samples=n,
            correct=correct, frequency_correct=freq, current_correct=current,
            accuracy=accuracy, frequency_accuracy=freq/n, current_accuracy=current/n,
            chance=1/k, chance_adjusted=(accuracy-1/k)/(1-1/k),
            baseline=baseline, baseline_excess=accuracy-baseline))
    return rows


def finite_differences(dynamics, pre, inputs, direction, tangent, envelope,
                       observed, c, index, anchor, out):
    fd, last_plus, last_minus, plus_hashes, minus_hashes = [], [], [], [], []
    for epsilon in c['finite_difference_epsilons']:
        plus, ph = replay(dynamics, pre, inputs, direction, epsilon)
        minus, mh = replay(dynamics, pre, inputs, direction, -epsilon)
        fd.append((plus-minus)/(2*epsilon))
        last_plus.append(plus[-1].copy())
        last_minus.append(minus[-1].copy())
        plus_hashes.append(ph)
        minus_hashes.append(mh)
    fd = np.asarray(fd)
    checks = fd_gate(tangent, fd, c['finite_difference_atol'], c['finite_difference_rtol'])
    saved = dict(full_tangent=tangent, full_envelope=envelope,
        fd_observed=fd[:, :, observed], epsilon_values=np.asarray(c['finite_difference_epsilons']),
        max_abs_error=checks['max_abs_error'], max_tolerance_excess=checks['max_tolerance_excess'],
        last_plus_states=np.asarray(last_plus), last_minus_states=np.asarray(last_minus))
    stem = f'fd-p{index:02d}'
    np.savez_compressed(out/(stem+'.npz'), **saved)
    write(out/(stem+'.json'), dict(probe_index=index, anchor=int(anchor),
        all_checks_pass=checks['all_checks_pass'],
        plus_full_state_trajectory_sha256=plus_hashes,
        minus_full_state_trajectory_sha256=minus_hashes,
        array_hashes={name: array_sha(value) for name, value in saved.items()}))
    assert (out/(stem+'.npz')).stat().st_size < 100_000_000
    return checks['all_checks_pass']


def probe_panel(a, model, observed, roles, c, block, level, out, resources):
    n, p, l, o = model.weights.shape[0], c['probe_count'], len(c['lags']), len(observed)
    dynamics = PulseDynamics(model.weights, roles, c['leak'], True)
    instant = PulseDynamics(model.weights, roles, c['leak'], False)
    patterns = a['input_patterns']
    prototypes = np.asarray([instant.step(np.zeros(n), pattern) for pattern in patterns])
    a['instantaneous_state_prototypes'] = prototypes
    pulse = a['test_symbols'][a['anchors']]
    alternative = (pulse+1) % c['alphabet_size']
    directions = patterns[alternative]-patterns[pulse]
    denominators = np.sum(directions*directions, axis=1)
    # Fixed binary mapping makes every squared norm an exactly represented quarter multiple.
    for direction, denominator in zip(directions, denominators):
        exact = sum((Fraction.from_float(float(x))**2 for x in direction if x), Fraction(0))
        assert exact == Fraction.from_float(float(denominator))
    a.update(pulse_symbols=pulse, alternative_symbols=alternative,
             contrast_input_squared_norm=denominators,
             contrast_support=dynamics.support(directions, observed, c['lags']))
    for name in ['probe_observed_tangents', 'probe_observed_envelopes',
                 'probe_observed_replacement_differences', 'instantaneous_observed_tangents',
                 'instantaneous_observed_replacement_differences']:
        a[name] = np.empty((p, l, o))
    for name in ['tangent_inf_norm', 'envelope_inf_norm', 'replacement_inf_norm']:
        a[name] = np.empty((p, l))
    a['replacement_final_states'] = np.empty((p, n))
    records, rows, fd_passes = [], [], []
    bounds = .3 * .54**np.asarray(c['lags'])
    for index, anchor in enumerate(a['anchors']):
        inputs = patterns[a['test_symbols'][anchor:anchor+l]]
        pre, direction = a['pre_states'][index], directions[index]
        state, v, e = pre.copy(), np.zeros(n), np.zeros(n)
        states, tangents, envelopes = [], [], []
        digest = hashlib.sha256()
        for lag, stimulation in enumerate(inputs):
            state, v, e = dynamics.linearized_step(state, stimulation, v, e,
                direction if lag == 0 else np.zeros(n))
            digest.update(state.tobytes())
            states.append(state.copy())
            tangents.append(v.copy())
            envelopes.append(e.copy())
        states, tangents, envelopes = map(np.asarray, [states, tangents, envelopes])
        np.testing.assert_array_equal(states[:, observed], a['test_features'][anchor:anchor+l])
        replacement, rh = replay(dynamics, pre, inputs, direction, c['replacement_eta'])
        difference = replacement-states
        a['replacement_final_states'][index] = replacement[-1]
        a['probe_observed_tangents'][index] = tangents[:, observed]
        a['probe_observed_envelopes'][index] = envelopes[:, observed]
        a['probe_observed_replacement_differences'][index] = difference[:, observed]
        a['tangent_inf_norm'][index] = np.abs(tangents).max(axis=1)
        a['envelope_inf_norm'][index] = envelopes.max(axis=1)
        a['replacement_inf_norm'][index] = np.abs(difference).max(axis=1)
        tolerance = c['reduction_atol']+c['reduction_rtol']*envelopes
        envelope_pass = bool(np.all(np.abs(tangents) <= envelopes+tolerance))
        bound_tolerance = c['reduction_atol']+c['reduction_rtol']*bounds
        contraction_pass = bool(np.all(a['tangent_inf_norm'][index] <= bounds+bound_tolerance)
            and np.all(a['envelope_inf_norm'][index] <= bounds+bound_tolerance)
            and np.all(a['replacement_inf_norm'][index] <= bounds+bound_tolerance))
        instant_base, ih = replay(instant, pre, inputs)
        instant_replacement, irh = replay(instant, pre, inputs, direction, c['replacement_eta'])
        iv, ie, instant_tangents = np.zeros(n), np.zeros(n), []
        instant_state = pre.copy()
        for lag, stimulation in enumerate(inputs):
            instant_state, iv, ie = instant.linearized_step(instant_state, stimulation, iv, ie,
                direction if lag == 0 else np.zeros(n))
            instant_tangents.append(iv.copy())
        instant_tangents = np.asarray(instant_tangents)
        instant_difference = instant_replacement-instant_base
        prototype_pass = bool(np.array_equal(instant_base, prototypes[a['test_symbols'][anchor:anchor+l]]))
        zero_pass = bool(np.array_equal(instant_tangents[1:], np.zeros_like(instant_tangents[1:]))
            and np.array_equal(instant_difference[1:], np.zeros_like(instant_difference[1:])))
        a['instantaneous_observed_tangents'][index] = instant_tangents[:, observed]
        a['instantaneous_observed_replacement_differences'][index] = instant_difference[:, observed]
        if index in c['finite_difference_probe_indices']:
            fd_passes.append(finite_differences(dynamics, pre, inputs, direction, tangents,
                envelopes, observed, c, index, anchor, out))
        records.append(dict(probe_index=index, anchor=int(anchor), symbol=int(pulse[index]),
            alternative_symbol=int(alternative[index]), baseline_full_state_trajectory_sha256=digest.hexdigest(),
            replacement_full_state_trajectory_sha256=rh,
            instantaneous_baseline_full_state_trajectory_sha256=ih,
            instantaneous_replacement_full_state_trajectory_sha256=irh,
            envelope_bound_pass=envelope_pass, contraction_bound_pass=contraction_pass,
            instantaneous_zero_past_exact=zero_pass, instantaneous_prototype_exact=prototype_pass))
        denominator = denominators[index]
        for j, lag in enumerate(c['lags']):
            gain = float(np.linalg.norm(tangents[j, observed])/np.sqrt(denominator)) if denominator else None
            unsigned = float(np.linalg.norm(envelopes[j, observed])/np.sqrt(denominator)) if denominator else None
            finite = float(np.linalg.norm(difference[j, observed])/np.sqrt(denominator)) if denominator else None
            rows.append(dict(level=level, seed=block['seed'], probe_index=index, anchor=int(anchor),
                lag=int(lag), symbol=int(pulse[index]), alternative_symbol=int(alternative[index]),
                contrast_squared_norm=float(denominator), gain=gain, unsigned_gain=unsigned,
                replacement_gain=finite, signed_to_envelope=gain/unsigned if unsigned else None,
                replacement_to_tangent=finite/gain if gain else None,
                tangent_inf_norm=float(a['tangent_inf_norm'][index, j]),
                envelope_inf_norm=float(a['envelope_inf_norm'][index, j]),
                replacement_inf_norm=float(a['replacement_inf_norm'][index, j]),
                global_inf_bound=float(bounds[j]), structural_observed_count=int(a['contrast_support'][index, j].sum())))
        resources.check()
    fd_valid = bool(len(fd_passes) == len(c['finite_difference_probe_indices']) and all(fd_passes))
    technical = bool(fd_valid and all(all(record[key] for key in ['envelope_bound_pass',
        'contraction_bound_pass', 'instantaneous_zero_past_exact', 'instantaneous_prototype_exact']) for record in records))
    info = dict(technical_checks_pass=technical, finite_difference_checks_pass=fd_valid,
        probes=records, array_hashes={name: array_sha(value) for name, value in a.items()})
    cell = primary_cell(a['probe_observed_tangents'], denominators, block['seed'], level,
        c['minimum_current_gain'], c['primary_lag'], technical)
    return rows, info, cell


def execute(c, block, level, out, resources):
    start = time.perf_counter()
    base, observed, graph = build(c, level, block)
    ids, roles = identifiers(c, level)
    model = FactorReservoir(base, False, True)
    a = dict(input_patterns=model.encoder.patterns.copy(), observed_indices=observed,
             root_ids=ids, roles=roles, anchors=anchors(c))
    traces = {}
    for split in ['train', 'test']:
        symbols = np.random.default_rng(block[split+'_seed']).integers(
            0, c['alphabet_size'], c['warmup']+c[split+'_samples'], dtype=np.int64)
        requested = a['anchors'] if split == 'test' else np.empty(0, dtype=np.int64)
        features, norms, final, digest, pre = collect_with_pre_states(model, observed, symbols, requested, resources)
        times = np.arange(c['warmup'], len(symbols))
        a.update({split+'_symbols': symbols, split+'_features': features,
            split+'_norms': norms, split+'_final_state': final, split+'_times': times,
            'y'+split: symbols[times[:, None]-np.asarray(c['lags'])[None, :]]})
        traces[split+'_full_state_trajectory_sha256'] = digest
        if split == 'test':
            a['pre_states'] = pre
    tdc_rows = decoding_rows(a, c, block, level)
    sensitivity_rows, probe_info, cell = probe_panel(a, model, observed, roles, c, block, level, out, resources)
    graph.update(**traces, direct_carry_multiplier=0., drive_multiplier=.6,
        same_step_kc_to_mbon_retained=True, decoder_parameters_per_lag=490,
        synaptic_history=True, array_hashes={name: array_sha(value) for name, value in a.items()},
        seconds=time.perf_counter()-start)
    assert core.weight_hash(model.weights) == graph['weight_sha256']
    return a, tdc_rows, sensitivity_rows, graph, probe_info, cell


def case_job(job):
    c, block, level, out = job['config'], job['block'], job['level'], Path(job['out'])
    with threadpool_limits(1):
        with attempt(out, c, 'functional-sensitivity-case') as resources:
            arrays, tdc, sensitivity, graph, probes, cell = execute(c, block, level, out, resources)
            np.savez_compressed(out/'case.npz', **arrays)
            assert (out/'case.npz').stat().st_size < 100_000_000
            write(out/'metrics.json', tdc)
            write(out/'sensitivity-metrics.json', sensitivity)
            write(out/'graph.json', graph)
            write(out/'probe-info.json', probes)
            write(out/'primary-cell.json', cell)
            assert probes['technical_checks_pass'], 'M5 technical gate failed; retain this attempt'
        seal(out, dict(complete=True, block=block, level=level))
    return dict(identity=f'{level}/s{block["seed"]}', rows=tdc,
        sensitivity_rows=sensitivity, primary_cell=cell, resources=read(out/'resources.json'))


def block_curves(raw, c):
    rows = []
    for (level, seed), part in raw.groupby(['level', 'seed'], sort=True):
        gains = {}
        for lag, group in part.groupby('lag', sort=True):
            gain = float(np.sqrt(np.mean(np.square(group.gain.to_numpy(dtype=float)))))
            envelope = float(np.sqrt(np.mean(np.square(group.unsigned_gain.to_numpy(dtype=float)))))
            finite = float(np.sqrt(np.mean(np.square(group.replacement_gain.to_numpy(dtype=float)))))
            gains[int(lag)] = gain
            rows.append(dict(level=level, seed=int(seed), lag=int(lag), gain=gain,
                unsigned_gain=envelope, replacement_gain=finite,
                signed_to_envelope=gain/envelope if envelope else None,
                replacement_to_tangent=finite/gain if gain else None,
                structural_observed_fraction=float(group.structural_observed_count.mean()/48)))
        for row in rows:
            if row['level'] == level and row['seed'] == int(seed):
                row['relative_gain'] = row['gain']/gains[0] if gains[0] else None
                row['relative_replacement_gain'] = row['replacement_gain']/gains[0] if gains[0] else None
    return pd.DataFrame(rows)


def sensitivity_statistics(frame, c):
    rows = []
    for (level, lag), part in frame.groupby(['level', 'lag'], sort=True):
        row = dict(level=level, lag=int(lag), n_blocks=len(part))
        for metric in ['gain', 'relative_gain', 'unsigned_gain', 'replacement_gain',
                       'signed_to_envelope', 'replacement_to_tangent', 'relative_replacement_gain',
                       'structural_observed_fraction']:
            values = part.sort_values('seed')[metric].to_numpy(dtype=float)
            if np.isfinite(values).all():
                row.update({metric+'_'+key: value for key, value in estimates(values, c).items()})
            else:
                row.update({metric+'_'+key: None for key in ['mean', 'median', 'sd', 'bootstrap_lo', 'bootstrap_hi']})
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(cells, c, smoke):
    levels = {}
    for level in c['levels']:
        selected = sorted((row for row in cells if row['level'] == level), key=lambda row: row['seed'])
        assert [row['seed'] for row in selected] == c['blocks_by_level'][level]
        levels[level] = dict(**endpoint(selected, smoke), cells=selected,
            relative_lag5_gain=estimates([row['relative_gain'] for row in selected], c)
                if all(row['relative_gain'] is not None for row in selected) else None)
    primary = levels[c['primary_level']]
    return dict(smoke=bool(smoke), outcome=primary['outcome'], primary_pass=primary['registered_rule_pass'],
        levels=levels, cases=len(cells), lag_heads=len(cells)*len(c['lags']),
        probe_pairs=len(cells)*c['probe_count'], selected_full_fd_probes=len(cells)*len(c['finite_difference_probe_indices']),
        criterion_unchanged=True, biological_plasticity_performed=False,
        unique_cycle_effect_claim=False, memory_loss_from_attenuation_claim=False,
        m1_outcome_unchanged='assay-invalid', m4_outcome_unchanged='INFEASIBLE')


def pair_checks(out, c):
    for block in c['blocks']:
        paired = []
        for level in c['levels']:
            if block['seed'] not in c['blocks_by_level'][level]:
                continue
            case = out/f'{level}_s{block["seed"]}'
            with np.load(case/'case.npz', allow_pickle=False) as a:
                paired.append({key: a[key] for key in ['train_symbols', 'test_symbols', 'ytrain', 'ytest', 'anchors']})
            graph = read(case/'graph.json')
            paired[-1].update({key: graph[key] for key in ['input_root_ids', 'observation_root_ids', 'input_mapping_sha256']})
        for other in paired[1:]:
            for key, value in paired[0].items():
                if isinstance(value, np.ndarray):
                    np.testing.assert_array_equal(value, other[key])
                else:
                    assert value == other[key]


def plot(curves, tdc, out, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for column, level in enumerate(['legacy5', 'brain5']):
        part = curves[curves.level == level]
        ax = axes[0, column]
        for _, block in part.groupby('seed'):
            ax.semilogy(block.lag, block.relative_gain, color='#17679e', alpha=.2, lw=.8)
        for metric, label, color in [('relative_gain', 'Local derivative / immediate gain', '#17679e'),
                                    ('relative_replacement_gain', 'Valid replacement / immediate local gain', '#ba6822')]:
            mean = part.groupby('lag')[metric].mean()
            ax.semilogy(mean.index, mean, 'o-', color=color, markersize=3, label=label)
        ax.axhline(.1, color='gray', ls=':', label='Registered lag5 attenuation threshold')
        ax.axvline(5, color='gray', ls='--', lw=.8)
        ax.set(title=level, xlabel='Steps since one symbol contrast', ylabel='Observed normalized relative gain')
        ax.legend(fontsize=8)
        ax = axes[1, column]
        for _, block in tdc[tdc.level == level].groupby('seed'):
            ax.plot(block.lag, block.accuracy, color='#188c72', alpha=.2, lw=.8)
        mean = tdc[tdc.level == level].groupby('lag').accuracy.mean()
        ax.plot(mean.index, mean, 'o-', color='#188c72', markersize=3, label='Fresh iid ridge decoding')
        ax.axhline(.1, color='gray', ls=':', label='Uniform chance')
        ax.set(xlabel='Historical symbol lag', ylabel='Independent test accuracy', ylim=(0, 1.02))
        ax.legend(fontsize=8)
    fig.suptitle('M5 finite-lag sensitivity - registered outcome: '+summary['outcome'])
    fig.tight_layout()
    fig.savefig(out/'functional-sensitivity.png', dpi=180)
    plt.close(fig)


def run(out, smoke=False):
    c = config(smoke)
    source = source_record(c)
    with attempt(out, c, 'functional-sensitivity-smoke' if smoke else 'functional-sensitivity-main'):
        write(out/'config.json', c)
        write(out/'source.json', source)
        write(out/'historical-preservation.json', history_preserved(c))
        jobs = [dict(config=c, block=block, level=level, out=str(out/f'{level}_s{block["seed"]}'))
            for level in c['levels'] for block in c['blocks'] if block['seed'] in c['blocks_by_level'][level]]
        results, usage = pooled(case_job, jobs, c['neural_workers'], c, 'M5 functional sensitivity')
        tdc = pd.DataFrame([row for result in results for row in result['rows']]).sort_values(['level', 'seed', 'lag']).reset_index(drop=True)
        raw = pd.DataFrame([row for result in results for row in result['sensitivity_rows']]).sort_values(['level', 'seed', 'probe_index', 'lag']).reset_index(drop=True)
        cells = [result['primary_cell'] for result in results]
        assert len(tdc) == len(jobs)*len(c['lags']) and len(raw) == len(jobs)*c['probe_count']*len(c['lags'])
        curves = block_curves(raw, c)
        summary = summarize(cells, c, smoke)
        for name, frame in [('raw-lags', tdc), ('tdc-summary', statistics(tdc, c)),
                            ('raw-sensitivity', raw), ('block-sensitivity', curves),
                            ('sensitivity-summary', sensitivity_statistics(curves, c))]:
            frame.to_csv(out/(name+'.csv'), index=False)
        write(out/'primary-cells.json', sorted(cells, key=lambda row: (row['level'], row['seed'])))
        write(out/'summary.json', summary)
        write(out/'process-tree-resources.json', usage)
        write(out/'job-resources.json', [dict(identity=result['identity'], resources=result['resources']) for result in results])
        pair_checks(out, c)
        plot(curves, tdc, out, summary)
        history_preserved(c)
        assert_source_unchanged(source)
    seal(out, dict(complete=True, smoke=smoke, config=c, source_commit=source['source_commit']))
    print(dict(outcome=summary['outcome'], primary_pass=summary['primary_pass'], cases=summary['cases']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    with threadpool_limits(1):
        run(args.out, args.smoke)
