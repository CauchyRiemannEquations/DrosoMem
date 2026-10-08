"""M6 frozen-readout tracking after one actual past-symbol replacement.

No decoder is fitted here. Every fixed anchor is included in the paired gate.
"""
import argparse
import hashlib
from pathlib import Path
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from flying.training import whole_brain_memory as core
from temporal_memory_curve import build
from temporal_mechanism import FactorReservoir, estimates
from tdc_pool import pooled
from counterfactual_support import (config, read, write, sha, array_sha, attempt,
    seal, source_record, assert_source_unchanged, history_preserved)


def scoring(features, mean, scale, coefficients, intercept):
    """Apply the archived scaler and affine head without any fitting."""
    features = np.asarray(features)
    mean, scale = np.asarray(mean), np.asarray(scale)
    coefficients, intercept = np.asarray(coefficients), np.asarray(intercept)
    assert features.shape[-1] == mean.size == scale.size
    assert coefficients.shape == (mean.size, intercept.size)
    assert np.isfinite(features).all() and np.isfinite(scale).all() and (scale > 0).all()
    return ((features-mean)/scale) @ coefficients + intercept


def extract_head(coefficients, intercept, lag, k=10):
    """Parent targets are flattened in ascending lag-major, then class order."""
    start = int(lag)*k
    assert coefficients.ndim == 2 and intercept.ndim == 1
    assert coefficients.shape[1] == len(intercept) and start+k <= len(intercept)
    return coefficients[:, start:start+k].copy(), intercept[start:start+k].copy()


def joint_pass(joint_correct, samples):
    assert isinstance(joint_correct, (int, np.integer)) and isinstance(samples, (int, np.integer))
    assert samples > 0 and 0 <= joint_correct <= samples
    return bool(10*int(joint_correct) >= 9*int(samples))


def trial_counts(original, replacement, original_targets, replacement_targets):
    original, replacement = np.asarray(original), np.asarray(replacement)
    s, r = np.asarray(original_targets), np.asarray(replacement_targets)
    assert original.shape == replacement.shape == s.shape == r.shape and s.ndim == 1
    assert len(s) and np.all(s != r)
    first, second = original == s, replacement == r
    return dict(samples=len(s), original_correct=int(first.sum()),
        replacement_correct=int(second.sum()), joint_correct=int((first & second).sum()),
        original_accuracy=float(first.mean()), replacement_accuracy=float(second.mean()),
        joint_accuracy=float((first & second).mean()),
        prediction_changes=int(np.count_nonzero(original != replacement)))


def endpoint(cells, smoke=False):
    assert cells and all(row['samples'] == 64 for row in cells)
    assert all(row['technical_checks_pass'] for row in cells)
    passed = None if smoke else all(joint_pass(row['joint_correct'], row['samples']) for row in cells)
    return dict(outcome='smoke' if smoke else 'PASS' if passed else 'FAIL',
                registered_rule_pass=passed)


def anchor_times(c):
    return c['warmup'] + np.floor(np.linspace(
        0, c['test_samples']-1-c['primary_lag'], c['probe_count'])).astype(np.int64)


def replacement_window(symbols, anchor, lag=2, k=10):
    original = np.asarray(symbols[anchor:anchor+lag+1], dtype=np.int64).copy()
    assert original.shape == (lag+1,)
    replacement = original.copy()
    replacement[0] = (int(original[0])+1) % k
    assert replacement[0] != original[0] and np.array_equal(replacement[1:], original[1:])
    return original, replacement


def state_sha(state):
    return hashlib.sha256(np.ascontiguousarray(state).tobytes()).hexdigest()


def parent_arrays(c, block, level):
    """Read only authenticated frozen arrays; never read parent replacement outcomes."""
    parent = Path(c['parent_namespace'])/f'{level}_s{block["parent_seed"]}'
    main = parent.parent
    assert sha(main/'manifest.json') == c['parent_manifest_sha256']
    parent_main = read(main/'manifest.json')
    source = read(main/'source.json')
    assert source['source_commit'] == c['parent_source_commit']
    assert source['protocol_commit'] == c['parent_protocol_commit']
    assert parent_main['source_commit'] == c['parent_source_commit']
    parent_manifest = read(parent/'manifest.json')
    assert parent_manifest['complete'] and parent_manifest['level'] == level
    assert parent_manifest['block']['seed'] == block['parent_seed']
    assert parent_manifest['block']['input_seed'] == block['input_seed']
    for name in ['manifest.json', 'graph.json', 'case.npz']:
        relative = parent.name+'/'+name
        assert sha(parent/name) == parent_main['artifacts'][relative]
    graph = read(parent/'graph.json')
    assert sha(parent/'case.npz') == parent_manifest['artifacts']['case.npz']
    names = ['input_patterns', 'observed_indices', 'root_ids', 'roles', 'mean',
             'scale', 'coefficients', 'intercept', 'current_tables', 'majority']
    with np.load(parent/'case.npz', allow_pickle=False) as archive:
        old = {name: archive[name].copy() for name in names}
    assert all(array_sha(value) == graph['array_hashes'][name] for name, value in old.items())
    assert old['coefficients'].shape == (48, 210) and old['intercept'].shape == (210,)
    assert old['mean'].shape == old['scale'].shape == (48,)
    assert old['current_tables'].shape == (21, 10, 10) and old['majority'].shape == (21,)
    assert old['input_patterns'].shape == (10, graph['neurons'])
    assert old['root_ids'].shape == old['roles'].shape == (graph['neurons'],)
    assert old['observed_indices'].shape == (48,)
    assert all(np.isfinite(value).all() for value in old.values() if value.dtype.kind in 'iuf')
    assert (old['scale'] >= 1e-5).all()
    patterns = old['input_patterns']
    assert len({row.tobytes() for row in patterns}) == 10, 'Non-distinct parent input patterns'
    assert np.all(np.count_nonzero(patterns, axis=1) == 51) and np.all((patterns == 0) | (patterns == .5))
    assert np.all(old['roles'][np.any(patterns != 0, axis=0)] == 'KC')
    assert np.all(old['roles'][old['observed_indices']] == 'MBON')
    assert [str(x) for x in old['root_ids'][old['observed_indices']]] == graph['observation_root_ids']
    assert [list(map(str, old['root_ids'][np.flatnonzero(row)])) for row in patterns] == graph['input_root_ids']
    frozen = {name: old[name] for name in ['input_patterns', 'observed_indices', 'root_ids', 'roles', 'mean', 'scale']}
    for lag in [0, 2]:
        frozen[f'coefficients_lag{lag}'], frozen[f'intercept_lag{lag}'] = extract_head(
            old['coefficients'], old['intercept'], lag)
    frozen['current_table_lag2'] = old['current_tables'][2].copy()
    frozen['majority_lag2'] = np.asarray(old['majority'][2], dtype=np.int64)
    mapping = frozen['current_table_lag2'].argmax(axis=1)
    mapping[frozen['current_table_lag2'].sum(axis=1) == 0] = frozen['majority_lag2']
    frozen['current_lookup_lag2'] = mapping
    lineage = dict(parent_case_path=parent.as_posix(), parent_case_sha256=sha(parent/'case.npz'),
        parent_case_manifest_sha256=sha(parent/'manifest.json'), parent_graph_sha256=sha(parent/'graph.json'),
        parent_main_manifest_sha256=c['parent_manifest_sha256'],
        parent_source_commit=c['parent_source_commit'], parent_protocol_commit=c['parent_protocol_commit'],
        parent_array_hashes={name: array_sha(value) for name, value in old.items()},
        frozen_array_hashes={name: array_sha(value) for name, value in frozen.items()})
    return frozen, graph, lineage


def replay_window(base, previous, symbols, observed, history=True):
    model = FactorReservoir(base, carry=False, synaptic_history=history)
    model.state = previous.copy()
    features, digest = [], hashlib.sha256()
    for symbol in symbols:
        state = model.step(int(symbol))
        features.append(state[observed].copy())
        digest.update(state.tobytes())
    return np.asarray(features), model.state.copy(), digest.hexdigest()


def apply_head(features, frozen, lag=2):
    return scoring(features, frozen['mean'], frozen['scale'],
                   frozen[f'coefficients_lag{lag}'], frozen[f'intercept_lag{lag}'])


def target_margin(scores, targets):
    masked = scores.copy()
    masked[np.arange(len(scores)), targets] = -np.inf
    return scores[np.arange(len(scores)), targets]-masked.max(axis=1)


def score_trials(a, c, block, level, smoke=False):
    s, r = a['original_symbols'], a['alternative_symbols']
    worlds = ['original', 'counterfactual', 'instantaneous_original', 'instantaneous_counterfactual']
    for world in worlds:
        a['scores_'+world] = apply_head(a['trial_'+world+'_features'][:, -1], a)
        a['predictions_'+world] = a['scores_'+world].argmax(axis=1)
    a['frequency_trial_predictions'] = np.full(len(s), int(a['majority_lag2']), dtype=np.int64)
    a['current_trial_predictions'] = a['current_lookup_lag2'][a['current_symbols']]
    counts = trial_counts(a['predictions_original'], a['predictions_counterfactual'], s, r)
    controls = {}
    for name in ['frequency', 'current']:
        predictions = a[name+'_trial_predictions']
        controls[name+'_joint_correct'] = trial_counts(predictions, predictions, s, r)['joint_correct']
    controls['instantaneous_joint_correct'] = trial_counts(a['predictions_instantaneous_original'],
        a['predictions_instantaneous_counterfactual'], s, r)['joint_correct']
    np.testing.assert_array_equal(a['scores_instantaneous_original'], a['scores_instantaneous_counterfactual'])
    np.testing.assert_array_equal(a['predictions_instantaneous_original'], a['predictions_instantaneous_counterfactual'])
    assert all(value == 0 for value in controls.values())
    a['original_target_margin'] = target_margin(a['scores_original'], s)
    a['replacement_target_margin'] = target_margin(a['scores_counterfactual'], r)
    ix = np.arange(len(s))
    a['original_contrast_margin'] = a['scores_original'][ix, r]-a['scores_original'][ix, s]
    a['replacement_contrast_margin'] = a['scores_counterfactual'][ix, r]-a['scores_counterfactual'][ix, s]
    a['contrast_margin_change'] = a['replacement_contrast_margin']-a['original_contrast_margin']
    difference = a['trial_counterfactual_features'][:, -1]-a['trial_original_features'][:, -1]
    normalized = difference/a['scale']
    head = a['coefficients_lag2']
    a['projected_contrast_margin_change'] = np.asarray([
        normalized[j] @ (head[:, r[j]]-head[:, s[j]]) for j in range(len(s))])
    np.testing.assert_allclose(a['contrast_margin_change'], a['projected_contrast_margin_change'],
        atol=c['score_atol'], rtol=c['score_rtol'])
    a['observed_delta_norm'] = np.linalg.norm(difference, axis=1)
    cell = dict(level=level, seed=block['seed'], parent_seed=block['parent_seed'], **counts, **controls,
        scientific_rule_pass=None if smoke else joint_pass(counts['joint_correct'], counts['samples']),
        technical_checks_pass=True)
    rows = []
    for j, anchor in enumerate(a['anchors']):
        row = dict(level=level, seed=block['seed'], probe_index=j, anchor=int(anchor),
            evaluation_time=int(a['evaluation_times'][j]), original_symbol=int(s[j]),
            replacement_symbol=int(r[j]), current_symbol=int(a['current_symbols'][j]),
            original_correct=bool(a['predictions_original'][j] == s[j]),
            replacement_correct=bool(a['predictions_counterfactual'][j] == r[j]),
            joint_correct=bool(a['predictions_original'][j] == s[j] and a['predictions_counterfactual'][j] == r[j]),
            prediction_changed=bool(a['predictions_original'][j] != a['predictions_counterfactual'][j]))
        for world in worlds:
            row['prediction_'+world] = int(a['predictions_'+world][j])
        for name in ['frequency', 'current']:
            row['prediction_'+name] = int(a[name+'_trial_predictions'][j])
        for name in ['original_target_margin', 'replacement_target_margin', 'original_contrast_margin',
                     'replacement_contrast_margin', 'contrast_margin_change',
                     'projected_contrast_margin_change', 'observed_delta_norm']:
            row[name] = float(a[name][j])
        rows.append(row)
    return cell, rows


def execute(c, block, level, resources, smoke=False):
    started = time.perf_counter()
    frozen, parent_graph, lineage = parent_arrays(c, block, level)
    base, observed, graph = build(c, level, block)
    model = FactorReservoir(base, carry=False, synaptic_history=True)
    np.testing.assert_array_equal(observed, frozen['observed_indices'])
    np.testing.assert_array_equal(model.encoder.patterns, frozen['input_patterns'])
    np.testing.assert_array_equal(model.kc, np.flatnonzero(frozen['roles'] == 'KC'))
    np.testing.assert_array_equal(model.mbon, np.flatnonzero(frozen['roles'] == 'MBON'))
    assert core.weight_hash(model.weights) == parent_graph['weight_sha256'] == graph['weight_sha256']
    assert graph['input_root_ids'] == parent_graph['input_root_ids']
    assert graph['observation_root_ids'] == parent_graph['observation_root_ids']
    a = dict(frozen)
    w, n = c['warmup'], c['test_samples']
    symbols = np.random.default_rng(block['stream_seed']).integers(0, 10, w+n, dtype=np.int64)
    anchors = anchor_times(c)
    assert len(anchors) == len(set(anchors.tolist())) == 64 and anchors[-1]+2 < len(symbols)
    a.update(stream_symbols=symbols, anchors=anchors, evaluation_times=anchors+2,
        original_symbols=symbols[anchors], alternative_symbols=(symbols[anchors]+1)%10,
        current_symbols=symbols[anchors+2])
    a['features'], a['norms'] = np.empty((w+n, 48)), np.empty(w+n)
    for world in ['original', 'counterfactual', 'instantaneous_original', 'instantaneous_counterfactual']:
        a['trial_'+world+'_features'] = np.empty((64, 3, 48))
    a['trial_original_symbols'] = np.empty((64, 3), dtype=np.int64)
    a['trial_counterfactual_symbols'] = np.empty((64, 3), dtype=np.int64)
    lookup = {int(anchor): index for index, anchor in enumerate(anchors)}
    pending, trial_info, digest = {}, [], hashlib.sha256()
    for t, symbol in enumerate(symbols):
        if t in lookup:
            index = lookup[t]
            previous = model.state.copy()
            original, replacement = replacement_window(symbols, t)
            a['trial_original_symbols'][index], a['trial_counterfactual_symbols'][index] = original, replacement
            info = dict(probe_index=index, anchor=t, evaluation_time=t+2, pre_state_sha256=state_sha(previous))
            finals = {}
            for world, window, history in [
                    ('original', original, True), ('counterfactual', replacement, True),
                    ('instantaneous_original', original, False), ('instantaneous_counterfactual', replacement, False)]:
                features, final, trace = replay_window(base, previous, window, observed, history)
                a['trial_'+world+'_features'][index] = features
                info[world+'_full_state_trajectory_sha256'] = trace
                info[world+'_final_state_sha256'] = state_sha(final)
                finals[world] = final
            np.testing.assert_array_equal(finals['instantaneous_original'], finals['instantaneous_counterfactual'])
            assert info['instantaneous_original_final_state_sha256'] == info['instantaneous_counterfactual_final_state_sha256']
            trial_info.append(info)
            pending[index] = hashlib.sha256()
        state = model.step(int(symbol))
        a['features'][t], a['norms'][t] = state[observed], np.linalg.norm(state)
        digest.update(state.tobytes())
        for index in list(pending):
            pending[index].update(state.tobytes())
            if t == int(anchors[index])+2:
                info = trial_info[index]
                info['baseline_window_sha256'] = pending.pop(index).hexdigest()
                info['baseline_endpoint_full_state_sha256'] = state_sha(state)
                assert info['baseline_window_sha256'] == info['original_full_state_trajectory_sha256']
                assert info['baseline_endpoint_full_state_sha256'] == info['original_final_state_sha256']
                np.testing.assert_array_equal(a['trial_original_features'][index], a['features'][t-2:t+1])
        if t % 50 == 0:
            resources.check()
    assert not pending and len(trial_info) == 64
    a['final_state'] = model.state.copy()
    a['times'] = np.arange(w, w+n, dtype=np.int64)
    a['y0'], a['y2'] = symbols[a['times']], symbols[a['times']-2]
    metrics = []
    for lag in [0, 2]:
        scores = apply_head(a['features'][w:], a, lag)
        predictions = scores.argmax(axis=1)
        a[f'scores_lag{lag}'], a[f'predictions_lag{lag}'] = scores, predictions
        metrics.append(dict(level=level, seed=block['seed'], lag=lag, samples=n,
            correct=int(np.count_nonzero(predictions == a[f'y{lag}'])),
            accuracy=float(np.mean(predictions == a[f'y{lag}'])), chance=.1))
    a['frequency_predictions'] = np.full(n, int(a['majority_lag2']), dtype=np.int64)
    a['current_predictions'] = a['current_lookup_lag2'][a['y0']]
    metrics[-1].update(frequency_correct=int(np.count_nonzero(a['frequency_predictions'] == a['y2'])),
        current_correct=int(np.count_nonzero(a['current_predictions'] == a['y2'])),
        frequency_accuracy=float(np.mean(a['frequency_predictions'] == a['y2'])),
        current_accuracy=float(np.mean(a['current_predictions'] == a['y2'])))
    cell, rows = score_trials(a, c, block, level, smoke)
    for index, anchor in enumerate(anchors):
        np.testing.assert_allclose(a['scores_original'][index], a['scores_lag2'][int(anchor)+2-w],
            atol=c['score_atol'], rtol=c['score_rtol'])
        assert a['predictions_original'][index] == a['predictions_lag2'][int(anchor)+2-w]
    assert all(np.isfinite(value).all() for value in a.values() if value.dtype.kind in 'iuf')
    assert core.weight_hash(model.weights) == graph['weight_sha256']
    graph.update(**lineage, baseline_full_state_trajectory_sha256=digest.hexdigest(),
        final_state_sha256=state_sha(a['final_state']), direct_carry_multiplier=0., drive_multiplier=.6,
        same_step_kc_to_mbon_retained=True, synaptic_history=True, decoder_parameters_per_lag=490,
        new_trained_parameters=0, array_hashes={name: array_sha(value) for name, value in a.items()},
        seconds=time.perf_counter()-started)
    probes = dict(technical_checks_pass=True, trials=trial_info,
        every_fixed_anchor_counted=True, new_trained_parameters=0,
        original_replay_equals_baseline=True, instantaneous_full_endpoint_pair_equal=True,
        fixed_head_margin_projection_verified=True)
    return a, metrics, cell, rows, graph, probes


def case_job(job):
    c, block, level, out = job['config'], job['block'], job['level'], Path(job['out'])
    with threadpool_limits(1):
        with attempt(out, c, 'counterfactual-tracking-case') as resources:
            arrays, metrics, cell, rows, graph, probes = execute(c, block, level, resources, job['smoke'])
            np.savez_compressed(out/'case.npz', **arrays)
            assert (out/'case.npz').stat().st_size < 100_000_000
            write(out/'metrics.json', metrics)
            write(out/'primary-cell.json', cell)
            write(out/'trial-metrics.json', dict(counts=cell, rows=rows))
            write(out/'graph.json', graph)
            write(out/'trial-info.json', probes)
        seal(out, dict(complete=True, block=block, level=level, smoke=job['smoke']))
    return dict(identity=f'{level}/s{block["seed"]}', cell=cell, rows=rows, metrics=metrics,
                resources=read(out/'resources.json'))


def summarize(cells, c, smoke=False):
    levels = {}
    for level in c['levels']:
        selected = sorted((row for row in cells if row['level'] == level), key=lambda row: row['seed'])
        assert [row['seed'] for row in selected] == c['blocks_by_level'][level]
        levels[level] = dict(**endpoint(selected, smoke), cells=selected,
            **{name: estimates([row[name] for row in selected], c) for name in
               ['joint_accuracy', 'original_accuracy', 'replacement_accuracy']})
    primary = levels[c['primary_level']]
    return dict(smoke=bool(smoke), outcome=primary['outcome'], primary_pass=primary['registered_rule_pass'],
        levels=levels, cases=len(cells), probe_pairs=len(cells)*c['probe_count'],
        frozen_heads_used=len(cells), new_trained_parameters=0, decoder_parameters_per_lag=490,
        criterion_unchanged=True, every_fixed_anchor_counted=True, performance_eligibility_gate=False,
        biological_plasticity_performed=False, unique_cycle_effect_claim=False,
        m1_outcome_unchanged='assay-invalid', m4_outcome_unchanged='INFEASIBLE')


def pair_checks(out, c):
    checked = []
    for block in c['blocks']:
        paths = [out/f'{level}_s{block["seed"]}' for level in c['levels']
                 if block['seed'] in c['blocks_by_level'][level]]
        if len(paths) != 2:
            continue
        with np.load(paths[0]/'case.npz', allow_pickle=False) as first, np.load(paths[1]/'case.npz', allow_pickle=False) as second:
            for name in ['stream_symbols', 'times', 'y0', 'y2', 'anchors', 'evaluation_times',
                         'original_symbols', 'alternative_symbols', 'current_symbols',
                         'trial_original_symbols', 'trial_counterfactual_symbols']:
                np.testing.assert_array_equal(first[name], second[name])
        first, second = read(paths[0]/'graph.json'), read(paths[1]/'graph.json')
        for name in ['input_root_ids', 'observation_root_ids', 'input_mapping_sha256']:
            assert first[name] == second[name]
        checked.append(block['seed'])
    return dict(paired_block_seeds=checked, inputs_targets_anchors_identical=True,
                computational_blocks_are_not_biological_replicates=True)


def plot(cells, out, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, level in zip(axes, ['legacy5', 'brain5']):
        selected = sorted((r for r in cells if r['level'] == level), key=lambda r: r['seed'])
        x = np.arange(len(selected))
        ax.plot(x, [r['original_accuracy'] for r in selected], 'o-', label='Original target accuracy')
        ax.plot(x, [r['replacement_accuracy'] for r in selected], 's-', label='Replacement target accuracy')
        ax.plot(x, [r['joint_accuracy'] for r in selected], 'D-', label='Paired transition accuracy')
        ax.axhline(.9, color='gray', ls=':', label='Registered paired threshold')
        ax.axhline(0, color='#666666', ls='--', label='Unchanged paired controls (exact 0)')
        ax.set(xticks=x, xticklabels=[str(r['seed']) for r in selected], xlabel='Fixed computational block',
               ylabel='Fraction of all 64 anchors', title=level, ylim=(-.04, 1.04))
        ax.tick_params(axis='x', labelrotation=25)
        ax.legend(fontsize=8)
    fig.suptitle('M6 frozen lag2 readout tracking — registered outcome: '+summary['outcome'])
    fig.tight_layout()
    fig.savefig(out/'counterfactual-tracking.png', dpi=180)
    plt.close(fig)


def run(out, smoke=False):
    c = config(smoke)
    source = source_record(c)
    with attempt(out, c, 'counterfactual-tracking-smoke' if smoke else 'counterfactual-tracking-main'):
        write(out/'config.json', c)
        write(out/'source.json', source)
        write(out/'historical-preservation.json', history_preserved(c))
        jobs = [dict(config=c, block=block, level=level, smoke=smoke,
                     out=str(out/f'{level}_s{block["seed"]}'))
                for level in c['levels'] for block in c['blocks']
                if block['seed'] in c['blocks_by_level'][level]]
        results, usage = pooled(case_job, jobs, c['neural_workers'], c, 'M6 frozen tracking')
        cells = sorted((result['cell'] for result in results), key=lambda row: (row['level'], row['seed']))
        raw = pd.DataFrame([row for result in results for row in result['rows']]).sort_values(['level', 'seed', 'probe_index'])
        metrics = pd.DataFrame([row for result in results for row in result['metrics']]).sort_values(['level', 'seed', 'lag'])
        assert len(raw) == len(jobs)*64 and len(metrics) == len(jobs)*2
        raw.to_csv(out/'raw-trials.csv', index=False)
        metrics.to_csv(out/'stream-metrics.csv', index=False)
        pd.DataFrame(cells).to_csv(out/'block-scores.csv', index=False)
        summary = summarize(cells, c, smoke)
        write(out/'primary-cells.json', cells)
        write(out/'summary.json', summary)
        write(out/'pairing.json', pair_checks(out, c))
        write(out/'process-tree-resources.json', usage)
        write(out/'job-resources.json', [dict(identity=result['identity'], resources=result['resources']) for result in results])
        plot(cells, out, summary)
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
