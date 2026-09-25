"""Post-training weight interpolation and factorial readout diagnosis.

Interpolation is NOT a smaller-learning-rate training trajectory. Only training
prefix states/labels are available here; novel-digit prediction is not evaluated.
"""
import copy
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from flying.brain.diagnostics import normalize_condition
from flying.brain.mushroom_body import CircuitReservoir, KCEncoder, SelectedReadout, role_shuffled
from flying.brain.plasticity import weight_hash
from flying.brain.reward_plasticity import RewardPlasticity, policy_hash
from flying.data.connectome import load_connectome, sha256
from flying.data.mushroom_body import load_roles
from flying.data.pi_digits import pi_digits
from flying.evaluation.free_recall import evaluate_recall

VIEWS = ('fixed', 'stats_only', 'coefficients_only', 'full_refit')
PAIR = ['circuit', 'seed', 'normalization', 'model', 'reward_delay', 'alpha', 'view']


def interpolate(initial, endpoint, alpha, roles):
    """Preserve exact CSR support and nonplastic entries, including at endpoints."""
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('alpha must be in [0, 1]')
    a = initial.tocsr(copy=True); b = endpoint.tocsr(copy=True)
    a.sort_indices(); b.sort_indices()
    if a.shape != b.shape or not np.array_equal(a.indices, b.indices) or not np.array_equal(a.indptr, b.indptr):
        raise ValueError('Interpolation requires identical edge support')
    roles = np.asarray(roles)
    row = np.repeat(np.arange(a.shape[0]), np.diff(a.indptr))
    mask = (roles[row] == 'MBON') & (roles[a.indices] == 'KC')
    if not np.array_equal(a.data[~mask], b.data[~mask]) or not np.array_equal(np.sign(a.data), np.sign(b.data)):
        raise ValueError('Endpoint changes nonplastic edges or signs')
    if not np.allclose(abs(a).sum(axis=1), abs(b).sum(axis=1), rtol=1e-12, atol=1e-14):
        raise ValueError('Endpoint violates row budgets')
    out = a.copy() if alpha == 0 else b.copy()
    if 0 < alpha < 1:
        out.data[mask] = a.data[mask] + alpha * (b.data[mask] - a.data[mask])
    assert np.array_equal(out.data[~mask], a.data[~mask])
    assert np.array_equal(np.sign(out.data), np.sign(a.data))
    error = float(np.max(abs(np.asarray(abs(out).sum(axis=1) - abs(a).sum(axis=1)))))
    assert error < 1e-12
    return out, dict(relative_plastic_change=float(np.linalg.norm(out.data[mask] - a.data[mask]) / np.linalg.norm(a.data[mask])),
                     max_budget_error=error)


def readout_views(policy, train_states, train_labels, cfg):
    """Four cells: old/new statistics crossed with old/new coefficients."""
    before = policy_hash(policy)
    selected = np.asarray(train_states)[:, policy.indices]
    stats = copy.deepcopy(policy)
    stats.model.mean = selected.mean(axis=0)
    stats.model.scale = np.maximum(selected.std(axis=0), 1e-5)
    kwargs = dict(epochs=cfg['readout_epochs'], learning_rate=cfg['readout_learning_rate'], l2=cfg['readout_l2'])
    coefficients = SelectedReadout(policy.indices)
    coefficients.fit(train_states, train_labels, feature_stats=(policy.model.mean, policy.model.scale), **kwargs)
    full = SelectedReadout(policy.indices)
    full.fit(train_states, train_labels, **kwargs)
    assert before == policy_hash(policy)
    assert np.array_equal(stats.model.weights, policy.model.weights)
    assert np.array_equal(coefficients.model.mean, policy.model.mean)
    assert np.array_equal(coefficients.model.scale, policy.model.scale)
    return dict(fixed=copy.deepcopy(policy), stats_only=stats, coefficients_only=coefficients, full_refit=full)


def paired_table(frame):
    trace = frame[frame.direction == 'reward_trace'].copy()
    yoked = frame[frame.direction == 'yoked_reward'][PAIR + ['pi_memory_score']].rename(columns={'pi_memory_score': 'yoked_score'})
    paired = trace.merge(yoked, on=PAIR, validate='one_to_one')
    base_keys = ['circuit', 'seed', 'normalization', 'model', 'reward_delay', 'view']
    baseline = trace[trace.alpha == 0][base_keys + ['pi_memory_score']].rename(columns={'pi_memory_score': 'frozen_score'})
    paired = paired.merge(baseline, on=base_keys, validate='many_to_one')
    paired['gain_frozen'] = paired.pi_memory_score - paired.frozen_score
    paired['gain_yoked'] = paired.pi_memory_score - paired.yoked_score
    return paired


def select_candidates(frame):
    paired = paired_table(frame)
    real = paired[(paired.model == 'fly') & (paired.alpha > 0)]
    candidates = []
    for keys, group in real.groupby(['normalization', 'reward_delay', 'view'], sort=True):
        means = group.groupby('alpha')[['gain_frozen', 'gain_yoked']].mean()
        eligible = means[(means.gain_frozen > 0) & (means.gain_yoked > 0)].copy()
        record = dict(zip(['normalization', 'reward_delay', 'view'], keys))
        record['reward_delay'] = int(record['reward_delay'])
        if eligible.empty:
            record.update(alpha=None, status='no_positive_candidate')
        else:
            eligible['minimum_gain'] = eligible.min(axis=1)
            ranked = eligible.reset_index().sort_values(['minimum_gain', 'alpha'], ascending=[False, True])
            best = ranked.iloc[0]
            record.update(alpha=float(best.alpha), status='candidate', discovery_gain_frozen=float(best.gain_frozen),
                          discovery_gain_yoked=float(best.gain_yoked))
        candidates.append(record)
    return candidates


def endpoint_training(initial, roles, encoder, policy, digits, cfg, seed, delay, yoked=None):
    trainer = RewardPlasticity(initial, roles, encoder, policy, cfg['leak'], cfg['microsteps'],
                              cfg['reward_learning_rate'], cfg['plastic_floor'], cfg['trace_decay'],
                              delay, cfg['temperature'], cfg['baseline_rate'])
    history, events = trainer.fit_reward(digits[:-1], digits[1:], cfg['reward_epochs'],
                                        seed=np.random.SeedSequence([seed, delay, 9501]), yoked_rewards=yoked)
    audit = trainer.audit()
    assert np.array_equal(events['true_rewards'], events['actions'] == digits[1:])
    if yoked is not None:
        assert np.array_equal(events['applied_rewards'], yoked)
    return trainer.weights.copy(), history, events, audit


def run(config_path, output, stage='discovery', selection_path=None):
    cfg = json.loads(Path(config_path).read_text())
    if stage not in ('discovery', 'confirmation'):
        raise ValueError(stage)
    if set(cfg['discovery_seeds']) & set(cfg['confirmation_seeds']):
        raise ValueError('Confirmation seeds overlap discovery')
    selection = None
    if stage == 'confirmation':
        if selection_path is None:
            raise ValueError('Confirmation requires locked discovery selection')
        selection = json.loads(Path(selection_path).read_text())
        if selection['config_sha256'] != sha256(config_path):
            raise ValueError('Configuration differs from discovery selection')
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    (out / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
    if selection:
        (out / 'locked_selection.json').write_text(json.dumps(selection, indent=2) + '\n')
    digits = pi_digits(cfg['pi_length'])  # No held-out suffix enters the experiment.
    seeds = cfg[stage + '_seeds']
    rows, recalls, endpoints, histories, event_keys, events_list, sources, checkpoints = [], [], [], [], [], [], [], {}
    legacy = pd.read_csv('results/phase5/results.csv') if stage == 'discovery' else None
    legacy_checks = 0
    total_groups = len(cfg['circuits']) * len(seeds) * len(cfg['normalizations']) * len(cfg['models']) * len(cfg['reward_delays'])
    completed = 0
    with threadpool_limits(limits=1):
        for directory in cfg['circuits']:
            raw, ids, provenance = load_connectome(directory)
            roles, _ = load_roles(directory, ids)
            circuit = Path(directory).name
            sources.append(dict(circuit=circuit, **provenance))
            mbon = np.flatnonzero(roles == 'MBON')
            for seed in seeds:
                encoder = KCEncoder(roles, seed, cfg['input_fraction'], cfg['input_amplitude'])
                graphs = {'fly': raw}
                if 'role_shuffled' in cfg['models']:
                    graphs['role_shuffled'], _ = role_shuffled(raw, roles, seed)
                for norm in cfg['normalizations']:
                    for model in cfg['models']:
                        initial = normalize_condition(graphs[model], norm, cfg['gain']); initial.sort_indices()
                        base_reservoir = CircuitReservoir(initial, encoder, cfg['leak'], cfg['microsteps'])
                        base_states = base_reservoir.states(digits[:-1])
                        policy = SelectedReadout(mbon)
                        policy.fit(base_states, digits[1:], epochs=cfg['readout_epochs'],
                                   learning_rate=cfg['readout_learning_rate'], l2=cfg['readout_l2'])
                        original_policy_hash = policy_hash(policy)
                        base_views = readout_views(policy, base_states, digits[1:], cfg)
                        assert all(policy_hash(v) == original_policy_hash for v in base_views.values())
                        for delay in cfg['reward_delays']:
                            key = dict(circuit=circuit, seed=seed, normalization=norm, model=model, reward_delay=delay)
                            if stage == 'discovery':
                                alphas = cfg['alphas']
                            else:
                                chosen = [r['alpha'] for r in selection['candidates'] if r['normalization'] == norm
                                          and r['reward_delay'] == delay and r['alpha'] is not None]
                                alphas = sorted(set([0.] + cfg['confirmation_anchors'] + chosen))
                            trace, th, te, ta = endpoint_training(initial, roles, encoder, policy, digits, cfg, seed, delay)
                            rng = np.random.default_rng(np.random.SeedSequence([seed, delay, 9502]))
                            yoked_signal = np.array([rng.permutation(r) for r in te['true_rewards']])
                            yoked, yh, ye, ya = endpoint_training(initial, roles, encoder, policy, digits, cfg, seed, delay, yoked_signal)
                            assert np.array_equal(te['true_rewards'].sum(axis=1), ye['applied_rewards'].sum(axis=1))
                            for direction, endpoint, history, events, audit in [
                                ('reward_trace', trace, th, te, ta), ('yoked_reward', yoked, yh, ye, ya)]:
                                event_index = len(events_list); events_list.append(events)
                                event_keys.append(dict(**key, direction=direction, event_index=event_index))
                                endpoints.append(dict(**key, direction=direction, event_index=event_index, **audit))
                                histories.extend(dict(**key, direction=direction, **h) for h in history)
                                checkpoints[f'endpoint_{event_index}'] = endpoint.data
                                # Keep structure/encoder/warm policy for independent replay.
                                checkpoints[f'initial_{event_index}'] = initial.data
                                checkpoints[f'indices_{event_index}'] = initial.indices
                                checkpoints[f'indptr_{event_index}'] = initial.indptr
                                checkpoints[f'encoder_{event_index}'] = encoder.patterns
                                checkpoints[f'policy_weights_{event_index}'] = policy.model.weights
                                checkpoints[f'policy_mean_{event_index}'] = policy.model.mean
                                checkpoints[f'policy_scale_{event_index}'] = policy.model.scale
                                for alpha in alphas:
                                    weights, interpolation_audit = interpolate(initial, endpoint, alpha, roles)
                                    frozen_weight_hash = weight_hash(weights)
                                    reservoir = CircuitReservoir(weights, encoder, cfg['leak'], cfg['microsteps'])
                                    states = reservoir.states(digits[:-1])
                                    views = base_views if alpha == 0 else readout_views(policy, states, digits[1:], cfg)
                                    for view in cfg['views']:
                                        readout = views[view]; before = policy_hash(readout)
                                        recall = evaluate_recall(reservoir, readout, digits, cfg['prompt_length'], len(digits) - cfg['prompt_length'])
                                        assert frozen_weight_hash == weight_hash(reservoir.weights) and before == policy_hash(readout)
                                        record = dict(**key, direction=direction, alpha=float(alpha), view=view, event_index=event_index,
                                                      pi_memory_score=recall['pi_memory_score'],
                                                      train_accuracy=float(np.mean(readout.predict(states) == digits[1:])),
                                                      weight_sha256=frozen_weight_hash, readout_sha256=before, **interpolation_audit)
                                        rows.append(record)
                                        recalls.append(dict(**key, direction=direction, alpha=float(alpha), view=view, **recall))
                                        if legacy is not None and alpha in (0, 1) and view in ('fixed', 'full_refit'):
                                            old = legacy
                                            for k, v in key.items(): old = old[old[k] == v]
                                            old = old[(old.condition == ('frozen' if alpha == 0 else direction)) &
                                                      (old.readout == ('frozen_policy' if view == 'fixed' else 'refit_readout'))]
                                            assert len(old) == 1
                                            old = old.iloc[0]
                                            assert int(old.pi_memory_score) == recall['pi_memory_score']
                                            assert old.final_weight_sha256 == frozen_weight_hash
                                            assert np.isclose(old.train_accuracy, record['train_accuracy'], atol=1e-14, rtol=0)
                                            legacy_checks += 1
                            assert policy_hash(policy) == original_policy_hash
                            completed += 1
                            print(f'{stage}: {completed}/{total_groups} groups, {len(rows)} evaluations', flush=True)
                            pd.DataFrame(rows).to_csv(out / 'results.csv', index=False)
    pd.DataFrame(endpoints).to_csv(out / 'endpoints.csv', index=False)
    pd.DataFrame(histories).to_csv(out / 'reward_training.csv', index=False)
    (out / 'recalls.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in recalls))
    (out / 'event_keys.json').write_text(json.dumps(event_keys, indent=2) + '\n')
    np.savez_compressed(out / 'endpoints.npz', **checkpoints)
    np.savez_compressed(out / 'reward_events.npz', **{k: np.array([e[k] for e in events_list]) for k in ('actions', 'true_rewards', 'applied_rewards')})
    frame = pd.DataFrame(rows)
    paired_table(frame).to_csv(out / 'paired.csv', index=False)
    if stage == 'discovery':
        (out / 'selection.json').write_text(json.dumps(dict(config_sha256=sha256(config_path),
            discovery_results_sha256=sha256(out / 'results.csv'), rule=cfg['selection_rule'], candidates=select_candidates(frame)), indent=2) + '\n')
    manifest = dict(stage=stage, source_commit=cfg['source_commit'], sources=sources, evaluations=len(rows),
                    endpoint_training_runs=len(endpoints), legacy_endpoint_checks=legacy_checks,
                    python=platform.python_version(), packages={n: version(n) for n in ('numpy','scipy','pandas','mpmath','threadpoolctl')},
                    config_sha256=sha256(config_path), selection_sha256=sha256(selection_path) if selection_path else None,
                    code_sha256={str(p): sha256(p) for p in sorted(Path('src/flying').rglob('*.py'))},
                    file_sha256={p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file()})
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return frame
