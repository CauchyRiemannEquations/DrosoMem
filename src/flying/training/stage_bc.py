"""Locked small-circuit LIF validation and descriptive pi feasibility study."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import time
from importlib.metadata import version

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from flying.brain.lif import LIFParameters, LIFReservoir
from flying.brain.lif_reference import reference_trace
from flying.brain.mushroom_body import KCEncoder, role_shuffled
from flying.brain.timed_reservoir import TimedReservoir
from flying.brain.diagnostics import normalize_condition
from flying.brain.plasticity import weight_hash
from flying.data.connectome import load_connectome, sha256
from flying.data.mushroom_body import load_roles
from flying.data.pi_digits import pi_digits, decimal_pi_digits
from flying.evaluation.free_recall import evaluate_recall
from flying.models.nonlinear_readout import NonlinearReadout
from flying.training.phase5_readout import head_arrays, nonlinear_seed
from flying.training.phase5_prefix import sample_weights
from flying.training.phase5_diagnostic import save_archive
from flying.training.phase5_prefix_confirmation import (
    atomic_json, context as base_context, fingerprint, load_chunk, decode_chunk, first_error,
)


def context(cfg):
    ctx = base_context(cfg)
    ctx['packages'].update({n: version(n) for n in ['brian2', 'sympy', 'pyparsing']})
    ctx['requirements_lif_sha256'] = sha256('requirements-lif-lock.txt')
    return ctx


def grid(cfg):
    for directory in cfg['circuits']:
        raw, ids, _ = load_connectome(directory)
        roles, _ = load_roles(directory, ids)
        for seed in cfg['seeds']:
            encoder = KCEncoder(roles, seed, cfg['input_fraction'], .5)
            shuffled, mixing = role_shuffled(raw, roles, seed)
            for model in cfg['models']:
                weights = {'fly': raw, 'role_shuffled': shuffled}[model].copy()
                weights.sort_indices()
                yield (dict(circuit=Path(directory).name, seed=seed, model=model),
                       weights, roles, encoder, mixing if model == 'role_shuffled' else {})


def make_rate(weights, roles, encoder):
    normalized = normalize_condition(weights, 'incoming_l1', .9)
    normalized.sort_indices()
    return TimedReservoir(normalized, encoder, roles, .6, 'mbon_after_kc')


def fit(states, labels, indices, seed, cfg):
    head = NonlinearReadout(indices, 8, nonlinear_seed(seed, 0))
    _, history = head.fit(states, labels, epochs=cfg['epochs'], checkpoints=(cfg['epochs'],),
                          learning_rate=.03, l2=1e-5,
                          sample_weight=sample_weights(len(labels), 3, 32, 4.))
    return head, history


def numerical_checks(weights, roles, encoder, digits, cfg):
    p = LIFParameters(**cfg['lif'])
    short = digits[:cfg['reference_digits']]
    r = LIFReservoir(weights, encoder, roles, p, trace=True)
    ids, ticks = r._events(short)
    expected = reference_trace(weights, roles, ids, ticks,
                               round(len(short)*p.digit_ms/p.dt_ms), p)
    r.advance(short)
    observed = r.spike_arrays()
    v_error = float(np.max(np.abs(r.trace.v.T/r.b.mV-expected['v_mv'])))
    g_error = float(np.max(np.abs(r.trace.g.T/r.b.mV-expected['g_mv'])))
    spikes_equal = all(np.array_equal(observed[k], expected[k]) for k in observed)
    reference_pass = spikes_equal and max(v_error, g_error) < cfg['reference_atol_mv']
    del r, expected
    # A separate, prespecified convergence diagnostic; do not adjust dt after
    # seeing its results or mistake exact integration for continuous spike timing.
    features, rates = [], []
    for dt in cfg['convergence_dt_ms']:
        r = LIFReservoir(weights, encoder, roles, replace(p, dt_ms=dt))
        states = r.states(digits[:cfg['convergence_digits']])
        features.append(states[:, roles == 'MBON'])
        rates.append({role: float(states[:, roles == role].mean()/(p.digit_ms/1000))
                      for role in sorted(set(roles))})
        del r
    differences = [float(np.abs(a-z).sum()/max(z.sum(), 1.))
                   for a, z in zip(features[:-1], features[1:])]
    convergence_pass = all(d <= cfg['convergence_count_relative_l1_max'] for d in differences)
    return dict(reference=dict(max_v_error_mv=v_error, max_g_error_mv=g_error,
                               spikes_equal=spikes_equal, passed=reference_pass),
                convergence=dict(dt_ms=cfg['convergence_dt_ms'],
                                 adjacent_count_relative_l1=differences,
                                 mean_role_hz=rates, passed=convergence_pass))


def train_condition(cfg, key, weights, roles, encoder, mixing, signature):
    started = time.monotonic()
    digits = pi_digits(cfg['pi_length']); labels = digits[1:]
    mbon = np.flatnonzero(roles == 'MBON')
    assert len(mbon) == 48
    diagnostics = numerical_checks(weights, roles, encoder, digits, cfg)
    if not diagnostics['reference']['passed']:
        raise RuntimeError('LIF reference mismatch; do not run readout comparisons')
    rows, recalls, histories, arrays = [], [], [], {}
    for kind in ['lif', 'rate']:
        r = (LIFReservoir(weights, encoder, roles, LIFParameters(**cfg['lif'])) if kind == 'lif'
             else make_rate(weights, roles, encoder))
        states = r.states(digits[:-1]); arrays[f'states_{kind}'] = states
        frozen = weight_hash(r.weights)
        if kind == 'lif':
            arrays.update({f'train_{k}': v for k, v in r.spike_arrays().items()})
            diagnostics['training_activity'] = dict(
                mean_role_hz={role: float(states[:, roles == role].mean()/(r.parameters.digit_ms/1000))
                              for role in sorted(set(roles))},
                active_mbon_features=int(np.count_nonzero(states[:, mbon].std(axis=0))),
                distinct_mbon_vectors=int(len(np.unique(states[:, mbon], axis=0))))
        head, history = fit(states, labels, mbon, key['seed'], cfg)
        saved_digest = head.digest()
        teacher = head.predict(states)
        recall = evaluate_recall(r, head, digits, 3, len(digits)-3)
        assert recall['pi_memory_score'] == first_error(digits[3:], teacher[2:])
        assert head.parameter_count == 482 and head.digest() == saved_digest
        assert frozen == weight_hash(r.weights)
        row = dict(**key, dynamics=kind, pi_memory_score=recall['pi_memory_score'],
                   train_accuracy=float((teacher == labels).mean()),
                   early_accuracy=float((teacher[2:34] == labels[2:34]).mean()),
                   later_accuracy=float((teacher[34:] == labels[34:]).mean()),
                   readout_sha256=saved_digest, weight_sha256=frozen, parameter_count=482)
        rows.append(row); recalls.append(dict(dynamics=kind, **recall))
        histories.append(dict(dynamics=kind, history=history))
        arrays.update({f'{kind}_{name}': a for name, a in head_arrays(head).items()})
        arrays[f'teacher_{kind}'] = teacher
        if kind == 'lif':
            arrays.update({f'recall_{k}': v for k, v in r.spike_arrays().items()})
        print(f"  {kind}: recall={recall['pi_memory_score']}, accuracy={row['train_accuracy']:.3f}", flush=True)
        del r
    payload = dict(fingerprint=signature, key=key, rows=rows, recalls=recalls,
                   diagnostics=diagnostics, histories=histories, mixing=mixing,
                   raw_weight_sha256=weight_hash(weights), elapsed_seconds=time.monotonic()-started)
    arrays['payload'] = np.array(json.dumps(payload, allow_nan=False))
    return arrays


def run(config_path, output, resume=False, max_conditions=None):
    cfg = json.loads(Path(config_path).read_text())
    if cfg['experiment'] != 'stage_bc_lif' or cfg['pi_length'] != 200:
        raise ValueError('Expected locked 200-digit Stage B/C configuration')
    if cfg['models'] != ['fly', 'role_shuffled'] or not cfg['seeds'] or len(set(cfg['seeds'])) != len(cfg['seeds']):
        raise ValueError('Invalid graph/seed grid')
    if max_conditions is not None and max_conditions < 1:
        raise ValueError('Invalid condition budget')
    assert np.array_equal(pi_digits(200), decimal_pi_digits(200))
    ctx = context(cfg); signature = fingerprint(ctx); out = Path(output)
    if resume:
        ledger = json.loads((out/'progress.json').read_text())
        if ledger['context'] != ctx or ledger['fingerprint'] != signature:
            raise ValueError('Resume config/code/data/protocol/runtime changed')
    else:
        out.mkdir(parents=True, exist_ok=False); (out/'checkpoints').mkdir()
        ledger = dict(context=ctx, fingerprint=signature, completed={},
                      source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
        atomic_json(out/'config.json', cfg); atomic_json(out/'progress.json', ledger)
    rows, diagnostics = [], []; trained = reused = 0; started = time.monotonic()
    expected = len(cfg['circuits'])*len(cfg['models'])*len(cfg['seeds'])
    with threadpool_limits(limits=1):
        for number, (key, weights, roles, encoder, mixing) in enumerate(grid(cfg)):
            name = f'checkpoints/condition_{number:03d}.npz'; path = out/name
            if name in ledger['completed']:
                arrays = load_chunk(path, ledger['completed'][name]); reused += 1
            else:
                print(f'{number+1}/{expected}: {key}', flush=True)
                arrays = train_condition(cfg, key, weights, roles, encoder, mixing, signature)
                save_archive(path, arrays); ledger['completed'][name] = sha256(path)
                atomic_json(out/'progress.json', ledger); trained += 1
            payload = decode_chunk(arrays, signature, key)
            rows.extend(payload['rows']); diagnostics.append(dict(**key, **payload['diagnostics']))
            print(f'{number+1}/{expected} completed; {time.monotonic()-started:.1f}s', flush=True)
            if max_conditions is not None and trained >= max_conditions and number+1 < expected:
                return
    assert len(ledger['completed']) == expected and len(rows) == expected*2
    pd.DataFrame(rows).to_csv(out/'evaluations.csv', index=False)
    atomic_json(out/'diagnostics.json', diagnostics)
    frame = pd.DataFrame(rows)
    atomic_json(out/'summary.json', dict(
        mean_metrics=frame.groupby(['model', 'dynamics'])[['pi_memory_score', 'train_accuracy']].mean().reset_index().to_dict('records'),
        reference_all_passed=all(d['reference']['passed'] for d in diagnostics),
        convergence_all_passed=all(d['convergence']['passed'] for d in diagnostics),
        scope='four graph conditions; one encoder/head seed; descriptive feasibility only'))
    files = {p.relative_to(out).as_posix(): sha256(p) for p in sorted(out.rglob('*'))
             if p.is_file() and p.suffix != '.part' and p.name not in ['manifest.json', 'verification.json', 'pytest.txt']}
    atomic_json(out/'manifest.json', dict(context=ctx, fingerprint=signature,
                source_commit=ledger['source_commit'], conditions=expected, evaluations=len(rows),
                file_sha256=files, elapsed_this_invocation=time.monotonic()-started))


def verify(output):
    out = Path(output); manifest = json.loads((out/'manifest.json').read_text())
    cfg = manifest['context']['config']
    if context(cfg) != manifest['context']:
        raise ValueError('Verification runtime/code/data/protocol changed')
    for name, digest in manifest['file_sha256'].items():
        if sha256(out/name) != digest:
            raise ValueError(f'Artifact checksum mismatch: {name}')
    digits = pi_digits(cfg['pi_length']); checked = refits = 0
    assert np.array_equal(digits, decimal_pi_digits(cfg['pi_length']))
    with threadpool_limits(limits=1):
        for number, (key, weights, roles, encoder, _) in enumerate(grid(cfg)):
            name = f'checkpoints/condition_{number:03d}.npz'
            arrays = load_chunk(out/name, manifest['file_sha256'][name])
            payload = decode_chunk(arrays, manifest['fingerprint'], key)
            assert weight_hash(weights) == payload['raw_weight_sha256']
            mbon = np.flatnonzero(roles == 'MBON')
            for kind in ['lif', 'rate']:
                r = (LIFReservoir(weights, encoder, roles, LIFParameters(**cfg['lif'])) if kind == 'lif'
                     else make_rate(weights, roles, encoder))
                states = r.states(digits[:-1])
                np.testing.assert_array_equal(states, arrays[f'states_{kind}'])
                if kind == 'lif':
                    for k, value in r.spike_arrays().items():
                        np.testing.assert_array_equal(value, arrays[f'train_{k}'])
                head = NonlinearReadout(mbon, 8, nonlinear_seed(key['seed'], 0))
                head.mean = arrays[f'{kind}_mean']; head.scale = arrays[f'{kind}_scale']
                head.parameters = {k: arrays[f'{kind}_{k}'] for k in ['w1', 'b1', 'w2', 'b2']}
                row = next(row for row in payload['rows'] if row['dynamics'] == kind)
                assert head.digest() == row['readout_sha256'] and weight_hash(r.weights) == row['weight_sha256']
                np.testing.assert_array_equal(head.predict(states), arrays[f'teacher_{kind}'])
                recall = evaluate_recall(r, head, digits, 3, len(digits)-3)
                assert dict(dynamics=kind, **recall) == next(c for c in payload['recalls'] if c['dynamics'] == kind)
                if kind == 'lif':
                    for k, value in r.spike_arrays().items():
                        np.testing.assert_array_equal(value, arrays[f'recall_{k}'])
                refit, _ = fit(states, digits[1:], mbon, key['seed'], cfg)
                assert refit.digest() == head.digest()
                checked += 1; refits += 1
                del r
            print(f'{number+1}/{manifest["conditions"]}: states, spikes, recalls and refits exact', flush=True)
    assert checked == manifest['evaluations']
    atomic_json(out/'verification.json', dict(manifest_sha256=sha256(out/'manifest.json'),
                exact_saved_head_replays=checked, exact_refits=refits,
                exact_training_state_rebuilds=checked, full_rollout_digits=197,
                lif_training_and_recall_spikes_exact=True, pi_generators_equal=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/stage_bc.json')
    parser.add_argument('--output', default='outputs/stage_bc')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--max-conditions', type=int)
    args = parser.parse_args()
    verify(args.output) if args.verify else run(args.config, args.output, args.resume, args.max_conditions)
