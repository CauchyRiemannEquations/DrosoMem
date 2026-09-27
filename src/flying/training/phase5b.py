"""Source-constrained gamma1/pedc LTD assay and separate pi transfer diagnostic."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits

from flying.brain.dopamine import DopamineLIF, replay_depression
from flying.brain.lif import LIFReservoir
from flying.brain.mushroom_body import KCEncoder
from flying.brain.plasticity import weight_hash
from flying.data.compartments import resolve_compartments
from flying.data.connectome import load_connectome, sha256
from flying.data.pi_digits import pi_digits, decimal_pi_digits
from flying.evaluation.free_recall import evaluate_recall
from flying.training.stage_bc import context as lif_context, fit
from flying.training.phase5_readout import head_arrays
from flying.training.phase5_diagnostic import save_archive
from flying.training.phase5_prefix_confirmation import atomic_json, fingerprint, load_chunk, decode_chunk, first_error


def context(cfg):
    ctx = lif_context(cfg)
    ctx['registry_sha256'] = sha256(cfg['registry'])
    return ctx


def circuit_setup(directory, cfg):
    weights, ids, _ = load_connectome(directory)
    weights.sort_indices()
    roles, mapping, audit = resolve_compartments(directory, ids, cfg['registry'])
    encoder = KCEncoder(roles, cfg['seed'], cfg['input_fraction'], .5)
    return weights, roles, mapping, audit, encoder


def probe(weights, roles, encoder, cfg):
    r = LIFReservoir(weights, encoder, roles)
    return np.asarray([r.states([d]*cfg['probe_digits']).sum(axis=0)
                       for d in [cfg['conditioned_digit'], cfg['control_digit']]])


def baseline(weights, roles, encoder, cfg):
    digits = pi_digits(cfg['pi_length'])
    r = LIFReservoir(weights, encoder, roles)
    states = r.states(digits[:-1])
    head, _ = fit(states, digits[1:], np.flatnonzero(roles == 'MBON'), cfg['seed'], cfg)
    return head, probe(weights, roles, encoder, cfg)


def condition(cfg, key, setup, base_head, base_probe, signature):
    weights, roles, mapping, audit, encoder = setup
    arm = key['arm']
    rate = 0. if arm == 'blocked' else cfg['learning_rate']
    r = DopamineLIF(weights, encoder, roles, mapping, eligibility_ms=cfg['eligibility_ms'],
                    learning_rate=rate, floor_fraction=cfg['floor_fraction'])
    params = dict(digit=cfg['conditioned_digit'], cs_start_ms=0., cs_duration_ms=cfg['cs_ms'],
                  dan_times_ms=cfg['forward_dan_ms'], duration_ms=cfg['conditioning_ms'])
    if arm == 'no_dopamine': params['dan_times_ms'] = []
    if arm == 'backward':
        params.update(cs_start_ms=cfg['backward_cs_start_ms'], dan_times_ms=cfg['backward_dan_ms'])
    if arm == 'dopamine_only': params['digit'] = None
    if arm == 'wrong_compartment': params['dan_compartment'] = 'alpha3'
    adapted, spikes, gates = r.condition(**params)
    rule_parameters = dict(dt_ms=r.parameters.dt_ms, eligibility_ms=cfg['eligibility_ms'],
                           learning_rate=rate, floor_fraction=cfg['floor_fraction'])
    closed = replay_depression(weights, roles, mapping['gamma1_pedc']['MBON'],
                               mapping['gamma1_pedc']['DAN'], spikes, **rule_parameters)
    maximum_error = float(np.max(np.abs(closed.data-adapted.data)))
    assert maximum_error <= cfg['weight_atol']
    mask = r.rule.mask.copy()
    assert np.array_equal(weights.indices, adapted.indices) and np.array_equal(weights.indptr, adapted.indptr)
    assert np.array_equal(weights.data[~mask], adapted.data[~mask])
    assert np.all(adapted.data[mask] <= weights.data[mask])
    assert np.all(adapted.data[mask] >= weights.data[mask]*cfg['floor_fraction'])
    post_probe = probe(adapted, roles, encoder, cfg)
    post = np.repeat(np.arange(len(roles)), np.diff(weights.indptr))
    pre = weights.indices
    active = mask & encoder.patterns[cfg['conditioned_digit']].astype(bool)[pre]
    inactive = mask & ~encoder.patterns[cfg['conditioned_digit']].astype(bool)[pre]
    depression = 1.-adapted.data[mask]/weights.data[mask]
    mbon = mapping['gamma1_pedc']['MBON']
    base_counts = base_probe[:, mbon].sum(axis=1)
    post_counts = post_probe[:, mbon].sum(axis=1)
    # Ratios are unavailable when the baseline has no spikes; never replace the
    # denominator with an epsilon and present a silent neuron as learned behavior.
    suppression = [None if a == 0 else float(1-z/a) for a, z in zip(base_counts, post_counts)]
    mapped_dan = mapping['gamma1_pedc']['DAN']; wrong_dan = mapping['alpha3']['DAN']
    metrics = dict(**key, plastic_edges=int(mask.sum()), changed_edges=int(np.count_nonzero(depression)),
                   outside_compartment_changed=0, mean_plastic_depression=float(depression.mean()),
                   active_edge_depression=float(np.mean(1-adapted.data[active]/weights.data[active])) if active.any() else None,
                   inactive_edge_depression=float(np.mean(1-adapted.data[inactive]/weights.data[inactive])) if inactive.any() else None,
                   independent_weight_max_error=maximum_error,
                   matching_dan_spikes=int(np.isin(spikes['spike_indices'], mapped_dan).sum()),
                   other_dan_spikes=int(np.isin(spikes['spike_indices'], wrong_dan).sum()),
                   cs_plus_before=int(base_counts[0]), cs_plus_after=int(post_counts[0]),
                   cs_minus_before=int(base_counts[1]), cs_minus_after=int(post_counts[1]),
                   cs_plus_suppression=suppression[0], cs_minus_suppression=suppression[1])
    arrays = dict(adapted_data=adapted.data.copy(), indices=adapted.indices.copy(), indptr=adapted.indptr.copy(),
                  plastic_mask=mask, probe_before=base_probe, probe_after=post_probe, **spikes)
    rows, recalls = [], []
    if arm in cfg['pi_arms']:
        frozen = LIFReservoir(adapted, encoder, roles)
        digits = pi_digits(cfg['pi_length'])
        states = frozen.states(digits[:-1]); arrays['pi_states'] = states
        arrays.update({f'pi_train_{k}': v for k, v in frozen.spike_arrays().items()})
        refit, _ = fit(states, digits[1:], np.flatnonzero(roles == 'MBON'), cfg['seed'], cfg)
        for view, head in [('frozen_head', base_head), ('refit_head', refit)]:
            digest = head.digest()
            teacher = head.predict(states)
            recall = evaluate_recall(frozen, head, digits, 3, len(digits)-3)
            assert recall['pi_memory_score'] == first_error(digits[3:], teacher[2:])
            assert head.digest() == digest and weight_hash(frozen.weights) == weight_hash(adapted)
            rows.append(dict(**key, view=view, pi_memory_score=recall['pi_memory_score'],
                             train_accuracy=float(np.mean(teacher == digits[1:])),
                             readout_sha256=digest, weight_sha256=weight_hash(adapted)))
            recalls.append(dict(view=view, **recall))
            arrays.update({f'{view}_{k}': v for k, v in head_arrays(head).items()})
            arrays[f'{view}_teacher'] = teacher
            arrays.update({f'{view}_{k}': v for k, v in frozen.spike_arrays().items()})
        del frozen
    payload = dict(key=key, fingerprint=signature, metrics=metrics, rows=rows, recalls=recalls,
                   mapping=audit, gates=gates, conditioning_schedule=params,
                   baseline_weight_sha256=weight_hash(weights), adapted_weight_sha256=weight_hash(adapted))
    arrays['payload'] = np.array(json.dumps(payload, allow_nan=False))
    return arrays


def summary(payloads, cfg):
    metrics = [p['metrics'] for p in payloads]
    criteria = {}
    for circuit in sorted(set(m['circuit'] for m in metrics)):
        arms = {m['arm']: m for m in metrics if m['circuit'] == circuit}
        paired = arms['paired']
        nulls = all(arms[a]['changed_edges'] == 0 for a in cfg['arms'] if a != 'paired')
        causal = (paired['matching_dan_spikes'] > 0 and paired['active_edge_depression'] is not None
                  and paired['active_edge_depression'] >= cfg['minimum_active_depression']
                  and paired['inactive_edge_depression'] == 0 and nulls)
        behavior = (paired['cs_plus_before'] > 0 and paired['cs_minus_before'] > 0
                    and paired['cs_plus_suppression'] > 0
                    and paired['cs_plus_suppression'] > paired['cs_minus_suppression'])
        criteria[circuit] = dict(causal_locality_and_nulls=bool(causal),
                                 selective_mbon_suppression=bool(behavior))
    return dict(criteria=criteria, all_rule_checks_passed=all(c['causal_locality_and_nulls'] for c in criteria.values()),
                all_response_checks_passed=all(c['selective_mbon_suppression'] for c in criteria.values()),
                scope='gamma1_pedc qualitative LTD approximation; alpha3 mapping is a control; one seed, two overlapping circuits')


def run(config_path, output, resume=False, max_conditions=None):
    cfg = json.loads(Path(config_path).read_text())
    if cfg['experiment'] != 'gamma1_pedc_causal_dopamine' or cfg['pi_length'] != 200:
        raise ValueError('Unexpected Phase 5B design')
    expected_arms = ['no_dopamine', 'paired', 'backward', 'dopamine_only', 'wrong_compartment', 'blocked']
    if cfg['arms'] != expected_arms or cfg['pi_arms'] != ['no_dopamine', 'paired']:
        raise ValueError('Missing locked controls')
    if max_conditions is not None and max_conditions < 1:
        raise ValueError('Invalid condition budget')
    ctx = context(cfg); signature = fingerprint(ctx); out = Path(output)
    assert np.array_equal(pi_digits(200), decimal_pi_digits(200))
    if resume:
        ledger = json.loads((out/'progress.json').read_text())
        if ledger['context'] != ctx or ledger['fingerprint'] != signature:
            raise ValueError('Resume code/data/protocol/registry/runtime changed')
    else:
        out.mkdir(parents=True, exist_ok=False); (out/'checkpoints').mkdir()
        ledger = dict(context=ctx, fingerprint=signature, completed={},
                      source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
        atomic_json(out/'config.json', cfg); atomic_json(out/'progress.json', ledger)
    payloads = []; trained = 0; started = time.monotonic()
    expected = len(cfg['circuits'])*len(cfg['arms'])
    with threadpool_limits(limits=1):
        for c, directory in enumerate(cfg['circuits']):
            setup = circuit_setup(directory, cfg)
            print(f'Baseline {c+1}/{len(cfg["circuits"])}', flush=True)
            base_head, base_probe = baseline(setup[0], setup[1], setup[4], cfg)
            for a, arm in enumerate(cfg['arms']):
                number = c*len(cfg['arms'])+a
                key = dict(circuit=Path(directory).name, arm=arm, seed=cfg['seed'])
                name = f'checkpoints/condition_{number:03d}.npz'
                if name in ledger['completed']:
                    arrays = load_chunk(out/name, ledger['completed'][name])
                else:
                    print(f'{number+1}/{expected}: {key}', flush=True)
                    arrays = condition(cfg, key, setup, base_head, base_probe, signature)
                    save_archive(out/name, arrays); ledger['completed'][name] = sha256(out/name)
                    atomic_json(out/'progress.json', ledger); trained += 1
                payload = decode_chunk(arrays, signature, key); payloads.append(payload)
                print(f"  changed={payload['metrics']['changed_edges']}; CS+ {payload['metrics']['cs_plus_before']}->{payload['metrics']['cs_plus_after']}; {time.monotonic()-started:.1f}s", flush=True)
                if max_conditions is not None and trained >= max_conditions and number+1 < expected:
                    return
    assert len(ledger['completed']) == len(payloads) == expected
    pd.DataFrame([p['metrics'] for p in payloads]).to_csv(out/'conditioning.csv', index=False)
    pd.DataFrame([r for p in payloads for r in p['rows']]).to_csv(out/'pi_transfer.csv', index=False)
    atomic_json(out/'mapping.json', {p['key']['circuit']: p['mapping'] for p in payloads})
    atomic_json(out/'summary.json', summary(payloads, cfg))
    files = {p.relative_to(out).as_posix(): sha256(p) for p in sorted(out.rglob('*'))
             if p.is_file() and p.suffix != '.part' and p.name not in ['manifest.json', 'verification.json', 'pytest.txt']}
    atomic_json(out/'manifest.json', dict(context=ctx, fingerprint=signature, source_commit=ledger['source_commit'],
                conditioning_cases=expected, pi_evaluations=sum(len(p['rows']) for p in payloads),
                file_sha256=files, elapsed_this_invocation=time.monotonic()-started))


def verify(output):
    out = Path(output); manifest = json.loads((out/'manifest.json').read_text())
    cfg = manifest['context']['config']
    if context(cfg) != manifest['context']:
        raise ValueError('Verification code/data/protocol/registry/runtime changed')
    for name, digest in manifest['file_sha256'].items():
        if sha256(out/name) != digest:
            raise ValueError(f'Artifact checksum mismatch: {name}')
    checked = recalls = 0
    with threadpool_limits(limits=1):
        for c, directory in enumerate(cfg['circuits']):
            setup = circuit_setup(directory, cfg)
            base_head, base_probe = baseline(setup[0], setup[1], setup[4], cfg)
            for a, arm in enumerate(cfg['arms']):
                number = c*len(cfg['arms'])+a
                key = dict(circuit=Path(directory).name, arm=arm, seed=cfg['seed'])
                name = f'checkpoints/condition_{number:03d}.npz'
                saved = load_chunk(out/name, manifest['file_sha256'][name])
                payload = decode_chunk(saved, manifest['fingerprint'], key)
                rebuilt = condition(cfg, key, setup, base_head, base_probe, manifest['fingerprint'])
                assert saved.keys() == rebuilt.keys()
                for field in saved:
                    np.testing.assert_array_equal(saved[field], rebuilt[field], err_msg=f'{name}: {field}')
                checked += 1; recalls += len(payload['rows'])
                print(f'{checked}/{manifest["conditioning_cases"]}: conditioning, weights, probes, refits and full recalls exact', flush=True)
    assert checked == manifest['conditioning_cases'] and recalls == manifest['pi_evaluations']
    atomic_json(out/'verification.json', dict(manifest_sha256=sha256(out/'manifest.json'),
                exact_conditioning_rebuilds=checked, exact_pi_rollouts_and_head_arrays=recalls,
                fresh_readout_fits_per_rebuild=6, independent_closed_form_ltd_checks=checked,
                all_saved_arrays_and_payloads_exact=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/phase5b.json')
    parser.add_argument('--output', default='outputs/phase5b')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--max-conditions', type=int)
    args = parser.parse_args()
    verify(args.output) if args.verify else run(args.config, args.output, args.resume, args.max_conditions)
