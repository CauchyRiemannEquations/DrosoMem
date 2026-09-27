"""Budget-matched cumulative-prefix training with auditable stage checkpoints."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from flying.training import phase5_prefix_confirmation as checkpoint
from flying.training.phase5_prefix import (
    KEYS, NonlinearReadout, evaluate_recall, head_arrays, metrics,
    sample_weights, save_archive, sha256, weight_hash,
)
from flying.training.phase5_readout import nonlinear_seed

ARMS = ('uniform', 'fixed', 'curriculum')
MEASURES = ['pi_memory_score', 'train_accuracy', 'early_accuracy', 'later_accuracy',
            'cross_entropy', 'accuracy_33_64', 'accuracy_65_128', 'accuracy_129_197']


def validate(cfg):
    for name in ['circuits', 'seeds', 'models', 'normalizations', 'schedules', 'offsets', 'initializations']:
        values = cfg[name]
        if not values or len(set(values)) != len(values):
            raise ValueError(f'Empty or duplicate {name}')
    if any(type(v) is not int or v < 0 for v in cfg['offsets']):
        raise ValueError('Invalid offsets')
    if cfg['pi_length'] != 200 or cfg['prompt_length'] != 3 or cfg['prefix_window'] != 32:
        raise ValueError('This protocol requires 200 digits, prompt 3 and fixed window 32')
    if cfg['windows'] != [32, 64, 128] or cfg['prefix_weight'] != 4 or cfg['hidden_units'] != 8:
        raise ValueError('Invalid curriculum protocol')
    if type(cfg['stage_epochs']) is not int or cfg['stage_epochs'] < 1:
        raise ValueError('Invalid stage epochs')
    if set(cfg['models']) != {'fly', 'role_shuffled'}:
        raise ValueError('Both topology controls are required')


def endpoints(cfg):
    return [cfg['stage_epochs'] * i for i in (1, 2, 3)]


def fit_head(states, labels, mbon, key, initialization, arm, cfg):
    if arm not in ARMS:
        raise ValueError('Unknown treatment')
    schedule = []
    for end, window in zip(endpoints(cfg), cfg['windows']):
        weights = sample_weights(len(labels), cfg['prompt_length'],
                                 window if arm == 'curriculum' else cfg['prefix_window'],
                                 1. if arm == 'uniform' else cfg['prefix_weight'])
        schedule.append((end, weights))
    head = NonlinearReadout(mbon, cfg['hidden_units'], nonlinear_seed(key['seed'], initialization))
    return head.fit(states, labels, epochs=endpoints(cfg)[-1], checkpoints=endpoints(cfg),
                    learning_rate=cfg['learning_rate'], l2=cfg['l2'], sample_weight_schedule=schedule)


def measure(head, states, labels, cfg):
    correct = head.predict(states)[2:] == labels[2:]
    return dict(**metrics(head, states, labels, cfg),
                accuracy_33_64=float(correct[32:64].mean()),
                accuracy_65_128=float(correct[64:128].mean()),
                accuracy_129_197=float(correct[128:].mean()))


def train_condition(cfg, key, reservoir, mbon, digits, signature):
    states = reservoir.states(digits[:-1]); labels = digits[1:]
    before = weight_hash(reservoir.weights)
    state_hash = hashlib.sha256(states.tobytes()).hexdigest()
    rows, recalls, histories, arrays = [], [], [], {}
    for initialization in cfg['initializations']:
        for arm in ARMS:
            saved, history = fit_head(states, labels, mbon, key, initialization, arm, cfg)
            histories.append(dict(initialization=initialization, treatment=arm, history=history))
            for epoch, head in saved.items():
                ix = len(rows); digest = head.digest(); pred = head.predict(states)
                recall = evaluate_recall(reservoir, head, digits, cfg['prompt_length'], 197)
                score = checkpoint.first_error(digits[3:], pred[2:])
                assert score == recall['pi_memory_score']
                assert head.digest() == digest and weight_hash(reservoir.weights) == before
                assert head.parameter_count == 482
                rows.append(dict(**key, head_index=ix, initialization=initialization, treatment=arm,
                                 epoch=epoch, pi_memory_score=score, **measure(head, states, labels, cfg),
                                 weight_sha256=before, state_sha256=state_hash, readout_sha256=digest))
                recalls.append(dict(head_index=ix, **recall))
                arrays.update({f'{name}_{ix}':value for name, value in head_arrays(head).items()})
                arrays[f'teacher_prediction_{ix}'] = pred.astype(np.uint8)
    payload = dict(fingerprint=signature, key=key, rows=rows, recalls=recalls, histories=histories,
                   segment=''.join(map(str, digits.tolist())))
    arrays['payload'] = np.array(json.dumps(payload, allow_nan=False))
    return arrays


def summary(frame, cfg):
    final = frame[frame.epoch == endpoints(cfg)[-1]]
    groups = KEYS + ['initialization']
    by_offset = {}
    for offset, part in final.groupby('offset'):
        models = {}
        for model, subset in part.groupby('model'):
            means = subset.groupby('treatment')[MEASURES].mean().to_dict('index')
            pairs = subset.groupby(KEYS + ['treatment']).pi_memory_score.mean().unstack('treatment')
            comparisons = {}
            for control in ['uniform', 'fixed']:
                delta = pairs.curriculum - pairs[control]
                comparisons[control] = dict(paired_mean_delta=float(delta.mean()), wins=int((delta > 0).sum()),
                                            ties=int((delta == 0).sum()), losses=int((delta < 0).sum()))
            completed = {arm:{str(n):int((rows.pi_memory_score >= n).sum()) for n in [32,64,128,197]}
                         for arm, rows in subset.groupby('treatment')}
            stages = frame[(frame.offset == offset) & (frame.model == model)]
            retention = {}
            for arm, rows in stages.groupby('treatment'):
                wide = rows.pivot(index=groups, columns='epoch', values='pi_memory_score')
                first = wide[endpoints(cfg)[0]] >= 32
                retention[arm] = dict(initial_32_complete=int(first.sum()),
                    lost_at_stage2=int((first & (wide[endpoints(cfg)[1]] < 32)).sum()),
                    lost_at_final=int((first & (wide[endpoints(cfg)[2]] < 32)).sum()))
            models[model] = dict(conditions=len(pairs), mean_metrics=means, comparisons=comparisons,
                                 completion_counts=completed, early_retention=retention)
        by_offset[str(int(offset))] = models
    recall_criteria, joint_criteria = {}, {}
    for offset in [0,1000,2000]:
        if str(offset) not in by_offset:
            continue
        real = by_offset[str(offset)]['fly']
        passed = real['conditions'] == 6 and all(
            c['paired_mean_delta'] > 0 and c['wins'] >= 4 for c in real['comparisons'].values())
        recall_criteria[str(offset)] = bool(passed)
        joint_criteria[str(offset)] = bool(passed and real['mean_metrics']['curriculum']['later_accuracy'] >=
                                          real['mean_metrics']['fixed']['later_accuracy'])
    return dict(by_offset=by_offset, recall_criteria=recall_criteria, joint_criteria=joint_criteria,
                all_offsets_recall_passed=len(recall_criteria)==3 and all(recall_criteria.values()),
                all_offsets_joint_passed=len(joint_criteria)==3 and all(joint_criteria.values()))


def run(config_path, output, resume=False, max_conditions=None):
    cfg = json.loads(Path(config_path).read_text()); validate(cfg)
    if max_conditions is not None and (type(max_conditions) is not int or max_conditions < 1):
        raise ValueError('Invalid condition budget')
    ctx = checkpoint.context(cfg); signature = checkpoint.fingerprint(ctx); out = Path(output)
    if resume:
        ledger = json.loads((out/'progress.json').read_text())
        if ledger['context'] != ctx or ledger['fingerprint'] != signature:
            raise ValueError('Resume config/code/data/protocol/runtime changed')
    else:
        out.mkdir(parents=True, exist_ok=False); (out/'checkpoints').mkdir()
        ledger = dict(context=ctx, fingerprint=signature, completed={})
        checkpoint.atomic_json(out/'config.json', cfg); checkpoint.atomic_json(out/'progress.json', ledger)
    maximum = max(cfg['offsets']) + cfg['pi_length']
    assert np.array_equal(checkpoint.pi_digits(maximum), checkpoint.decimal_pi_digits(maximum))
    expected = int(np.prod([len(cfg[k]) for k in KEYS_TO_GRID]))
    rows, recalls = [], []; started = time.monotonic(); trained = reused = 0
    with threadpool_limits(limits=1):
        for number, (key, reservoir, mbon) in enumerate(checkpoint.grid(cfg)):
            name = f'checkpoints/condition_{number:03d}.npz'; path = out/name
            if name in ledger['completed']:
                arrays = checkpoint.load_chunk(path, ledger['completed'][name]); reused += 1
            else:
                arrays = train_condition(cfg, key, reservoir, mbon,
                                         checkpoint.segment_digits(key['offset'], cfg['pi_length']), signature)
                save_archive(path, arrays); ledger['completed'][name] = sha256(path)
                checkpoint.atomic_json(out/'progress.json', ledger); trained += 1
            payload = checkpoint.decode_chunk(arrays, signature, key)
            rows.extend(dict(checkpoint=name, **r) for r in payload['rows'])
            recalls.extend(dict(checkpoint=name, **key, **r) for r in payload['recalls'])
            print(f'{number+1}/{expected} conditions; {len(rows)} stage evaluations; '
                  f'{trained} trained/{reused} reused; {time.monotonic()-started:.1f}s', flush=True)
            if max_conditions is not None and trained >= max_conditions and number+1 < expected:
                return dict(complete=False, trained=trained, reused=reused)
    assert len(ledger['completed']) == expected and len(rows) == expected * len(cfg['initializations']) * 9
    frame = pd.DataFrame(rows)
    temporary = out/'evaluations.csv.part'; frame.to_csv(temporary, index=False); temporary.replace(out/'evaluations.csv')
    temporary = out/'recalls.jsonl.part'
    temporary.write_text(''.join(json.dumps(r)+'\n' for r in recalls)); temporary.replace(out/'recalls.jsonl')
    checkpoint.atomic_json(out/'summary.json', summary(frame, cfg))
    files = {p.relative_to(out).as_posix():sha256(p) for p in sorted(out.rglob('*'))
             if p.is_file() and p.suffix != '.part' and p.name not in ['manifest.json','verification.json','pytest.txt']}
    checkpoint.atomic_json(out/'manifest.json', dict(context=ctx, fingerprint=signature, conditions=expected,
        fits=expected*len(cfg['initializations'])*3, evaluations=len(rows), pi_generators_equal_digits=maximum,
        trained_this_invocation=trained, reused_this_invocation=reused,
        elapsed_this_invocation=time.monotonic()-started, file_sha256=files))
    return dict(complete=True, trained=trained, reused=reused)


KEYS_TO_GRID = ['circuits','seeds','models','normalizations','schedules','offsets']


def verify(output):
    out = Path(output); manifest = json.loads((out/'manifest.json').read_text()); cfg = manifest['context']['config']
    validate(cfg)
    if checkpoint.context(cfg) != manifest['context']:
        raise ValueError('Verification context changed')
    for name, digest in manifest['file_sha256'].items():
        checkpoint.load_chunk(out/name, digest) if name.endswith('.npz') else verify_hash(out/name, digest)
    maximum = max(cfg['offsets']) + cfg['pi_length']
    assert np.array_equal(checkpoint.pi_digits(maximum), checkpoint.decimal_pi_digits(maximum))
    frame = pd.read_csv(out/'evaluations.csv')
    assert not frame.duplicated(KEYS+['offset','initialization','treatment','epoch']).any()
    recorded = [json.loads(line) for line in (out/'recalls.jsonl').read_text().splitlines()]
    rows, recalls = [], []; refitted = replayed = 0
    with threadpool_limits(limits=1):
        for number, (key, reservoir, mbon) in enumerate(checkpoint.grid(cfg)):
            name = f'checkpoints/condition_{number:03d}.npz'
            arrays = checkpoint.load_chunk(out/name, manifest['file_sha256'][name])
            payload = checkpoint.decode_chunk(arrays, manifest['fingerprint'], key)
            digits = checkpoint.segment_digits(key['offset'], cfg['pi_length']); labels = digits[1:]
            assert payload['segment'] == ''.join(map(str, digits.tolist()))
            states = reservoir.states(digits[:-1]); before = weight_hash(reservoir.weights)
            refit = key['circuit'] == Path(cfg['circuits'][0]).name and key['seed'] == cfg['seeds'][0] and key['model'] == 'fly'
            fitted = {}
            if refit:
                for initialization in cfg['initializations']:
                    for arm in ARMS:
                        fitted[(initialization, arm)] = fit_head(states, labels, mbon, key, initialization, arm, cfg)
                        refitted += 1
            assert len(payload['rows']) == len(payload['recalls']) == len(cfg['initializations'])*9
            expected_ids = {(i,a,e) for i in cfg['initializations'] for a in ARMS for e in endpoints(cfg)}
            assert {(r['initialization'],r['treatment'],r['epoch']) for r in payload['rows']} == expected_ids
            for row, rec in zip(payload['rows'], payload['recalls']):
                ix = row['head_index']; head = NonlinearReadout(mbon, cfg['hidden_units'])
                head.mean = arrays[f'mean_{ix}']; head.scale = arrays[f'scale_{ix}']
                head.parameters = {n:arrays[f'{n}_{ix}'] for n in ['w1','b1','w2','b2']}
                assert head.parameter_count == 482
                assert np.array_equal(head.mean, states[:,mbon].mean(axis=0))
                assert np.array_equal(head.scale, np.maximum(states[:,mbon].std(axis=0),1e-5))
                assert row['state_sha256'] == hashlib.sha256(states.tobytes()).hexdigest()
                assert row['weight_sha256'] == before and row['readout_sha256'] == head.digest()
                pred = head.predict(states); assert np.array_equal(pred, arrays[f'teacher_prediction_{ix}'])
                for k,v in measure(head, states, labels, cfg).items():
                    assert np.isclose(v, row[k], atol=1e-12, rtol=0)
                got = evaluate_recall(reservoir, head, digits, cfg['prompt_length'], 197)
                assert rec == dict(head_index=ix, **got)
                assert checkpoint.first_error(digits[3:], pred[2:]) == got['pi_memory_score'] == row['pi_memory_score']
                assert head.digest() == row['readout_sha256'] and weight_hash(reservoir.weights) == before
                if refit:
                    assert fitted[(row['initialization'], row['treatment'])][0][row['epoch']].digest() == head.digest()
                rows.append(dict(checkpoint=name, **row)); recalls.append(dict(checkpoint=name, **key, **rec)); replayed += 1
            if refit:
                for history in payload['histories']:
                    assert fitted[(history['initialization'], history['treatment'])][1] == history['history']
    pd.testing.assert_frame_equal(frame, pd.DataFrame(rows), check_exact=False, atol=1e-12, rtol=0)
    assert recorded == recalls and summary(pd.DataFrame(rows), cfg) == json.loads((out/'summary.json').read_text())
    assert number+1 == manifest['conditions'] and replayed == manifest['evaluations']
    assert refitted == len(cfg['offsets'])*len(cfg['initializations'])*3
    result = dict(states_rebuilt=number+1, heads_replayed=replayed, independently_refitted=refitted,
                  refitted_stage_heads=refitted*3, exact_recall_strings=True, frozen_weights_verified=True,
                  aggregate_and_per_position_metrics_verified=True, pi_generators_equal_digits=maximum)
    checkpoint.atomic_json(out/'verification.json', result); print(json.dumps(result, indent=2)); return result


def verify_hash(path, digest):
    if not path.exists() or sha256(path) != digest:
        raise ValueError(f'Artifact hash mismatch: {path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/phase5_curriculum.json')
    parser.add_argument('--output', default='outputs/phase5_curriculum')
    parser.add_argument('--resume', action='store_true'); parser.add_argument('--verify', action='store_true')
    parser.add_argument('--max-conditions', type=int)
    args = parser.parse_args()
    verify(args.output) if args.verify else run(args.config, args.output, args.resume, args.max_conditions)
