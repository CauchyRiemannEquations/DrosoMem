"""Prespecified weighted-prefix diagnostic on frozen connectome states."""
import hashlib
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
from flying.training.phase5_readout import (
    KEYS, conditions, nonlinear_seed, head_arrays, NonlinearReadout,
    weight_hash, sha256, pi_digits, evaluate_recall, pi_memory_score, save_archive,
)


def sample_weights(count, prompt_length, window, multiplier):
    start = prompt_length - 1
    if not 0 <= start < count or window < 1 or start + window > count:
        raise ValueError('Invalid scored prefix window')
    if not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError('Invalid prefix multiplier')
    weights = np.ones(count)
    weights[start:start + window] = multiplier
    return weights


def metrics(head, states, labels, cfg):
    prediction = head.predict(states)
    raw = head.logits(states)
    correct = prediction == labels
    start = cfg['prompt_length'] - 1
    end = start + cfg['prefix_window']
    return dict(train_accuracy=float(correct.mean()),
                early_accuracy=float(correct[start:end].mean()),
                later_accuracy=float(correct[end:].mean()),
                cross_entropy=float(np.mean(logsumexp(raw, axis=1) - raw[np.arange(len(labels)), labels])))


def fit_head(states, labels, mbon, key, initialization, treatment, cfg):
    head = NonlinearReadout(mbon, cfg['hidden_units'], nonlinear_seed(key['seed'], initialization))
    weights = sample_weights(len(labels), cfg['prompt_length'], cfg['prefix_window'],
                             cfg['prefix_weight'] if treatment == 'weighted' else 1.)
    _, history = head.fit(states, labels, epochs=cfg['epochs'], learning_rate=cfg['learning_rate'],
                          l2=cfg['l2'], checkpoints=(cfg['epochs'],), sample_weight=weights)
    return head, history


def summarize(frame):
    grouped = frame.groupby(KEYS + ['treatment'])[['pi_memory_score', 'train_accuracy', 'early_accuracy', 'later_accuracy', 'cross_entropy']].mean()
    result = {}
    for model in ['fly', 'role_shuffled']:
        subset = frame[frame.model == model]
        pairs = grouped.xs(model, level='model')['pi_memory_score'].unstack('treatment')
        delta = pairs.weighted - pairs.uniform
        individual = subset.pivot(index=KEYS + ['initialization'], columns='treatment', values='pi_memory_score')
        robust = (individual.weighted > individual.uniform).groupby(level=KEYS).all()
        result[model] = dict(conditions=len(pairs), mean_metrics=subset.groupby('treatment')[['pi_memory_score', 'train_accuracy', 'early_accuracy', 'later_accuracy', 'cross_entropy']].mean().to_dict('index'),
                             paired_mean_delta=float(delta.mean()), wins=int((delta > 0).sum()),
                             ties=int((delta == 0).sum()), losses=int((delta < 0).sum()),
                             all_initializations_improve=int(robust.sum()))
    return result


def run(config_path, output):
    cfg = json.loads(Path(config_path).read_text())
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    (out/'config.json').write_text(json.dumps(cfg, indent=2)+'\n')
    source_hashes = {str(p): sha256(p) for p in sorted(Path('src/flying').rglob('*.py'))}
    digits = pi_digits(cfg['pi_length']); labels = digits[1:]
    rows, recalls, histories, archive = [], [], [], {}
    started = time.monotonic()
    expected = int(np.prod([len(cfg[k]) for k in ['circuits', 'seeds', 'normalizations', 'models', 'schedules']]))
    with threadpool_limits(limits=1):
        for number, (key, reservoir, mbon) in enumerate(conditions(cfg), 1):
            states = reservoir.states(digits[:-1]); before = weight_hash(reservoir.weights)
            state_hash = hashlib.sha256(states.tobytes()).hexdigest()
            for initialization in cfg['initializations']:
                for treatment in ['uniform', 'weighted']:
                    head, history = fit_head(states, labels, mbon, key, initialization, treatment, cfg)
                    assert head.parameter_count == 482
                    prediction = head.predict(states); digest = head.digest()
                    recall = evaluate_recall(reservoir, head, digits, cfg['prompt_length'], len(digits)-cfg['prompt_length'])
                    teacher = pi_memory_score(digits[cfg['prompt_length']:], prediction[cfg['prompt_length']-1:])
                    assert recall['pi_memory_score'] == teacher
                    assert head.digest() == digest and weight_hash(reservoir.weights) == before
                    ix = len(rows)
                    rows.append(dict(**key, evaluation_index=ix, initialization=initialization, treatment=treatment,
                                     pi_memory_score=teacher, **metrics(head, states, labels, cfg),
                                     weight_sha256=before, state_sha256=state_hash, readout_sha256=digest))
                    recalls.append(dict(evaluation_index=ix, **recall))
                    histories.extend(dict(evaluation_index=ix, **h) for h in history)
                    for name, array in head_arrays(head).items(): archive[f'{name}_{ix}'] = array
                    archive[f'teacher_prediction_{ix}'] = prediction.astype(np.uint8)
            print(f'{number}/{expected} states; {len(rows)} fits; {time.monotonic()-started:.1f}s', flush=True)
    assert len(rows) == expected * len(cfg['initializations']) * 2
    frame = pd.DataFrame(rows); frame.to_csv(out/'evaluations.csv', index=False)
    pd.DataFrame(histories).to_csv(out/'training.csv', index=False)
    (out/'recalls.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in recalls))
    (out/'summary.json').write_text(json.dumps(summarize(frame), indent=2)+'\n')
    save_archive(out/'readouts.npz', archive)
    manifest = dict(base_commit=cfg['source_commit'], code_sha256=source_hashes,
                    python=platform.python_version(), packages={n: version(n) for n in ['numpy','scipy','pandas','threadpoolctl','mpmath','pytest']},
                    conditions=expected, evaluations=len(rows), elapsed_seconds=time.monotonic()-started,
                    data_sha256={str(p):sha256(p) for d in cfg['circuits'] for p in sorted(Path(d).glob('*')) if p.is_file()},
                    protocol_sha256=sha256('docs/phase5-prefix-protocol.md'),
                    file_sha256={p.name:sha256(p) for p in sorted(out.iterdir()) if p.is_file()})
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


def verify(directory):
    out = Path(directory); cfg = json.loads((out/'config.json').read_text())
    manifest = json.loads((out/'manifest.json').read_text())
    for name, digest in manifest['file_sha256'].items(): assert sha256(out/name) == digest, name
    for name, digest in {**manifest['code_sha256'], **manifest['data_sha256']}.items(): assert sha256(name) == digest, name
    assert sha256('docs/phase5-prefix-protocol.md') == manifest['protocol_sha256']
    frame = pd.read_csv(out/'evaluations.csv'); saved = np.load(out/'readouts.npz')
    recalls = {r['evaluation_index']:r for r in map(json.loads, (out/'recalls.jsonl').read_text().splitlines())}
    assert not frame.duplicated(KEYS + ['initialization','treatment']).any()
    assert len(frame) == len(recalls) == manifest['evaluations']
    digits = pi_digits(cfg['pi_length']); labels = digits[1:]; replayed = refitted = count = 0
    with threadpool_limits(limits=1):
        for key, reservoir, mbon in conditions(cfg):
            count += 1; states = reservoir.states(digits[:-1]); subset = frame
            for k, v in key.items(): subset = subset[subset[k] == v]
            assert len(subset) == len(cfg['initializations']) * 2
            for row in subset.to_dict('records'):
                ix = row['evaluation_index']; head = NonlinearReadout(mbon, cfg['hidden_units'])
                head.mean = saved[f'mean_{ix}']; head.scale = saved[f'scale_{ix}']
                head.parameters = {n:saved[f'{n}_{ix}'] for n in ['w1','b1','w2','b2']}
                assert head.parameter_count == 482
                assert np.array_equal(head.mean, states[:,mbon].mean(axis=0))
                assert np.array_equal(head.scale, np.maximum(states[:,mbon].std(axis=0),1e-5))
                assert hashlib.sha256(states.tobytes()).hexdigest() == row['state_sha256']
                assert head.digest() == row['readout_sha256'] and weight_hash(reservoir.weights) == row['weight_sha256']
                prediction = head.predict(states)
                assert np.array_equal(prediction, saved[f'teacher_prediction_{ix}'])
                for k, v in metrics(head, states, labels, cfg).items(): assert np.isclose(v,row[k],atol=1e-12,rtol=0)
                recall = evaluate_recall(reservoir, head, digits, cfg['prompt_length'], len(digits)-cfg['prompt_length'])
                for k,v in recall.items(): assert recalls[ix][k] == v, (ix,k)
                target = digits[cfg['prompt_length']:]; pred = prediction[cfg['prompt_length']-1:]
                prefix = next((j for j,(a,b) in enumerate(zip(target,pred)) if a != b),len(target))
                assert prefix == recall['pi_memory_score'] == row['pi_memory_score']
                assert head.digest() == row['readout_sha256'] and weight_hash(reservoir.weights) == row['weight_sha256']
                if key['circuit'] == Path(cfg['circuits'][0]).name and key['seed'] == cfg['seeds'][0] and key['model'] == 'fly':
                    fitted,_ = fit_head(states, labels, mbon, key, row['initialization'], row['treatment'], cfg)
                    assert fitted.digest() == head.digest(); refitted += 1
                replayed += 1
    assert count == manifest['conditions'] and replayed == manifest['evaluations'] and refitted == 6
    result = dict(states_rebuilt=count, heads_replayed=replayed, independently_refitted=refitted,
                  exact_recall_strings=True, first_error_equivalence=True, frozen_weights_verified=True)
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/phase5_prefix.json')
    parser.add_argument('--output', default='outputs/phase5_prefix')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    verify(args.output) if args.verify else run(args.config, args.output)
