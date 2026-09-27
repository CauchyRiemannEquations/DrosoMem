"""Cross-segment replication with atomic condition checkpoints and strict resume."""
import argparse
import hashlib
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from flying.data.pi_digits import pi_digits, decimal_pi_digits
from flying.training.phase5_prefix import (
    KEYS, conditions, fit_head, head_arrays, metrics, NonlinearReadout,
    weight_hash, sha256, evaluate_recall, save_archive, summarize,
)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def segment_digits(offset, length):
    if type(offset) is not int or offset < 0 or type(length) is not int or length < 2:
        raise ValueError('Invalid segment')
    return pi_digits(offset + length)[offset:].copy()


def context(cfg):
    return dict(config=cfg, python=platform.python_version(),
                packages={n:version(n) for n in ['numpy','scipy','pandas','threadpoolctl','mpmath']},
                code={str(p):sha256(p) for p in sorted(Path('src/flying').rglob('*.py'))},
                data={str(p):sha256(p) for d in cfg['circuits'] for p in sorted(Path(d).glob('*')) if p.is_file()},
                protocol_sha256=sha256(cfg['protocol']))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_chunk(path, expected_hash):
    if not path.exists() or sha256(path) != expected_hash:
        raise ValueError(f'Checkpoint missing or checksum mismatch: {path}')
    with np.load(path, allow_pickle=False) as saved:
        return {name:saved[name].copy() for name in saved.files}


def decode_chunk(arrays, signature, key):
    payload = json.loads(str(arrays['payload'].item()))
    if payload['fingerprint'] != signature or payload['key'] != key:
        raise ValueError('Checkpoint context mismatch')
    return payload


def first_error(target, prediction):
    return next((i for i,(a,b) in enumerate(zip(target,prediction)) if a != b),len(target))


def train_condition(cfg, key, reservoir, mbon, digits, signature):
    states = reservoir.states(digits[:-1]); labels = digits[1:]
    before = weight_hash(reservoir.weights)
    state_digest = hashlib.sha256(states.tobytes()).hexdigest()
    rows, recalls, histories, arrays = [], [], [], {}
    for initialization in cfg['initializations']:
        for treatment in ['uniform','weighted']:
            ix = len(rows)
            head, history = fit_head(states,labels,mbon,key,initialization,treatment,cfg)
            digest = head.digest()
            pred = head.predict(states)
            recall = evaluate_recall(reservoir,head,digits,cfg['prompt_length'],len(digits)-cfg['prompt_length'])
            prefix = first_error(digits[cfg['prompt_length']:],pred[cfg['prompt_length']-1:])
            assert recall['pi_memory_score'] == prefix
            assert head.digest() == digest and weight_hash(reservoir.weights) == before
            assert head.parameter_count == 482
            rows.append(dict(**key,head_index=ix,initialization=initialization,treatment=treatment,
                             pi_memory_score=prefix,**metrics(head,states,labels,cfg),
                             weight_sha256=before,state_sha256=state_digest,readout_sha256=digest))
            recalls.append(dict(head_index=ix,**recall))
            histories.append(dict(head_index=ix,history=history))
            for name,array in head_arrays(head).items(): arrays[f'{name}_{ix}'] = array
            arrays[f'teacher_prediction_{ix}'] = pred.astype(np.uint8)
    payload = dict(fingerprint=signature,key=key,rows=rows,recalls=recalls,histories=histories,
                   segment=''.join(map(str,digits.tolist())))
    arrays['payload'] = np.array(json.dumps(payload,allow_nan=False))
    return arrays


def summary(frame):
    by_offset = {str(int(offset)):summarize(part) for offset,part in frame.groupby('offset')}
    criteria = {}
    for offset in [1000,2000]:
        if str(offset) not in by_offset: continue
        result = by_offset[str(offset)]['fly']
        criteria[str(offset)] = bool(result['paired_mean_delta'] > 0 and result['wins'] >= 4
                                    and result['all_initializations_improve'] >= 4)
    return dict(by_offset=by_offset,prespecified_new_segment_criteria=criteria,
                both_new_segments_confirmed=len(criteria)==2 and all(criteria.values()))


def grid(cfg):
    for key,reservoir,mbon in conditions(cfg):
        for offset in cfg['offsets']:
            yield dict(**key,offset=offset),reservoir,mbon


def run(config_path, output, resume=False, max_conditions=None):
    cfg = json.loads(Path(config_path).read_text())
    if len(set(cfg['offsets'])) != len(cfg['offsets']) or any(type(v) is not int or v < 0 for v in cfg['offsets']):
        raise ValueError('Invalid/duplicate offsets')
    if max_conditions is not None and max_conditions < 1: raise ValueError('Invalid condition budget')
    ctx = context(cfg); signature = fingerprint(ctx); out = Path(output)
    if resume:
        ledger = json.loads((out/'progress.json').read_text())
        if ledger['context'] != ctx or ledger['fingerprint'] != signature:
            raise ValueError('Resume config/code/data/protocol/runtime changed')
    else:
        out.mkdir(parents=True,exist_ok=False); (out/'checkpoints').mkdir()
        ledger = dict(context=ctx,fingerprint=signature,completed={})
        atomic_json(out/'config.json',cfg); atomic_json(out/'progress.json',ledger)
    maximum = max(cfg['offsets']) + cfg['pi_length']
    assert np.array_equal(pi_digits(maximum),decimal_pi_digits(maximum))
    rows,recalls = [],[]; started = time.monotonic(); trained = reused = 0
    expected = int(np.prod([len(cfg[k]) for k in ['circuits','seeds','models','normalizations','schedules','offsets']]))
    with threadpool_limits(limits=1):
        for number,(key,reservoir,mbon) in enumerate(grid(cfg)):
            name = f'checkpoints/condition_{number:03d}.npz'; path = out/name
            if name in ledger['completed']:
                arrays = load_chunk(path,ledger['completed'][name]); reused += 1
            else:
                digits = segment_digits(key['offset'],cfg['pi_length'])
                arrays = train_condition(cfg,key,reservoir,mbon,digits,signature)
                save_archive(path,arrays)
                ledger['completed'][name] = sha256(path)
                atomic_json(out/'progress.json',ledger); trained += 1
            payload = decode_chunk(arrays,signature,key)
            rows.extend(dict(checkpoint=name,**r) for r in payload['rows'])
            recalls.extend(dict(checkpoint=name,**key,**r) for r in payload['recalls'])
            print(f'{number+1}/{expected} conditions; {len(rows)} evaluations; {trained} trained/{reused} reused; {time.monotonic()-started:.1f}s',flush=True)
            if max_conditions is not None and trained >= max_conditions and number+1 < expected:
                return dict(complete=False,trained=trained,reused=reused)
    assert len(ledger['completed']) == expected and len(rows) == expected * 2 * len(cfg['initializations'])
    frame = pd.DataFrame(rows)
    temporary = out/'evaluations.csv.part'; frame.to_csv(temporary,index=False); temporary.replace(out/'evaluations.csv')
    temporary = out/'recalls.jsonl.part'; temporary.write_text(''.join(json.dumps(r)+'\n' for r in recalls)); temporary.replace(out/'recalls.jsonl')
    atomic_json(out/'summary.json',summary(frame))
    files = {p.relative_to(out).as_posix():sha256(p) for p in sorted(out.rglob('*'))
             if p.is_file() and p.suffix != '.part' and p.name not in ['manifest.json','verification.json','pytest.txt']}
    atomic_json(out/'manifest.json',dict(context=ctx,fingerprint=signature,conditions=expected,evaluations=len(rows),
                trained_this_invocation=trained,reused_this_invocation=reused,elapsed_this_invocation=time.monotonic()-started,
                pi_generators_equal_digits=maximum,file_sha256=files))
    return dict(complete=True,trained=trained,reused=reused)


def verify(output):
    out = Path(output); manifest = json.loads((out/'manifest.json').read_text()); cfg = manifest['context']['config']
    # Accept manifests emitted by older Windows runs, which used backslashes.
    manifest['file_sha256'] = {name.replace('\\', '/'):digest for name,digest in manifest['file_sha256'].items()}
    if context(cfg) != manifest['context']: raise ValueError('Verification context changed')
    for name,digest in manifest['file_sha256'].items():
        if sha256(out/name) != digest: raise ValueError(f'Artifact hash mismatch: {name}')
    maximum = max(cfg['offsets']) + cfg['pi_length']
    assert np.array_equal(pi_digits(maximum),decimal_pi_digits(maximum))
    frame = pd.read_csv(out/'evaluations.csv')
    assert len(frame)==manifest['evaluations'] and not frame.duplicated(KEYS+['offset','initialization','treatment']).any()
    recorded_recalls = [json.loads(line) for line in (out/'recalls.jsonl').read_text().splitlines()]
    rebuilt_recalls = []; rebuilt_rows = []; refitted = replayed = 0
    with threadpool_limits(limits=1):
        for number,(key,reservoir,mbon) in enumerate(grid(cfg)):
            name = f'checkpoints/condition_{number:03d}.npz'
            arrays = load_chunk(out/name,manifest['file_sha256'][name]); payload = decode_chunk(arrays,manifest['fingerprint'],key)
            digits = segment_digits(key['offset'],cfg['pi_length']); labels = digits[1:]
            assert payload['segment'] == ''.join(map(str,digits.tolist()))
            states = reservoir.states(digits[:-1]); before = weight_hash(reservoir.weights)
            refit = key['circuit'] == Path(cfg['circuits'][0]).name and key['seed']==cfg['seeds'][0] and key['model']=='fly'
            assert len(payload['rows']) == len(payload['recalls']) == len(cfg['initializations'])*2
            for row,rec in zip(payload['rows'],payload['recalls']):
                ix = row['head_index']; head = NonlinearReadout(mbon,cfg['hidden_units'])
                head.mean = arrays[f'mean_{ix}']; head.scale = arrays[f'scale_{ix}']
                head.parameters = {n:arrays[f'{n}_{ix}'] for n in ['w1','b1','w2','b2']}
                assert head.parameter_count == 482
                assert np.array_equal(head.mean,states[:,mbon].mean(axis=0))
                assert np.array_equal(head.scale,np.maximum(states[:,mbon].std(axis=0),1e-5))
                assert row['state_sha256']==hashlib.sha256(states.tobytes()).hexdigest() and row['weight_sha256']==before
                assert head.digest() == row['readout_sha256']
                pred = head.predict(states); assert np.array_equal(pred,arrays[f'teacher_prediction_{ix}'])
                for k,v in metrics(head,states,labels,cfg).items(): assert np.isclose(v,row[k],atol=1e-12,rtol=0)
                got = evaluate_recall(reservoir,head,digits,cfg['prompt_length'],len(digits)-cfg['prompt_length'])
                assert rec == dict(head_index=ix,**got)
                assert first_error(digits[cfg['prompt_length']:],pred[cfg['prompt_length']-1:])==got['pi_memory_score']==row['pi_memory_score']
                assert head.digest()==row['readout_sha256'] and weight_hash(reservoir.weights)==before
                if refit:
                    fitted,_ = fit_head(states,labels,mbon,key,row['initialization'],row['treatment'],cfg)
                    assert fitted.digest()==head.digest(); refitted += 1
                rebuilt_rows.append(dict(checkpoint=name,**row)); rebuilt_recalls.append(dict(checkpoint=name,**key,**rec)); replayed += 1
    pd.testing.assert_frame_equal(frame,pd.DataFrame(rebuilt_rows),check_exact=False,atol=1e-12,rtol=0)
    assert recorded_recalls == rebuilt_recalls
    assert summary(pd.DataFrame(rebuilt_rows)) == json.loads((out/'summary.json').read_text())
    assert replayed == manifest['evaluations'] and refitted == len(cfg['offsets'])*len(cfg['initializations'])*2
    result = dict(states_rebuilt=number+1,heads_replayed=replayed,independently_refitted=refitted,
                  exact_recall_strings=True,frozen_weights_verified=True,pi_generators_equal_digits=maximum,
                  aggregate_and_per_position_metrics_verified=True)
    atomic_json(out/'verification.json',result); print(json.dumps(result,indent=2))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config',default='configs/phase5_prefix_confirmation.json')
    p.add_argument('--output',default='outputs/phase5_prefix_confirmation')
    p.add_argument('--resume',action='store_true'); p.add_argument('--verify',action='store_true')
    p.add_argument('--max-conditions',type=int)
    a = p.parse_args()
    verify(a.output) if a.verify else run(a.config,a.output,a.resume,a.max_conditions)
