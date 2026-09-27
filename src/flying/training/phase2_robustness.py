"""Locked perturbation study on two independently seeded saved model cohorts."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from flying.evaluation.perturbation import KINDS, recall
from flying.evaluation.metrics import pi_memory_score
from flying.models.nonlinear_readout import NonlinearReadout
from flying.training import phase5_prefix_confirmation as checkpoint
from flying.training.phase5_readout import load_roles, load_connectome
from flying.training.phase5_prefix import weight_hash, sha256


def validate(cfg):
    if cfg.get('experiment') != 'frozen_model_perturbation_robustness':
        raise ValueError('Wrong experiment')
    if cfg['prompt'] != '314' or cfg['horizon'] != 197 or cfg['treatment'] != 'fixed' or cfg['epoch'] != 6000:
        raise ValueError('Expected the fixed-32 game-opening baseline')
    if set(cfg['strengths']) != set(KINDS) or set(cfg['primary']) != set(KINDS):
        raise ValueError('Missing perturbation family')
    for kind in KINDS:
        values = cfg['strengths'][kind]
        if (not values or values[0] != 0 or len(set(values)) != len(values)
                or any(type(x) not in (int,float) or not np.isfinite(x) or x < 0 for x in values)
                or cfg['primary'][kind] not in values or cfg['primary'][kind] <= 0):
            raise ValueError('Invalid strengths or primary endpoint')
        if kind == 'edge_dropout' and max(values) > 1:
            raise ValueError('Invalid dropout fraction')
    if len(cfg['cohorts']) != 2 or len({c['name'] for c in cfg['cohorts']}) != 2:
        raise ValueError('Two distinct cohorts required')
    for field in ['model_seeds','noise_seeds']:
        seen = set()
        for cohort in cfg['cohorts']:
            values = cohort[field]
            if (not values or len(set(values)) != len(values) or seen.intersection(values)
                    or any(type(x) is not int or x < 0 for x in values)):
                raise ValueError('Cohorts need distinct nonnegative seeds')
            seen.update(values)
    if type(cfg['expected_models']) is not int or cfg['expected_models'] < 1:
        raise ValueError('Invalid model count')
    if cfg['retention_fraction'] != .8 or cfg['required_blocks'] != 5:
        raise ValueError('The locked robustness thresholds must not change')


def source_manifest(cohort):
    source = Path(cohort['source'])
    if sha256(source/'manifest.json') != cohort['manifest_sha256']:
        raise ValueError('Source manifest changed')
    manifest = json.loads((source/'manifest.json').read_text())
    return source, manifest


def study_context(cfg):
    ctx = checkpoint.context(cfg)
    for cohort in cfg['cohorts']:
        source, manifest = source_manifest(cohort)
        for name, digest in manifest['file_sha256'].items():
            if sha256(source/name.replace('\\','/')) != digest:
                raise ValueError('Source artifact changed')
        for name, digest in manifest['context']['data'].items():
            if sha256(name.replace('\\','/')) != digest:
                raise ValueError('Source data changed')
    return ctx


def models(cfg):
    """Reconstruct every eligible head; verify source states and clean outputs."""
    for cohort in cfg['cohorts']:
        source, manifest = source_manifest(cohort)
        original = manifest['context']['config']
        if (original['seeds'] != cohort['model_seeds'] or original['circuits'] != cfg['circuits']
                or original['initializations'] != [0,1,2] or original['pi_length'] != 200
                or original['stage_epochs'] != 2000 or original['prompt_length'] != 3
                or original['normalizations'] != ['incoming_l1'] or original['schedules'] != ['mbon_after_kc']
                or original['prefix_window'] != 32 or original['prefix_weight'] != 4
                or set(original['models']) != {'fly','role_shuffled'}):
            raise ValueError('Source is not the prespecified model population')
        roles = {Path(d).name:load_roles(d,load_connectome(d)[1])[0] for d in cfg['circuits']}
        hashes = {k.replace('\\','/'):v for k,v in manifest['file_sha256'].items()}
        for number, (key, reservoir, mbon) in enumerate(checkpoint.grid(original)):
            if key['offset'] != 0:
                continue
            name = f'checkpoints/condition_{number:03d}.npz'
            arrays = checkpoint.load_chunk(source/name,hashes[name])
            payload = checkpoint.decode_chunk(arrays,manifest['fingerprint'],key)
            target = payload['segment']
            assert target == ''.join(map(str,checkpoint.pi_digits(200)))
            states = reservoir.states([int(d) for d in target[:-1]])
            state_hash = hashlib.sha256(states.tobytes()).hexdigest()
            selected = [r for r in payload['rows'] if r['treatment'] == cfg['treatment'] and r['epoch'] == cfg['epoch']]
            assert sorted(r['initialization'] for r in selected) == [0,1,2]
            for row in selected:
                ix = row['head_index']
                assert state_hash == row['state_sha256'] and weight_hash(reservoir.weights) == row['weight_sha256']
                head = NonlinearReadout(mbon,hidden=original['hidden_units'])
                head.mean, head.scale = arrays[f'mean_{ix}'], arrays[f'scale_{ix}']
                head.parameters = {k:arrays[f'{k}_{ix}'] for k in ['w1','b1','w2','b2']}
                assert head.digest() == row['readout_sha256']
                assert np.array_equal(head.predict(states),arrays[f'teacher_prediction_{ix}'])
                expected = next(r['prediction'] for r in payload['recalls'] if r['head_index'] == ix)
                identity = f"{cohort['name']}_{key['circuit']}_{key['model']}_s{key['seed']}_i{row['initialization']}"
                yield dict(key=dict(cohort=cohort['name'],model_id=identity,**key,initialization=row['initialization'],
                                    source_checkpoint=name,source_head_index=ix,readout_sha256=head.digest(),
                                    weight_sha256=row['weight_sha256'],state_sha256=state_hash),
                           reservoir=reservoir,head=head,roles=roles[key['circuit']],target=target[3:],expected=expected,
                           circuit_index=[Path(d).name for d in cfg['circuits']].index(key['circuit']),
                           state_std=dict(all_median=float(np.median(states.std(axis=0))),
                                          mbon_min=float(states[:,mbon].std(axis=0).min()),
                                          mbon_median=float(np.median(states[:,mbon].std(axis=0))),
                                          mbon_max=float(states[:,mbon].std(axis=0).max())))


def trials(cfg, cohort):
    yield 'clean',0.,None
    for kind in KINDS:
        for strength in cfg['strengths'][kind]:
            for seed in ([None] if strength == 0 else cohort['noise_seeds']):
                yield kind,float(strength),seed


def evaluate(model, cfg, signature):
    key = model['key']
    cohort = next(c for c in cfg['cohorts'] if c['name'] == key['cohort'])
    rows = []
    target = np.array([int(d) for d in model['target']])
    for kind,strength,seed in trials(cfg,cohort):
        result = recall(model['reservoir'],model['head'],model['roles'],cfg['prompt'],cfg['horizon'],
                        kind,strength,[0 if seed is None else seed,key['seed'],model['circuit_index']])
        score = pi_memory_score(target,np.array([int(d) for d in result['prediction']]))
        if strength == 0:
            assert result['prediction'] == model['expected']
            assert result['perturbed_weight_sha256'] == key['weight_sha256']
        rows.append(dict(kind=kind,strength=strength,noise_seed=seed,pi_memory_score=score,
                         censored=score == cfg['horizon'],first_error_position=None if score == cfg['horizon'] else score+1,
                         **result))
    assert weight_hash(model['reservoir'].weights) == key['weight_sha256']
    assert model['head'].digest() == key['readout_sha256']
    return dict(fingerprint=signature,key=key,state_std=model['state_std'],trials=rows)


def flatten(payload):
    return [dict(**payload['key'],**{k:v for k,v in row.items() if k != 'prediction'}) for row in payload['trials']]


def summary(frame,cfg):
    baseline = frame[frame.kind == 'clean'][['model_id','pi_memory_score']].rename(columns={'pi_memory_score':'clean_score'})
    assert not baseline.model_id.duplicated().any()
    joined = frame.merge(baseline,on='model_id',validate='many_to_one')
    curves = {}
    for (cohort,graph,kind,strength),part in joined.groupby(['cohort','model','kind','strength']):
        clean = float(part.clean_score.mean())
        blocks = part.groupby(['circuit','seed'])[['pi_memory_score','clean_score']].mean()
        kept = int(((blocks.clean_score > 0) & (blocks.pi_memory_score >= cfg['retention_fraction']*blocks.clean_score)).sum())
        eligible = part[part.clean_score >= 32]
        rate = float((eligible.pi_memory_score >= 32).mean()) if len(eligible) else None
        result = dict(trials=len(part),models=int(part.model_id.nunique()),blocks=len(blocks),
                      mean_recall=float(part.pi_memory_score.mean()),mean_clean_recall=clean,
                      mean_ratio=float(part.pi_memory_score.mean()/clean) if clean > 0 else None,
                      blocks_retaining_80_percent=kept,eligible_32_trials=len(eligible),
                      eligible_32_models=int(eligible.model_id.nunique()),retained_32_fraction=rate,
                      censored_trials=int(part.censored.sum()),clipped_coordinates=int(part.clipped_coordinates.sum()))
        curves.setdefault(cohort,{}).setdefault(graph,{}).setdefault(kind,{})[str(float(strength))] = result
    decisions = {}
    for cohort in cfg['cohorts']:
        name = cohort['name']
        decisions[name] = {}
        for kind in KINDS:
            value = curves.get(name,{}).get('fly',{}).get(kind,{}).get(str(float(cfg['primary'][kind])))
            criteria = dict(mean_retention=False,condition_retention=False,prefix_32_retention=False)
            if value:
                criteria = dict(mean_retention=value['mean_ratio'] is not None and value['mean_ratio'] >= cfg['retention_fraction'],
                                condition_retention=value['blocks'] == 6 and value['blocks_retaining_80_percent'] >= cfg['required_blocks'],
                                prefix_32_retention=value['retained_32_fraction'] is not None and value['retained_32_fraction'] >= cfg['retention_fraction'])
            decisions[name][kind] = dict(**criteria,passed=all(criteria.values()))
    return dict(by_cohort=curves,primary_by_cohort=decisions,
                both_cohorts_all_primary_passed=all(d['passed'] for c in decisions.values() for d in c.values()))


def run(config_path,output,resume=False,max_models=None):
    cfg = json.loads(Path(config_path).read_text()); validate(cfg)
    if max_models is not None and (type(max_models) is not int or max_models < 1):
        raise ValueError('Invalid model budget')
    ctx = study_context(cfg); signature = checkpoint.fingerprint(ctx)
    out = Path(output)
    if resume:
        ledger = json.loads((out/'progress.json').read_text())
        if ledger['context'] != ctx or ledger['fingerprint'] != signature:
            raise ValueError('Resume context changed')
    else:
        out.mkdir(parents=True,exist_ok=False); (out/'checkpoints').mkdir()
        ledger = dict(context=ctx,fingerprint=signature,completed={})
        checkpoint.atomic_json(out/'config.json',cfg); checkpoint.atomic_json(out/'progress.json',ledger)
    assert np.array_equal(checkpoint.pi_digits(200),checkpoint.decimal_pi_digits(200))
    rows, names = [], set(); created = reused = 0; started = time.monotonic()
    with threadpool_limits(limits=1):
        for number,model in enumerate(models(cfg)):
            name = f'checkpoints/model_{number:03d}.json'; names.add(name)
            if name in ledger['completed']:
                if not (out/name).exists() or sha256(out/name) != ledger['completed'][name]:
                    raise ValueError('Checkpoint missing or checksum mismatch')
                payload = json.loads((out/name).read_text()); reused += 1
            else:
                payload = evaluate(model,cfg,signature)
                checkpoint.atomic_json(out/name,payload)
                ledger['completed'][name] = sha256(out/name)
                checkpoint.atomic_json(out/'progress.json',ledger); created += 1
            if payload['key'] != model['key'] or payload['fingerprint'] != signature:
                raise ValueError('Checkpoint identity mismatch')
            rows.extend(flatten(payload))
            print(f'{number+1}/{cfg["expected_models"]} heads; {len(rows)} recalls; '
                  f'{created} new/{reused} reused; {time.monotonic()-started:.1f}s',flush=True)
            if max_models is not None and created >= max_models and number+1 < cfg['expected_models']:
                return dict(complete=False,created=created,reused=reused)
    assert number+1 == cfg['expected_models'] and set(ledger['completed']) == names
    frame = pd.DataFrame(rows)
    assert not frame.duplicated(['model_id','kind','strength','noise_seed']).any()
    frame.to_csv(out/'evaluations.csv',index=False)
    checkpoint.atomic_json(out/'summary.json',summary(frame,cfg))
    files = {p.relative_to(out).as_posix():sha256(p) for p in sorted(out.rglob('*')) if p.is_file()
             and p.suffix != '.part' and p.name not in ['manifest.json','verification.json','overview.png']}
    checkpoint.atomic_json(out/'manifest.json',dict(context=ctx,fingerprint=signature,models=number+1,
        recalls=len(rows),new_fits=0,created_this_invocation=created,reused_this_invocation=reused,
        elapsed_this_invocation=time.monotonic()-started,file_sha256=files))
    return dict(complete=True,created=created,reused=reused)


def verify(output):
    out = Path(output); manifest = json.loads((out/'manifest.json').read_text())
    cfg = manifest['context']['config']; validate(cfg)
    if study_context(cfg) != manifest['context']:
        raise ValueError('Verification context changed')
    for name,digest in manifest['file_sha256'].items():
        if not (out/name).exists() or sha256(out/name) != digest:
            raise ValueError('Artifact checksum mismatch')
    rows, count = [], 0
    with threadpool_limits(limits=1):
        for number,model in enumerate(models(cfg)):
            recorded = json.loads((out/f'checkpoints/model_{number:03d}.json').read_text())
            replayed = evaluate(model,cfg,manifest['fingerprint'])
            assert replayed == recorded
            rows.extend(flatten(recorded)); count += 1
            if count % 6 == 0:
                print(f'Verified {count}/{manifest["models"]} heads / {len(rows)} full recalls',flush=True)
    frame = pd.DataFrame(rows)
    pd.testing.assert_frame_equal(frame,pd.read_csv(out/'evaluations.csv'),check_dtype=False,check_exact=False,atol=1e-12,rtol=0)
    assert summary(frame,cfg) == json.loads((out/'summary.json').read_text())
    assert count == manifest['models'] == cfg['expected_models'] and len(rows) == manifest['recalls']
    assert np.array_equal(checkpoint.pi_digits(200),checkpoint.decimal_pi_digits(200))
    result = dict(source_heads_reconstructed=count,full_recalls_replayed=len(rows),
                  null_perturbations_equal_clean=3*count,exact_predictions_and_perturbations=True,
                  original_weights_and_readouts_unchanged=True,summary_recomputed=True,pi_generators_equal_digits=200)
    checkpoint.atomic_json(out/'verification.json',result)
    print(json.dumps(result,indent=2)); return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/phase2_robustness.json')
    parser.add_argument('--output',default='outputs/phase2_robustness')
    parser.add_argument('--resume',action='store_true'); parser.add_argument('--verify',action='store_true')
    parser.add_argument('--max-models',type=int)
    args = parser.parse_args()
    verify(args.output) if args.verify else run(args.config,args.output,args.resume,args.max_models)
