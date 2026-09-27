"""Offline export boundary: research artifacts in, target-free opponents out."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from flying.game.opponent import OpponentCatalog, file_hash
from flying.training.phase5_prefix_confirmation import grid, load_chunk
from flying.training.phase5_prefix import weight_hash, head_arrays, NonlinearReadout
from flying.training.phase5_readout import load_roles, load_connectome


def export_opponents(source, output, report_path):
    source, out, report_path = Path(source),Path(output),Path(report_path)
    if report_path.exists():
        raise FileExistsError(report_path)
    manifest = json.loads((source/'manifest.json').read_text())
    cfg = manifest['context']['config']
    if (cfg.get('experiment') != 'prefix_share_retention' or cfg['offsets'] != [0]
            or cfg['pi_length'] != 200 or cfg['prompt_length'] != 3
            or cfg['stage_epochs'] != 2000 or cfg['initializations'] != [0,1,2]
            or cfg['normalizations'] != ['incoming_l1'] or cfg['schedules'] != ['mbon_after_kc']
            or cfg['prefix_window'] != 32 or cfg['prefix_weight'] != 4):
        raise ValueError('Expected the validated fixed-32 game-opening study')
    hashes = {name.replace('\\','/'):digest for name,digest in manifest['file_sha256'].items()}
    for name,digest in hashes.items():
        if file_hash(source/name) != digest:
            raise ValueError(f'Research artifact checksum mismatch: {name}')
    for name,digest in manifest['context']['data'].items():
        if file_hash(name.replace('\\','/')) != digest:
            raise ValueError(f'Source data changed: {name}')
    out.mkdir(parents=True,exist_ok=False)
    roles = {Path(d).name:load_roles(d,load_connectome(d)[1])[0] for d in cfg['circuits']}
    entries, evidence, expected = [], [], {}
    with threadpool_limits(limits=1):
        for number,(key,reservoir,mbon) in enumerate(grid(cfg)):
            if key['model'] != 'fly':
                continue
            name = f'checkpoints/condition_{number:03d}.npz'
            arrays = load_chunk(source/name,hashes[name])
            payload = json.loads(str(arrays['payload'].item()))
            if payload['key'] != key or payload['fingerprint'] != manifest['fingerprint']:
                raise ValueError('Research checkpoint context mismatch')
            selected = [r for r in payload['rows'] if r['treatment']=='fixed' and r['epoch']==6000]
            if sorted(r['initialization'] for r in selected) != [0,1,2]:
                raise ValueError('Incomplete fixed-32 population')
            states = reservoir.states([int(d) for d in payload['segment'][:-1]])
            for row in selected:
                ix = row['head_index']
                if row['state_sha256'] != hashlib.sha256(states.tobytes()).hexdigest() or row['weight_sha256'] != weight_hash(reservoir.weights):
                    raise ValueError('Reconstructed connectome states differ')
                head = NonlinearReadout(mbon,cfg['hidden_units'])
                head.mean = arrays[f'mean_{ix}']; head.scale = arrays[f'scale_{ix}']
                head.parameters = {n:arrays[f'{n}_{ix}'] for n in ['w1','b1','w2','b2']}
                if head.digest() != row['readout_sha256'] or not np.array_equal(head.predict(states),arrays[f'teacher_prediction_{ix}']):
                    raise ValueError('Readout replay differs')
                identity = f"{key['circuit']}_seed{key['seed']}_init{row['initialization']}_fixed6000"
                artifact = identity+'.npz'; w = reservoir.weights
                with (out/artifact).open('wb') as stream:
                    np.savez_compressed(stream,weights_data=w.data,weights_indices=w.indices,weights_indptr=w.indptr,
                        roles=np.asarray(roles[key['circuit']],dtype=str),patterns=reservoir.encoder.patterns,indices=mbon,
                        leak=np.array(cfg['leak']),schedule=np.array(key['schedule']),**head_arrays(head))
                entries.append(dict(id=identity,artifact=artifact,sha256=file_hash(out/artifact),
                    readout_sha256=head.digest(),source_checkpoint=name,source_head_index=ix))
                recall = next(r for r in payload['recalls'] if r['head_index']==ix)
                expected[identity] = recall['prediction']
                evidence.append(dict(opponent_id=identity,recorded_score=row['pi_memory_score']))
        if len(entries) != 18:
            raise ValueError('Expected all 18 fixed-32 real-graph models')
        catalog = dict(format_version=1,mode='pretrained',prompt='314',horizon=197,trained_digits=200,
            selection='uniform_by_match_seed_before_play',source_manifest_sha256=file_hash(source/'manifest.json'),
            opponents=entries)
        (out/'catalog.json').write_text(json.dumps(catalog,indent=2)+'\n',encoding='utf-8',newline='\n')
        loaded = OpponentCatalog(out/'catalog.json')
        for row in evidence:
            model = loaded.load(row['opponent_id'])
            generated = ''.join(str(model.next_digit()) for _ in range(model.horizon))
            if generated != expected[model.identity]:
                raise ValueError('Exported opponent does not reproduce the full saved rollout')
            row['rollout_sha256'] = hashlib.sha256(generated.encode()).hexdigest()
    report = dict(source_manifest_sha256=file_hash(source/'manifest.json'),catalog_sha256=loaded.digest,
                  opponents_exported=len(entries),full_197_digit_replays=len(evidence),
                  selection_uses_scores=False,targets_or_rollouts_in_inference_artifacts=False,models=evidence)
    report_path.parent.mkdir(parents=True,exist_ok=True)
    report_path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',default='results/phase5_retention')
    parser.add_argument('--output',default='assets/opponents/fixed32')
    parser.add_argument('--report',default='results/game_adapter/export.json')
    args = parser.parse_args()
    result = export_opponents(args.source,args.output,args.report)
    print(f"Exported and replayed all {result['opponents_exported']} opponents")


if __name__ == '__main__':
    main()
