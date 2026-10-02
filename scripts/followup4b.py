"""One whole-brain degree/role preserving graph null with paired decoding."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits

from flying.brain.mushroom_body import role_shuffled
from flying.brain.timed_reservoir import TimedReservoir
from flying.training import whole_brain_memory as core
from alphabet_memory import SymbolEncoder,read
from frozen_state_probe import Budget
import followup4 as previous

CONFIG=Path('configs/followup4b.json')


def source_hashes(c):
    paths=[CONFIG,Path(c['protocol']),Path('scripts/followup4b.py'),Path('scripts/verify_followup4b.py')]
    paths+=sorted(Path('src/flying').rglob('*.py'))
    paths += [Path(c['cache'])/n for n in ('nodes.npz','brain5.npz','provenance.json')]
    return {p.as_posix():core.sha256(p) for p in paths}


def structural_checks(raw,shuffled,roles,log):
    assert raw.shape==shuffled.shape and raw.nnz==shuffled.nnz and not shuffled.diagonal().any()
    assert log['accepted_swaps']==log['requested_swaps']==raw.nnz
    for axis in (0,1):
        np.testing.assert_array_equal(np.asarray((raw!=0).sum(axis=axis)).ravel(),
                                      np.asarray((shuffled!=0).sum(axis=axis)).ravel())
    np.testing.assert_array_equal(np.sort(raw.data),np.sort(shuffled.data))
    roles=np.asarray(roles)
    a,b=raw.tocoo(),shuffled.tocoo()
    block_counts={}
    for pre in np.unique(roles):
        for post in np.unique(roles):
            key=f'{pre}->{post}'
            x=int(np.sum((roles[a.col]==pre)&(roles[a.row]==post)))
            y=int(np.sum((roles[b.col]==pre)&(roles[b.row]==post)))
            assert x==y,key
            block_counts[key]=x
    return dict(source_edges=int(raw.nnz),shuffled_edges=int(shuffled.nnz),
                source_weight_sha256=core.weight_hash(raw),shuffled_weight_sha256=core.weight_hash(shuffled),
                degree_and_role_blocks_exact=True,global_weight_multiset_exact=True,
                swap_log=log,role_blocks=block_counts)


def prepare(c):
    root=Path(c['graph']);root.mkdir(parents=True,exist_ok=False);budget=Budget(c)
    try:
        cache=Path(c['cache']);raw=sparse.load_npz(cache/'brain5.npz').tocsr()
        with np.load(cache/'nodes.npz',allow_pickle=False) as nodes:roles=nodes['roles'].copy()
        shuffled,log=role_shuffled(raw,roles,c['graph_seed'],c['swaps_per_edge'])
        audit=structural_checks(raw,shuffled,roles,log)
        sparse.save_npz(root/'shuffled-raw.npz',shuffled)
        core.write_json(root/'audit.json',audit)
        core.write_json(root/'source-hashes.json',source_hashes(c))
        usage=budget.close();core.write_json(root/'verification.json',dict(complete=True,usage=usage))
        core.write_json(root/'manifest.json',dict(config=c,complete=True,
            artifacts={p.name:core.sha256(p) for p in root.iterdir() if p.is_file()}))
        print(json.dumps(dict(edges=audit['source_edges'],swap_log=log,usage=usage)),flush=True)
    except BaseException:
        budget.close();raise


def graph(c):
    root=Path(c['graph']);manifest=read(root/'manifest.json')
    for name,digest in manifest['artifacts'].items():assert core.sha256(root/name)==digest,name
    assert manifest['config']==c
    raw=sparse.load_npz(root/'shuffled-raw.npz').tocsr()
    weights=core.normalize_condition(raw,c['normalization'],c['gain']).tocsr()
    weights.sort_indices()
    return weights,read(root/'audit.json')


def run(out,smoke=False):
    c=read(CONFIG);parent=Path(c['parent']);old=read(parent/'manifest.json')
    for name,digest in old['artifacts'].items():assert core.sha256(parent/name)==digest,name
    shuffled,audit=graph(c)
    out.mkdir(parents=True,exist_ok=False);budget=Budget(c)
    try:
        core.write_json(out/'config.json',c)
        sources=source_hashes(c);sources[(Path(c['graph'])/'manifest.json').as_posix()]=core.sha256(Path(c['graph'])/'manifest.json')
        sources[(parent/'manifest.json').as_posix()]=core.sha256(parent/'manifest.json')
        core.write_json(out/'source-hashes.json',sources)
        prior=pd.read_csv(parent/'raw-cases.csv')
        rows=[];lags=[]
        for block in c['blocks'][:1] if smoke else c['blocks']:
            for circuit in c['circuit_seeds'][:1] if smoke else c['circuit_seeds']:
                budget.check();base,observed,info,roles=previous.base_model(c,c['level'],circuit,block)
                assert base.weights.shape==shuffled.shape
                model=TimedReservoir(shuffled,SymbolEncoder(base.encoder.patterns.copy()),roles,c['leak'],c['schedule'])
                train=previous.symbols(block,'train',c['warmup']+c['train_samples'])
                test=previous.symbols(block,'test',c['warmup']+c['test_samples'])
                a,per_lag,frequency=previous.train_and_score(c,model,observed,train,test)
                existing=prior[(prior.seed==block['seed'])&(prior.circuit==circuit)&(prior.level=='brain5')&(prior.arm=='intact')]
                assert len(existing)==1
                real=float(existing.iloc[0].accuracy);score=float(per_lag.mean())
                row=dict(seed=block['seed'],circuit=circuit,real_accuracy=real,shuffled_accuracy=score,
                         real_minus_shuffled=real-score,frequency_accuracy=float(frequency.mean()),
                         graph_weight_sha256=core.weight_hash(shuffled),input_pattern_sha256=info['k4_pattern_sha256'])
                p=out/f's{block["seed"]}_c{circuit}';p.mkdir()
                np.savez_compressed(p/'case.npz',**a)
                core.write_json(p/'metrics.json',row)
                core.write_json(p/'manifest.json',dict(graph_weight_sha256=row['graph_weight_sha256'],
                    parent_case_sha256=core.sha256(parent/f's{block["seed"]}_c{circuit}_brain5_intact/case.npz'),
                    artifacts={q.name:core.sha256(q) for q in p.iterdir()}))
                rows.append(row)
                lags.extend(dict(seed=block['seed'],circuit=circuit,lag=lag,shuffled_accuracy=float(per_lag[j]),
                                 frequency_accuracy=float(frequency[j])) for j,lag in enumerate(c['primary_lags']))
                print(f'completed s{block["seed"]} c{circuit}',flush=True)
        data=pd.DataFrame(rows);data.to_csv(out/'raw-cases.csv',index=False)
        pd.DataFrame(lags).to_csv(out/'raw-lags.csv',index=False)
        blocks=data.groupby('seed')[['real_accuracy','shuffled_accuracy','real_minus_shuffled']].mean().reset_index()
        blocks.to_csv(out/'seed-blocks.csv',index=False)
        gate=None if smoke else bool(blocks.real_minus_shuffled.mean()>=.03 and (blocks.real_minus_shuffled>0).all())
        summary=dict(real_wiring_advantage=gate,mean_difference=float(blocks.real_minus_shuffled.mean()),
                     per_seed=blocks.real_minus_shuffled.tolist(),cases=len(data),edge_overlap=audit['swap_log']['edge_overlap'])
        core.write_json(out/'summary.json',summary)
        usage=budget.close();core.write_json(out/'verification.json',dict(complete=True,smoke=smoke,usage=usage))
        core.write_json(out/'manifest.json',dict(config=c,smoke=smoke,complete=True,
            artifacts={p.relative_to(out).as_posix():core.sha256(p) for p in out.rglob('*') if p.is_file()}))
        print(json.dumps(summary),flush=True)
    except BaseException:
        budget.close();raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run']);p.add_argument('--out',type=Path)
    p.add_argument('--smoke',action='store_true');a=p.parse_args()
    with threadpool_limits(1):
        if a.command=='prepare':prepare(read(CONFIG))
        else:
            if a.out is None:p.error('run requires --out')
            run(a.out,a.smoke)
