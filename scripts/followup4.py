"""Preregistered whole-brain expansion and DAN→MBON pathway test."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits

from flying.brain.timed_reservoir import TimedReservoir
from flying.models.ridge import RidgeDecoder
from flying.training import whole_brain_memory as core
from alphabet_memory import SymbolEncoder,read
from frozen_state_probe import Budget

CONFIG=Path('configs/followup4.json')


def roles_for(cache,level,circuit):
    if level=='legacy5':
        root=Path(f'data/flywire_783_mb_left_kc512_s{circuit}')
        _,ids,_=core.load_connectome(root)
        roles,_=core.load_roles(root,ids)
        return np.asarray(roles)
    with np.load(cache/'nodes.npz',allow_pickle=False) as nodes:return nodes['roles'].copy()


def base_model(c,level,circuit,block):
    condition=core.NetworkCondition(level,circuit,block['input_seed'])
    base,observed,graph=core.build_model(c['cache'],condition,c)
    bank=base.encoder.patterns[:4].copy()
    base.encoder=SymbolEncoder(bank);base.reset()
    graph['k4_pattern_sha256']=hashlib.sha256(bank.tobytes()).hexdigest()
    roles=roles_for(Path(c['cache']),level,circuit)
    assert base.weights.shape==(len(roles),len(roles))
    return base,observed,graph,roles


def masks(base,roles,seed):
    coo=base.weights.tocoo()
    dan=(roles[coo.row]=='MBON')&(roles[coo.col]=='DAN')
    eligible=(roles[coo.row]=='MBON')&(roles[coo.col]!='DAN')
    n=int(dan.sum());target_mass=float(abs(coo.data[dan]).sum())
    assert n>0 and int(eligible.sum())>=n
    indices=np.flatnonzero(eligible)
    values=np.abs(coo.data[indices]);rng=np.random.default_rng(seed)
    # Fixed candidate family: contiguous n-edge windows in absolute-weight order.
    # This avoids any access to activity or labels and preserves target edge count.
    order=np.lexsort((rng.random(len(indices)),values))
    cumulative=np.r_[0,np.cumsum(values[order])]
    masses=cumulative[n:]-cumulative[:-n]
    start=int(np.argmin(np.abs(masses-target_mass)))
    matched=np.zeros(len(coo.data),bool);matched[indices[order[start:start+n]]]=True
    control_mass=float(abs(coo.data[matched]).sum())
    info=dict(edges=n,dan_mass=target_mass,control_mass=control_mass,
              mass_residual_fraction=abs(control_mass-target_mass)/target_mass,
              valid_mass_match=abs(control_mass-target_mass)<=.1*target_mass,
              matched_window_start=start,eligible_non_dan=int(eligible.sum()))
    return coo,dan,matched,info


def arm_model(base,roles,coo,dan,matched,arm):
    removed={'intact':np.zeros(len(coo.data),bool),'dan_cut':dan,'matched_cut':matched}[arm]
    keep=~removed
    weights=sparse.csr_matrix((coo.data[keep],(coo.row[keep],coo.col[keep])),shape=base.weights.shape)
    weights.sort_indices()
    return TimedReservoir(weights,SymbolEncoder(base.encoder.patterns.copy()),roles,base.leak,base.schedule),removed


def symbols(block,split,n):return np.random.default_rng(block[split+'_seed']).integers(0,4,n,dtype=np.int64)


def train_and_score(c,model,observed,train,test):
    xtrain,_=core.collect(model,observed,train,c['activity_epsilon'])
    xtest,_=core.collect(model,observed,test,c['activity_epsilon'])
    w=c['warmup'];lags=c['primary_lags']
    ytrain=np.asarray([[train[t-l] for l in lags] for t in range(w,len(train))],np.int64)
    ytest=np.asarray([[test[t-l] for l in lags] for t in range(w,len(test))],np.int64)
    target=np.eye(4)[ytrain].reshape(len(ytrain),-1)
    ridge=RidgeDecoder(c['alpha']).fit(xtrain[w:],target)
    scores=ridge.scores(xtest[w:]).reshape(len(ytest),len(lags),4)
    prediction=scores.argmax(axis=2)
    base=target.mean(axis=0).reshape(len(lags),4).argmax(axis=1)
    per_lag=np.mean(prediction==ytest,axis=0)
    frequency=np.mean(ytest==base,axis=0)
    arrays=dict(train_symbols=train,test_symbols=test,train_features=xtrain,test_features=xtest,
                ytrain=ytrain,ytest=ytest,mean=ridge.mean,scale=ridge.scale,weights=ridge.weights,
                bias=ridge.target_mean,predictions=prediction,scores=scores,frequency=base)
    return arrays,per_lag,frequency


def summarize(rows,c):
    data=pd.DataFrame(rows)
    blocks=data.groupby(['seed','level','arm'])[['accuracy','frequency_accuracy','frozen_accuracy']].mean().reset_index()
    intact=blocks[blocks.arm=='intact'].pivot(index='seed',columns='level',values='accuracy')
    scale=(intact['brain5']-intact['legacy5']).tolist()
    pathway={}
    for arm in ('intact','dan_cut','matched_cut'):
        pathway[arm]=blocks[(blocks.level=='brain5')&(blocks.arm==arm)].set_index('seed').accuracy
    contrast=(pathway['matched_cut']-pathway['dan_cut']).tolist()
    all_valid=bool(data[data.arm=='matched_cut'].valid_mass_match.all())
    expansion=bool(len(scale)==3 and np.mean(scale)>=.05 and all(v>0 for v in scale)
                   and all((blocks[blocks.arm=='intact'].accuracy-blocks[blocks.arm=='intact'].frequency_accuracy)>=.05))
    specificity=bool(all_valid and len(contrast)==3 and np.mean(contrast)>=.03 and all(v>0 for v in contrast))
    return blocks,dict(expansion=expansion,pathway_specificity=specificity,
                       brain5_minus_legacy5_by_seed=scale,matched_minus_dan_by_seed=contrast,
                       valid_mass_match=all_valid,cases=len(data))


def run(out,smoke=False):
    c=read(CONFIG);out.mkdir(parents=True,exist_ok=False);budget=Budget(c)
    rows=[];lag_rows=[]
    try:
        core.write_json(out/'config.json',c)
        source=[CONFIG,Path(c['protocol']),Path('scripts/followup4.py'),Path('scripts/verify_followup4.py')]
        source+=sorted(Path('src/flying').rglob('*.py'))
        source+=[p for circuit in c['circuit_seeds'] for p in sorted(Path(f'data/flywire_783_mb_left_kc512_s{circuit}').glob('*'))]
        source+=[Path(c['cache'])/name for name in ('nodes.npz','brain5.npz','provenance.json')]
        core.write_json(out/'source-hashes.json',{p.as_posix():core.sha256(p) for p in source})
        for block in c['blocks'][:1] if smoke else c['blocks']:
            for circuit in c['circuit_seeds'][:1] if smoke else c['circuit_seeds']:
                train=symbols(block,'train',c['warmup']+c['train_samples'])
                test=symbols(block,'test',c['warmup']+c['test_samples'])
                for level in c['levels']:
                    budget.check();base,observed,graph,roles=base_model(c,level,circuit,block)
                    coo,dan,matched,matching=masks(base,roles,block['control_seed']+circuit)
                    intact_head=None;intact_test=None
                    for arm in c['arms']:
                        budget.check();model,removed=arm_model(base,roles,coo,dan,matched,arm)
                        a,per_lag,frequency=train_and_score(c,model,observed,train,test)
                        if arm=='intact':intact_head={key:a[key] for key in ('mean','scale','weights','bias')}
                        frozen_scores=(a['test_features'][c['warmup']:]-intact_head['mean'])/intact_head['scale']@intact_head['weights']+intact_head['bias']
                        frozen=frozen_scores.reshape(len(a['ytest']),len(c['primary_lags']),4).argmax(axis=2)
                        row=dict(seed=block['seed'],circuit=circuit,level=level,arm=arm,
                                 accuracy=float(per_lag.mean()),frequency_accuracy=float(frequency.mean()),
                                 frozen_accuracy=float(np.mean(frozen==a['ytest'])),
                                 graph_sha256=graph['weight_sha256'],weight_sha256=core.weight_hash(model.weights),
                                 removed_edges=int(removed.sum()),removed_mass=float(abs(coo.data[removed]).sum()),
                                 **matching)
                        stem=f's{block["seed"]}_c{circuit}_{level}_{arm}';p=out/stem;p.mkdir()
                        np.savez_compressed(p/'case.npz',**a,frozen_predictions=frozen,
                                            removed_rows=coo.row[removed],removed_cols=coo.col[removed],removed_values=coo.data[removed])
                        core.write_json(p/'metrics.json',row)
                        core.write_json(p/'manifest.json',dict(graph=graph,matching=matching,
                            artifacts={q.name:core.sha256(q) for q in p.iterdir()}))
                        rows.append(row)
                        lag_rows.extend(dict(seed=block['seed'],circuit=circuit,level=level,arm=arm,lag=lag,
                                             accuracy=float(per_lag[j]),frequency_accuracy=float(frequency[j]))
                                        for j,lag in enumerate(c['primary_lags']))
                        del model,a
                    del base
                    print(f'completed s{block["seed"]} c{circuit} {level}',flush=True)
        data=pd.DataFrame(rows);data.to_csv(out/'raw-cases.csv',index=False)
        pd.DataFrame(lag_rows).to_csv(out/'raw-lags.csv',index=False)
        blocks,s=summarize(rows,c)
        blocks.to_csv(out/'seed-blocks.csv',index=False)
        if smoke:s['expansion']=None;s['pathway_specificity']=None
        core.write_json(out/'summary.json',s)
        usage=budget.close();core.write_json(out/'verification.json',dict(complete=True,smoke=smoke,usage=usage))
        core.write_json(out/'manifest.json',dict(config=c,complete=True,smoke=smoke,
            artifacts={p.relative_to(out).as_posix():core.sha256(p) for p in out.rglob('*') if p.is_file()}))
        print(json.dumps(s),flush=True)
    except BaseException:
        budget.close();raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    with threadpool_limits(1):run(args.out,args.smoke)
