"""Preregistered length × alphabet × family sequence-memory experiment."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from flying.models.ridge import RidgeDecoder
from flying.training import whole_brain_memory as core
from alphabet_memory import SymbolReadout, read
from frozen_state_probe import Budget
from structural_k4 import build as build_partial

CONFIG = Path('configs/followup3.json')


def sequence(family, k, n, seed):
    rng = np.random.default_rng(seed)
    if family == 'iid':
        return rng.integers(0, k, n, dtype=np.int64)
    if family == 'markov':
        result = np.empty(n, np.int64); result[0] = rng.integers(k)
        for i in range(1, n):
            result[i] = result[i-1] if rng.random() < .7 else (result[i-1] + 1 + rng.integers(k-1)) % k
        return result
    if family == 'motif':
        motif = rng.integers(0, k, 8, dtype=np.int64)
        return motif[np.arange(n) % 8]
    raise ValueError(family)


def train_control(symbols, k, prompt=3):
    counts = np.bincount(symbols[1:], minlength=k)
    trans = np.zeros((k,k), dtype=np.int64)
    np.add.at(trans, (symbols[:-1], symbols[1:]), 1)
    majority = int(counts.argmax())
    preds = {}
    for name in ('majority', 'markov1'):
        previous = int(symbols[prompt-1]); out = []
        for _ in range(len(symbols)-prompt):
            previous = majority if name == 'majority' or trans[previous].sum() == 0 else int(trans[previous].argmax())
            out.append(previous)
        preds[name] = np.asarray(out, np.int64)
    return preds, counts, trans


def reverse_control(train, test, k):
    counts = np.bincount(train[100-2:len(train)-2], minlength=k)
    table = np.zeros((k,k), dtype=np.int64)
    for t in range(100, len(train)): table[train[t], train[t-2]] += 1
    majority = int(counts.argmax())
    pred = np.asarray([int(table[s].argmax()) if table[s].sum() else majority for s in test[100:]], np.int64)
    freq = np.full(len(pred), majority, dtype=np.int64)
    return pred, freq, table


def model(c, circuit, input_seed, k):
    reservoir, observed, graph, _ = build_partial(c, circuit, input_seed, k=k)
    assert len(observed) == 48
    return reservoir, observed, graph


def train_case(c, block, circuit, k, family, n, root):
    reservoir, observed, graph = model(c,circuit,block['input_seed'],k)
    symbols = sequence(family,k,256,block['sequence_seed'] + 1000*k + 100*['iid','markov','motif'].index(family))[:n]
    features, _ = core.collect(reservoir,observed,symbols[:-1],c['activity_epsilon'])
    seed = block['seed'] + 1000*k + 100*['iid','markov','motif'].index(family) + n
    head = SymbolReadout(np.arange(48),c['hidden_units'],seed,k)
    weights = core.sample_weights(n-1,3,c['prefix_window'],c['prefix_weight'])
    _,history = head.fit(features,symbols[1:],epochs=c['epochs'],learning_rate=c['learning_rate'],l2=c['l2'],
                         checkpoints=(c['epochs'],),sample_weight=weights)
    generated,probs,_ = core.rollout(reservoir,observed,head,symbols[:3],n-3)
    controls,counts,trans = train_control(symbols,k)
    score = core.prefix_score(symbols[3:],generated)
    control_scores = {name:core.prefix_score(symbols[3:],value) for name,value in controls.items()}
    row = dict(seed=block['seed'],circuit=circuit,k=k,family=family,length=n,prefix=score,
               majority_prefix=control_scores['majority'],markov1_prefix=control_scores['markov1'],
               control_excess=score-max(control_scores.values()),teacher_accuracy=float(np.mean(head.predict(features)==symbols[1:])),
               graph_sha256=graph['weight_sha256'])
    arrays = dict(symbols=symbols,features=features,generated=generated,probabilities=probs,
                  mean=head.mean,scale=head.scale,**head.parameters,majority=controls['majority'],
                  markov1=controls['markov1'],counts=counts,transitions=trans)
    np.savez_compressed(root/'case.npz',**arrays)
    core.write_json(root/'metrics.json',row);core.write_json(root/'training.json',history)
    core.write_json(root/'manifest.json',dict(graph=graph,artifacts={p.name:core.sha256(p) for p in root.iterdir()}))
    return row


def probe_case(c,block,circuit,k,family,root):
    reservoir,observed,graph=model(c,circuit,block['input_seed'],k)
    family_index=['iid','markov','motif'].index(family)
    train=sequence(family,k,2100,block['probe_train_seed']+1000*k+100*family_index)
    test=sequence(family,k,1100,block['probe_test_seed']+1000*k+100*family_index)
    xtrain,_=core.collect(reservoir,observed,train,c['activity_epsilon'])
    xtest,_=core.collect(reservoir,observed,test,c['activity_epsilon'])
    target=train[np.arange(100,len(train))-2];truth=test[np.arange(100,len(test))-2]
    ridge=RidgeDecoder(c['alpha']).fit(xtrain[100:],np.eye(k)[target])
    pred=ridge.scores(xtest[100:]).argmax(axis=1)
    control,freq,table=reverse_control(train,test,k)
    accuracy=float(np.mean(pred==truth));base=max(float(np.mean(control==truth)),float(np.mean(freq==truth)))
    row=dict(seed=block['seed'],circuit=circuit,k=k,family=family,accuracy=accuracy,
             reverse_markov_accuracy=float(np.mean(control==truth)),frequency_accuracy=float(np.mean(freq==truth)),
             excess=accuracy-base,phase8_oracle_accuracy=float(np.mean(test[(np.arange(100,len(test))-2)%8]==truth)) if family=='motif' else None,
             graph_sha256=graph['weight_sha256'])
    np.savez_compressed(root/'probe.npz',train=train,test=test,xtrain=xtrain,xtest=xtest,target=target,truth=truth,
                        ridge_mean=ridge.mean,ridge_scale=ridge.scale,ridge_weights=ridge.weights,
                        ridge_bias=ridge.target_mean,pred=pred,control=control,frequency=freq,reverse_table=table)
    core.write_json(root/'metrics.json',row)
    core.write_json(root/'manifest.json',dict(graph=graph,artifacts={p.name:core.sha256(p) for p in root.iterdir()}))
    return row


def summary(rows,probes,c):
    a=pd.DataFrame(rows);b=pd.DataFrame(probes)
    agg=a.groupby(['family','length','k'])[['prefix','majority_prefix','markov1_prefix','control_excess','teacher_accuracy']].mean().reset_index()
    pairs=b.groupby(['seed','k','family'])[['accuracy','reverse_markov_accuracy','frequency_accuracy','excess']].mean().reset_index()
    gates={}
    if len(b)==54:
        for family in ('iid','markov'):
            for cohort,seeds in [('discovery',[810001,810002]),('confirmation',[810003])]:
                sample=pairs[(pairs.family==family)&(pairs.k==10)&pairs.seed.isin(seeds)]
                gates[f'{cohort}/{family}']=bool(len(sample)==len(seeds) and (sample.excess>=.05).all())
    return a,b,agg,pairs,dict(gates=gates,primary_confirmed=bool(gates) and all(gates.values()),
                               cases=len(a),independent_probes=len(b))


def run(out,smoke=False):
    c=read(CONFIG);out.mkdir(parents=True,exist_ok=False);budget=Budget(c)
    rows=[];probes=[]
    try:
        core.write_json(out/'config.json',c)
        source_paths=[CONFIG,Path(c['protocol']),Path('scripts/followup3.py'),Path('scripts/verify_followup3.py')]
        source_paths+=sorted(Path('src/flying').rglob('*.py'))
        source_paths+=[p for circuit in c['circuit_seeds'] for p in sorted(Path(f'data/flywire_783_mb_left_kc512_s{circuit}').glob('*'))]
        core.write_json(out/'source-hashes.json',{p.as_posix():core.sha256(p) for p in source_paths})
        for block in c['blocks'][:1] if smoke else c['blocks']:
            for circuit in c['circuit_seeds'][:1] if smoke else c['circuit_seeds']:
                for k in c['alphabet_sizes'][:1] if smoke else c['alphabet_sizes']:
                    for family in c['families'][:1] if smoke else c['families']:
                        for n in c['lengths'][:1] if smoke else c['lengths']:
                            budget.check();p=out/f's{block["seed"]}_c{circuit}_k{k}_{family}_n{n}';p.mkdir()
                            rows.append(train_case(c,block,circuit,k,family,n,p))
                        p=out/f'probe_s{block["seed"]}_c{circuit}_k{k}_{family}';p.mkdir()
                        probes.append(probe_case(c,block,circuit,k,family,p))
                print(f'completed block {block["seed"]} circuit {circuit}',flush=True)
        a,b,agg,pairs,s=summary(rows,probes,c)
        a.to_csv(out/'raw-cases.csv',index=False);b.to_csv(out/'raw-probes.csv',index=False)
        agg.to_csv(out/'grid-summary.csv',index=False);pairs.to_csv(out/'seed-probes.csv',index=False)
        core.write_json(out/'summary.json',s)
        usage=budget.close();core.write_json(out/'verification.json',dict(complete=True,smoke=smoke,usage=usage))
        core.write_json(out/'manifest.json',dict(config=c,complete=True,smoke=smoke,
            artifacts={p.relative_to(out).as_posix():core.sha256(p) for p in out.rglob('*') if p.is_file()}))
        print(json.dumps(s),flush=True)
    except BaseException:
        budget.close();raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--smoke',action='store_true')
    a=parser.parse_args()
    with threadpool_limits(1):run(a.out,a.smoke)
