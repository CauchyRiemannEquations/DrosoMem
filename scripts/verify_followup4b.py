"""Structural-invariant and independent neural/refit audit for 4B."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits

from flying.brain.timed_reservoir import TimedReservoir
from flying.training import whole_brain_memory as core
from alphabet_memory import SymbolEncoder,read
import followup4 as previous
import followup4b as study


def verify(result,out):
    main=read(result/'manifest.json');c=main['config']
    for name,digest in main['artifacts'].items():assert core.sha256(result/name)==digest,name
    for name,digest in read(result/'source-hashes.json').items():assert core.sha256(name)==digest,name
    graphroot=Path(c['graph']);graphmeta=read(graphroot/'manifest.json')
    for name,digest in graphmeta['artifacts'].items():assert core.sha256(graphroot/name)==digest,name
    original=sparse.load_npz(Path(c['cache'])/'brain5.npz').tocsr()
    shuffled=sparse.load_npz(graphroot/'shuffled-raw.npz').tocsr()
    with np.load(Path(c['cache'])/'nodes.npz',allow_pickle=False) as f:roles=f['roles']
    audit=read(graphroot/'audit.json')
    assert study.structural_checks(original,shuffled,roles,audit['swap_log'])==audit
    weights,_=study.graph(c)
    parent=Path(c['parent']);prior=read(parent/'manifest.json')
    for name,digest in prior['artifacts'].items():assert core.sha256(parent/name)==digest,name
    out.mkdir(parents=True,exist_ok=False);counts=dict(cases=0,test_trajectories=0,ridge_refits=0,structural_invariant_sets=1)
    for block in c['blocks'][:1] if main['smoke'] else c['blocks']:
        for circuit in c['circuit_seeds'][:1] if main['smoke'] else c['circuit_seeds']:
            base,observed,info,roles=previous.base_model(c,c['level'],circuit,block)
            model=TimedReservoir(weights,SymbolEncoder(base.encoder.patterns.copy()),roles,c['leak'],c['schedule'])
            train=previous.symbols(block,'train',c['warmup']+c['train_samples'])
            test=previous.symbols(block,'test',c['warmup']+c['test_samples'])
            p=result/f's{block["seed"]}_c{circuit}';meta=read(p/'manifest.json')
            for name,digest in meta['artifacts'].items():assert core.sha256(p/name)==digest
            assert meta['parent_case_sha256']==core.sha256(parent/f's{block["seed"]}_c{circuit}_brain5_intact/case.npz')
            row=read(p/'metrics.json')
            with np.load(p/'case.npz',allow_pickle=False) as a:
                np.testing.assert_array_equal(a['train_symbols'],train)
                np.testing.assert_array_equal(a['test_symbols'],test)
                model.reset();features=[]
                for symbol in test:features.append(model.step(int(symbol))[observed].copy())
                np.testing.assert_array_equal(a['test_features'],features)
                counts['test_trajectories']+=1
                w=c['warmup'];lags=c['primary_lags'];ytrain=np.asarray([[train[t-l] for l in lags] for t in range(w,len(train))])
                ytest=np.asarray([[test[t-l] for l in lags] for t in range(w,len(test))])
                np.testing.assert_array_equal(a['ytrain'],ytrain);np.testing.assert_array_equal(a['ytest'],ytest)
                x=a['train_features'][w:];mean=x.mean(axis=0);scale=np.maximum(x.std(axis=0),1e-5)
                np.testing.assert_array_equal(a['mean'],mean);np.testing.assert_array_equal(a['scale'],scale)
                target=np.eye(4)[ytrain].reshape(len(ytrain),-1);bias=target.mean(axis=0)
                z=(x-mean)/scale
                coef=np.linalg.lstsq(np.vstack([z,np.sqrt(c['alpha'])*np.eye(z.shape[1])]),
                    np.vstack([target-bias,np.zeros((z.shape[1],target.shape[1]))]),rcond=None)[0]
                np.testing.assert_allclose(a['weights'],coef,atol=1e-10,rtol=1e-10)
                scores=((a['test_features'][w:]-mean)/scale@coef+bias).reshape(len(ytest),len(lags),4)
                pred=scores.argmax(axis=2)
                np.testing.assert_array_equal(a['predictions'],pred)
                assert row['shuffled_accuracy']==float(np.mean(pred==ytest))
                counts['ridge_refits']+=1
            counts['cases']+=1
            print(f'verified s{block["seed"]} c{circuit}',flush=True)
    assert counts['cases']==(1 if main['smoke'] else 6)
    checks=dict(all_checks_pass=True,counts=counts,result_manifest_sha256=core.sha256(result/'manifest.json'),
                graph_manifest_sha256=core.sha256(graphroot/'manifest.json'))
    core.write_json(out/'checks.json',checks)
    core.write_json(out/'manifest.json',dict(artifacts={'checks.json':core.sha256(out/'checks.json')}))
    print(json.dumps(checks),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('result',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    with threadpool_limits(1):verify(a.result,a.out)
