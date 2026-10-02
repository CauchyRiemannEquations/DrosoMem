"""Independent test-stream replay and all-head refit for follow-up 4."""
import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from flying.training import whole_brain_memory as core
from alphabet_memory import read
import followup4 as study


def replay(model,observed,symbols):
    model.reset();features=[]
    for symbol in symbols:features.append(model.step(int(symbol))[observed].copy())
    return np.asarray(features)


def verify(result,out):
    main=read(result/'manifest.json');c=main['config']
    for name,digest in main['artifacts'].items():assert core.sha256(result/name)==digest,name
    for name,digest in read(result/'source-hashes.json').items():assert core.sha256(name)==digest,name
    assert c==read(result/'config.json')
    out.mkdir(parents=True,exist_ok=False);counts=dict(cases=0,test_trajectories=0,ridge_refits=0,matched_controls=0)
    for block in c['blocks'][:1] if main['smoke'] else c['blocks']:
        for circuit in c['circuit_seeds'][:1] if main['smoke'] else c['circuit_seeds']:
            train=study.symbols(block,'train',c['warmup']+c['train_samples'])
            test=study.symbols(block,'test',c['warmup']+c['test_samples'])
            for level in c['levels']:
                base,observed,graph,roles=study.base_model(c,level,circuit,block)
                coo,dan,matched,matching=study.masks(base,roles,block['control_seed']+circuit)
                intact=None
                for arm in c['arms']:
                    model,removed=study.arm_model(base,roles,coo,dan,matched,arm)
                    p=result/f's{block["seed"]}_c{circuit}_{level}_{arm}'
                    meta=read(p/'manifest.json');assert meta['graph']==graph and meta['matching']==matching
                    for name,digest in meta['artifacts'].items():assert core.sha256(p/name)==digest
                    row=read(p/'metrics.json')
                    assert row['weight_sha256']==core.weight_hash(model.weights)
                    assert row['removed_edges']==int(removed.sum())
                    assert row['removed_mass']==float(abs(coo.data[removed]).sum())
                    with np.load(p/'case.npz',allow_pickle=False) as a:
                        for key,value in [('train_symbols',train),('test_symbols',test),
                                          ('removed_rows',coo.row[removed]),('removed_cols',coo.col[removed]),
                                          ('removed_values',coo.data[removed])]:np.testing.assert_array_equal(a[key],value)
                        np.testing.assert_array_equal(a['test_features'],replay(model,observed,test))
                        counts['test_trajectories']+=1
                        w=c['warmup'];lags=c['primary_lags']
                        ytrain=np.asarray([[train[t-l] for l in lags] for t in range(w,len(train))])
                        ytest=np.asarray([[test[t-l] for l in lags] for t in range(w,len(test))])
                        np.testing.assert_array_equal(a['ytrain'],ytrain)
                        np.testing.assert_array_equal(a['ytest'],ytest)
                        x=a['train_features'][w:];mean=x.mean(axis=0);scale=np.maximum(x.std(axis=0),1e-5)
                        np.testing.assert_array_equal(a['mean'],mean);np.testing.assert_array_equal(a['scale'],scale)
                        z=(x-mean)/scale;target=np.eye(4)[ytrain].reshape(len(ytrain),-1)
                        bias=target.mean(axis=0)
                        weights=np.linalg.lstsq(np.vstack([z,np.sqrt(c['alpha'])*np.eye(z.shape[1])]),
                            np.vstack([target-bias,np.zeros((z.shape[1],target.shape[1]))]),rcond=None)[0]
                        np.testing.assert_allclose(a['weights'],weights,atol=1e-10,rtol=1e-10)
                        np.testing.assert_allclose(a['bias'],bias,atol=1e-15,rtol=0)
                        scores=((a['test_features'][w:]-mean)/scale@weights+bias).reshape(len(ytest),len(lags),4)
                        pred=scores.argmax(axis=2)
                        np.testing.assert_array_equal(a['predictions'],pred)
                        assert abs(np.mean(pred==ytest)-row['accuracy'])<1e-15
                        if arm=='intact':intact=(mean,scale,weights,bias)
                        m,s,coef,b=intact
                        frozen=((a['test_features'][w:]-m)/s@coef+b).reshape(len(ytest),len(lags),4).argmax(axis=2)
                        np.testing.assert_array_equal(a['frozen_predictions'],frozen)
                        assert abs(np.mean(frozen==ytest)-row['frozen_accuracy'])<1e-15
                    counts['cases']+=1;counts['ridge_refits']+=1
                    if arm=='matched_cut':counts['matched_controls']+=1
                print(f'verified s{block["seed"]} c{circuit} {level}',flush=True)
    assert counts['cases']==(6 if main['smoke'] else 36)
    checks=dict(all_checks_pass=True,counts=counts,result_manifest_sha256=core.sha256(result/'manifest.json'))
    core.write_json(out/'checks.json',checks)
    core.write_json(out/'manifest.json',dict(artifacts={'checks.json':core.sha256(out/'checks.json')}))
    print(json.dumps(checks),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('result',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    with threadpool_limits(1):verify(a.result,a.out)
