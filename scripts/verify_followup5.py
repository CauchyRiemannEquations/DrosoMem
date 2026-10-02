"""Independent LTD reconstruction, LIF probe replay and lag-2 refit."""
import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from flying.brain.dopamine import replay_depression
from flying.brain.lif import LIFReservoir
from flying.training import whole_brain_memory as core
from alphabet_memory import read
import followup5 as study


def verify(result,out):
    main=read(result/'manifest.json');c=main['config']
    for name,digest in main['artifacts'].items():assert core.sha256(result/name)==digest,name
    for name,digest in read(result/'source-hashes.json').items():assert core.sha256(name)==digest,name
    out.mkdir(parents=True,exist_ok=False)
    counts=dict(cases=0,closed_form_updates=0,response_replays=0,lag_ridge_refits=0)
    for seed in c['seeds'][:1] if main['smoke'] else c['seeds']:
        for circuit in c['circuits'][:1] if main['smoke'] else c['circuits']:
            weights,roles,mapping,audit,encoder=study.setup(c,circuit,seed)
            mbon=mapping['gamma1_pedc']['MBON']
            baseline=study.response(weights,roles,encoder,c,mbon)
            for arm in c['arms']:
                p=result/f's{seed}_c{circuit}_{arm}';meta=read(p/'manifest.json')
                assert meta['mapping']==audit and meta['parameters']==study.schedule(c,arm)
                for name,digest in meta['artifacts'].items():assert core.sha256(p/name)==digest
                row=read(p/'metrics.json')
                with np.load(p/'case.npz',allow_pickle=False) as a:
                    for key,value in [('initial_data',weights.data),('indices',weights.indices),
                                      ('indptr',weights.indptr),('probe_before',baseline)]:
                        np.testing.assert_array_equal(a[key],value)
                    rate=0. if arm=='blocked' else c['learning_rate']
                    spikes=dict(spike_ticks=a['spike_ticks'],spike_indices=a['spike_indices'])
                    fresh=replay_depression(weights,roles,mbon,mapping['gamma1_pedc']['DAN'],spikes,
                        dt_ms=.1,eligibility_ms=c['eligibility_ms'],learning_rate=rate,
                        floor_fraction=c['floor_fraction'])
                    np.testing.assert_allclose(a['adapted_data'],fresh.data,atol=1e-10,rtol=0)
                    adapted=weights.copy();adapted.data[:]=a['adapted_data']
                    assert row['adapted_weight_sha256']==core.weight_hash(adapted)
                    assert row['changed_edges']==int(np.count_nonzero(adapted.data!=weights.data))
                    counts['closed_form_updates']+=1
                    after=study.response(adapted,roles,encoder,c,mbon)
                    np.testing.assert_array_equal(a['probe_after'],after)
                    np.testing.assert_array_equal(a['drive_before'],study.drive(weights,encoder,c,mbon))
                    np.testing.assert_array_equal(a['drive_after'],study.drive(adapted,encoder,c,mbon))
                    assert row['cs_plus_after']==int(after[0].sum())
                    assert row['cs_minus_after']==int(after[1].sum())
                    assert row['fixed_anatomical_accuracy']==study.fixed_accuracy(after)
                    counts['response_replays']+=1
                    if arm in ('paired','no_dopamine'):
                        train=a['lag_train'];test=a['lag_test'];observed=np.flatnonzero(np.asarray(roles)=='MBON')
                        simulator=LIFReservoir(adapted,encoder,roles)
                        xtrain=simulator.states(train)[:,observed];xtest=simulator.states(test)[:,observed]
                        np.testing.assert_array_equal(a['lag_xtrain'],xtrain)
                        np.testing.assert_array_equal(a['lag_xtest'],xtest)
                        w=c['lag_warmup'];ytrain=train[np.arange(w,len(train))-2];ytest=test[np.arange(w,len(test))-2]
                        np.testing.assert_array_equal(a['lag_ytrain'],ytrain)
                        np.testing.assert_array_equal(a['lag_ytest'],ytest)
                        x=xtrain[w:];mean=x.mean(axis=0);scale=np.maximum(x.std(axis=0),1e-5)
                        z=(x-mean)/scale;target=np.eye(4)[ytrain];bias=target.mean(axis=0)
                        coefficient=np.linalg.lstsq(np.vstack([z,np.sqrt(c['lag_alpha'])*np.eye(z.shape[1])]),
                            np.vstack([target-bias,np.zeros((z.shape[1],4))]),rcond=None)[0]
                        np.testing.assert_allclose(a['lag_weights'],coefficient,atol=1e-10,rtol=1e-10)
                        np.testing.assert_array_equal(a['lag_pred'],((xtest[w:]-mean)/scale@coefficient+bias).argmax(axis=1))
                        assert row['lag2_accuracy']==float(np.mean(a['lag_pred']==ytest))
                        counts['lag_ridge_refits']+=1
                counts['cases']+=1
                print(f'verified s{seed} c{circuit} {arm}',flush=True)
    assert counts['cases']==(5 if main['smoke'] else 30)
    checks=dict(all_checks_pass=True,counts=counts,result_manifest_sha256=core.sha256(result/'manifest.json'))
    core.write_json(out/'checks.json',checks)
    core.write_json(out/'manifest.json',dict(artifacts={'checks.json':core.sha256(out/'checks.json')}))
    print(json.dumps(checks),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('result',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    with threadpool_limits(1):verify(a.result,a.out)
