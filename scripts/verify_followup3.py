"""Independent replay of every follow-up-3 state, head, rollout and probe."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import softmax
from threadpoolctl import threadpool_limits

from flying.training import whole_brain_memory as core
from alphabet_memory import SymbolReadout,read
import followup3 as study


def manual_features(model,observed,symbols):
    model.reset();out=[]
    for symbol in symbols:out.append(model.step(int(symbol))[observed].copy())
    return np.asarray(out)


def verify(out,target):
    c=read(target/'config.json');main=read(target/'manifest.json')
    for name,digest in main['artifacts'].items():assert core.sha256(target/name)==digest,name
    for name,digest in read(target/'source-hashes.json').items():assert core.sha256(name)==digest,name
    assert c==main['config']
    out.mkdir(parents=True,exist_ok=False);counts=dict(cases=0,probes=0,rollouts=0,ridge_refits=0)
    for block in c['blocks'][:1] if main['smoke'] else c['blocks']:
        for circuit in c['circuit_seeds'][:1] if main['smoke'] else c['circuit_seeds']:
            for k in c['alphabet_sizes'][:1] if main['smoke'] else c['alphabet_sizes']:
                for family in c['families'][:1] if main['smoke'] else c['families']:
                    for n in c['lengths'][:1] if main['smoke'] else c['lengths']:
                        p=target/f's{block["seed"]}_c{circuit}_k{k}_{family}_n{n}'
                        meta=read(p/'manifest.json')
                        for name,digest in meta['artifacts'].items():assert core.sha256(p/name)==digest
                        model,obs,graph=study.model(c,circuit,block['input_seed'],k)
                        assert graph==meta['graph']
                        with np.load(p/'case.npz',allow_pickle=False) as a:
                            expected=study.sequence(family,k,256,block['sequence_seed']+1000*k+100*['iid','markov','motif'].index(family))[:n]
                            np.testing.assert_array_equal(a['symbols'],expected)
                            np.testing.assert_array_equal(a['features'],manual_features(model,obs,expected[:-1]))
                            head=SymbolReadout(np.arange(48),c['hidden_units'],0,k)
                            head.mean=a['mean'];head.scale=a['scale'];head.parameters={key:a[key] for key in ('w1','b1','w2','b2')}
                            model.reset();state=None
                            for digit in expected[:3]:state=model.step(int(digit))
                            generated=[];prob=[]
                            for _ in range(n-3):
                                value=softmax(head.logits(state[obs])[0]);digit=int(np.argmax(value))
                                prob.append(value);generated.append(digit);state=model.step(digit)
                            np.testing.assert_array_equal(a['generated'],generated)
                            np.testing.assert_allclose(a['probabilities'],prob,atol=1e-15,rtol=0)
                            controls,counts0,trans=study.train_control(expected,k)
                            for name in ('majority','markov1'):np.testing.assert_array_equal(a[name],controls[name])
                            np.testing.assert_array_equal(a['counts'],counts0);np.testing.assert_array_equal(a['transitions'],trans)
                            row=read(p/'metrics.json');assert row['prefix']==core.prefix_score(expected[3:],generated)
                            assert row['control_excess']==row['prefix']-max(core.prefix_score(expected[3:],controls[x]) for x in controls)
                        counts['cases']+=1;counts['rollouts']+=1
                    p=target/f'probe_s{block["seed"]}_c{circuit}_k{k}_{family}'
                    meta=read(p/'manifest.json')
                    for name,digest in meta['artifacts'].items():assert core.sha256(p/name)==digest
                    model,obs,graph=study.model(c,circuit,block['input_seed'],k);assert graph==meta['graph']
                    with np.load(p/'probe.npz',allow_pickle=False) as a:
                        idx=['iid','markov','motif'].index(family)
                        train=study.sequence(family,k,2100,block['probe_train_seed']+1000*k+100*idx)
                        test=study.sequence(family,k,1100,block['probe_test_seed']+1000*k+100*idx)
                        np.testing.assert_array_equal(a['train'],train);np.testing.assert_array_equal(a['test'],test)
                        np.testing.assert_array_equal(a['xtrain'],manual_features(model,obs,train))
                        np.testing.assert_array_equal(a['xtest'],manual_features(model,obs,test))
                        target_labels=train[np.arange(100,len(train))-2];truth=test[np.arange(100,len(test))-2]
                        np.testing.assert_array_equal(a['target'],target_labels);np.testing.assert_array_equal(a['truth'],truth)
                        x=a['xtrain'][100:];z=(x-x.mean(axis=0))/np.maximum(x.std(axis=0),1e-5)
                        y=np.eye(k)[target_labels];b=y.mean(axis=0);alpha=c['alpha']
                        coef=np.linalg.lstsq(np.vstack([z,np.sqrt(alpha)*np.eye(z.shape[1])]),
                                             np.vstack([y-b,np.zeros((z.shape[1],k))]),rcond=None)[0]
                        np.testing.assert_allclose(coef,a['ridge_weights'],atol=1e-10,rtol=1e-10)
                        scores=(a['xtest'][100:]-a['ridge_mean'])/a['ridge_scale']@coef+b
                        pred=scores.argmax(axis=1)
                        np.testing.assert_array_equal(a['pred'],pred)
                        control,freq,table=study.reverse_control(train,test,k)
                        for key,value in [('control',control),('frequency',freq),('reverse_table',table)]:np.testing.assert_array_equal(a[key],value)
                        row=read(p/'metrics.json');assert np.mean(pred==truth)==row['accuracy']
                    counts['probes']+=1;counts['ridge_refits']+=1
    assert counts['cases']==(1 if main['smoke'] else 162)
    assert counts['probes']==1 if main['smoke'] else counts['probes']==54
    checks=dict(all_checks_pass=True,counts=counts,result_manifest_sha256=core.sha256(target/'manifest.json'))
    core.write_json(out/'checks.json',checks)
    core.write_json(out/'manifest.json',dict(artifacts={'checks.json':core.sha256(out/'checks.json')}))
    print(json.dumps(checks),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('result',type=Path);parser.add_argument('--out',type=Path,required=True)
    a=parser.parse_args()
    with threadpool_limits(1):verify(a.out,a.result)
