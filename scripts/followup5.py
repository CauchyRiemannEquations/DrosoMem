"""Fixed-parameter γ1/pedc LTD plausibility and internal-learning assay."""
import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits

from flying.brain.dopamine import DopamineLIF,replay_depression
from flying.brain.lif import LIFReservoir
from flying.models.ridge import RidgeDecoder
from flying.training import whole_brain_memory as core
from flying.training.phase5b import circuit_setup
from alphabet_memory import read

CONFIG=Path('configs/followup5.json')


def setup(c,circuit,seed):
    local=dict(c,seed=seed)
    weights,roles,mapping,audit,encoder=circuit_setup(Path(f'data/flywire_783_mb_left_kc512_s{circuit}'),local)
    return weights,roles,mapping,audit,encoder


def schedule(c,arm):
    params=dict(digit=c['conditioned_digit'],cs_start_ms=0.,cs_duration_ms=c['cs_ms'],
                dan_times_ms=c['forward_dan_ms'],duration_ms=c['conditioning_ms'])
    if arm=='backward':params.update(cs_start_ms=c['backward_cs_start_ms'],dan_times_ms=c['backward_dan_ms'])
    if arm=='no_dopamine':params['dan_times_ms']=[]
    if arm=='wrong_compartment':params['dan_compartment']='alpha3'
    return params


def response(weights,roles,encoder,c,mbon):
    r=LIFReservoir(weights,encoder,roles)
    per=[]
    for digit in (c['conditioned_digit'],c['control_digit']):
        states=r.states([digit]*c['probe_digits'])
        per.append(states[:,mbon].sum(axis=1).astype(np.int64))
    del r;gc.collect()
    return np.asarray(per)


def drive(weights,encoder,c,mbon):
    result=[]
    for digit in (c['conditioned_digit'],c['control_digit']):
        active=np.flatnonzero(encoder.patterns[digit])
        result.append(float(weights[mbon,:][:,active].sum()))
    return np.asarray(result)


def fixed_accuracy(probe):
    # Anatomy-fixed binary rule: no MBON11 spike predicts the conditioned CS+.
    return float((np.count_nonzero(probe[0]==0)+np.count_nonzero(probe[1]>0))/(2*probe.shape[1]))


def lag_probe(weights,roles,encoder,c,seed,circuit):
    observed=np.flatnonzero(np.asarray(roles)=='MBON')
    assert len(observed)==48
    rng1=np.random.default_rng(seed+831000+circuit);rng2=np.random.default_rng(seed+832000+circuit)
    train=rng1.integers(0,4,c['lag_warmup']+c['lag_train_samples'],dtype=np.int64)
    test=rng2.integers(0,4,c['lag_warmup']+c['lag_test_samples'],dtype=np.int64)
    r=LIFReservoir(weights,encoder,roles)
    xtrain=r.states(train)[:,observed];xtest=r.states(test)[:,observed]
    del r;gc.collect()
    w=c['lag_warmup'];ytrain=train[np.arange(w,len(train))-2];ytest=test[np.arange(w,len(test))-2]
    ridge=RidgeDecoder(c['lag_alpha']).fit(xtrain[w:],np.eye(4)[ytrain])
    pred=ridge.scores(xtest[w:]).argmax(axis=1)
    return dict(train=train,test=test,xtrain=xtrain,xtest=xtest,ytrain=ytrain,ytest=ytest,
                mean=ridge.mean,scale=ridge.scale,weights=ridge.weights,bias=ridge.target_mean,pred=pred),float(np.mean(pred==ytest))


def one_case(c,circuit,seed,arm,root,base_probe=None,base_drive=None):
    weights,roles,mapping,audit,encoder=setup(c,circuit,seed)
    mbon=mapping['gamma1_pedc']['MBON'];assert len(mbon)==1
    if base_probe is None:base_probe=response(weights,roles,encoder,c,mbon)
    if base_drive is None:base_drive=drive(weights,encoder,c,mbon)
    rate=0. if arm=='blocked' else c['learning_rate']
    learner=DopamineLIF(weights,encoder,roles,mapping,eligibility_ms=c['eligibility_ms'],
                         learning_rate=rate,floor_fraction=c['floor_fraction'])
    adapted,spikes,gates=learner.condition(**schedule(c,arm))
    replay=replay_depression(weights,roles,mbon,mapping['gamma1_pedc']['DAN'],spikes,
                             dt_ms=learner.parameters.dt_ms,eligibility_ms=c['eligibility_ms'],
                             learning_rate=rate,floor_fraction=c['floor_fraction'])
    np.testing.assert_allclose(replay.data,adapted.data,atol=1e-10,rtol=0)
    mask=learner.rule.mask
    assert np.array_equal(weights.data[~mask],adapted.data[~mask])
    after=response(adapted,roles,encoder,c,mbon)
    after_drive=drive(adapted,encoder,c,mbon)
    plus_before=int(base_probe[0].sum());minus_before=int(base_probe[1].sum())
    plus_after=int(after[0].sum());minus_after=int(after[1].sum())
    plus_loss=None if plus_before==0 else 1-plus_after/plus_before
    minus_change=None if minus_before==0 else abs(minus_after/minus_before-1)
    changed=int(np.count_nonzero(adapted.data!=weights.data))
    row=dict(circuit=circuit,seed=seed,arm=arm,plastic_edges=int(mask.sum()),changed_edges=changed,
             matching_dan_spikes=int(np.isin(spikes['spike_indices'],mapping['gamma1_pedc']['DAN']).sum()),
             cs_plus_before=plus_before,cs_plus_after=plus_after,cs_minus_before=minus_before,
             cs_minus_after=minus_after,cs_plus_suppression=plus_loss,cs_minus_absolute_change=minus_change,
             cs_plus_drive_before=float(base_drive[0]),cs_plus_drive_after=float(after_drive[0]),
             cs_minus_drive_before=float(base_drive[1]),cs_minus_drive_after=float(after_drive[1]),
             fixed_anatomical_accuracy=fixed_accuracy(after),
             initial_weight_sha256=core.weight_hash(weights),adapted_weight_sha256=core.weight_hash(adapted),
             lag2_accuracy=None)
    arrays=dict(initial_data=weights.data,adapted_data=adapted.data,indices=weights.indices,indptr=weights.indptr,
                plastic_mask=mask,probe_before=base_probe,probe_after=after,drive_before=base_drive,drive_after=after_drive,
                spike_ticks=spikes['spike_ticks'],spike_indices=spikes['spike_indices'])
    if arm in ('paired','no_dopamine'):
        lag,accuracy=lag_probe(adapted,roles,encoder,c,seed,circuit)
        arrays.update({'lag_'+key:value for key,value in lag.items()})
        row['lag2_accuracy']=accuracy
    np.savez_compressed(root/'case.npz',**arrays)
    core.write_json(root/'metrics.json',row);core.write_json(root/'gates.json',gates)
    core.write_json(root/'manifest.json',dict(mapping=audit,parameters=schedule(c,arm),
        artifacts={p.name:core.sha256(p) for p in root.iterdir()}))
    del learner;gc.collect()
    return row,base_probe,base_drive


def summarize(rows,smoke):
    f=pd.DataFrame(rows);blocks=[];criteria={}
    for (seed,circuit),sub in f.groupby(['seed','circuit']):
        arms={r.arm:r for r in sub.itertuples()}
        paired=arms['paired'];nulls=all(arms[a].changed_edges==0 for a in ('backward','no_dopamine','wrong_compartment','blocked'))
        response=bool(paired.cs_plus_before>0 and paired.cs_minus_before>0 and
                      paired.cs_plus_suppression>=.2 and paired.cs_minus_absolute_change<=.1)
        fixed_gain=paired.fixed_anatomical_accuracy-arms['no_dopamine'].fixed_anatomical_accuracy
        blocks.append(dict(seed=seed,circuit=circuit,causal_nulls=nulls,response_gate=response,
                           fixed_gain=fixed_gain,paired_lag2=paired.lag2_accuracy,
                           no_dopamine_lag2=arms['no_dopamine'].lag2_accuracy))
    b=pd.DataFrame(blocks)
    criteria=dict(functional_response=bool(len(b)==6 and b.causal_nulls.all() and b.response_gate.all()),
                  useful_fixed_readout=bool(len(b)==6 and (b.fixed_gain>=.05).all()),cases=len(f))
    if smoke:criteria['functional_response']=None;criteria['useful_fixed_readout']=None
    return f,b,criteria


def run(out,smoke=False):
    c=read(CONFIG);out.mkdir(parents=True,exist_ok=False);start=time.perf_counter();process=psutil.Process();peak=0
    core.write_json(out/'config.json',c)
    source=[CONFIG,Path(c['protocol']),Path(c['registry']),Path('scripts/followup5.py'),Path('scripts/verify_followup5.py')]
    source+=sorted(Path('src/flying').rglob('*.py'))
    source+=[p for circuit in c['circuits'] for p in sorted(Path(f'data/flywire_783_mb_left_kc512_s{circuit}').glob('*'))]
    core.write_json(out/'source-hashes.json',{p.as_posix():core.sha256(p) for p in source})
    rows=[]
    for seed in c['seeds'][:1] if smoke else c['seeds']:
        for circuit in c['circuits'][:1] if smoke else c['circuits']:
            before=None;drive0=None
            for arm in c['arms']:
                peak=max(peak,process.memory_info().rss)
                assert time.perf_counter()-start<c['max_seconds'] and peak<c['max_rss_bytes'],'Resource budget exceeded; partial output preserved'
                p=out/f's{seed}_c{circuit}_{arm}';p.mkdir()
                row,before,drive0=one_case(c,circuit,seed,arm,p,before,drive0)
                rows.append(row)
                print(f'completed s{seed} c{circuit} {arm}',flush=True)
    f,b,s=summarize(rows,smoke)
    f.to_csv(out/'raw-cases.csv',index=False);b.to_csv(out/'seed-blocks.csv',index=False)
    core.write_json(out/'summary.json',s)
    usage=dict(seconds=time.perf_counter()-start,peak_sampled_rss_bytes=max(peak,process.memory_info().rss))
    core.write_json(out/'verification.json',dict(complete=True,smoke=smoke,usage=usage))
    core.write_json(out/'manifest.json',dict(config=c,complete=True,smoke=smoke,
        artifacts={p.relative_to(out).as_posix():core.sha256(p) for p in out.rglob('*') if p.is_file()}))
    print(json.dumps(s),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    with threadpool_limits(1):run(a.out,a.smoke)
