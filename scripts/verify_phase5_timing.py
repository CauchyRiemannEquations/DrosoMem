import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from flying.brain.timed_reservoir import TimedReservoir
from flying.brain.mushroom_body import KCEncoder,SelectedReadout,role_shuffled
from flying.brain.diagnostics import normalize_condition
from flying.brain.plasticity import weight_hash
from flying.brain.reward_plasticity import policy_hash
from flying.data.connectome import load_connectome,sha256
from flying.data.mushroom_body import load_roles
from flying.data.pi_digits import pi_digits
from flying.evaluation.free_recall import evaluate_recall
from flying.evaluation.delayed_memory import decode_delays


def verify(directory):
    root=Path(directory);cfg=json.loads((root/'config.json').read_text());manifest=json.loads((root/'manifest.json').read_text())
    for name,h in manifest['file_sha256'].items():assert sha256(root/name)==h,name
    frame=pd.read_csv(root/'pi.csv');memory=pd.read_csv(root/'memory.csv');recalls=[json.loads(s) for s in (root/'recalls.jsonl').read_text().splitlines()]
    keys=['circuit','seed','normalization','model','schedule']
    expected=int(np.prod([len(cfg[k]) for k in ['circuits','seeds','normalizations','models','schedules']]))
    assert len(frame)==len(recalls)==expected and len(memory)==expected*len(cfg['lags'])
    assert not frame.duplicated(keys).any() and not memory.duplicated(keys+['lag']).any()
    archive=np.load(root/'delayed_predictions.npz');saved=np.load(root/'readouts.npz')
    digits=pi_digits(cfg['pi_length']);labels=digits[1:];bykey={tuple(r[k] for k in keys):r for r in recalls}
    for row in frame.to_dict('records'):
        ix=row['prediction_index'];r=bykey[tuple(row[k] for k in keys)]
        assert len(r['target'])==len(r['prediction'])==cfg['pi_length']-cfg['prompt_length']
        score=next((i for i,(a,b) in enumerate(zip(r['target'],r['prediction'])) if a!=b),len(r['target']))
        assert score==row['pi_memory_score']==row['teacher_forced_prefix']==r['pi_memory_score']
        assert r['censored']==(score==r['horizon'])
        assert r['first_error_digit_index']==(None if score==r['horizon'] else cfg['prompt_length']+score)
        pred=saved[f'teacher_prediction_{ix}'][cfg['prompt_length']-1:]
        target=digits[cfg['prompt_length']:]
        teacher_score=next((i for i,(a,b) in enumerate(zip(pred,target)) if a!=b),len(target))
        assert teacher_score==score
        for j,lag in enumerate(cfg['lags']):
            logged=memory[(memory.prediction_index==ix)&(memory.lag==lag)].iloc[0]
            assert np.isclose(logged.accuracy,np.mean(archive['predictions'][ix,:,j]==archive['targets'][ix,:,j]),atol=1e-15,rtol=0)
    replays=0;probes=0
    with threadpool_limits(limits=1):
        for directory in cfg['circuits']:
            a,ids,_=load_connectome(directory);roles,_=load_roles(directory,ids);mbon=np.flatnonzero(roles=='MBON')
            circuit=Path(directory).name
            for seed in cfg['seeds']:
                encoder=KCEncoder(roles,seed,cfg['input_fraction'],cfg['input_amplitude']);shuffled,_=role_shuffled(a,roles,seed)
                rng=np.random.default_rng(np.random.SeedSequence([seed,9801]));warm=cfg['memory_warmup']
                train=rng.integers(0,10,warm+cfg['memory_train_count']);test=rng.integers(0,10,warm+cfg['memory_test_count'])
                for norm in cfg['normalizations']:
                    for model,w in [('fly',a),('role_shuffled',shuffled)]:
                        weights=normalize_condition(w,norm,cfg['gain']);weights.sort_indices()
                        for schedule in cfg['schedules']:
                            key=dict(circuit=circuit,seed=seed,normalization=norm,model=model,schedule=schedule)
                            selected=frame
                            for k,v in key.items():selected=selected[selected[k]==v]
                            assert len(selected)==1;row=selected.iloc[0];ix=int(row.prediction_index)
                            r=TimedReservoir(weights,encoder,roles,cfg['leak'],schedule);states=r.states(digits[:-1])
                            readout=SelectedReadout(mbon);readout.fit(states,labels,epochs=cfg['readout_epochs'],learning_rate=cfg['readout_learning_rate'],l2=cfg['readout_l2'])
                            assert policy_hash(readout)==row.readout_sha256 and weight_hash(weights)==row.weight_sha256
                            for name in ['mean','scale','weights']:assert np.array_equal(getattr(readout.model,name),saved[f'{name}_{ix}'])
                            assert np.array_equal(readout.predict(states),saved[f'teacher_prediction_{ix}'])
                            got=evaluate_recall(r,readout,digits,cfg['prompt_length'],len(digits)-cfg['prompt_length'])
                            assert got['prediction']==bykey[tuple(key[k] for k in keys)]['prediction'];replays+=1
                            metrics,pred,truth=decode_delays(r.states(train)[:,mbon],r.states(test)[:,mbon],train,test,cfg['lags'],warm,cfg['memory_ridge_alpha'])
                            assert np.array_equal(pred,archive['predictions'][ix]) and np.array_equal(truth,archive['targets'][ix])
                            for metric in metrics:
                                logged=memory[(memory.prediction_index==ix)&(memory.lag==metric['lag'])].iloc[0]
                                for k in ['accuracy','frequency_baseline_accuracy','r2_vs_training_frequency']:assert np.isclose(logged[k],metric[k],atol=1e-12,rtol=0)
                            probes+=len(metrics)
    result=dict(pi_runs_rebuilt_and_replayed=replays,all_readout_arrays_exact=True,all_recall_strings_exact=True,
                first_error_equivalence_checked=expected,lag_decoders_independently_refitted=probes,
                iid_digit_predictions_checked=expected*len(cfg['lags'])*cfg['memory_test_count'],all_iid_predictions_exact=True)
    (root/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');a=p.parse_args();verify(a.directory)
