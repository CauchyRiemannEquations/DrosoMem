"""Temporal diagnostics with matched 48-feature readout budgets."""
import json,platform
from pathlib import Path
from importlib.metadata import version
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from flying.data.connectome import load_connectome,sha256
from flying.data.mushroom_body import load_roles
from flying.data.pi_digits import pi_digits
from flying.brain.mushroom_body import KCEncoder,CircuitReservoir,SelectedReadout,ablate,role_shuffled
from flying.brain.random_network import random_network
from flying.brain.diagnostics import normalize_condition,state_diagnostics
from flying.models.ridge import RidgeReadout
from flying.evaluation.delayed_memory import decode_delays
from flying.evaluation.free_recall import evaluate_recall
from flying.training.phase3 import snapshot

def run(config_path,output):
    cfg=json.loads(Path(config_path).read_text());out=Path(output);out.mkdir(parents=True,exist_ok=True)
    snapshot(out/'config.json',json.dumps(cfg,indent=2))
    start=cfg['heldout_start']; count=cfg['heldout_count'];length=cfg['pi_length']
    if length>=start: raise ValueError('Held-out targets overlap training')
    digits=pi_digits(start+count+1);warmup=cfg['memory_warmup']
    pi_rows=[];memory_rows=[];recalls=[];predictions=[];targets=[];observations=[];sources=[];mixing=[]
    with threadpool_limits(limits=1):
        for directory in cfg['circuits']:
            a,ids,meta=load_connectome(directory);roles,_=load_roles(directory,ids);circuit=Path(directory).name
            sources.append(dict(circuit=circuit,**meta))
            for seed in cfg['seeds']:
                # Independent RNG streams; iid inputs and observation sets shared across matched conditions.
                rng=np.random.default_rng(np.random.SeedSequence([seed,901]))
                train_digits=rng.integers(0,10,cfg['memory_train_count']+warmup)
                test_digits=rng.integers(0,10,cfg['memory_test_count']+warmup)
                obs_rng=np.random.default_rng(np.random.SeedSequence([seed,902]))
                views={'mbon':np.flatnonzero(roles=='MBON'),'sampled48':np.sort(obs_rng.choice(len(ids),48,replace=False))}
                assert len(views['mbon'])==48
                for name,ix in views.items():
                    observations.append(dict(circuit=circuit,seed=seed,view=name,root_ids=[ids[i] for i in ix],
                        role_counts={r:int(np.sum(roles[ix]==r)) for r in np.unique(roles)}))
                encoder=KCEncoder(roles,seed,cfg['input_fraction'],cfg['input_amplitude'])
                shuffled,log=role_shuffled(a,roles,seed);mixing.append(dict(circuit=circuit,seed=seed,**log))
                raw={'fly':a,'role_shuffled':shuffled,'random':random_network(a,seed)}
                for norm in cfg['normalizations']:
                    matrices={name:normalize_condition(w,norm,cfg['gain']) for name,w in raw.items()}
                    matrices['leaky_only']=ablate(matrices['fly'],roles,'leaky_only')
                    for model in cfg['models']:
                        for microsteps in cfg['microsteps']:
                            reservoir=CircuitReservoir(matrices[model],encoder,cfg['leak'],microsteps)
                            pi_states=reservoir.states(digits[:-1])
                            train_states=reservoir.states(train_digits)  # reset, independent iid training
                            test_states=reservoir.states(test_digits)    # reset, independent iid test
                            for view in cfg['views']:
                                ix=views[view];key=dict(circuit=circuit,seed=seed,normalization=norm,model=model,microsteps=microsteps,view=view)
                                metrics,pred,truth=decode_delays(train_states[:,ix],test_states[:,ix],train_digits,test_digits,
                                    cfg['lags'],warmup,cfg['memory_ridge_alpha'])
                                index=len(predictions);predictions.append(pred);targets.append(truth)
                                diagnostics=state_diagnostics(train_states[warmup:,ix])
                                memory_rows.extend(dict(**key,**m,**diagnostics,prediction_index=index) for m in metrics)
                                for kind in cfg['pi_readouts']:
                                    if kind=='softmax':
                                        readout=SelectedReadout(ix)
                                        readout.fit(pi_states[:length-1],digits[1:length],epochs=cfg['epochs'],learning_rate=cfg['learning_rate'],l2=cfg['l2'])
                                    elif kind=='ridge':
                                        readout=RidgeReadout(ix,cfg['pi_ridge_alpha']);readout.fit(pi_states[:length-1],digits[1:length])
                                    else:raise ValueError(kind)
                                    recall=evaluate_recall(reservoir,readout,digits,cfg['prompt_length'],length-cfg['prompt_length'])
                                    pi_rows.append(dict(**key,readout=kind,train_accuracy=float(np.mean(readout.predict(pi_states[:length-1])==digits[1:length])),
                                        heldout_accuracy=float(np.mean(readout.predict(pi_states[start-1:start+count-1])==digits[start:start+count])),
                                        pi_memory_score=recall['pi_memory_score'],censored=recall['censored'],features=len(ix)))
                                    recalls.append(dict(**key,readout=kind,**recall))
                        print(circuit,seed,norm,model,'pi runs',len(pi_rows),flush=True)
            snapshot(out/'pi.csv',pd.DataFrame(pi_rows).to_csv(index=False))
            snapshot(out/'memory.csv',pd.DataFrame(memory_rows).to_csv(index=False))
            snapshot(out/'recalls.jsonl',''.join(json.dumps(r)+'\n' for r in recalls))
        expected=int(np.prod([len(cfg[k]) for k in ('circuits','seeds','normalizations','models','microsteps','views')]))
        assert len(pi_rows)==expected*len(cfg['pi_readouts']) and len(memory_rows)==expected*len(cfg['lags'])
        np.savez_compressed(out/'delayed_predictions.npz',predictions=np.array(predictions),targets=np.array(targets))
        snapshot(out/'observations.json',json.dumps(observations,indent=2));snapshot(out/'mixing.json',json.dumps(mixing,indent=2))
        snapshot(out/'manifest.json',json.dumps(dict(pi_runs=len(pi_rows),memory_conditions=expected,memory_lag_measurements=len(memory_rows),
            python=platform.python_version(),packages={n:version(n) for n in ['numpy','scipy','pandas','matplotlib','mpmath','threadpoolctl']},
            sources=sources,config_sha256=sha256(out/'config.json'),pi_sha256=sha256(out/'pi.csv'),memory_sha256=sha256(out/'memory.csv')),indent=2))
