"""Reward stage with fixed warm-started policy and separate refit diagnostic."""
import json,platform
from pathlib import Path
from importlib.metadata import version
import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits
from flying.data.connectome import load_connectome,sha256
from flying.data.mushroom_body import load_roles
from flying.data.pi_digits import pi_digits
from flying.brain.mushroom_body import KCEncoder,CircuitReservoir,SelectedReadout,role_shuffled
from flying.brain.role_random import role_random
from flying.brain.reward_plasticity import RewardPlasticity,policy_hash
from flying.brain.plasticity import weight_hash
from flying.brain.diagnostics import normalize_condition
from flying.evaluation.free_recall import evaluate_recall
from flying.training.phase3 import snapshot

def fit_readout(states,labels,indices,cfg):
    ro=SelectedReadout(indices);ro.fit(states,labels,epochs=cfg['readout_epochs'],learning_rate=cfg['readout_learning_rate'],l2=cfg['readout_l2']);return ro

def run(config_path,output):
    cfg=json.loads(Path(config_path).read_text());out=Path(output);out.mkdir(parents=True,exist_ok=True);(out/'checkpoints').mkdir(exist_ok=True)
    snapshot(out/'config.json',json.dumps(cfg,indent=2));length=cfg['pi_length'];start=cfg['heldout_start'];count=cfg['heldout_count']
    if length>=start:raise ValueError('Held-out overlap')
    if cfg['conditions'].index('reward_trace')>cfg['conditions'].index('yoked_reward'):raise ValueError('Trace run must precede yoked control')
    digits=pi_digits(start+count+1);labels=digits[1:length]
    rows=[];recalls=[];histories=[];sources=[];mixing=[];event_logs=[];event_keys=[]
    with threadpool_limits(limits=1):
        for directory in cfg['circuits']:
            a,ids,meta=load_connectome(directory);roles,_=load_roles(directory,ids);circuit=Path(directory).name
            sources.append(dict(circuit=circuit,**meta));mbon=np.flatnonzero(roles=='MBON')
            for seed in cfg['seeds']:
                encoder=KCEncoder(roles,seed,cfg['input_fraction'],cfg['input_amplitude'])
                shuffled,log=role_shuffled(a,roles,seed);mixing.append(dict(circuit=circuit,seed=seed,**log))
                raw={'fly':a,'role_shuffled':shuffled,'role_random':role_random(a,roles,seed)}
                for norm in cfg['normalizations']:
                    for model in cfg['models']:
                        initial=normalize_condition(raw[model],norm,cfg['gain']);initial.sort_indices()
                        baseline=CircuitReservoir(initial,encoder,cfg['leak'],cfg['microsteps'])
                        policy=fit_readout(baseline.states(digits[:length-1]),labels,mbon,cfg);initial_policy_hash=policy_hash(policy)
                        for delay in cfg['reward_delays']:
                            trace_events=None
                            for condition in cfg['conditions']:
                                key=dict(circuit=circuit,seed=seed,normalization=norm,model=model,reward_delay=delay,condition=condition)
                                decay=0. if condition=='reward_no_trace' else cfg['trace_decay']
                                trainer=RewardPlasticity(initial,roles,encoder,policy,cfg['leak'],cfg['microsteps'],cfg['reward_learning_rate'],
                                    cfg['plastic_floor'],decay,delay,cfg['temperature'],cfg['baseline_rate'])
                                event_index=-1
                                if condition!='frozen':
                                    if condition not in ('reward_trace','reward_no_trace','yoked_reward'):raise ValueError(condition)
                                    yoked=None
                                    if condition=='yoked_reward':
                                        rng=np.random.default_rng(np.random.SeedSequence([seed,delay,9502]))
                                        yoked=np.array([rng.permutation(row) for row in trace_events['true_rewards']])
                                    history,events=trainer.fit_reward(digits[:length-1],labels,cfg['reward_epochs'],seed=np.random.SeedSequence([seed,delay,9501]),yoked_rewards=yoked)
                                    if condition=='reward_trace':trace_events=events
                                    event_index=len(event_logs);event_logs.append(events);event_keys.append(dict(**key,event_index=event_index))
                                    histories.extend(dict(**key,**h) for h in history)
                                audit=trainer.audit();assert initial_policy_hash==policy_hash(policy)
                                reservoir=CircuitReservoir(trainer.weights,encoder,cfg['leak'],cfg['microsteps']);frozen_hash=weight_hash(reservoir.weights)
                                states=reservoir.states(digits[:-1])
                                refit=fit_readout(states[:length-1],labels,mbon,cfg)
                                if condition=='frozen':assert policy_hash(refit)==initial_policy_hash
                                for view in cfg['readouts']:
                                    ro={'frozen_policy':policy,'refit_readout':refit}[view]
                                    recall=evaluate_recall(reservoir,ro,digits,cfg['prompt_length'],length-cfg['prompt_length'])
                                    assert frozen_hash==weight_hash(reservoir.weights)
                                    rows.append(dict(**key,readout=view,event_index=event_index,train_accuracy=float(np.mean(ro.predict(states[:length-1])==labels)),
                                        heldout_accuracy=float(np.mean(ro.predict(states[start-1:start+count-1])==digits[start:start+count])),
                                        pi_memory_score=recall['pi_memory_score'],censored=recall['censored'],evaluation_weights_frozen=True,
                                        policy_frozen_during_reward=True,initial_policy_sha256=initial_policy_hash,**audit))
                                    recalls.append(dict(**key,readout=view,**recall))
                                    if directory==cfg['circuits'][0] and seed==cfg['seeds'][0] and norm=='incoming_l1' and model=='fly' and delay==3:
                                        name=f'{condition}_{view}';np.savez_compressed(out/'checkpoints'/f'{name}_readout.npz',patterns=encoder.patterns,
                                            indices=mbon,weights=ro.model.weights,mean=ro.model.mean,scale=ro.model.scale)
                                if directory==cfg['circuits'][0] and seed==cfg['seeds'][0] and norm=='incoming_l1' and model=='fly' and delay==3:
                                    sparse.save_npz(out/'checkpoints'/f'{condition}_weights.npz',reservoir.weights)
                        print(circuit,seed,norm,model,'evaluations',len(rows),flush=True)
            snapshot(out/'results.csv',pd.DataFrame(rows).to_csv(index=False));snapshot(out/'recalls.jsonl',''.join(json.dumps(r)+'\n' for r in recalls))
        networks=int(np.prod([len(cfg[k]) for k in ('circuits','seeds','normalizations','models','reward_delays','conditions')]))
        assert len(rows)==len(recalls)==networks*len(cfg['readouts'])
        np.savez_compressed(out/'reward_events.npz',**{k:np.array([e[k] for e in event_logs]) for k in ['actions','true_rewards','applied_rewards']})
        snapshot(out/'event_keys.json',json.dumps(event_keys,indent=2));snapshot(out/'reward_training.csv',pd.DataFrame(histories).to_csv(index=False))
        snapshot(out/'mixing.json',json.dumps(mixing,indent=2))
        snapshot(out/'manifest.json',json.dumps(dict(network_conditions=networks,evaluations=len(rows),reward_training_runs=len(event_logs),
            python=platform.python_version(),packages={n:version(n) for n in ['numpy','scipy','pandas','matplotlib','mpmath','threadpoolctl']},sources=sources,
            config_sha256=sha256(out/'config.json'),results_sha256=sha256(out/'results.csv')),indent=2))
