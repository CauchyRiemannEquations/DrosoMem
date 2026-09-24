"""Train anatomical synapses on a training prefix, then freeze before readout/evaluation."""
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
from flying.brain.plasticity import KCMBONPlasticity,teacher_codes,weight_hash
from flying.brain.diagnostics import normalize_condition
from flying.evaluation.free_recall import evaluate_recall
from flying.training.phase3 import snapshot

def run(config_path,output):
    cfg=json.loads(Path(config_path).read_text());out=Path(output);out.mkdir(parents=True,exist_ok=True)
    snapshot(out/'config.json',json.dumps(cfg,indent=2));(out/'checkpoints').mkdir(exist_ok=True)
    start=cfg['heldout_start'];count=cfg['heldout_count']
    if max(cfg['lengths'])>=start:raise ValueError('Held-out overlap')
    digits=pi_digits(start+count+1);rows=[];recalls=[];histories=[];sources=[];mixing=[];codes_log=[]
    with threadpool_limits(limits=1):
        for directory in cfg['circuits']:
            a,ids,meta=load_connectome(directory);roles,_=load_roles(directory,ids);circuit=Path(directory).name
            sources.append(dict(circuit=circuit,**meta));mbon=np.flatnonzero(roles=='MBON')
            for seed in cfg['seeds']:
                encoder=KCEncoder(roles,seed,cfg['input_fraction'],cfg['input_amplitude'])
                codes=teacher_codes(len(mbon),seed,cfg['teacher_fraction'],cfg['teacher_amplitude'])
                codes_log.append(dict(circuit=circuit,seed=seed,mbon_ids=[ids[i] for i in mbon],codes=codes.tolist()))
                shuffled,log=role_shuffled(a,roles,seed);mixing.append(dict(circuit=circuit,seed=seed,**log))
                raw={'fly':a,'role_shuffled':shuffled,'role_random':role_random(a,roles,seed)}
                edge_counts=[]
                for matrix in raw.values():
                    c=matrix.tocoo();edge_counts.append(int(((roles[c.row]=='MBON')&(roles[c.col]=='KC')).sum()))
                assert len(set(edge_counts))==1
                for norm in cfg['normalizations']:
                    for model in cfg['models']:
                        initial=normalize_condition(raw[model],norm,cfg['gain'])
                        for steps in cfg['microsteps']:
                            for length in cfg['lengths']:
                                labels=digits[1:length]
                                permuted=np.random.default_rng(np.random.SeedSequence([seed,length,9402])).permutation(labels)
                                for condition in cfg['conditions']:
                                    key=dict(circuit=circuit,seed=seed,normalization=norm,model=model,microsteps=steps,length=length,condition=condition)
                                    trainer=KCMBONPlasticity(initial,roles,encoder,cfg['leak'],steps,cfg['plastic_learning_rate'],cfg['plastic_floor'])
                                    if condition!='frozen':
                                        if condition not in ('supervised','permuted_teacher'):raise ValueError(condition)
                                        target_labels=labels if condition=='supervised' else permuted
                                        history=trainer.fit(digits[:length-1],target_labels,codes,cfg['plastic_epochs'])
                                        histories.extend(dict(**key,**h) for h in history)
                                    audit=trainer.audit()
                                    # NEW fixed reservoir: no teacher signal or plasticity method exposed to evaluation.
                                    reservoir=CircuitReservoir(trainer.weights,encoder,cfg['leak'],steps)
                                    frozen_hash=weight_hash(reservoir.weights)
                                    states=reservoir.states(digits[:-1])
                                    readout=SelectedReadout(mbon)
                                    readout.fit(states[:length-1],labels,epochs=cfg['readout_epochs'],learning_rate=cfg['readout_learning_rate'],l2=cfg['readout_l2'])
                                    recall=evaluate_recall(reservoir,readout,digits,cfg['prompt_length'],length-cfg['prompt_length'])
                                    assert frozen_hash==weight_hash(reservoir.weights)
                                    rows.append(dict(**key,train_accuracy=float(np.mean(readout.predict(states[:length-1])==labels)),
                                        heldout_accuracy=float(np.mean(readout.predict(states[start-1:start+count-1])==digits[start:start+count])),
                                        pi_memory_score=recall['pi_memory_score'],censored=recall['censored'],evaluation_weights_frozen=True,**audit))
                                    recalls.append(dict(**key,**recall))
                                    # Predetermined audit examples, not selected for good performance.
                                    if directory==cfg['circuits'][0] and seed==cfg['seeds'][0] and norm=='incoming_l1' and steps==1 and length==200:
                                        name=f'{model}_{condition}';sparse.save_npz(out/'checkpoints'/f'{name}_weights.npz',reservoir.weights)
                                        np.savez_compressed(out/'checkpoints'/f'{name}_readout.npz',patterns=encoder.patterns,indices=mbon,
                                            weights=readout.model.weights,mean=readout.model.mean,scale=readout.model.scale)
                        print(circuit,seed,norm,model,'runs',len(rows),flush=True)
            snapshot(out/'results.csv',pd.DataFrame(rows).to_csv(index=False));snapshot(out/'recalls.jsonl',''.join(json.dumps(r)+'\n' for r in recalls))
        expected=int(np.prod([len(cfg[k]) for k in ('circuits','seeds','normalizations','models','microsteps','lengths','conditions')]))
        assert len(rows)==len(recalls)==expected
        snapshot(out/'plastic_training.csv',pd.DataFrame(histories).to_csv(index=False))
        snapshot(out/'teacher_codes.json',json.dumps(codes_log,indent=2));snapshot(out/'mixing.json',json.dumps(mixing,indent=2))
        snapshot(out/'manifest.json',json.dumps(dict(completed_runs=len(rows),expected_runs=expected,python=platform.python_version(),
            packages={n:version(n) for n in ['numpy','scipy','pandas','matplotlib','mpmath','threadpoolctl']},sources=sources,
            config_sha256=sha256(out/'config.json'),results_sha256=sha256(out/'results.csv')),indent=2))
