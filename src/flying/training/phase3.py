"""Predeclared mushroom-body experiment; only readouts are optimized."""
import json, platform, time
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from flying.data.connectome import load_connectome, sha256
from flying.data.mushroom_body import load_roles
from flying.data.pi_digits import pi_digits
from flying.brain.mushroom_body import KCEncoder, CircuitReservoir, SelectedReadout, ablate, role_shuffled
from flying.brain.random_network import random_network
from flying.brain.diagnostics import normalize_condition, graph_diagnostics
from flying.evaluation.free_recall import evaluate_recall

def snapshot(path, text):
    tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(text); tmp.replace(path)

def run(config_path, output):
    cfg=json.loads(Path(config_path).read_text()); out=Path(output); out.mkdir(parents=True,exist_ok=True)
    snapshot(out/'config.json',json.dumps(cfg,indent=2))
    start=cfg['heldout_start']; count=cfg['heldout_count']; digits=pi_digits(start+count+1)
    if max(cfg['lengths'])>=start: raise ValueError('Training overlaps held-out targets')
    rows=[]; recalls=[]; mixing=[]; histories=[]; provenance=[]
    with threadpool_limits(limits=1):
        for directory in cfg['circuits']:
            a,ids,meta=load_connectome(directory); roles,_=load_roles(directory,ids)
            circuit=Path(directory).name; provenance.append(dict(circuit=circuit,**meta))
            for seed in cfg['seeds']:
                encoder=KCEncoder(roles,seed,cfg["input_fraction"],cfg["input_amplitude"])
                shuffled,log=role_shuffled(a,roles,seed)
                mixing.append(dict(circuit=circuit,seed=seed,**log))
                raw={'fly':a,'role_shuffled':shuffled,'random':random_network(a,seed)}
                for norm in cfg['normalizations']:
                    matrices={k:normalize_condition(v,norm,cfg['gain']) for k,v in raw.items()}
                    for name in ('no_dan','no_feedback','leaky_only'):
                        matrices[name]=ablate(matrices['fly'],roles,name)
                    for model in cfg['models']:
                        reservoir=CircuitReservoir(matrices[model],encoder,cfg['leak'],cfg['microsteps'])
                        states=reservoir.states(digits[:-1])
                        if seed==cfg['seeds'][0] and model=='fly' and norm=='spectral':
                            np.savez_compressed(out/(circuit+'_activity.npz'),states=states[:100],roles=np.asarray(roles,dtype="U8"))
                        for length in cfg['lengths']:
                            for view in cfg['views']:
                                tick=time.perf_counter()
                                ix=np.flatnonzero(roles=='MBON') if view=='mbon' else np.arange(len(roles))
                                readout=SelectedReadout(ix)
                                history=readout.fit(states[:length-1],digits[1:length],epochs=cfg['epochs'],learning_rate=cfg['learning_rate'],l2=cfg['l2'])
                                recall=evaluate_recall(reservoir,readout,digits,cfg['prompt_length'],length-cfg['prompt_length'])
                                key=dict(circuit=circuit,seed=seed,normalization=norm,model=model,length=length,view=view)
                                row=dict(**key,train_accuracy=float(np.mean(readout.predict(states[:length-1])==digits[1:length])),
                                    heldout_accuracy=float(np.mean(readout.predict(states[start-1:start+count-1])==digits[start:start+count])),
                                    pi_memory_score=recall['pi_memory_score'],censored=recall['censored'],
                                    seconds=time.perf_counter()-tick,readout_neurons=len(ix),**graph_diagnostics(matrices[model]))
                                rows.append(row); recalls.append(dict(**key,**recall))
                                if seed==cfg['seeds'][0] and length==200 and circuit==Path(cfg['circuits'][0]).name:
                                    histories.extend(dict(**key,**h) for h in history)
                        print(circuit,seed,norm,model,'runs',len(rows),flush=True)
            snapshot(out/'results.csv',pd.DataFrame(rows).to_csv(index=False))
            snapshot(out/'recalls.jsonl',''.join(json.dumps(r)+'\n' for r in recalls))
    expected=int(np.prod([len(cfg[k]) for k in ('circuits','seeds','normalizations','models','lengths','views')]))
    assert len(rows)==len(recalls)==expected
    snapshot(out/'mixing.json',json.dumps(mixing,indent=2))
    snapshot(out/'training.csv',pd.DataFrame(histories).to_csv(index=False))
    snapshot(out/'manifest.json',json.dumps(dict(expected_runs=expected,completed_runs=len(rows),
        python=platform.python_version(),numpy=np.__version__,sources=provenance,
        config_sha256=sha256(out/'config.json'),results_sha256=sha256(out/'results.csv')) ,indent=2))
    return pd.DataFrame(rows)
