"""Paired scheduling experiment on fixed connectome reservoirs."""
import json,time,platform
from pathlib import Path
from importlib.metadata import version
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
from flying.evaluation.metrics import pi_memory_score
from flying.evaluation.delayed_memory import decode_delays
from flying.training.phase5_diagnostic import save_archive


def run(config_path,output):
    cfg=json.loads(Path(config_path).read_text());out=Path(output);out.mkdir(parents=True,exist_ok=False)
    (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    source_hashes={str(p):sha256(p) for p in sorted(Path('src/flying').rglob('*.py'))}
    digits=pi_digits(cfg['pi_length']);labels=digits[1:];warmup=cfg['memory_warmup']
    rows=[];memory=[];recalls=[];predictions=[];targets=[];checkpoints={};sources=[];mixing=[]
    started=time.monotonic()
    with threadpool_limits(limits=1):
        for directory in cfg['circuits']:
            a,ids,meta=load_connectome(directory);roles,_=load_roles(directory,ids);mbon=np.flatnonzero(roles=='MBON')
            assert len(mbon)==48
            circuit=Path(directory).name;sources.append(dict(circuit=circuit,**meta))
            for seed in cfg['seeds']:
                encoder=KCEncoder(roles,seed,cfg['input_fraction'],cfg['input_amplitude'])
                shuffled,log=role_shuffled(a,roles,seed);graphs={'fly':a,'role_shuffled':shuffled};mixing.append(dict(circuit=circuit,seed=seed,**log))
                rng=np.random.default_rng(np.random.SeedSequence([seed,9801]))
                train_digits=rng.integers(0,10,warmup+cfg['memory_train_count'])
                test_digits=rng.integers(0,10,warmup+cfg['memory_test_count'])
                for norm in cfg['normalizations']:
                    for model in cfg['models']:
                        weights=normalize_condition(graphs[model],norm,cfg['gain']);weights.sort_indices();before=weight_hash(weights)
                        for schedule in cfg['schedules']:
                            key=dict(circuit=circuit,seed=seed,normalization=norm,model=model,schedule=schedule)
                            r=TimedReservoir(weights,encoder,roles,cfg['leak'],schedule)
                            states=r.states(digits[:-1]);readout=SelectedReadout(mbon)
                            readout.fit(states,labels,epochs=cfg['readout_epochs'],learning_rate=cfg['readout_learning_rate'],l2=cfg['readout_l2'])
                            before_readout=policy_hash(readout)
                            teacher_prediction=readout.predict(states)
                            teacher_prefix=pi_memory_score(digits[cfg['prompt_length']:],teacher_prediction[cfg['prompt_length']-1:])
                            recall=evaluate_recall(r,readout,digits,cfg['prompt_length'],len(digits)-cfg['prompt_length'])
                            assert recall['pi_memory_score']==teacher_prefix
                            assert before_readout==policy_hash(readout) and before==weight_hash(r.weights)
                            ix=len(rows)
                            for name in ['mean','scale','weights']:checkpoints[f'{name}_{ix}']=getattr(readout.model,name)
                            checkpoints[f'teacher_prediction_{ix}']=teacher_prediction.astype(np.uint8)
                            rows.append(dict(**key,prediction_index=ix,pi_memory_score=recall['pi_memory_score'],teacher_forced_prefix=teacher_prefix,
                                train_accuracy=float(np.mean(teacher_prediction==labels)),weight_sha256=before,readout_sha256=before_readout,features=len(mbon)))
                            recalls.append(dict(**key,**recall))
                            train_states=r.states(train_digits)[:,mbon];test_states=r.states(test_digits)[:,mbon]
                            metrics,pred,truth=decode_delays(train_states,test_states,train_digits,test_digits,cfg['lags'],warmup,cfg['memory_ridge_alpha'])
                            assert before==weight_hash(r.weights)
                            memory.extend(dict(**key,prediction_index=ix,**m) for m in metrics);predictions.append(pred);targets.append(truth)
                        print(f'{len(rows)}/72 pi runs, {len(memory)} delay measurements; {time.monotonic()-started:.1f}s',flush=True)
    pd.DataFrame(rows).to_csv(out/'pi.csv',index=False);pd.DataFrame(memory).to_csv(out/'memory.csv',index=False)
    (out/'recalls.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in recalls))
    (out/'mixing.json').write_text(json.dumps(mixing,indent=2)+'\n')
    save_archive(out/'readouts.npz',checkpoints)
    save_archive(out/'delayed_predictions.npz',dict(predictions=np.array(predictions),targets=np.array(targets)))
    expected=int(np.prod([len(cfg[k]) for k in ['circuits','seeds','normalizations','models','schedules']]))
    assert len(rows)==expected and len(memory)==expected*len(cfg['lags'])
    manifest=dict(source_commit=cfg['source_commit'],sources=sources,code_sha256=source_hashes,python=platform.python_version(),
                  packages={n:version(n) for n in ['numpy','scipy','pandas','threadpoolctl']},pi_runs=len(rows),memory_measurements=len(memory),
                  pi_generator='mpmath when installed; verified decimal AGM fallback otherwise',
                  elapsed_seconds=time.monotonic()-started,file_sha256={p.name:sha256(p) for p in sorted(out.iterdir()) if p.is_file()})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
