"""Matched-parameter decoder experiment on identical, frozen MBON states."""
import hashlib,json,time,platform
from pathlib import Path
from importlib.metadata import version
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
from flying.training.phase5_timing import (TimedReservoir,KCEncoder,SelectedReadout,role_shuffled,normalize_condition,
    weight_hash,policy_hash,load_connectome,sha256,load_roles,pi_digits,evaluate_recall,pi_memory_score,save_archive)
from flying.models.nonlinear_readout import NonlinearReadout

KEYS=['circuit','seed','normalization','model','schedule']

def conditions(cfg):
    for directory in cfg['circuits']:
        a,ids,_=load_connectome(directory);roles,_=load_roles(directory,ids);mbon=np.flatnonzero(roles=='MBON')
        for seed in cfg['seeds']:
            encoder=KCEncoder(roles,seed,cfg['input_fraction'],cfg['input_amplitude'])
            shuffled,_=role_shuffled(a,roles,seed)
            for norm in cfg['normalizations']:
                for model in cfg['models']:
                    weights=normalize_condition({'fly':a,'role_shuffled':shuffled}[model],norm,cfg['gain']);weights.sort_indices()
                    for schedule in cfg['schedules']:
                        key=dict(circuit=Path(directory).name,seed=seed,normalization=norm,model=model,schedule=schedule)
                        yield key,TimedReservoir(weights,encoder,roles,cfg['leak'],schedule),mbon

def nonlinear_seed(seed,initialization):return np.random.SeedSequence([seed,9901,initialization])

def head_arrays(head):
    if isinstance(head,NonlinearReadout):return dict(mean=head.mean,scale=head.scale,**head.parameters)
    return {name:getattr(head.model,name) for name in ['mean','scale','weights']}

def digest(head):return head.digest() if isinstance(head,NonlinearReadout) else policy_hash(head)

def logits(head,states):
    if isinstance(head,NonlinearReadout):return head.logits(states)
    return head.model.features(states[:,head.indices])@head.model.weights

def run(config_path,output):
    cfg=json.loads(Path(config_path).read_text());out=Path(output);out.mkdir(parents=True,exist_ok=False)
    (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    source_hashes={str(p):sha256(p) for p in sorted(Path('src/flying').rglob('*.py'))}
    digits=pi_digits(cfg['pi_length']);labels=digits[1:];rows=[];recalls=[];histories=[];archive={};started=time.monotonic()
    expected=int(np.prod([len(cfg[k]) for k in ['circuits','seeds','normalizations','models','schedules']]))
    with threadpool_limits(limits=1):
        for condition_index,(key,reservoir,mbon) in enumerate(conditions(cfg)):
            states=reservoir.states(digits[:-1]);before=weight_hash(reservoir.weights)
            state_hash=hashlib.sha256(states.tobytes()).hexdigest();mean=states[:,mbon].mean(axis=0);scale=np.maximum(states[:,mbon].std(axis=0),1e-5)
            def record(head,kind,initialization,epoch):
                arrays=head_arrays(head)
                assert np.array_equal(arrays['mean'],mean) and np.array_equal(arrays['scale'],scale)
                pred=head.predict(states);raw=logits(head,states);ce=float(np.mean(logsumexp(raw,axis=1)-raw[np.arange(len(labels)),labels]))
                h=digest(head);recall=evaluate_recall(reservoir,head,digits,cfg['prompt_length'],len(digits)-cfg['prompt_length'])
                teacher=pi_memory_score(digits[cfg['prompt_length']:],pred[cfg['prompt_length']-1:])
                assert recall['pi_memory_score']==teacher and h==digest(head) and before==weight_hash(reservoir.weights)
                ix=len(rows);parameters=head.parameter_count if kind=='nonlinear' else head.model.weights.size
                row=dict(**key,evaluation_index=ix,readout_type=kind,initialization=initialization,epochs=epoch,parameter_count=parameters,
                         pi_memory_score=teacher,teacher_forced_prefix=teacher,train_accuracy=float(np.mean(pred==labels)),cross_entropy=ce,
                         weight_sha256=before,state_sha256=state_hash,readout_sha256=h)
                rows.append(row);recalls.append(dict(evaluation_index=ix,**recall))
                for name,array in arrays.items():archive[f'{name}_{ix}']=array
                archive[f'teacher_prediction_{ix}']=pred.astype(np.uint8)
            for epoch in cfg['checkpoints']:
                head=SelectedReadout(mbon);history=head.fit(states,labels,epochs=epoch,learning_rate=cfg['learning_rate'],l2=cfg['l2'])
                record(head,'affine',-1,epoch)
                if epoch==max(cfg['checkpoints']):histories.extend(dict(**key,readout_type='affine',initialization=-1,**h) for h in history if h['epoch']==1 or h['epoch']%50==0)
            for initialization in cfg['initializations']:
                head=NonlinearReadout(mbon,cfg['hidden_units'],nonlinear_seed(key['seed'],initialization))
                saved,history=head.fit(states,labels,epochs=max(cfg['checkpoints']),learning_rate=cfg['learning_rate'],l2=cfg['l2'],checkpoints=cfg['checkpoints'])
                histories.extend(dict(**key,readout_type='nonlinear',initialization=initialization,**h) for h in history)
                for epoch in cfg['checkpoints']:record(saved[epoch],'nonlinear',initialization,epoch)
            print(f'{condition_index+1}/{expected} fixed states; {len(rows)} evaluations; {time.monotonic()-started:.1f}s',flush=True)
    assert len(rows)==expected*len(cfg['checkpoints'])*(1+len(cfg['initializations']))
    pd.DataFrame(rows).to_csv(out/'evaluations.csv',index=False);pd.DataFrame(histories).to_csv(out/'training.csv',index=False)
    (out/'recalls.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in recalls));save_archive(out/'readouts.npz',archive)
    manifest=dict(source_commit=cfg['source_commit'],code_sha256=source_hashes,python=platform.python_version(),
        packages={n:version(n) for n in ['numpy','scipy','pandas','threadpoolctl']},fixed_state_conditions=expected,evaluations=len(rows),
        sources=[dict(directory=d,**load_connectome(d)[2]) for d in cfg['circuits']],
        elapsed_seconds=time.monotonic()-started,file_sha256={p.name:sha256(p) for p in sorted(out.iterdir()) if p.is_file()})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
