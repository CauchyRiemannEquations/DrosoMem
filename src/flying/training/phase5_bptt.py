"""Bounded positive-control study; no hyperparameter or recall-score selection."""
import json
import platform
import time
from pathlib import Path
from importlib.metadata import version
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from flying.brain.constrained_bptt import ConstrainedBPTT
from flying.brain.diagnostics import normalize_condition
from flying.brain.mushroom_body import KCEncoder,CircuitReservoir,SelectedReadout,role_shuffled
from flying.brain.plasticity import weight_hash
from flying.brain.reward_plasticity import policy_hash
from flying.data.connectome import load_connectome,sha256
from flying.data.mushroom_body import load_roles
from flying.data.pi_digits import pi_digits
from flying.evaluation.free_recall import evaluate_recall
from flying.training.phase5_diagnostic import save_archive


def run(config_path,output):
    cfg=json.loads(Path(config_path).read_text());out=Path(output);out.mkdir(parents=True,exist_ok=False)
    (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    code_hashes={str(p):sha256(p) for p in sorted(Path('src/flying').rglob('*.py'))}
    digits=pi_digits(cfg['pi_length']);labels=digits[1:]
    rows=[];recalls=[];histories=[];checkpoints={};sources=[];condition_keys=[];mixing=[]
    started=time.monotonic();completed=0
    with threadpool_limits(limits=1):
        for directory in cfg['circuits']:
            a,ids,meta=load_connectome(directory);roles,_=load_roles(directory,ids)
            circuit=Path(directory).name;mbon=np.flatnonzero(roles=='MBON');sources.append(dict(circuit=circuit,**meta))
            for seed in cfg['seeds']:
                encoder=KCEncoder(roles,seed,cfg['input_fraction'],cfg['input_amplitude'])
                shuffled,log=role_shuffled(a,roles,seed);mixing.append(dict(circuit=circuit,seed=seed,**log))
                graphs={'fly':a,'role_shuffled':shuffled}
                for normalization in cfg['normalizations']:
                    for model in cfg['models']:
                        initial=normalize_condition(graphs[model],normalization,cfg['gain']);initial.sort_indices()
                        base=CircuitReservoir(initial,encoder,cfg['leak'],1)
                        policy=SelectedReadout(mbon)
                        fit_args=dict(epochs=cfg['readout_epochs'],learning_rate=cfg['readout_learning_rate'],l2=cfg['readout_l2'])
                        policy.fit(base.states(digits[:-1]),labels,**fit_args);before=policy_hash(policy)
                        rng=np.random.default_rng(np.random.SeedSequence([seed,9701]));permuted=rng.permutation(labels)
                        for condition in cfg['conditions']:
                            key=dict(circuit=circuit,seed=seed,normalization=normalization,model=model,condition=condition)
                            trainer=ConstrainedBPTT(initial,roles,encoder,policy,cfg['leak'],cfg['floor'])
                            targets=permuted if condition=='permuted_bptt' else labels
                            if condition=='frozen':
                                loss,acc=trainer.objective(digits[:-1],targets,gradient=False)
                                selection=dict(selected_epoch=0,initial_loss=loss,selected_loss=loss,selection='frozen baseline')
                                history=[dict(epoch=0,loss=loss,accuracy=acc,best_loss=loss)]
                            else:
                                history,selection=trainer.fit(digits[:-1],targets,cfg['epochs'],cfg['learning_rate'],
                                    temporal=condition!='local_gradient',theta_cap=cfg['theta_cap'],gradient_clip=cfg['gradient_clip'])
                            assert before==policy_hash(policy)
                            audit=trainer.audit();weights,_,_=trainer.materialize(trainer.theta)
                            reservoir=CircuitReservoir(weights,encoder,cfg['leak'],1);states=reservoir.states(digits[:-1])
                            refit=SelectedReadout(mbon);refit.fit(states,labels,**fit_args)
                            if condition=='frozen':assert policy_hash(refit)==before
                            checkpoint_index=len(condition_keys);condition_keys.append(dict(**key,checkpoint_index=checkpoint_index))
                            checkpoints[f'theta_{checkpoint_index}']=trainer.theta
                            histories.extend(dict(**key,**h) for h in history)
                            for view in cfg['views']:
                                readout=policy if view=='fixed_policy' else refit
                                rohash=policy_hash(readout);whash=weight_hash(weights)
                                recall=evaluate_recall(reservoir,readout,digits,cfg['prompt_length'],len(digits)-cfg['prompt_length'])
                                assert whash==weight_hash(reservoir.weights) and rohash==policy_hash(readout)
                                from scipy.special import logsumexp
                                logits=readout.model.features(states[:,mbon])@readout.model.weights
                                true_loss=float(np.mean(logsumexp(logits,axis=1)-logits[np.arange(len(labels)),labels]))
                                rows.append(dict(**key,view=view,checkpoint_index=checkpoint_index,pi_memory_score=recall['pi_memory_score'],
                                    train_accuracy=float(np.mean(logits.argmax(axis=1)==labels)),true_train_cross_entropy=true_loss,
                                    readout_sha256=rohash,**selection,**audit))
                                recalls.append(dict(**key,view=view,**recall))
                            completed+=1
                        print(f'{completed}/96 network conditions; {len(rows)} evaluations; {time.monotonic()-started:.1f}s',flush=True)
    pd.DataFrame(rows).to_csv(out/'results.csv',index=False)
    pd.DataFrame(histories).to_csv(out/'training.csv',index=False)
    (out/'recalls.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in recalls))
    (out/'checkpoint_keys.json').write_text(json.dumps(condition_keys,indent=2)+'\n')
    (out/'mixing.json').write_text(json.dumps(mixing,indent=2)+'\n')
    save_archive(out/'checkpoints.npz',checkpoints)
    expected=len(cfg['circuits'])*len(cfg['seeds'])*len(cfg['normalizations'])*len(cfg['models'])*len(cfg['conditions'])*len(cfg['views'])
    assert len(rows)==expected
    manifest=dict(source_commit=cfg['source_commit'],sources=sources,code_sha256=code_hashes,python=platform.python_version(),
        packages={n:version(n) for n in ['numpy','scipy','pandas','mpmath','threadpoolctl']},evaluations=len(rows),network_conditions=completed,
        learned_conditions=sum(k['condition']!='frozen' for k in condition_keys),elapsed_seconds=time.monotonic()-started,
        file_sha256={p.name:sha256(p) for p in sorted(out.iterdir()) if p.is_file()})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return pd.DataFrame(rows)
