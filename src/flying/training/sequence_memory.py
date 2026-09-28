"""ACT II four-family comparison, using the unchanged ACT I numerical backend."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pandas as pd
import psutil
from scipy.special import softmax
from threadpoolctl import threadpool_limits
from flying.data.sequences import SequenceDataset
from flying.training import whole_brain_memory as core


def validate(c):
    proxy = dict(c, seeds=[b['model_seed'] for b in c['blocks']])
    core.ExperimentConfig(proxy).validate()
    if set(c['conditions']) != {'legacy5', 'brain1'}:
        raise ValueError('ACT II compares legacy5 and brain1')
    if not c['families'] or not set(c['families']) <= {'pi','random','shuffled_pi','periodic'}:
        raise ValueError('Unknown family')
    if len(set(c['families'])) != len(c['families']):
        raise ValueError('Duplicate families')
    if len({b['dataset_seed'] for b in c['blocks']}) != len(c['blocks']):
        raise ValueError('Duplicate dataset seed')
    if len(c['dataset']['prompt']) != c['prompt_length']:
        raise ValueError('Prompt mismatch')


def prediction_controls(symbols, k, prompt_length, weights):
    """Weighted majority and order-1 training-set predictors, no sequence lookup."""
    global_counts = np.bincount(symbols[1:], weights=weights, minlength=k)
    table = np.zeros((k,k))
    np.add.at(table, (symbols[:-1], symbols[1:]), weights)
    for row in table:
        if row.sum() == 0: row[:] = global_counts
    predictions = {}
    for name in ('majority','markov1'):
        teacher = (np.full(len(symbols)-1, np.argmax(global_counts)) if name == 'majority'
                   else table[symbols[:-1]].argmax(axis=1))
        previous = int(symbols[prompt_length-1]); generated = []
        for _ in range(len(symbols)-prompt_length):
            previous = int(np.argmax(global_counts if name=='majority' else table[previous]))
            generated.append(previous)
        predictions[name] = dict(generated=generated, exact_prefix_symbols=core.prefix_score(symbols[prompt_length:],generated),
                                 teacher_forced_accuracy=float(np.mean(teacher==symbols[1:])))
    return dict(predictions=predictions,global_counts=global_counts.tolist(),transition_counts=table.tolist())


def fit(features, symbols, condition, c):
    head = core.NonlinearReadout(np.arange(48),c['hidden_units'],core.nonlinear_seed(condition.seed,c['initialization']))
    weights = core.sample_weights(len(symbols)-1,c['prompt_length'],c['prefix_window'],c['prefix_weight'])
    _, history = head.fit(features,symbols[1:],epochs=c['epochs'],learning_rate=c['learning_rate'],
                          l2=c['l2'],checkpoints=(c['epochs'],),sample_weight=weights)
    return head, history, weights


def score_metrics(symbols, prediction, teacher, probabilities, c):
    truth = symbols[c['prompt_length']:]
    score = core.prefix_score(truth,prediction); correct = prediction == truth
    return dict(exact_prefix_symbols=score,autonomous_prefix_bits=float(score*np.log2(c['dataset']['alphabet_size'])),
                first_error_position=None if score==len(truth) else score+1,
                first_error_sequence_index=None if score==len(truth) else c['dataset']['offset']+c['prompt_length']+score,
                censored=score==len(truth),next_digit_accuracy=float(np.mean(teacher==symbols[1:])),
                teacher_forced_accuracy=float(np.mean(teacher[c['prompt_length']-1:]==truth)),
                autonomous_accuracy=float(correct.mean()),mean_confidence=float(probabilities.max(axis=1).mean()),
                region_accuracy={f'{a+1}-{min(b,len(truth))}':float(correct[a:b].mean())
                     for a,b in [(0,32),(32,64),(64,128),(128,len(truth))] if a<len(truth) and b>a})


def worker(config,cache,out,condition,family,dataset_seed):
    start=time.perf_counter(); c=json.loads(Path(config).read_text()); validate(c)
    out=Path(out);out.mkdir(exist_ok=False)
    with threadpool_limits(1):
        dataset=SequenceDataset(family,dataset_seed,**c['dataset']);symbols=dataset.symbols()
        model,observed,graph=core.build_model(cache,condition,c)
        features,diagnostics=core.collect(model,observed,symbols[:-1],c['activity_epsilon'])
        decay_full,decay_observed=core.decay_probe(model,observed,c['decay_steps'])
        head,history,weights=fit(features,symbols,condition,c)
        assert head.parameter_count==482
        teacher=head.predict(features);teacher_probs=softmax(head.logits(features),axis=1)
        prediction,probabilities,recall_features=core.rollout(model,observed,head,symbols[:c['prompt_length']],c['eval_length'])
        scores=score_metrics(symbols,prediction,teacher,probabilities,c)
        assert scores['exact_prefix_symbols']==core.prefix_score(symbols[c['prompt_length']:],teacher[c['prompt_length']-1:])
        assert core.weight_hash(model.weights)==graph['weight_sha256']
        controls=prediction_controls(symbols,c['dataset']['alphabet_size'],c['prompt_length'],weights)
        objective,_,training=head.objective(features,symbols[1:],c['l2'],weights)
        row=dict(**condition.__dict__,family=family,dataset_seed=dataset_seed,**scores,
                 pi_memory_score=scores['exact_prefix_symbols'] if family=='pi' else None,
                 train_loss=objective,weighted_cross_entropy=training['cross_entropy'],
                 effective_rank=core.state_diagnostics(features)['effective_rank'],
                 active_neurons_mean=float(diagnostics['active_counts'].mean()),
                 activity_sparsity=float(1-diagnostics['active_counts'].mean()/graph['neurons']),
                 mean_observed_cosine=float(diagnostics['observed_cosine'].mean()),
                 observed_mean_abs=float(np.abs(features).mean()),
                 observed_decay_ratio_32=float(decay_observed[-1]/decay_observed[0]) if decay_observed[0] else 0.,
                 majority_prefix=controls['predictions']['majority']['exact_prefix_symbols'],
                 markov1_prefix=controls['predictions']['markov1']['exact_prefix_symbols'],
                 runtime_seconds=time.perf_counter()-start)
        np.savez_compressed(out/'checkpoint.npz',mean=head.mean,scale=head.scale,**head.parameters,symbols=symbols,
                            features=features,teacher=teacher,teacher_probabilities=teacher_probs,prediction=prediction,
                            probabilities=probabilities,recall_features=recall_features,
                            position_accuracy=prediction==symbols[c['prompt_length']:],observed_indices=observed,
                            singular_values=np.linalg.svd(features-features.mean(axis=0),compute_uv=False),
                            decay_full=decay_full,decay_observed=decay_observed,**diagnostics)
        core.write_json(out/'metrics.json',row);core.write_json(out/'training.json',history)
        core.write_json(out/'controls.json',controls);core.write_json(out/'dataset.json',dataset.identity(symbols))
        core.write_json(out/'recall.json',dict(prompt=symbols[:c['prompt_length']].tolist(),target=symbols[c['prompt_length']:].tolist(),generated=prediction.tolist()))
        core.write_json(out/'manifest.json',dict(config=c,config_sha256=core.fingerprint(c),context=core.source_context(c),
             timestamp=datetime.now(timezone.utc).isoformat(),environment=core.environment(),graph=graph,
             dataset=dataset.identity(symbols),metrics=row,readout_parameters=482,checkpoint='checkpoint.npz',
             artifacts={p.name:core.sha256(p) for p in out.iterdir() if p.is_file()}))


def supervise(args,log,c):
    start=time.monotonic();peak=0
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    with Path(log).open('w',encoding='utf-8') as f:
        process=subprocess.Popen([sys.executable,'-m','flying.training.sequence_memory',*args],stdout=f,stderr=subprocess.STDOUT,env=env)
        root=psutil.Process(process.pid)
        while process.poll() is None:
            try:
                tree=[root]+root.children(recursive=True)
                rss=sum(p.memory_info().rss for p in tree if p.is_running());peak=max(peak,rss)
                if rss>c['worker_rss_bytes'] or time.monotonic()-start>c['worker_seconds']:
                    for p in reversed(tree):
                        try:p.kill()
                        except psutil.NoSuchProcess:pass
                    process.wait();raise RuntimeError('Resource limit exceeded; partial artifacts preserved')
            except psutil.NoSuchProcess:pass
            time.sleep(.2)
        if process.returncode:raise RuntimeError(f'Worker failed: {log}')
    return dict(peak_process_tree_rss_bytes=peak,wall_seconds=time.monotonic()-start,sample_seconds=.2)


def run(config,cache,out):
    c=json.loads(Path(config).read_text());validate(c)
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    context=core.source_context(c);core.write_json(out/'config.json',c);core.write_json(out/'started.json',context)
    rows=[];mapping_reference={};dataset_reference={}
    for block in c['blocks']:
        seed,ds=block['model_seed'],block['dataset_seed']
        for family in c['families']:
            for circuit in c['circuit_seeds']:
                for level in c['conditions']:
                    stem=f'{family}_{level}_c{circuit}_s{seed}_d{ds}';path=out/stem
                    usage=supervise(['worker','--config',str(out/'config.json'),'--cache',str(cache),'--out',str(path),
                         '--level',level,'--circuit',str(circuit),'--seed',str(seed),'--family',family,'--dataset-seed',str(ds)],out/(stem+'.log'),c)
                    m=json.loads((path/'manifest.json').read_text());g=m['graph']
                    mapping={k:g[k] for k in ['input_root_ids','observation_root_ids','input_mapping_sha256']}
                    key=(seed,circuit); reference=mapping_reference.setdefault(key,mapping)
                    if mapping!=reference:raise ValueError('Unmatched input/observation across family or graph')
                    dkey=(family,ds);reference=dataset_reference.setdefault(dkey,m['dataset']['sha256'])
                    if m['dataset']['sha256']!=reference:raise ValueError('Unmatched dataset')
                    core.write_json(path/'resources.json',usage)
                    rows.append(dict(**m['metrics'],**usage,neurons=g['neurons'],edges=g['edges'],checkpoint=f'{stem}/checkpoint.npz'))
                    print(f'{len(rows)} completed: {stem}',flush=True)
    if core.source_context(c)!=context:raise ValueError('Source changed during run')
    pd.DataFrame(rows).to_csv(out/'evaluations.csv',index=False)
    core.write_json(out/'manifest.json',dict(config=c,context=context,
         artifacts={p.relative_to(out).as_posix():core.sha256(p) for p in out.rglob('*') if p.is_file()}))


def verify(out,cache,refit=False):
    out=Path(out);m=json.loads((out/'manifest.json').read_text());c=m['config'];validate(c)
    context=core.source_context(c)
    if context['source_sha256']!=m['context']['source_sha256'] or context['config_sha256']!=m['context']['config_sha256']:
        raise ValueError('Numerical source/protocol/config changed')
    for name,digest in m['artifacts'].items():
        if core.sha256(out/name)!=digest:raise ValueError('Artifact changed: '+name)
    destination=out/('verification-refit.json' if refit else 'verification.json')
    if destination.exists():raise FileExistsError(destination)
    replayed=refitted=0
    with threadpool_limits(1):
        for path in sorted(out.glob('*/manifest.json')):
            record=json.loads(path.read_text());row=record['metrics']
            cond=core.NetworkCondition(row['level'],row['circuit_seed'],row['seed'])
            dataset=SequenceDataset(row['family'],row['dataset_seed'],**c['dataset']);symbols=dataset.symbols()
            # JSON converts dataclass tuple defaults to arrays; config explicitly uses arrays.
            assert json.loads(json.dumps(dataset.identity(symbols)))==record['dataset']
            model,obs,graph=core.build_model(cache,cond,c);assert graph==record['graph']
            features,diagnostics=core.collect(model,obs,symbols[:-1],c['activity_epsilon'])
            decay_full,decay_obs=core.decay_probe(model,obs,c['decay_steps'])
            with np.load(path.parent/'checkpoint.npz',allow_pickle=False) as a:
                for key,value in dict(symbols=symbols,features=features,decay_full=decay_full,decay_observed=decay_obs,**diagnostics).items():
                    np.testing.assert_array_equal(value,a[key])
                head=core.NonlinearReadout(np.arange(48),c['hidden_units'],core.nonlinear_seed(cond.seed,c['initialization']))
                head.mean=a['mean'];head.scale=a['scale'];head.parameters={k:a[k] for k in ['w1','b1','w2','b2']}
                teacher=head.predict(features);tp=softmax(head.logits(features),axis=1)
                pred,probs,states=core.rollout(model,obs,head,symbols[:c['prompt_length']],c['eval_length'])
                for key,value in dict(teacher=teacher,teacher_probabilities=tp,prediction=pred,probabilities=probs,recall_features=states).items():
                    np.testing.assert_array_equal(value,a[key])
                assert score_metrics(symbols,pred,teacher,probs,c)=={k:row[k] for k in score_metrics(symbols,pred,teacher,probs,c)}
                weights=core.sample_weights(len(symbols)-1,c['prompt_length'],c['prefix_window'],c['prefix_weight'])
                assert prediction_controls(symbols,c['dataset']['alphabet_size'],c['prompt_length'],weights)==json.loads((path.parent/'controls.json').read_text())
                if refit and cond.seed==c['blocks'][0]['model_seed'] and cond.circuit_seed==c['circuit_seeds'][0]:
                    fitted,_,_=fit(features,symbols,cond,c);assert fitted.digest()==head.digest();refitted+=1
            replayed+=1;print(f'{replayed} exact replays',flush=True)
    assert replayed==len(c['blocks'])*len(c['families'])*len(c['conditions'])*len(c['circuit_seeds'])
    core.write_json(destination,dict(exact_replays=replayed,exact_refits=refitted,
                                    exact_predictions_probabilities_features_decay_controls=True,source_artifacts_checked=True))


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['run','worker','verify'])
    p.add_argument('--config',type=Path,default=Path('configs/sequence_memory.json'))
    p.add_argument('--cache',type=Path,default=Path('outputs/act1-graphs'));p.add_argument('--out',type=Path,required=True)
    p.add_argument('--level');p.add_argument('--circuit',type=int);p.add_argument('--seed',type=int)
    p.add_argument('--family');p.add_argument('--dataset-seed',type=int);p.add_argument('--refit',action='store_true');a=p.parse_args()
    if a.command=='run':run(a.config,a.cache,a.out)
    elif a.command=='verify':verify(a.out,a.cache,a.refit)
    else:worker(a.config,a.cache,a.out,core.NetworkCondition(a.level,a.circuit,a.seed),a.family,a.dataset_seed)


if __name__=='__main__':main()
