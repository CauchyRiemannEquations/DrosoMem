"""Bounded grouped alphabet fits and independent replay with specified refits."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import os,subprocess,sys,time
import numpy as np
import pandas as pd
import psutil
from scipy.special import softmax
from threadpoolctl import threadpool_limits
from flying.training import whole_brain_memory as core
from flying.training.sequence_memory import prediction_controls,score_metrics
from alphabet_memory import read,check,context,AlphabetExperimentConfig,SymbolReadout,build_bank,configure,dataset,fit


def evaluate(model,observed,graph,features,diagnostics,df,do,head,symbols,cond,c,k,ds):
    teacher=head.predict(features);tp=softmax(head.logits(features),axis=1)
    pred,probs,states=core.rollout(model,observed,head,symbols[:3],c['eval_length'])
    effective=dict(c,dataset=dict(c['dataset'],alphabet_size=k))
    scores=score_metrics(symbols,pred,teacher,probs,effective)
    assert scores['exact_prefix_symbols']==core.prefix_score(symbols[3:],teacher[2:])
    weights=core.sample_weights(len(symbols)-1,3,c['prefix_window'],c['prefix_weight'])
    controls=prediction_controls(symbols,k,3,weights)
    objective,_,training=head.objective(features,symbols[1:],c['l2'],weights)
    neural=dict(effective_rank=core.state_diagnostics(features)['effective_rank'],
        active_neurons_mean=float(diagnostics['active_counts'].mean()),active_neurons_max=int(diagnostics['active_counts'].max()),
        activity_sparsity=float(1-diagnostics['active_counts'].mean()/graph['neurons']),mean_observed_cosine=float(diagnostics['observed_cosine'].mean()),
        observed_mean_abs=float(np.abs(features).mean()),observed_decay_ratio_32=float(do[-1]/do[0]) if do[0] else 0.)
    row=dict(**cond.__dict__,dataset_seed=ds,alphabet_size=k,**scores,**neural,train_loss=objective,
        weighted_cross_entropy=training['cross_entropy'],readout_parameters=head.parameter_count,
        prefix_fraction=scores['exact_prefix_symbols']/125,full_completion=scores['exact_prefix_symbols']==125,
        majority_prefix=controls['predictions']['majority']['exact_prefix_symbols'],
        markov1_prefix=controls['predictions']['markov1']['exact_prefix_symbols'])
    row['control_excess']=row['exact_prefix_symbols']-max(row['majority_prefix'],row['markov1_prefix'])
    a=dict(mean=head.mean,scale=head.scale,**head.parameters,symbols=symbols,features=features,teacher=teacher,
        teacher_probabilities=tp,prediction=pred,probabilities=probs,recall_features=states,position_accuracy=pred==symbols[3:],
        observed_indices=observed,singular_values=np.linalg.svd(features-features.mean(axis=0),compute_uv=False),
        decay_full=df,decay_observed=do,**diagnostics)
    assert probs.shape==(125,k) and pred.min()>=0 and pred.max()<k
    assert core.weight_hash(model.weights)==graph['weight_sha256']
    return row,a,controls


def worker(config,cache,out,block,circuit,level,source=None):
    c=read(config);AlphabetExperimentConfig(c).validate();b=c['blocks'][block]
    cond=core.NetworkCondition(level,circuit,b['model_seed']);out.mkdir(exist_ok=False)
    start=time.perf_counter()
    with threadpool_limits(1):
        model,observed,base,bank,meta=build_bank(cache,cond,c);build_seconds=time.perf_counter()-start
        for k in c['alphabet_sizes']:
            started=time.perf_counter();p=out/f'k{k}';p.mkdir(exist_ok=False)
            graph=configure(model,base,bank,meta,k);ds=dataset(c,k,b['dataset_seed']);symbols=ds.symbols()
            features,diagnostics=core.collect(model,observed,symbols[:-1],c['activity_epsilon'])
            df,do=core.decay_probe(model,observed,c['decay_steps'])
            refitted=False
            if source:
                old=check(source/f'k{k}');assert old['config']==c and old['graph']==graph
                assert old['context']['source_sha256']==context(c)['source_sha256']
                assert old['context']['adapter_sha256']==context(c)['adapter_sha256']
                assert old['dataset']==__import__('json').loads(__import__('json').dumps(ds.identity(symbols)))
                with np.load(source/f'k{k}/checkpoint.npz',allow_pickle=False) as saved:
                    head=SymbolReadout(np.arange(48),8,0,k);head.mean=saved['mean'];head.scale=saved['scale'];head.parameters={n:saved[n] for n in ['w1','b1','w2','b2']}
                    selected=features[:,head.indices]
                    np.testing.assert_array_equal(head.mean,selected.mean(axis=0));np.testing.assert_array_equal(head.scale,np.maximum(selected.std(axis=0),1e-5))
                    row,a,controls=evaluate(model,observed,graph,features,diagnostics,df,do,head,symbols,cond,c,k,b['dataset_seed'])
                    assert set(saved.files)==set(a)
                    for name,value in a.items():np.testing.assert_array_equal(value,saved[name],err_msg=name)
                assert row==old['metrics'] and controls==read(source/f'k{k}/controls.json')
                if block==0 and circuit==c['circuit_seeds'][0]:
                    fresh,history,_=fit(features,symbols,cond,c,k);assert fresh.digest()==head.digest()
                    assert history==read(source/f'k{k}/training.json');refitted=True
                core.write_json(p/'verification.json',dict(level=level,circuit_seed=circuit,seed=cond.seed,alphabet_size=k,
                    source_manifest_sha256=core.sha256(source/f'k{k}/manifest.json'),exact_replay=True,exact_refit=refitted,
                    arrays_checked=len(a),features_controls_probabilities_losses_exact=True))
            else:
                head,history,_=fit(features,symbols,cond,c,k)
                row,a,controls=evaluate(model,observed,graph,features,diagnostics,df,do,head,symbols,cond,c,k,b['dataset_seed'])
                np.savez_compressed(p/'checkpoint.npz',**a)
                core.write_json(p/'metrics.json',row);core.write_json(p/'training.json',history);core.write_json(p/'controls.json',controls)
                core.write_json(p/'recall.json',dict(prompt=symbols[:3].tolist(),target=symbols[3:].tolist(),generated=a['prediction'].tolist()))
                core.write_json(p/'manifest.json',dict(config=c,config_sha256=core.fingerprint(c),context=context(c),
                    timestamp=datetime.now(timezone.utc).isoformat(),environment=core.environment(),graph=graph,dataset=ds.identity(symbols),
                    metrics=row,runtime_excluding_shared_graph_build=time.perf_counter()-started,checkpoint='checkpoint.npz',
                    resource_scope='shared per graph/seed/stratum group, including all four K',
                    artifacts={q.name:core.sha256(q) for q in p.iterdir() if q.is_file()}))
            print(f'K{k} {"verified" if source else "fit"}: {level} / {circuit} / {cond.seed}',flush=True)
        core.write_json(out/'group.json',dict(level=level,circuit_seed=circuit,seed=cond.seed,build_seconds=build_seconds,
            total_worker_seconds=time.perf_counter()-start,graph_built_once=True,alphabets=c['alphabet_sizes']))


def supervise(args,log,c):
    start=time.monotonic();peak=0;env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    with log.open('w',encoding='utf-8') as f:
        proc=subprocess.Popen([sys.executable,__file__,'worker',*args],stdout=f,stderr=subprocess.STDOUT,env=env);root=psutil.Process(proc.pid)
        while proc.poll() is None:
            try:
                tree=[root]+root.children(recursive=True);rss=sum(p.memory_info().rss for p in tree if p.is_running());peak=max(peak,rss)
                if rss>c['worker_rss_bytes'] or time.monotonic()-start>c['worker_seconds']:
                    for q in reversed(tree):
                        try:q.kill()
                        except psutil.NoSuchProcess:pass
                    proc.wait();raise RuntimeError('Resource limit exceeded; partial artifacts retained')
            except psutil.NoSuchProcess:pass
            time.sleep(.2)
        if proc.returncode:raise RuntimeError(f'Worker failed: {log}')
    return dict(peak_process_tree_rss_bytes=peak,wall_seconds=time.monotonic()-start,sample_seconds=.2,scope='all four K in this group')


def run(config,cache,out,source=None):
    c=read(source/'config.json') if source else read(config);AlphabetExperimentConfig(c).validate()
    out.mkdir(parents=True,exist_ok=False);ctx=context(c);core.write_json(out/'config.json',c);core.write_json(out/'started.json',ctx)
    rows=[];records=[];started=time.monotonic();count=0
    def wait(path):
        while not path.exists():
            if time.monotonic()-started>14400:raise TimeoutError('Missing complete source; partial verification retained')
            time.sleep(2)
    for i,b in enumerate(c['blocks']):
        for circuit in c['circuit_seeds']:
            for level in c['conditions']:
                stem=f'{level}_c{circuit}_s{b["model_seed"]}';p=out/stem
                if source:wait(source/stem/'group.json')
                args=['--config',str(out/'config.json'),'--cache',str(cache),'--out',str(p),'--block',str(i),'--circuit',str(circuit),'--level',level]
                if source:args+=['--source',str(source/stem)]
                usage=supervise(args,out/(stem+'.log'),c);core.write_json(p/'resources.json',usage)
                for k in c['alphabet_sizes']:
                    if source:records.append(read(p/f'k{k}/verification.json'))
                    else:rows.append(read(p/f'k{k}/metrics.json'))
                count+=4;print(f'{count} {"replays" if source else "fits"} completed ({stem})',flush=True)
    if source:wait(source/'manifest.json');check(source)
    assert context(c)==ctx
    if rows:pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    core.write_json(out/'manifest.json',dict(config=c,context=ctx,complete=True,fits=0 if source else count,
        exact_replays=count if source else 0,exact_refits=sum(r['exact_refit'] for r in records),
        source=str(source) if source else None,source_manifest_sha256=core.sha256(source/'manifest.json') if source else None,
        artifacts={p.relative_to(out).as_posix():core.sha256(p) for p in out.rglob('*') if p.is_file()}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['run','worker','verify']);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path('configs/alphabet_memory.json'));p.add_argument('--cache',type=Path,default=Path('outputs/act1-graphs'))
    p.add_argument('--source',type=Path);p.add_argument('--block',type=int);p.add_argument('--circuit',type=int);p.add_argument('--level');a=p.parse_args()
    if a.command=='worker':worker(a.config,a.cache,a.out,a.block,a.circuit,a.level,a.source)
    else:
        if a.command=='verify' and not a.source:p.error('verify requires --source')
        run(a.config,a.cache,a.out,a.source if a.command=='verify' else None)
