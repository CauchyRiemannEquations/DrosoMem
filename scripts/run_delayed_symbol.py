"""Fixed-interface independent-stream delay probes; immutable cohorts and repeats."""
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
from threadpoolctl import threadpool_limits
from flying.evaluation.delayed_memory import delayed_labels, decode_delays
from flying.models.ridge import RidgeDecoder
from flying.training import whole_brain_memory as core


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def check(path):
    m=read(path/'manifest.json')
    for name,digest in m['artifacts'].items():
        assert core.sha256(path/name)==digest,(path,name)
    return m


def validate(c):
    assert c['conditions']==['legacy5','brain1']
    assert c['lags']==[0,1,2,3,4,5,8,12,16,24,32]
    assert c['primary_lags']==[1,2,3,4,5,8]
    assert c['warmup']==100 and c['alpha']==1.
    assert c['train_samples']==2*c['null_shift']
    assert c['train_samples'] in [200,2000] and c['test_samples']==c['train_samples']//2
    assert c['schedule']=='mbon_after_kc' and c['normalization']=='incoming_l1'
    assert (c['gain'],c['leak'],c['input_fraction'],c['input_amplitude'])==(.9,.6,.1,.5)
    seeds=[b[k] for b in c['blocks'] for k in ['seed','train_seed','test_seed']]
    assert len(set(seeds))==len(seeds)
    assert len(set(c['circuit_seeds']))==len(c['circuit_seeds'])


def fit_probe(train_features,test_features,train_symbols,test_symbols,c):
    w=c['warmup'];lags=c['lags']
    train=delayed_labels(train_symbols,lags,w);test=delayed_labels(test_symbols,lags,w)
    assert len(train_features)==len(train_symbols) and len(test_features)==len(test_symbols)
    targets=np.eye(10)[train].reshape(len(train),-1)
    real=RidgeDecoder(c['alpha']).fit(train_features[w:],targets)
    null=RidgeDecoder(c['alpha']).fit(train_features[w:],np.roll(targets,c['null_shift'],axis=0))
    np.testing.assert_array_equal(real.mean,null.mean)
    np.testing.assert_array_equal(real.scale,null.scale)
    np.testing.assert_array_equal(real.target_mean,null.target_mean)
    arrays=dict(train_features=train_features,test_features=test_features,
        train_symbols=train_symbols,test_symbols=test_symbols,train_targets=train,test_targets=test,
        mean=real.mean,scale=real.scale,target_mean=real.target_mean,weights=real.weights,
        null_weights=null.weights,null_target_mean=null.target_mean,
        train_scores=real.scores(train_features[w:]).reshape(len(train),len(lags),10),
        test_scores=real.scores(test_features[w:]).reshape(len(test),len(lags),10),
        null_scores=null.scores(test_features[w:]).reshape(len(test),len(lags),10))
    arrays['predictions']=arrays['test_scores'].argmax(axis=2).astype(np.uint8)
    arrays['null_predictions']=arrays['null_scores'].argmax(axis=2).astype(np.uint8)
    return arrays


def measures(a,c):
    truth=np.eye(10)[a['test_targets']]
    freq=a['target_mean'].reshape(len(c['lags']),10)
    denom=np.sum((truth-freq)**2,axis=(0,2))
    r2=1-np.sum((truth-a['test_scores'])**2,axis=(0,2))/denom
    return [dict(lag=lag,test_accuracy=float(np.mean(a['predictions'][:,j]==a['test_targets'][:,j])),
        train_accuracy=float(np.mean(a['train_scores'][:,j].argmax(axis=1)==a['train_targets'][:,j])),
        frequency_accuracy=float(np.mean(a['test_targets'][:,j]==freq[j].argmax())),
        null_accuracy=float(np.mean(a['null_predictions'][:,j]==a['test_targets'][:,j])),
        r2_vs_training_frequency=float(r2[j])) for j,lag in enumerate(c['lags'])]


def independent_checks(a,c):
    rows,pred,truth=decode_delays(a['train_features'],a['test_features'],a['train_symbols'],a['test_symbols'],c['lags'],c['warmup'],c['alpha'])
    np.testing.assert_array_equal(pred,a['predictions']);np.testing.assert_array_equal(truth,a['test_targets'])
    ours=measures(a,c)
    for r,o in zip(rows,ours):
        assert r['accuracy']==o['test_accuracy']
        assert r['frequency_baseline_accuracy']==o['frequency_accuracy']
        assert r['r2_vs_training_frequency']==o['r2_vs_training_frequency']
    # Reconstruct scores from saved coefficients, not stored predictions.
    z=(a['test_features'][c['warmup']:]-a['mean'])/a['scale']
    for key,weights,bias in [('test_scores','weights','target_mean'),('null_scores','null_weights','null_target_mean')]:
        rebuilt=(z@a[weights]+a[bias]).reshape(a[key].shape)
        np.testing.assert_array_equal(rebuilt,a[key])
    np.testing.assert_array_equal(a['null_predictions'],a['null_scores'].argmax(axis=2))


def worker(config,cache,out,block,circuit,level,source=None):
    started=time.perf_counter();c=read(config);validate(c);b=c['blocks'][block]
    if source:
        old=check(source);assert old['config']==c and old['script_sha256']==core.sha256(__file__)
        for name,digest in old['context']['source_sha256'].items():assert core.sha256(name)==digest,name
    out.mkdir(exist_ok=False)
    with threadpool_limits(1):
        model,observed,graph=core.build_model(cache,core.NetworkCondition(level,circuit,b['seed']),c)
        streams={key:np.random.default_rng(b[key+'_seed']).integers(0,10,c[key+'_samples']+c['warmup']).astype(np.uint8) for key in ['train','test']}
        train,td=core.collect(model,observed,streams['train'],c['activity_epsilon'])
        df,do=core.decay_probe(model,observed,c['decay_steps'])
        test,vd=core.collect(model,observed,streams['test'],c['activity_epsilon'])
        a=fit_probe(train,test,streams['train'],streams['test'],c)
        a.update(observed_indices=observed,decay_full=df,decay_observed=do,
            singular_values=np.linalg.svd(train[c['warmup']:]-train[c['warmup']:].mean(axis=0),compute_uv=False))
        a.update({'train_'+k:v for k,v in td.items()});a.update({'test_'+k:v for k,v in vd.items()})
        independent_checks(a,c);rows=measures(a,c)
        assert core.weight_hash(model.weights)==graph['weight_sha256']
        neural=dict(effective_rank=core.state_diagnostics(train[c['warmup']:])['effective_rank'],
            observed_mean_abs=float(np.abs(train[c['warmup']:]).mean()),
            active_neurons_mean=float(td['active_counts'].mean()),
            active_neurons_max=int(td['active_counts'].max()),
            activity_sparsity=float(1-td['active_counts'].mean()/graph['neurons']),
            observed_cosine_mean=float(td['observed_cosine'].mean()),
            standardized_max_abs=float(np.abs((train[c['warmup']:]-a['mean'])/a['scale']).max()),
            clipped_scale_count=int((a['scale']==1e-5).sum()),
            observed_decay_ratio=float(do[-1]/do[0]) if do[0] else 0.)
        identity=dict(**b,circuit_seed=circuit,level=level)
        if source:
            with np.load(source/'checkpoint.npz',allow_pickle=False) as prior:
                assert set(prior.files)==set(a)
                for name,value in a.items():np.testing.assert_array_equal(value,prior[name],err_msg=name)
            assert rows==old['metrics'] and graph==old['graph'] and neural==old['neural']
            core.write_json(out/'verification.json',dict(**identity,exact_trajectory_replay=True,exact_refit=True,
                numerical_arrays_checked=len(a),reference_decode_delays=True,source_manifest_sha256=core.sha256(source/'manifest.json')))
        else:
            np.savez_compressed(out/'checkpoint.npz',**a)
            core.write_json(out/'metrics.json',rows)
            core.write_json(out/'manifest.json',dict(config=c,context=core.source_context(c),identity=identity,
                script_sha256=core.sha256(__file__),timestamp=datetime.now(timezone.utc).isoformat(),
                graph=graph,neural=neural,metrics=rows,environment=core.environment(),
                rng='numpy.default_rng PCG64 integers(0,10), generated int64 then stored uint8',
                readout_coefficients_per_lag=490,total_coefficients_per_head=490*len(c['lags']),
                runtime_seconds=time.perf_counter()-started,
                dataset_sha256={key:__import__('hashlib').sha256(value.tobytes()).hexdigest() for key,value in streams.items()},
                artifacts={p.name:core.sha256(p) for p in out.iterdir() if p.is_file()}))


def supervise(args,log,c):
    start=time.monotonic();peak=0
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    with log.open('w',encoding='utf-8') as f:
        process=subprocess.Popen([sys.executable,__file__,'worker',*args],stdout=f,stderr=subprocess.STDOUT,env=env)
        root=psutil.Process(process.pid)
        while process.poll() is None:
            try:
                tree=[root]+root.children(recursive=True);rss=sum(p.memory_info().rss for p in tree if p.is_running());peak=max(peak,rss)
                if rss>c['worker_rss_bytes'] or time.monotonic()-start>c['worker_seconds']:
                    for p in reversed(tree):
                        try:p.kill()
                        except psutil.NoSuchProcess:pass
                    process.wait();raise RuntimeError('Resource limit exceeded; artifacts preserved')
            except psutil.NoSuchProcess:pass
            time.sleep(.2)
        if process.returncode:raise RuntimeError(f'Worker failed: {log}')
    return dict(peak_process_tree_rss_bytes=peak,wall_seconds=time.monotonic()-start,sample_seconds=.2)


def run(config,cache,out,source=None):
    c=read(source/'config.json') if source else read(config);validate(c)
    if source:check(source)
    out.mkdir(parents=True,exist_ok=False);context=core.source_context(c);script=core.sha256(__file__)
    core.write_json(out/'config.json',c);core.write_json(out/'started.json',dict(context=context,script_sha256=script))
    rows=[];count=0
    for i,b in enumerate(c['blocks']):
        for circuit in c['circuit_seeds']:
            for level in c['conditions']:
                stem=f'{level}_c{circuit}_s{b["seed"]}';p=out/stem
                args=['--config',str(out/'config.json'),'--cache',str(cache),'--out',str(p),'--block',str(i),'--circuit',str(circuit),'--level',level]
                if source:args+=['--source',str(source/stem)]
                usage=supervise(args,out/(stem+'.log'),c);core.write_json(p/'resources.json',usage)
                if not source:
                    m=read(p/'manifest.json')
                    rows.extend(dict(**m['identity'],**r,**m['neural'],**usage) for r in m['metrics'])
                count+=1;print(f'{count} {"verified" if source else "completed"}: {stem}',flush=True)
    assert core.source_context(c)==context and core.sha256(__file__)==script
    if rows:pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    core.write_json(out/'manifest.json',dict(config=c,context=context,script_sha256=script,complete=True,runs=count,
        source=str(source) if source else None,source_manifest_sha256=core.sha256(source/'manifest.json') if source else None,
        exact_replays=count if source else 0,exact_refits=count if source else 0,
        artifacts={p.relative_to(out).as_posix():core.sha256(p) for p in out.rglob('*') if p.is_file()}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['run','worker','verify'])
    p.add_argument('--config',type=Path,default=Path('configs/delayed_symbol.json'));p.add_argument('--cache',type=Path,default=Path('outputs/act1-graphs'))
    p.add_argument('--out',type=Path,required=True);p.add_argument('--source',type=Path);p.add_argument('--block',type=int);p.add_argument('--circuit',type=int);p.add_argument('--level')
    a=p.parse_args()
    if a.command=='worker':worker(a.config,a.cache,a.out,a.block,a.circuit,a.level,a.source)
    else:
        if a.command=='verify' and not a.source:p.error('verify requires --source')
        run(a.config,a.cache,a.out,a.source if a.command=='verify' else None)
