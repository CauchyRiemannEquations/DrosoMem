"""Refit only the decoder on verified N512 caches; independently replay normally."""
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
from flying.training import whole_brain_memory as core
from flying.training import sequence_memory as memory


def validate_pair(baseline,c):
    memory.validate(c)
    assert c['dataset']['length']==512 and c['prefix_window']==32
    assert c['prefix_weight']==4*479/167 and baseline['prefix_weight']==4
    assert c['families']==['random'] and c['prompt_length']==3
    ignored={'study','protocol','prefix_weight'}
    assert {k:v for k,v in baseline.items() if k not in ignored}=={k:v for k,v in c.items() if k not in ignored},'Unmatched condition'


def check_artifacts(source):
    m=json.loads((source/'manifest.json').read_text())
    for name,digest in m['artifacts'].items():assert core.sha256(source/name)==digest,name
    return m


def worker(config,source,cache,out):
    started=time.perf_counter();c=json.loads(config.read_text());m=check_artifacts(source)
    validate_pair(m['config'],c)
    context=core.source_context(c)
    for path,digest in m['context']['source_sha256'].items():
        if path.startswith('src/'):assert core.sha256(path)==digest,path
    out.mkdir(exist_ok=False)
    row=dict(m['metrics']);condition=core.NetworkCondition(row['level'],row['circuit_seed'],row['seed'])
    with threadpool_limits(1):
        with np.load(source/'checkpoint.npz',allow_pickle=False) as a:
            arrays={k:a[k] for k in a.files}
        symbols=arrays['symbols'];features=arrays['features']
        np.testing.assert_array_equal(symbols,memory.SequenceDataset('random',row['dataset_seed'],**c['dataset']).symbols())
        initial=core.NonlinearReadout(np.arange(48),c['hidden_units'],core.nonlinear_seed(condition.seed,c['initialization']))
        initial.initialize(features);initial_digest=initial.digest()
        head,history,weights=memory.fit(features,symbols,condition,c)
        np.testing.assert_array_equal(head.mean,arrays['mean']);np.testing.assert_array_equal(head.scale,arrays['scale'])
        model,observed,graph=core.build_model(cache,condition,c);assert graph==m['graph']
        np.testing.assert_array_equal(observed,arrays['observed_indices'])
        teacher=head.predict(features);teacher_probs=softmax(head.logits(features),axis=1)
        prediction,probabilities,recall_features=core.rollout(model,observed,head,symbols[:3],c['eval_length'])
        scores=memory.score_metrics(symbols,prediction,teacher,probabilities,c)
        assert scores['exact_prefix_symbols']==core.prefix_score(symbols[3:],teacher[2:])
        assert core.weight_hash(model.weights)==graph['weight_sha256']
        controls=memory.prediction_controls(symbols,10,3,weights)
        objective,_,training=head.objective(features,symbols[1:],c['l2'],weights)
        row.update(**scores,train_loss=objective,weighted_cross_entropy=training['cross_entropy'],
            majority_prefix=controls['predictions']['majority']['exact_prefix_symbols'],
            markov1_prefix=controls['predictions']['markov1']['exact_prefix_symbols'],runtime_seconds=time.perf_counter()-started)
        arrays.update(mean=head.mean,scale=head.scale,**head.parameters,teacher=teacher,teacher_probabilities=teacher_probs,
            prediction=prediction,probabilities=probabilities,recall_features=recall_features,position_accuracy=prediction==symbols[3:])
        np.savez_compressed(out/'checkpoint.npz',**arrays)
        core.write_json(out/'metrics.json',row);core.write_json(out/'training.json',history)
        core.write_json(out/'controls.json',controls);core.write_json(out/'dataset.json',m['dataset'])
        core.write_json(out/'recall.json',dict(prompt=symbols[:3].tolist(),target=symbols[3:].tolist(),generated=prediction.tolist()))
        core.write_json(out/'manifest.json',dict(config=c,config_sha256=core.fingerprint(c),context=context,
            timestamp=datetime.now(timezone.utc).isoformat(),environment=core.environment(),graph=graph,dataset=m['dataset'],
            metrics=row,readout_parameters=head.parameter_count,checkpoint='checkpoint.npz',
            cached_source=dict(path=str(source),manifest_sha256=core.sha256(source/'manifest.json'),
                checkpoint_sha256=core.sha256(source/'checkpoint.npz'),teacher_features_and_decay_reused=True,initial_head_digest=initial_digest),
            script_sha256=core.sha256(__file__),artifacts={p.name:core.sha256(p) for p in out.iterdir() if p.is_file()}))


def supervise(args,log,c):
    start=time.monotonic();peak=0
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    with log.open('w',encoding='utf-8') as f:
        process=subprocess.Popen([sys.executable,__file__,'worker',*args],stdout=f,stderr=subprocess.STDOUT,env=env)
        root=psutil.Process(process.pid)
        while process.poll() is None:
            try:
                tree=[root]+root.children(recursive=True)
                rss=sum(p.memory_info().rss for p in tree if p.is_running());peak=max(peak,rss)
                if rss>c['worker_rss_bytes'] or time.monotonic()-start>c['worker_seconds']:
                    for p in reversed(tree):
                        try:p.kill()
                        except psutil.NoSuchProcess:pass
                    process.wait();raise RuntimeError('Resource limit exceeded; artifacts preserved')
            except psutil.NoSuchProcess:pass
            time.sleep(.2)
        if process.returncode:raise RuntimeError(f'Worker failed: {log}')
    return dict(peak_process_tree_rss_bytes=peak,wall_seconds=time.monotonic()-start,sample_seconds=.2)


def run(config,source,cache,out):
    c=json.loads(config.read_text());source_meta=check_artifacts(source);validate_pair(source_meta['config'],c)
    out.mkdir(parents=True,exist_ok=False);context=core.source_context(c);script=core.sha256(__file__)
    core.write_json(out/'config.json',c)
    core.write_json(out/'started.json',dict(context=context,script_sha256=script,source_manifest_sha256=core.sha256(source/'manifest.json')))
    rows=[]
    for b in c['blocks']:
        for circuit in c['circuit_seeds']:
            for level in c['conditions']:
                stem=f'random_{level}_c{circuit}_s{b["model_seed"]}_d{b["dataset_seed"]}';p=out/stem
                usage=supervise(['--config',str(out/'config.json'),'--source',str(source/stem),'--cache',str(cache),'--out',str(p)],out/(stem+'.log'),c)
                m=json.loads((p/'manifest.json').read_text());core.write_json(p/'resources.json',usage)
                rows.append(dict(**m['metrics'],**usage,neurons=m['graph']['neurons'],edges=m['graph']['edges'],checkpoint=f'{stem}/checkpoint.npz'))
                print(f'{len(rows)} cached fits completed: {stem}',flush=True)
    assert core.source_context(c)==context and core.sha256(__file__)==script
    pd.DataFrame(rows).to_csv(out/'evaluations.csv',index=False)
    core.write_json(out/'manifest.json',dict(config=c,context=context,script_sha256=script,source=str(source),
        source_manifest_sha256=core.sha256(source/'manifest.json'),complete=True,
        artifacts={p.relative_to(out).as_posix():core.sha256(p) for p in out.rglob('*') if p.is_file()}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['run','worker'])
    p.add_argument('--config',type=Path,default=Path('configs/prefix_mass.json'));p.add_argument('--source',type=Path,required=True)
    p.add_argument('--cache',type=Path,default=Path('outputs/act1-graphs'));p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();(run if a.command=='run' else worker)(a.config,a.source,a.cache,a.out)
