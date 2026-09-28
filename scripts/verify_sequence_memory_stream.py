"""Independently replay completed atomic run artifacts while fitting continues.

Only finalized per-run manifests are consumed. A completion manifest is emitted
only after the complete main manifest and every planned run have been verified.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy.special import softmax
from threadpoolctl import threadpool_limits
from flying.data.sequences import SequenceDataset
from flying.training import whole_brain_memory as core
from flying.training.sequence_memory import validate, fit, prediction_controls, score_metrics


def verify_one(path,c,cache):
    m=json.loads(path.read_text());row=m['metrics']
    assert m['config']==c and m['config_sha256']==core.fingerprint(c)
    assert m['context']['source_sha256']==core.source_context(c)['source_sha256']
    for name,digest in m['artifacts'].items():assert core.sha256(path.parent/name)==digest
    cond=core.NetworkCondition(row['level'],row['circuit_seed'],row['seed'])
    ds=SequenceDataset(row['family'],row['dataset_seed'],**c['dataset']);symbols=ds.symbols()
    assert json.loads(json.dumps(ds.identity(symbols)))==m['dataset']
    model,observed,graph=core.build_model(cache,cond,c);assert graph==m['graph']
    features,diagnostics=core.collect(model,observed,symbols[:-1],c['activity_epsilon'])
    df,do=core.decay_probe(model,observed,c['decay_steps']);refitted=False
    with np.load(path.parent/'checkpoint.npz',allow_pickle=False) as a:
        for key,value in dict(symbols=symbols,features=features,decay_full=df,decay_observed=do,**diagnostics).items():
            np.testing.assert_array_equal(value,a[key])
        head=core.NonlinearReadout(np.arange(48),c['hidden_units'],core.nonlinear_seed(cond.seed,c['initialization']))
        head.mean=a['mean'];head.scale=a['scale'];head.parameters={k:a[k] for k in ['w1','b1','w2','b2']}
        assert head.parameter_count==482
        # Match the head's indexed feature layout: NumPy reductions can differ
        # at roundoff level between contiguous and advanced-indexed arrays.
        selected=features[:,head.indices]
        np.testing.assert_array_equal(head.mean,selected.mean(axis=0))
        np.testing.assert_array_equal(head.scale,np.maximum(selected.std(axis=0),1e-5))
        teacher=head.predict(features);tp=softmax(head.logits(features),axis=1)
        pred,probs,states=core.rollout(model,observed,head,symbols[:c['prompt_length']],c['eval_length'])
        for key,value in dict(teacher=teacher,teacher_probabilities=tp,prediction=pred,probabilities=probs,recall_features=states).items():
            np.testing.assert_array_equal(value,a[key])
        for k,v in score_metrics(symbols,pred,teacher,probs,c).items():assert v==row[k]
        weights=core.sample_weights(len(symbols)-1,c['prompt_length'],c['prefix_window'],c['prefix_weight'])
        assert prediction_controls(symbols,c['dataset']['alphabet_size'],c['prompt_length'],weights)==json.loads((path.parent/'controls.json').read_text())
        loss,_,training=head.objective(features,symbols[1:],c['l2'],weights)
        assert loss==row['train_loss'] and training['cross_entropy']==row['weighted_cross_entropy']
        assert core.weight_hash(model.weights)==graph['weight_sha256']
        if cond.seed==c['blocks'][0]['model_seed'] and cond.circuit_seed==c['circuit_seeds'][0]:
            independent,_,_=fit(features,symbols,cond,c);assert independent.digest()==head.digest();refitted=True
    return dict(run=path.parent.name,source_manifest_sha256=core.sha256(path),exact_replay=True,
                exact_refit=refitted,features_probabilities_controls_decay_and_losses=True)


def main(source,out,cache):
    c=json.loads((source/'config.json').read_text());validate(c)
    out.mkdir(parents=True,exist_ok=False);(out/'runs').mkdir()
    script_digest=core.sha256(__file__);context=core.source_context(c);started=time.monotonic()
    expected=[f'{family}_{level}_c{circuit}_s{b["model_seed"]}_d{b["dataset_seed"]}'
              for b in c['blocks'] for family in c['families'] for circuit in c['circuit_seeds'] for level in c['conditions']]
    records=[]
    with threadpool_limits(1):
        for stem in expected:
            path=source/stem/'manifest.json'
            while not path.exists():
                if time.monotonic()-started>7200:raise TimeoutError('Waiting for completed run; partial verification preserved')
                time.sleep(2)
            record=verify_one(path,c,cache);records.append(record)
            core.write_json(out/'runs'/(stem+'.json'),record)
            print(f'{len(records)}/{len(expected)} verified',flush=True)
    while not (source/'manifest.json').exists():
        if time.monotonic()-started>7200:raise TimeoutError('Main manifest unavailable')
        time.sleep(2)
    m=json.loads((source/'manifest.json').read_text());assert m['config']==c and m['context']==context
    for name,digest in m['artifacts'].items():assert core.sha256(source/name)==digest
    assert core.source_context(c)==context and core.sha256(__file__)==script_digest
    core.write_json(out/'manifest.json',dict(source=str(source),source_manifest_sha256=core.sha256(source/'manifest.json'),
        context=context,script_sha256=script_digest,exact_replays=len(records),exact_refits=sum(r['exact_refit'] for r in records),
        complete=True,artifacts={p.relative_to(out).as_posix():core.sha256(p) for p in out.rglob('*') if p.is_file()}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--cache',type=Path,default=Path('outputs/act1-graphs'));a=p.parse_args();main(a.source,a.out,a.cache)
