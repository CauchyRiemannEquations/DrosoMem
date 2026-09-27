"""ACT I: bounded, paired rate-model recall with fixed input/observation IDs."""
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import psutil
from scipy import sparse
from scipy.special import softmax
from threadpoolctl import threadpool_info, threadpool_limits

from flying.brain.diagnostics import normalize_condition, state_diagnostics
from flying.brain.mushroom_body import KCEncoder
from flying.brain.timed_reservoir import TimedReservoir
from flying.brain.plasticity import weight_hash
from flying.data.connectome import load_connectome, sha256
from flying.data.mushroom_body import load_roles
from flying.data.pi_digits import pi_digits, decimal_pi_digits
from flying.models.nonlinear_readout import NonlinearReadout
from flying.training.phase5_prefix import sample_weights
from flying.training.phase5_readout import nonlinear_seed


def write_json(path, value):
    path = Path(path)
    part = path.with_suffix(path.suffix+'.part')
    part.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    part.replace(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class SequenceDataset:
    family: str = 'pi'
    alphabet_size: int = 10
    offset: int = 0
    length: int = 200

    def symbols(self):
        # Extend sources in ACT II; do not silently map decimal digits to other K.
        if self.family != 'pi' or self.alphabet_size != 10 or self.offset < 0 or self.length < 2:
            raise ValueError('ACT I supports verified decimal pi only')
        n = self.offset+self.length
        a, b = pi_digits(n), decimal_pi_digits(n)
        np.testing.assert_array_equal(a, b)
        return a[self.offset:].astype(np.int64)


@dataclass(frozen=True)
class NetworkCondition:
    level: str
    circuit_seed: int
    seed: int


@dataclass(frozen=True)
class ExperimentConfig:
    values: dict

    def validate(self):
        c = self.values
        if not set(c['conditions']) <= {'legacy5','left5','left1','brain5','brain1'}:
            raise ValueError('Unknown graph')
        if c['normalization'] != 'incoming_l1' or c['schedule'] != 'mbon_after_kc':
            raise ValueError('Unregistered dynamics')
        if c['hidden_units'] != 8 or c['dataset']['alphabet_size'] != 10:
            raise ValueError('Unregistered readout')
        if c['eval_length'] != c['dataset']['length']-c['prompt_length']:
            raise ValueError('Recall must cover the trained suffix')
        for name in ['conditions','circuit_seeds','seeds']:
            if not c[name] or len(set(c[name])) != len(c[name]):
                raise ValueError('Empty or duplicate factor')
        sample_weights(c['dataset']['length']-1,c['prompt_length'],c['prefix_window'],c['prefix_weight'])


@dataclass
class ResultManifest:
    config: dict
    context: dict
    artifacts: dict


def source_context(c):
    paths = sorted(Path('src/flying').rglob('*.py'))+[Path(c['protocol'])]
    for seed in c['circuit_seeds']:
        paths += sorted(Path(f'data/flywire_783_mb_left_kc512_s{seed}').glob('*'))
    return dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                source_sha256={p.as_posix():sha256(p) for p in paths if p.is_file()},
                config_sha256=fingerprint(c))


def environment():
    return dict(python=sys.version,platform=platform.platform(),processor=platform.processor(),
                cpu_count=psutil.cpu_count(),ram_bytes=psutil.virtual_memory().total,
                packages={n:importlib.metadata.version(n) for n in
                          ['numpy','scipy','pandas','mpmath','threadpoolctl','psutil','pyarrow']},
                blas=threadpool_info(),numerical_threads=1)


class MappedEncoder:
    def __init__(self, patterns):
        self.patterns = patterns
        self.silent = False

    def __call__(self, digit):
        return np.zeros(self.patterns.shape[1]) if self.silent else self.patterns[int(digit)]


def build_model(cache, condition, c):
    cache = Path(cache)
    old_dir = Path(f'data/flywire_783_mb_left_kc512_s{condition.circuit_seed}')
    old, original_ids, _ = load_connectome(old_dir)
    original_roles, _ = load_roles(old_dir, original_ids)
    original_ids = np.asarray(original_ids, dtype=np.int64)
    original_patterns = KCEncoder(original_roles,condition.seed,c['input_fraction'],c['input_amplitude']).patterns
    meta = json.loads((cache/'provenance.json').read_text())
    required = ['nodes.npz']
    if condition.level != 'legacy5':
        required += [('brain1' if condition.level=='left1' else condition.level)+'.npz']
    for name in required:
        if sha256(cache/name) != meta['files'][name]:
            raise ValueError('Graph cache changed: '+name)
    with np.load(cache/'nodes.npz', allow_pickle=False) as nodes:
        if condition.level=='legacy5':
            raw, ids, roles = old, original_ids, original_roles
        else:
            graph = 'brain1' if condition.level=='left1' else condition.level
            raw = sparse.load_npz(cache/(graph+'.npz')).tocsr()
            ids, roles = nodes['ids'], nodes['roles']
            if condition.level.startswith('left'):
                ix = nodes['left_indices']
                ids, roles = ids[ix], roles[ix]
                if condition.level=='left1':
                    raw = raw[ix,:][:,ix].tocsr()
    ix = pd.Index(ids).get_indexer(original_ids)
    if (ix<0).any():
        raise ValueError('Original input/observation universe missing')
    observed = ix[np.asarray(original_roles)=='MBON']
    if len(observed)!=48:
        raise ValueError('Observation budget changed')
    patterns = np.zeros((10,len(ids)))
    patterns[:,ix] = original_patterns
    weights = normalize_condition(raw,c['normalization'],c['gain']); weights.sort_indices()
    encoder = MappedEncoder(patterns)
    model = TimedReservoir(weights,encoder,roles,c['leak'],c['schedule'])
    info = dict(level=condition.level,neurons=len(ids),edges=int(raw.nnz),
                threshold=1 if condition.level.endswith('1') else 5,
                weight_sha256=weight_hash(weights),source=meta['sources'],
                graph_cache_hashes={name:meta['files'][name] for name in required},
                observation_root_ids=ids[observed].astype(str).tolist(),
                input_root_ids=[original_ids[pattern!=0].astype(str).tolist() for pattern in original_patterns],
                input_amplitude=c['input_amplitude'],
                observed_features=48, input_mapping_sha256=hashlib.sha256(original_ids.tobytes()+original_patterns.tobytes()).hexdigest())
    return model, observed, info


def collect(model, observed, symbols, epsilon):
    model.reset(); features=[]; active=[]; norms=[]; cosine=[]; previous=None
    for symbol in symbols:
        state = model.step(int(symbol)); x = state[observed]
        features.append(x); active.append(np.count_nonzero(np.abs(state)>epsilon))
        norms.append(np.linalg.norm(state))
        if previous is not None:
            denominator = np.linalg.norm(x)*np.linalg.norm(previous)
            cosine.append(float(x@previous/denominator) if denominator else 0.)
        previous=x
    return np.asarray(features), dict(active_counts=np.asarray(active),full_norm=np.asarray(norms),
                                   observed_cosine=np.asarray(cosine))


def rollout(model, observed, head, prompt, horizon):
    """No reference sequence or target labels are accessible here."""
    model.reset()
    for digit in prompt:
        state=model.step(int(digit))
    generated=[]; probabilities=[]; features=[]
    for _ in range(horizon):
        x=state[observed]; p=softmax(head.logits(x)[0]); digit=int(np.argmax(p))
        generated.append(digit); probabilities.append(p); features.append(x)
        state=model.step(digit)
    return np.asarray(generated),np.asarray(probabilities),np.asarray(features)


def prefix_score(target, generated):
    if len(target)!=len(generated): raise ValueError('Mismatched recall horizon')
    errors=np.flatnonzero(np.asarray(target)!=np.asarray(generated))
    return int(errors[0]) if len(errors) else len(target)


def decay_probe(model, observed, steps):
    before=model.state.copy(); model.encoder.silent=True
    full=[float(np.linalg.norm(before))]; obs=[float(np.linalg.norm(before[observed]))]
    try:
        for _ in range(steps):
            state=model.step(0); full.append(float(np.linalg.norm(state))); obs.append(float(np.linalg.norm(state[observed])))
    finally:
        model.encoder.silent=False; model.state=before
    return np.asarray(full),np.asarray(obs)


def worker(config, cache, out, condition):
    started=time.perf_counter(); c=json.loads(Path(config).read_text()); ExperimentConfig(c).validate()
    out=Path(out); out.mkdir(exist_ok=False)
    with threadpool_limits(1):
        model,observed,graph=build_model(cache,condition,c)
        digits=SequenceDataset(**c['dataset']).symbols()
        features, diagnostics=collect(model,observed,digits[:-1],c['activity_epsilon'])
        decay_full,decay_observed=decay_probe(model,observed,c['decay_steps'])
        head=NonlinearReadout(np.arange(48),c['hidden_units'],nonlinear_seed(condition.seed,c['initialization']))
        weights=sample_weights(len(digits)-1,c['prompt_length'],c['prefix_window'],c['prefix_weight'])
        _,history=head.fit(features,digits[1:],epochs=c['epochs'],learning_rate=c['learning_rate'],
                           l2=c['l2'],checkpoints=(c['epochs'],),sample_weight=weights)
        assert head.parameter_count==482
        teacher=head.predict(features); teacher_probs=softmax(head.logits(features),axis=1)
        prediction,probabilities,recall_features=rollout(model,observed,head,digits[:c['prompt_length']],c['eval_length'])
        score=prefix_score(digits[c['prompt_length']:],prediction)
        assert score==prefix_score(digits[c['prompt_length']:],teacher[c['prompt_length']-1:])
        assert weight_hash(model.weights)==graph['weight_sha256']
        correct=prediction==digits[c['prompt_length']:]
        regions={f'{a+1}-{min(b,len(correct))}':float(correct[a:b].mean())
                 for a,b in [(0,32),(32,64),(64,128),(128,len(correct))] if a<len(correct) and b>a}
        singular=np.linalg.svd(features-features.mean(axis=0),compute_uv=False)
        objective,_,training=head.objective(features,digits[1:],c['l2'],weights)
        row=dict(**condition.__dict__,offset=c['dataset']['offset'],pi_memory_score=score,
                 autonomous_prefix_bits=float(score*np.log2(c['dataset']['alphabet_size'])),
                 first_error_position=None if score==len(correct) else score+1,
                 first_error_digit_index=None if score==len(correct) else c['dataset']['offset']+c['prompt_length']+score,
                 censored=score==len(correct),train_loss=objective,next_digit_accuracy=training['accuracy'],
                 teacher_forced_accuracy=float((teacher[c['prompt_length']-1:]==digits[c['prompt_length']:]).mean()),
                 autonomous_accuracy=float(correct.mean()),region_accuracy=regions,
                 effective_rank=state_diagnostics(features)['effective_rank'],
                 observed_mean_abs=float(np.abs(features).mean()),
                 active_neurons_mean=float(diagnostics['active_counts'].mean()),
                 active_neurons_max=int(diagnostics['active_counts'].max()),
                 activity_sparsity=float(1-diagnostics['active_counts'].mean()/graph['neurons']),
                 mean_observed_cosine=float(diagnostics['observed_cosine'].mean()),
                 full_decay_ratio_32=float(decay_full[-1]/decay_full[0]) if decay_full[0] else 0.,
                 observed_decay_ratio_32=float(decay_observed[-1]/decay_observed[0]) if decay_observed[0] else 0.,
                 runtime_seconds=time.perf_counter()-started)
        np.savez_compressed(out/'checkpoint.npz',mean=head.mean,scale=head.scale,**head.parameters,
                            features=features,teacher=teacher,teacher_probabilities=teacher_probs,
                            prediction=prediction,probabilities=probabilities,recall_features=recall_features,
                            position_accuracy=correct,digits=digits,observed_indices=observed,
                            singular_values=singular,decay_full=decay_full,decay_observed=decay_observed,**diagnostics)
        write_json(out/'metrics.json',row); write_json(out/'training.json',history)
        write_json(out/'recall.json',dict(prompt=digits[:c['prompt_length']].tolist(),target=digits[c['prompt_length']:].tolist(),generated=prediction.tolist()))
        write_json(out/'manifest.json',dict(config=c,config_sha256=fingerprint(c),context=source_context(c),
                   timestamp=datetime.now(timezone.utc).isoformat(),seed=condition.seed,graph=graph,
                   dataset=dict(**c['dataset'],sha256=hashlib.sha256(digits.tobytes()).hexdigest()),
                   environment=environment(),metrics=row,readout_parameters=482,checkpoint='checkpoint.npz',
                   artifacts={p.name:sha256(p) for p in out.iterdir() if p.is_file()}))


def supervise(args, log, c):
    start=time.monotonic(); peak=0
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    with Path(log).open('w',encoding='utf-8') as f:
        process=subprocess.Popen([sys.executable,'-m','flying.training.whole_brain_memory',*args],stdout=f,stderr=subprocess.STDOUT,env=env)
        root=psutil.Process(process.pid)
        while process.poll() is None:
            try:
                tree=[root]+root.children(recursive=True)
                rss=sum(p.memory_info().rss for p in tree if p.is_running())
                peak=max(peak,rss)
                if rss>c['worker_rss_bytes'] or time.monotonic()-start>c['worker_seconds']:
                    for p in reversed(tree):
                        try: p.kill()
                        except psutil.NoSuchProcess: pass
                    process.wait()
                    raise RuntimeError('Resource limit exceeded; partial artifacts retained')
            except psutil.NoSuchProcess:
                pass
            time.sleep(.2)
        if process.returncode:
            raise RuntimeError(f'Worker failed; inspect {log}')
    return dict(peak_process_tree_rss_bytes=peak,wall_seconds=time.monotonic()-start,sample_seconds=.2)


def run(config, cache, out):
    c=json.loads(Path(config).read_text()); ExperimentConfig(c).validate()
    out=Path(out); out.mkdir(parents=True,exist_ok=False)
    context=source_context(c); write_json(out/'config.json',c); write_json(out/'started.json',context)
    rows=[]; resources=[]
    for circuit in c['circuit_seeds']:
        for seed in c['seeds']:
            reference=None
            for level in c['conditions']:
                stem=f'{level}_c{circuit}_s{seed}'; path=out/stem
                usage=supervise(['worker','--config',str(out/'config.json'),'--cache',str(cache),'--out',str(path),
                                 '--level',level,'--circuit',str(circuit),'--seed',str(seed)],out/(stem+'.log'),c)
                manifest=json.loads((path/'manifest.json').read_text()); graph=manifest['graph']
                mapping={k:graph[k] for k in ['input_root_ids','observation_root_ids','input_mapping_sha256']}
                if reference is None: reference=mapping
                if mapping!=reference: raise ValueError('Unmatched input/observation mapping')
                write_json(path/'resources.json',usage)
                rows.append(dict(**manifest['metrics'],**usage,neurons=graph['neurons'],edges=graph['edges'],checkpoint=f'{stem}/checkpoint.npz'))
                resources.append(dict(run=stem,**usage))
                print(f'{len(rows)} completed: {stem}',flush=True)
    if source_context(c)!=context: raise ValueError('Source/config changed during run')
    pd.DataFrame(rows).to_csv(out/'evaluations.csv',index=False)
    write_json(out/'resources.json',resources)
    artifacts={p.relative_to(out).as_posix():sha256(p) for p in out.rglob('*') if p.is_file()}
    write_json(out/'manifest.json',ResultManifest(c,context,artifacts).__dict__)


def verify(out,cache,refit=False):
    out=Path(out); m=json.loads((out/'manifest.json').read_text()); c=m['config']
    # Context commits may differ after results/docs commits; exact code/protocol is authoritative.
    current=source_context(c)
    if current['source_sha256']!=m['context']['source_sha256'] or current['config_sha256']!=m['context']['config_sha256']:
        raise ValueError('Numerical source/protocol/config changed')
    for name,digest in m['artifacts'].items():
        if sha256(out/name)!=digest: raise ValueError('Artifact checksum mismatch: '+name)
    replayed=refitted=0
    with threadpool_limits(1):
        for path in sorted(out.glob('*/manifest.json')):
            run_manifest=json.loads(path.read_text()); row=run_manifest['metrics']
            cond=NetworkCondition(row['level'],row['circuit_seed'],row['seed'])
            model,observed,graph=build_model(cache,cond,c)
            if graph!=run_manifest['graph']: raise ValueError('Graph identity changed')
            with np.load(path.parent/'checkpoint.npz',allow_pickle=False) as a:
                digits=SequenceDataset(**c['dataset']).symbols(); np.testing.assert_array_equal(digits,a['digits'])
                features,diagnostics=collect(model,observed,digits[:-1],c['activity_epsilon'])
                np.testing.assert_array_equal(features,a['features'])
                for k,v in diagnostics.items(): np.testing.assert_array_equal(v,a[k])
                head=NonlinearReadout(np.arange(48),c['hidden_units'],nonlinear_seed(cond.seed,c['initialization']))
                head.mean=a['mean']; head.scale=a['scale']; head.parameters={k:a[k] for k in ['w1','b1','w2','b2']}
                np.testing.assert_array_equal(head.predict(features),a['teacher'])
                pred,probs,states=rollout(model,observed,head,digits[:c['prompt_length']],c['eval_length'])
                for key,value in [('prediction',pred),('probabilities',probs),('recall_features',states)]:
                    np.testing.assert_array_equal(value,a[key])
                assert prefix_score(digits[c['prompt_length']:],pred)==row['pi_memory_score']
                if refit and cond.seed==c['seeds'][0] and cond.circuit_seed==c['circuit_seeds'][0]:
                    fitted=NonlinearReadout(np.arange(48),c['hidden_units'],nonlinear_seed(cond.seed,c['initialization']))
                    fitted.fit(features,digits[1:],epochs=c['epochs'],learning_rate=c['learning_rate'],l2=c['l2'],
                               checkpoints=(c['epochs'],),sample_weight=sample_weights(len(digits)-1,c['prompt_length'],c['prefix_window'],c['prefix_weight']))
                    assert fitted.digest()==head.digest(); refitted+=1
            replayed+=1
            print(f'{replayed} exact replays',flush=True)
    expected=len(c['conditions'])*len(c['circuit_seeds'])*len(c['seeds'])
    assert replayed==expected
    result=dict(exact_replays=replayed,exact_refits=refitted,source_and_artifacts_checked=True)
    destination=out/('verification-refit.json' if refit else 'verification.json')
    if destination.exists(): raise FileExistsError(destination)
    write_json(destination,result)


def main():
    p=argparse.ArgumentParser(); p.add_argument('command',choices=['run','worker','verify'])
    p.add_argument('--config',type=Path,default=Path('configs/whole_brain_memory.json'))
    p.add_argument('--cache',type=Path,default=Path('outputs/act1-graphs'))
    p.add_argument('--out',type=Path,required=True); p.add_argument('--level'); p.add_argument('--circuit',type=int);p.add_argument('--seed',type=int)
    p.add_argument('--refit',action='store_true'); a=p.parse_args()
    if a.command=='run': run(a.config,a.cache,a.out)
    elif a.command=='verify': verify(a.out,a.cache,a.refit)
    else: worker(a.config,a.cache,a.out,NetworkCondition(a.level,a.circuit,a.seed))


if __name__=='__main__': main()
