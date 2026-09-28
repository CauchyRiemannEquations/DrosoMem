"""Finite suffix-count controls, separate from historical neural source."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import threading
import time

import numpy as np
import pandas as pd
import psutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def audit(path):
    m = read(path / 'manifest.json')
    for name, digest in m['artifacts'].items():
        assert sha(path / name) == digest, str(path / name)
    return m


def fit(symbols, k, order, weights):
    s = np.asarray(symbols)
    w = np.asarray(weights)
    if s.ndim != 1 or s.dtype.kind not in 'iu' or len(s) < 2 or k < 2 or order < 1:
        raise ValueError('Invalid symbols/alphabet/order')
    if np.any(s < 0) or np.any(s >= k) or w.shape != (len(s)-1,) or not np.all(np.isfinite(w)) or np.any(w <= 0):
        raise ValueError('Invalid symbols or training weights')
    table = defaultdict(lambda: np.zeros(k, dtype=float))
    for t in range(1, len(s)):
        for length in range(min(order, t)+1):
            table[tuple(int(x) for x in s[t-length:t])][s[t]] += w[t-1]
    return dict(table)


def predict(table, history, order):
    h = tuple(int(x) for x in history)
    for length in range(min(order, len(h)), -1, -1):
        key = h[-length:] if length else ()
        if key in table:
            counts = table[key]
            return int(np.argmax(counts)), counts/counts.sum(), length
    raise ValueError('Missing global fallback')


def rollout(table, prompt, length, order):
    # Deliberately no reference sequence argument or position-dependent table.
    history = list(prompt)
    generated, probs, used = [], [], []
    for _ in range(length):
        y, p, n = predict(table, history, order)
        generated.append(y)
        probs.append(p)
        used.append(n)
        history.append(y)
    return generated, probs, used


def ambiguity(symbols, order, start=3):
    counts = defaultdict(lambda: defaultdict(int))
    for t in range(start, len(symbols)):
        context = tuple(int(x) for x in symbols[max(0,t-order):t])
        counts[context][int(symbols[t])] += 1
    total = len(symbols)-start
    details = [dict(context=list(c), counts={str(y):n for y,n in sorted(v.items())}) for c,v in sorted(counts.items())]
    return dict(distinct_contexts=len(counts), singleton_contexts=sum(sum(v.values())==1 for v in counts.values()),
        conflicting_contexts=sum(len(v)>1 for v in counts.values()),
        conflicting_occurrence_fraction=sum(sum(v.values()) for v in counts.values() if len(v)>1)/total,
        ambiguity_error_floor=(total-sum(max(v.values()) for v in counts.values()))/total,
        counts=details)


def prefix(target, generated):
    assert len(target) == len(generated)
    return next((i for i,(a,b) in enumerate(zip(target,generated)) if a != b), len(target))


def serialize(table):
    return [dict(context=list(c), counts=v.tolist()) for c,v in sorted(table.items())]


def evaluate(symbols, k, order, weights):
    table = fit(symbols, k, order, weights)
    generated, probs, used = rollout(table, symbols[:3], len(symbols)-3, order)
    teacher = [predict(table, symbols[:t], order) for t in range(1,len(symbols))]
    target = symbols[3:]
    correct = np.asarray(generated) == target
    score = prefix(target, generated)
    amb = ambiguity(symbols, order)
    encoded = serialize(table)
    metrics = dict(order=order, exact_prefix_symbols=score, autonomous_prefix_bits=float(score*np.log2(k)),
        first_error_position=score+1 if score < len(target) else None,
        autonomous_accuracy=float(correct.mean()), teacher_accuracy=float(np.mean([p[0] for p in teacher]==symbols[1:])),
        teacher_eval_accuracy=float(np.mean([p[0] for p in teacher[2:]]==target)),
        accuracy_1_32=float(correct[:32].mean()), accuracy_33_64=float(correct[32:64].mean()), accuracy_65_125=float(correct[64:].mean()),
        fallback_fraction=float(np.mean(np.asarray(used) < np.minimum(order,np.arange(3,len(symbols))))),
        table_contexts=len(table), table_nonzero_cells=sum(int(np.count_nonzero(v)) for v in table.values()),
        table_dense_scalars=len(table)*k, table_json_bytes=len(json.dumps(encoded,separators=(',',':')).encode()),
        **{key:v for key,v in amb.items() if key!='counts'})
    return dict(metrics=metrics, symbols=symbols.tolist(), weights=weights.tolist(), tables=encoded,
        ambiguity_counts=amb['counts'], generated=generated, probabilities=[p.tolist() for p in probs],
        used_context_lengths=used, position_accuracy=correct.tolist(), teacher=[p[0] for p in teacher],
        teacher_probabilities=[p[1].tolist() for p in teacher])


def estimate(x, c):
    x = np.asarray(x,dtype=float)
    rng = np.random.default_rng(c['bootstrap_seed'])
    means = x[rng.integers(0,len(x),size=(c['bootstrap_draws'],len(x)))].mean(axis=1)
    sd = float(x.std(ddof=1))
    return dict(n=len(x),mean=float(x.mean()),median=float(np.median(x)),variance=sd*sd,
        bootstrap95=np.quantile(means,[.025,.975]).tolist(),paired_dz=float(x.mean()/sd) if sd else None,
        wins=int((x>0).sum()),ties=int((x==0).sum()),losses=int((x<0).sum()))


def summarize(rows, comparisons, out, c):
    f = pd.DataFrame(rows)
    d = pd.DataFrame(comparisons)
    f.to_csv(out/'raw-context-table.csv',index=False)
    d.to_csv(out/'neural-comparisons.csv',index=False)
    blocks = d.groupby(['alphabet_size','order','level','seed'])[['neural_prefix','context_prefix','neural_minus_context']].mean().reset_index()
    blocks.to_csv(out/'paired-blocks.csv',index=False)
    metrics = ['exact_prefix_symbols','autonomous_prefix_bits','ambiguity_error_floor','conflicting_occurrence_fraction',
        'teacher_accuracy','autonomous_accuracy','table_contexts','table_nonzero_cells','table_dense_scalars','table_json_bytes','fallback_fraction']
    cells = {f'k{k}_m{m}':{metric:estimate(g[metric],c) for metric in metrics} for (k,m),g in f.groupby(['alphabet_size','order'])}
    contrasts = {f'k{k}_m{m}_{level}':estimate(g.neural_minus_context,c) for (k,m,level),g in blocks.groupby(['alphabet_size','order','level'])}
    endpoints = {}
    for name,order,metric in [('H1',c['primary_ambiguity_order'],'ambiguity_error_floor'),('H2',c['primary_recall_order'],'exact_prefix_symbols')]:
        a = f[(f.alphabet_size==2)&(f.order==order)].set_index('dataset_seed')[metric]
        b = f[(f.alphabet_size==16)&(f.order==order)].set_index('dataset_seed')[metric]
        delta = b-a
        gate = delta.mean()<=-.10 and (delta<0).sum()>=4 if name=='H1' else delta.mean()>=25 and (delta>0).sum()>=4
        endpoints[name] = dict(**estimate(delta,c),passed=bool(gate),per_seed={str(k):float(v) for k,v in delta.items()})
    summary = dict(cells=cells,neural_minus_context=contrasts,endpoints=endpoints,
        joint_criterion=all(v['passed'] for v in endpoints.values()),unique_tasks=20,table_fits=len(f),neural_comparisons=len(d))
    write(out/'summary.json',summary)
    fig,axes = plt.subplots(1,3,figsize=(15,4.5))
    x = np.arange(4)
    for m in c['orders']:
        g = f[f.order==m].groupby('alphabet_size')
        axes[0].plot(x,g.ambiguity_error_floor.mean(),'o-',label=f'context {m}')
        axes[1].plot(x,g.exact_prefix_symbols.mean(),'o-',label=f'context {m}')
    base = blocks[blocks.order==3]
    for level,g in base.groupby('level'):
        means = g.groupby('alphabet_size').neural_prefix.mean()
        axes[2].plot(x,means,'o-',label=level)
    for seed,g in f[f.order==3].groupby('dataset_seed'):
        axes[2].plot(x,g.sort_values('alphabet_size').exact_prefix_symbols,color='gray',alpha=.25)
    axes[2].plot(x,f[f.order==3].groupby('alphabet_size').exact_prefix_symbols.mean(),'o--',color='black',label='context 3 (primary)')
    for ax in axes:
        ax.set(xticks=x,xticklabels=[2,4,10,16],xlabel='Alphabet size K (categorical)')
        ax.spines[['top','right']].set_visible(False)
        ax.legend(fontsize=8)
    axes[0].set(ylabel='Ambiguity error floor',title='Conflicting finite contexts')
    axes[1].set(ylabel='Exact-prefix symbols',title='All six registered orders',ylim=(-2,130))
    axes[2].set(ylabel='Exact-prefix symbols',title='Primary control vs saved neural means',ylim=(-2,130))
    fig.tight_layout(); fig.savefig(out/'context-curve.png',dpi=180); plt.close(fig)
    return summary


@threadpool_limits.wrap(limits=1)
def run(config, out):
    from alphabet_memory import dataset
    from context_reference import verify
    c = read(config)
    assert c['orders']==[1,2,3,4,5,8]
    out.mkdir(parents=True,exist_ok=False)
    start = time.perf_counter(); proc = psutil.Process()
    usage = dict(peak_sampled_rss_bytes=proc.memory_info().rss)
    stop = threading.Event()
    def monitor():
        while not stop.wait(.05):
            usage['peak_sampled_rss_bytes'] = max(usage['peak_sampled_rss_bytes'],proc.memory_info().rss)
    watcher = threading.Thread(target=monitor,daemon=True); watcher.start()
    def budget():
        if time.perf_counter()-start>c['max_seconds'] or usage['peak_sampled_rss_bytes']>c['max_rss_bytes']:
            raise RuntimeError('Registered resource budget exceeded; partial artifacts preserved')
    try:
        write(out/'config.json',c)
        prior = {p.as_posix():sha(p) for p in Path('results').rglob('*') if p.is_file() and out not in p.parents}
        write(out/'prior-results-sha256.json',prior)
        source = Path(c['source']); main = audit(source); old = main['config']
        baseline=[]; rows=[]; comparisons=[]; verified=[]
        for block in old['blocks']:
            for k in old['alphabet_sizes']:
                budget()
                copies=[]
                for level in old['conditions']:
                    for circuit in old['circuit_seeds']:
                        p=source/f'{level}_c{circuit}_s{block["model_seed"]}'/f'k{k}'
                        meta=audit(p)
                        with np.load(p/'checkpoint.npz',allow_pickle=False) as a:
                            s=a['symbols'].copy(); score=prefix(s[3:],a['prediction'])
                        assert score==meta['metrics']['exact_prefix_symbols']
                        copies.append((level,circuit,score,sha(p/'manifest.json')))
                        if len(copies)==1: symbols=s
                        else: np.testing.assert_array_equal(s,symbols)
                np.testing.assert_array_equal(symbols,dataset(old,k,block['dataset_seed']).symbols())
                assert len(symbols)==128 and symbols[:3].tolist()==[0,1,0]
                weights=np.ones(127); weights[2:34]=4
                oldcontrol=read(source/f'legacy5_c701_s{block["model_seed"]}'/f'k{k}'/'controls.json')['predictions']['markov1']
                for order in c['orders']:
                    record=evaluate(symbols,k,order,weights)
                    record['identity']=dict(alphabet_size=k,**block)
                    record['source_manifests']=[v[3] for v in copies]
                    verified.append(verify(record,k,order))
                    if order==1:
                        assert record['generated']==oldcontrol['generated']
                        assert record['metrics']['teacher_accuracy']==oldcontrol['teacher_forced_accuracy']
                        assert record['metrics']['exact_prefix_symbols']==oldcontrol['exact_prefix_symbols']
                        baseline.append(dict(**block,alphabet_size=k,prefix=oldcontrol['exact_prefix_symbols'],exact_match=True))
                    name=f'k{k}_d{block["dataset_seed"]}_m{order}.json'
                    write(out/name,record)
                    rows.append(dict(alphabet_size=k,**block,**record['metrics']))
                    for level,circuit,score,digest in copies:
                        comparisons.append(dict(alphabet_size=k,order=order,seed=block['model_seed'],dataset_seed=block['dataset_seed'],
                            level=level,circuit_seed=circuit,neural_prefix=score,context_prefix=record['metrics']['exact_prefix_symbols'],
                            neural_minus_context=score-record['metrics']['exact_prefix_symbols'],source_manifest_sha256=digest))
                print(f'K{k} data{block["dataset_seed"]}: all six orders independently checked',flush=True)
        budget()
        summary=summarize(rows,comparisons,out,c)
        for name,digest in prior.items(): assert sha(name)==digest,name
        write(out/'verification.json',dict(independent_reference_fits=len(verified),checks=verified,order1_baselines=baseline,
            prior_results_unchanged=len(prior),source_tasks_checked=80,unique_sequences_checked=20))
        budget()
        usage.update(wall_seconds=time.perf_counter()-start,sampling_seconds=.05,scope='entire diagnostic process including analysis and independent reference')
        write(out/'resources.json',usage)
        sources={str(p):sha(p) for p in [Path(__file__),Path(__file__).with_name('context_reference.py'),Path(config),Path(c['protocol'])]}
        write(out/'manifest.json',dict(study=c['study'],timestamp=datetime.now(timezone.utc).isoformat(),
            git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            git_status=subprocess.check_output(['git','status','--short','--untracked-files=no'],text=True).strip(),
            config_sha256=sha(config),source_manifest_sha256=sha(source/'manifest.json'),source_hashes=sources,
            environment=dict(python=platform.python_version(),platform=platform.platform(),packages=subprocess.check_output([__import__('sys').executable,'-m','pip','freeze'],text=True).splitlines()),
            artifacts={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
        print(json.dumps(dict(endpoints=summary['endpoints'],joint_criterion=summary['joint_criterion'],resources=usage),indent=2))
    finally:
        stop.set(); watcher.join()


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,default=Path('configs/context_memory.json')); p.add_argument('--out',type=Path,required=True)
    args=p.parse_args(); run(args.config,args.out)
