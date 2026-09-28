"""Bounded, performance-blind paired sequence construction."""
import argparse
from pathlib import Path
import json
import subprocess
import numpy as np
from alphabet_memory import read, check, context as old_context
from context_memory import ambiguity, evaluate
from context_reference import verify as verify_control
from flying.training import whole_brain_memory as core


def validate(c):
    assert c['alphabet_sizes']==[4] and c['arms']==['low','high']
    assert c['dataset']==dict(length=128,offset=0,prompt=[0,1,0])
    assert c['construction']==dict(order=3,proposals=2000,min_floor_gap=.1)
    assert c['conditions']==['legacy5','brain1'] and c['epochs'] in [20,2000]
    assert (c['hidden_units'],c['initialization'],c['learning_rate'],c['l2'])==(8,0,.03,1e-5)
    assert (c['prompt_length'],c['eval_length'],c['prefix_window'],c['prefix_weight'])==(3,125,32,4)


def context(c):
    r=old_context(c)
    r['intervention_sha256']={f'scripts/{n}.py':core.sha256(Path(__file__).parent/(n+'.py')) for n in
        ['context_intervention','run_context_intervention','context_memory','context_reference']}
    r['sequences_manifest_sha256']=core.sha256(Path(c['sequences'])/'manifest.json')
    return r


def error_count(s):
    # Four-symbol code -> 64 contexts and four next-symbol counts.
    code=(s[:-3]*16+s[1:-2]*4+s[2:-1])*4+s[3:]
    table=np.bincount(code,minlength=256).reshape(64,4)
    return int(125-table.max(axis=1).sum())


def construct(seed, proposals=2000):
    rng=np.random.default_rng(seed)
    initial=np.r_[[0,1,0],rng.permutation(np.repeat(np.arange(4),[30,31,32,32]))].astype(np.int64)
    swaps=rng.integers(3,128,size=(proposals,2)); result={}
    for arm in ['low','high']:
        s=initial.copy(); errors=[error_count(s)]; accepted=[]
        for i,j in swaps:
            candidate=s.copy(); candidate[i],candidate[j]=candidate[j],candidate[i]
            value=error_count(candidate)
            take=value<errors[-1] if arm=='low' else value>errors[-1]
            if take:s=candidate
            errors.append(value if take else errors[-1]); accepted.append(bool(take))
        result[arm]=dict(symbols=s.tolist(),error_counts=errors,accepted=accepted)
    return dict(dataset_seed=seed,initial=initial.tolist(),proposals=swaps.tolist(),arms=result)


def verify_construction(record):
    # Generic dictionary-based ambiguity count, not the optimized 4-symbol code.
    def generic(s):
        rows={}
        for t in range(3,128):
            key=tuple(s[t-3:t]); rows.setdefault(key,[0]*4)[s[t]]+=1
        return 125-sum(max(v) for v in rows.values())
    fresh=construct(record['dataset_seed'],len(record['proposals'])); assert fresh==record
    for arm,r in record['arms'].items():
        s=record['initial'].copy(); errors=[generic(s)]; accepts=[]
        for i,j in record['proposals']:
            candidate=s.copy(); candidate[i],candidate[j]=candidate[j],candidate[i]
            value=generic(candidate)
            take=value<errors[-1] if arm=='low' else value>errors[-1]
            if take:s=candidate
            accepts.append(take); errors.append(value if take else errors[-1])
        assert s==r['symbols'] and errors==r['error_counts'] and accepts==r['accepted']
        assert s[:3]==[0,1,0] and np.bincount(s,minlength=4).tolist()==[32]*4
    return (record['arms']['high']['error_counts'][-1]-record['arms']['low']['error_counts'][-1])/125


def prepare(config):
    c=read(config); validate(c); out=Path(c['sequences']); out.mkdir(parents=True,exist_ok=False)
    all_pass=True; rows=[]; weights=np.ones(127); weights[2:34]=4
    for b in c['blocks']:
        r=construct(b['dataset_seed']); gap=verify_construction(r); passed=gap>=.1; all_pass &= passed
        p=out/f'd{b["dataset_seed"]}'; p.mkdir(); core.write_json(p/'construction.json',r)
        diagnostics={}
        for arm in c['arms']:
            s=np.array(r['arms'][arm]['symbols'],dtype=np.int64)
            orders={}
            for order in [1,2,3,4,5,8]:
                record=evaluate(s,4,order,weights); verify_control(record,4,order)
                core.write_json(p/f'{arm}_m{order}.json',record); orders[str(order)]=record['metrics']
            trans=np.zeros((4,4),dtype=int); np.add.at(trans,(s[:-1],s[1:]),1)
            diagnostics[arm]=dict(orders=orders,early_ambiguity=ambiguity(s[:35],3),late_ambiguity=ambiguity(s,3,start=35),
                region_counts={name:np.bincount(s[a:z],minlength=4).tolist() for name,a,z in [('first32',3,35),('later93',35,128)]},
                transition_counts=trans.tolist(),repeated_neighbor_fraction=float(np.mean(s[:-1]==s[1:])),
                symbols_sha256=__import__('hashlib').sha256(s.tobytes()).hexdigest())
        core.write_json(p/'diagnostics.json',diagnostics)
        rows.append(dict(**b,floor_gap=gap,passed=passed))
    core.write_json(out/'manifest.json',dict(config=c,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        protocol_sha256=core.sha256(c['protocol']),generator_sha256=core.sha256(__file__),all_pairs_pass=bool(all_pass),pairs=rows,
        controls_verified=len(c['blocks'])*12,artifacts={p.relative_to(out).as_posix():core.sha256(p) for p in out.rglob('*') if p.is_file()}))
    print(json.dumps(dict(all_pairs_pass=all_pass,pairs=rows),indent=2))
    if not all_pass:raise RuntimeError('Construction gate failed; cohort stopped without replacement')


def sequence(c,seed,arm):
    p=Path(c['sequences'])/f'd{seed}'
    s=np.array(read(p/'construction.json')['arms'][arm]['symbols'],dtype=np.int64)
    identity=dict(family='fixed-k-context-conflict',dataset_seed=seed,arm=arm,alphabet_size=4,length=128,
        sha256=__import__('hashlib').sha256(s.tobytes()).hexdigest(),construction_sha256=core.sha256(p/'construction.json'))
    return s,identity


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,required=True); a=p.parse_args(); prepare(a.config)
