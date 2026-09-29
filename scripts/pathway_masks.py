"""Prospective minimum-overlap, mass-matched pathway cut bank."""
import argparse,subprocess,hashlib,time
from pathlib import Path
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from threadpoolctl import threadpool_limits
from flying.training import whole_brain_memory as core
from alphabet_memory import read
from edge_panel import anatomy,seal
from frozen_state_probe import Budget


def target_mask(raw,roles,family):
    pre,post=family.split('_');co=raw.tocoo()
    return (roles[co.col]==pre)&(roles[co.row]==post)


def constraints(raw,target,c):
    w=core.normalize_condition(raw,'incoming_l1',c['gain']);w.sort_indices()
    n=int(target.sum());neg=int(((raw.data<0)&target).sum())
    r=float(abs(raw.data[target]).sum());v=float(abs(w.data[target]).sum())
    assert n>0 and r>0 and v>0
    A=np.stack([np.ones(raw.nnz),(raw.data<0).astype(float),abs(raw.data)/r,abs(w.data)/v])
    e=c['relative_mass_tolerance'];return A,np.array([n,neg,1-e,1-e]),np.array([n,neg,1+e,1+e])


def solve(A,lo,hi,cost,c,gap):
    start=time.perf_counter()
    r=milp(cost,integrality=np.ones(len(cost)),bounds=Bounds(0,1),constraints=LinearConstraint(A,lo,hi),options={'time_limit':c['solver_seconds'],'mip_rel_gap':gap})
    if r.status!=0 or r.x is None:raise RuntimeError(('MILP did not solve',r.status,r.message))
    assert np.max(abs(r.x-np.rint(r.x)))<1e-5
    mask=r.x>.5;values=A@mask
    assert np.all(values>=lo-c['numerical_guard']) and np.all(values<=hi+c['numerical_guard'])
    assert r.mip_gap<=gap+1e-10
    return mask,dict(status=int(r.status),message=r.message,objective=float(r.fun),dual_bound=float(r.mip_dual_bound),gap=float(r.mip_gap),nodes=int(r.mip_node_count),seconds=time.perf_counter()-start)


def audit_mask(raw,target,mask,c,minimum=None):
    assert mask.dtype==bool and mask.shape==target.shape
    assert mask.sum()==target.sum() and ((raw.data<0)&mask).sum()==((raw.data<0)&target).sum()
    w=core.normalize_condition(raw,'incoming_l1',c['gain']);w.sort_indices()
    raw_error=float(abs(raw.data[mask]).sum()/abs(raw.data[target]).sum()-1)
    norm_error=float(abs(w.data[mask]).sum()/abs(w.data[target]).sum()-1)
    assert abs(raw_error)<=c['relative_mass_tolerance']+c['numerical_guard']
    assert abs(norm_error)<=c['relative_mass_tolerance']+c['numerical_guard']
    overlap=int((mask&target).sum())
    if minimum is not None:assert overlap==minimum
    return dict(count=int(mask.sum()),negative=int(((raw.data<0)&mask).sum()),overlap=overlap,overlap_fraction=float(overlap/target.sum()),raw_l1_error=raw_error,normalized_l1_error=norm_error,
        raw_l1=float(abs(raw.data[mask]).sum()),normalized_l1=float(abs(w.data[mask]).sum()))


@threadpool_limits.wrap(limits=1)
def build(config,out):
    c=read(config);assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
    context=core.source_context(c)
    for p in [config,Path(__file__),Path('docs/pathway-memory-seed-audit.json'),Path('docs/pathway-memory-graph-audit.json')]:
        p=p.resolve().relative_to(Path.cwd());context['source_sha256'][p.as_posix()]=core.sha256(p)
    for p,h in context['source_sha256'].items():assert hashlib.sha256(subprocess.check_output(['git','show',context['git_commit']+':'+p])).hexdigest()==h,p
    out.mkdir(parents=True,exist_ok=False);budget=Budget(c)
    try:
        graphs={ci:anatomy(ci) for ci in c['circuit_seeds']};minimum={};minmasks={}
        for ci,(raw,ids,roles) in graphs.items():
            minimum[str(ci)]={}
            for family in c['families']:
                target=target_mask(raw,roles,family);A,lo,hi=constraints(raw,target,c)
                mask,log=solve(A,lo,hi,target.astype(float),c,0);k=int((mask&target).sum())
                assert abs(log['objective']-k)<1e-5 and log['dual_bound']>k-1+1e-6
                minimum[str(ci)][family]=dict(**audit_mask(raw,target,mask,c,k),solver=log,eligible=k/target.sum()<=c['maximum_specificity_overlap'])
                minmasks[f'c{ci}_{family}']=mask
        np.savez_compressed(out/'minimum-masks.npz',**minmasks);core.write_json(out/'minimum-overlap.json',minimum)
        for cohort,settings in c['cohorts'].items():
            for block in settings['blocks']:
                for ci in settings['circuit_seeds']:
                    raw,ids,roles=graphs[ci];masks=dict(intact=np.zeros(raw.nnz,bool));info={}
                    for family in c['families']:
                        target=target_mask(raw,roles,family);masks[family]=target
                        A,lo,hi=constraints(raw,target,c);k=minimum[str(ci)][family]['overlap']
                        A=np.vstack([A,target]);lo=np.append(lo,k);hi=np.append(hi,k)
                        for j,seed in enumerate(block['edge_seeds'][family]):
                            budget.check();cost=np.random.default_rng(seed).uniform(-1,1,raw.nnz)
                            mask,log=solve(A,lo,hi,cost,c,c['draw_mip_gap']);arm=f'{family}_control{j}';masks[arm]=mask
                            info[arm]=dict(seed=seed,**audit_mask(raw,target,mask,c,k),solver=log,cost_sha256=hashlib.sha256(cost.tobytes()).hexdigest())
                    name=f'c{ci}_s{block["seed"]}';np.savez_compressed(out/(name+'.npz'),**masks)
                    core.write_json(out/(name+'.json'),dict(cohort=cohort,circuit_seed=ci,seed=block['seed'],raw_sha256=core.weight_hash(raw),controls=info))
                    print(f'Prepared {name}:15 prospective controls',flush=True)
        usage=budget.close();seal(out,c,context,environment=core.environment(),budget=usage,task_outcomes_inspected=False)
        print('Mask bank sealed; no neural task outcomes executed',flush=True)
    except Exception as exc:
        core.write_json(out/'failure.json',dict(error=repr(exc)));raise
    finally:budget.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=Path('configs/pathway_memory.json'));p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();build(args.config,args.out)
