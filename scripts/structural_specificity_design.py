"""M8 construction-only feasibility; no source neural stepping or decoder fit."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits
from flying.training import whole_brain_memory as core
from flying.brain.diagnostics import normalize_condition
from flying.encoding.digit_encoder import DigitEncoder
from tdc_pool import pooled
from specificity_design_support import config,read,write,sha,array_sha,attempt,seal,history_preserved,source_record,source_unchanged

def canonical_ids(values):
    ids=np.asarray(values,dtype=np.int64)
    assert ids.ndim==1 and len(np.unique(ids))==len(ids)
    return ids

def original(c,level):
    folder=Path('data/flywire_783_mb_left_kc512_s701');small,small_ids,_=core.load_connectome(folder);small_roles,_=core.load_roles(folder,small_ids)
    small_ids=canonical_ids(small_ids)
    if level=='legacy5':raw,ids,roles=small,small_ids,small_roles
    else:
        cache=Path(c['cache']);provenance=read(cache/'provenance.json')
        for name in ['nodes.npz','brain5.npz']:assert sha(cache/name)==provenance['files'][name]
        raw=sparse.load_npz(cache/'brain5.npz').tocsr()
        with np.load(cache/'nodes.npz',allow_pickle=False) as n:ids=n['ids'].copy();roles=n['roles'].copy()
    roles=np.asarray(roles,dtype='U5');ids=canonical_ids(ids);raw.sort_indices()
    local=pd.Index(ids).get_indexer(small_ids);assert (local>=0).all()
    pool=local[np.flatnonzero(small_roles=='KC')];obs=local[np.flatnonzero(small_roles=='MBON')]
    assert len(pool)==512 and len(obs)==48
    return raw,ids,roles,pool,obs

def normalize(raw,gain=.9):
    w=normalize_condition(raw,'incoming_l1',gain);w.sort_indices();return w

def protected(raw,roles):
    coo=raw.tocoo();return (roles[coo.col]=='KC')&(roles[coo.row]=='MBON')

def proposal_groups(raw,roles):
    a=raw.tocoo();locked=protected(raw,roles);groups={}
    for i in np.flatnonzero(~locked):
        key=(str(roles[a.col[i]]),str(roles[a.row[i]]),float(a.data[i]));groups.setdefault(key,[]).append(int(i))
    groups={k:np.asarray(v,dtype=np.int64) for k,v in groups.items() if len(np.unique(a.col[v]))>=2 and len(np.unique(a.row[v]))>=2}
    pool=np.asarray(sorted(i for v in groups.values() for i in v),dtype=np.int64)
    return groups,pool,int((~locked).sum())

def rewire(raw,roles,seed,swaps,maximum=200,resources=None):
    a=raw.tocoo();pre=a.col.copy();post=a.row.copy();groups,pool,eligible=proposal_groups(raw,roles)
    active=set(zip(pre.tolist(),post.tolist()));rng=np.random.default_rng(seed)
    requested=swaps*eligible;accepted=attempts=0
    while len(pool) and accepted<requested and attempts<maximum*requested:
        attempts+=1
        if resources is not None and attempts%10000==0:resources.check()
        i=int(pool[int(rng.integers(len(pool)))])
        key=(str(roles[pre[i]]),str(roles[post[i]]),float(a.data[i]));candidates=groups[key]
        j=int(candidates[int(rng.integers(len(candidates)))]);u,v,x,y=map(int,(pre[i],post[i],pre[j],post[j]))
        if u==x or v==y or u==y or x==v or (u,y) in active or (x,v) in active:continue
        active.remove((u,v));active.remove((x,y));active.add((u,y));active.add((x,v));post[i],post[j]=y,v;accepted+=1
    result=sparse.csr_matrix((a.data.copy(),(post,pre)),shape=raw.shape);result.sort_indices()
    return result,dict(requested_swaps=requested,accepted_swaps=accepted,attempts=attempts,eligible_delayed_edges=eligible,
        proposal_pool_edges=len(pool),proposal_groups=len(groups),proposal_pool_sha256=array_sha(pool),finite_budget_complete=accepted==requested)

def multisets(m,axis):
    a=m.tocsr(copy=True) if axis=='incoming' else m.tocsc(copy=True)
    for i in range(a.shape[0]):a.data[a.indptr[i]:a.indptr[i+1]].sort()
    return a.indptr,a.data

def role_counts(raw,roles):
    labels,code=np.unique(roles,return_inverse=True);a=raw.tocoo();r=len(labels)
    return labels,np.bincount(code[a.row]*r+code[a.col],minlength=r*r).reshape(r,r)

def audit_graph(raw,null,roles,log,c):
    assert raw.shape==null.shape and raw.nnz==null.nnz and null.has_canonical_format
    assert not null.diagonal().any() and (null.data!=0).all() and np.isfinite(null.data).all()
    for axis in ['incoming','outgoing']:
        for a,b in zip(multisets(raw,axis),multisets(null,axis)):np.testing.assert_array_equal(a,b)
    labels,blocks=role_counts(raw,roles);np.testing.assert_array_equal(blocks,role_counts(null,roles)[1])
    kc=np.flatnonzero(roles=='KC');mbon=np.flatnonzero(roles=='MBON')
    assert (raw[mbon][:,kc]!=null[mbon][:,kc]).nnz==0
    strength=np.asarray(abs(raw).sum(axis=1)).ravel();np.testing.assert_array_equal(strength,np.asarray(abs(null).sum(axis=1)).ravel())
    w,nw=normalize(raw,c['gain']),normalize(null,c['gain'])
    assert (w[mbon][:,kc]!=nw[mbon][:,kc]).nnz==0
    mass=np.asarray(abs(nw).sum(axis=1)).ravel();np.testing.assert_allclose(mass[mass>0],c['gain'],atol=1e-14,rtol=0)
    p=int(protected(raw,roles).sum());delayed=raw.nnz-p;common=(raw!=0).multiply(null!=0).nnz-p
    assert delayed==log['eligible_delayed_edges'] and 0<=common<=delayed
    ready=log['finite_budget_complete'] and 10*common<=9*delayed and core.weight_hash(raw)!=core.weight_hash(null)
    return dict(neurons=raw.shape[0],edges=raw.nnz,protected_current_edges=p,delayed_edges=delayed,common_delayed_edges=common,
        delayed_overlap=common/delayed,total_edge_overlap=(common+p)/raw.nnz,exact_degrees_and_both_raw_weight_multisets=True,
        exact_role_blocks=True,exact_incoming_strength_and_factors=True,exact_protected_raw_and_normalized_coefficients=True,no_invalid_edges=True,
        role_labels=labels.tolist(),role_block_counts=blocks.tolist(),original_raw_sha256=core.weight_hash(raw),null_raw_sha256=core.weight_hash(null),
        original_normalized_sha256=core.weight_hash(w),null_normalized_sha256=core.weight_hash(nw),protected_normalized_sha256=core.weight_hash(w[mbon][:,kc]),
        construction_rule_pass=bool(ready),normalized_outgoing_multisets_constrained=False,**log)

def delay_layout(n,observed_root_ids):
    assert n>=608 and len(observed_root_ids)==48 and len(set(observed_root_ids))==48
    order=np.argsort(np.asarray(observed_root_ids,dtype=np.int64),kind='stable')
    observed=np.empty(48,dtype=np.int64);observed[order[:24]]=np.arange(512,536);observed[order[24:]]=np.arange(584,608)
    rows=np.r_[np.arange(560,584),np.arange(584,608)];cols=np.r_[np.arange(536,560),np.arange(560,584)]
    d=sparse.csr_matrix((np.ones(48),(rows,cols)),shape=(n,n));assert (d@d).nnz==24 and (d@d@d).nnz==0
    return d,order,observed

def spectra(prototypes,order,tolerance):
    singular={name:np.linalg.svd(np.column_stack([prototypes[:,order[part]],np.ones(10)]),compute_uv=False) for name,part in [('current',slice(0,24)),('past',slice(24,48))]}
    ranks={name:int(np.sum(s>tolerance*s[0])) for name,s in singular.items()}
    return singular,ranks

def definition(c,level):
    raw,ids,roles,pool,obs=original(c,level);w=normalize(raw,c['gain']);n=len(ids)
    d,order,observed=delay_layout(n,ids[obs])
    tables=[]
    for seed in c['input_seeds'][level]:
        code=DigitEncoder(512,seed,c['input_fraction'],c['input_amplitude']).patterns.copy()
        p=c['drive']*np.tanh((c['drive']*np.tanh(code))@w[obs][:,pool].T)
        singular,ranks=spectra(p,order,c['rank_relative_tolerance'])
        tables.append(dict(seed=seed,prototypes=p,current_singular_values=singular['current'],past_singular_values=singular['past'],
            current_rank=ranks['current'],past_rank=ranks['past'],numerical_affine_certificate_pass=all(v==10 for v in ranks.values())))
    arrays=dict(root_ids=ids,roles=roles,input_pool_indices=pool,observed_source_indices=obs,canonical_order=order,delay_observed_indices=observed,
        original_incoming_strength=np.asarray(abs(raw).sum(axis=1)).ravel())
    info=dict(level=level,source_neurons=n,active_reference_coordinates=608,idle_reference_coordinates=n-608,delay_shift_edges=48,delay_squared_edges=24,
        delay_cubed_edges=0,delay_transition_sha256=core.weight_hash(d),observed_coordinates=48,decoder_parameter_budget_per_lag=490,
        source_raw_sha256=core.weight_hash(raw),source_normalized_sha256=core.weight_hash(w),observation_root_ids=ids[obs].astype(str).tolist(),
        full_current_access_not_empirically_tested=True,new_decoder_fits=0,source_graph_neural_trajectories=0)
    return d,arrays,tables,info

def definition_job(c,level,out):
    with attempt(out,c,'m8-analytic-delay-definition'):
        d,a,tables,info=definition(c,level);sparse.save_npz(out/'delay-transition.npz',d);np.savez_compressed(out/'definition.npz',**a)
        cells=[]
        for row in tables:
            seed=row['seed'];np.savez_compressed(out/f'prototypes-s{seed}.npz',**{k:v for k,v in row.items() if isinstance(v,np.ndarray)})
            cells.append({k:v for k,v in row.items() if not isinstance(v,np.ndarray)})
        write(out/'cells.json',cells);write(out/'definition.json',info)
    seal(out,dict(complete=True,level=level));return cells

def graph_job(job):
    c,l,seed,out=job['config'],job['level'],job['seed'],Path(job['out'])
    with threadpool_limits(1),attempt(out,c,'m8-equal-weight-graph') as resources:
        raw,ids,roles,pool,obs=original(c,l);null,log=rewire(raw,roles,seed,c['swaps_per_delayed_edge'][l],c['max_proposals_per_requested_swap'],resources)
        report=audit_graph(raw,null,roles,log,c);report.update(level=l,seed=seed)
        sparse.save_npz(out/'raw.npz',null);write(out/'audit.json',report)
    seal(out,dict(complete=True,level=l,seed=seed));return dict(identity=f'{l}/g{seed}',audit=report)

def summarize(graphs,maps,c,smoke):
    levels={}
    for l in c['levels']:
        a=sorted([x for x in graphs if x['level']==l],key=lambda x:x['seed']);p=maps[l]
        assert [x['seed'] for x in a]==c['graph_seeds'][l] and [x['seed'] for x in p]==c['input_seeds'][l]
        distinct=len({x['null_raw_sha256'] for x in a})==len(a)
        ready=None if smoke else all(x['construction_rule_pass'] for x in a) and distinct and all(x['numerical_affine_certificate_pass'] for x in p)
        levels[l]=dict(registered_ready=ready,graphs=len(a),maps=len(p),distinct_graphs=distinct,
            minimum_delayed_overlap=min(x['delayed_overlap'] for x in a),maximum_delayed_overlap=max(x['delayed_overlap'] for x in a),
            all_swap_budgets_complete=all(x['finite_budget_complete'] for x in a),all_delay_code_certificates_pass=all(x['numerical_affine_certificate_pass'] for x in p))
    ready=levels[c['primary_level']]['registered_ready']
    return dict(smoke=smoke,outcome='smoke' if smoke else 'READY' if ready else 'NOT_READY',primary_ready=ready,levels=levels,
        graph_constructions=len(graphs),input_maps=sum(len(v) for v in maps.values()),new_decoder_fits=0,source_graph_neural_trajectories=0,
        performance_experiment_executed=False,real_wiring_superiority_tested=False,criterion_unchanged=True)

def run(out,smoke=False):
    c=config(smoke);captured=source_record(c)
    with attempt(out,c,'m8-smoke' if smoke else 'm8-main'):
        write(out/'config.json',c);write(out/'source.json',captured);write(out/'historical-preservation.json',history_preserved())
        maps={l:definition_job(c,l,out/'definitions'/l) for l in c['levels']}
        jobs=[dict(config=c,level=l,seed=s,out=str(out/'graphs'/f'{l}_g{s}')) for l in c['levels'] for s in c['graph_seeds'][l]]
        results,usage=pooled(graph_job,jobs,1,c,'M8 construction')
        reports=sorted([v['audit'] for v in results],key=lambda x:(x['level'],x['seed']));summary=summarize(reports,maps,c,smoke)
        write(out/'structural-audits.json',reports);write(out/'summary.json',summary);write(out/'process-tree-resources.json',usage)
        pd.DataFrame(reports).drop(columns=['role_labels','role_block_counts']).to_csv(out/'graph-summary.csv',index=False)
        source_unchanged(captured)
    seal(out,dict(complete=True,smoke=smoke,source_commit=captured['source_commit']));print(summary,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    with threadpool_limits(1):run(a.out,a.smoke)
