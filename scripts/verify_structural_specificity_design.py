"""Independent M8 source, swap, invariant, code-rank and delay-layout replay.

No imports from the main M8 generator/normalizer/prototype/summary algorithms.
"""
import argparse
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse,linalg
from threadpoolctl import threadpool_limits
from verify_temporal_memory_curve import partial_source,validate_whole_source
from tdc_pool import pooled
from specificity_design_support import config,read,write,sha,array_sha,attempt,seal,check_manifest,history_preserved,source_record,source_unchanged

def matrix_hash(a):
    a=a.tocsr();return hashlib.sha256(a.data.tobytes()+a.indices.tobytes()+a.indptr.tobytes()).hexdigest()

def source(c,level):
    small,small_ids,small_roles=partial_source()
    if level=='legacy5':raw,ids,roles=small,small_ids,small_roles
    else:
        raw=sparse.load_npz(Path(c['cache'])/'brain5.npz').tocsr()
        with np.load(Path(c['cache'])/'nodes.npz',allow_pickle=False) as n:ids=n['ids'].copy();roles=n['roles'].copy()
    raw.sort_indices();roles=np.asarray(roles,dtype='U5')
    index=pd.Index(ids).get_indexer(small_ids);assert (index>=0).all()
    return raw,ids,roles,index[np.flatnonzero(small_roles=='KC')],index[np.flatnonzero(small_roles=='MBON')]

def normalize_independent(raw,gain):
    m=raw.copy().tocsr();strength=np.asarray(abs(m).sum(axis=1)).ravel();factor=np.zeros(len(strength))
    factor[strength>0]=gain/strength[strength>0];m.data*=np.repeat(factor,np.diff(m.indptr));m.sort_indices();return m

def independent_swaps(raw,roles,seed,swaps,maximum=200,resources=None):
    a=raw.tocoo();u=a.col.copy();v=a.row.copy();groups={};eligible=0
    for i in range(raw.nnz):
        if roles[u[i]]=='KC' and roles[v[i]]=='MBON':continue
        eligible+=1;key=(str(roles[u[i]]),str(roles[v[i]]),float(a.data[i]));groups.setdefault(key,[]).append(i)
    groups={key:indices for key,indices in groups.items() if len({int(u[i]) for i in indices})>1 and len({int(v[i]) for i in indices})>1}
    pool=sorted(i for indices in groups.values() for i in indices)
    existing={(int(x),int(y)) for x,y in zip(u,v)};random=np.random.Generator(np.random.PCG64(seed))
    requested=swaps*eligible;accepted=proposals=0
    while pool and accepted<requested and proposals<maximum*requested:
        proposals+=1
        if resources is not None and proposals%10000==0:resources.check()
        first=pool[int(random.integers(0,len(pool)))];key=(str(roles[u[first]]),str(roles[v[first]]),float(a.data[first]))
        candidates=groups[key];second=candidates[int(random.integers(0,len(candidates)))]
        x,y,z,t=int(u[first]),int(v[first]),int(u[second]),int(v[second])
        invalid=(x==z or y==t or x==t or z==y or (x,t) in existing or (z,y) in existing)
        if invalid:continue
        existing.difference_update([(x,y),(z,t)]);existing.update([(x,t),(z,y)]);v[first]=t;v[second]=y;accepted+=1
    output=sparse.csr_matrix((a.data.copy(),(v,u)),shape=raw.shape);output.sort_indices()
    return output,dict(requested_swaps=requested,accepted_swaps=accepted,attempts=proposals,eligible_delayed_edges=eligible,
        proposal_pool_edges=len(pool),proposal_groups=len(groups),proposal_pool_sha256=array_sha(np.asarray(pool,dtype=np.int64)),finite_budget_complete=accepted==requested)

def invariants(raw,null,roles,log,c):
    assert raw.shape==null.shape and raw.nnz==null.nnz and null.has_canonical_format and not null.diagonal().any()
    assert (null.data!=0).all() and np.isfinite(null.data).all()
    for axis in [0,1]:np.testing.assert_array_equal(np.asarray((raw!=0).sum(axis=axis)),np.asarray((null!=0).sum(axis=axis)))
    for method in ['tocsr','tocsc']:
        a,b=getattr(raw,method)(),getattr(null,method)()
        np.testing.assert_array_equal(a.indptr,b.indptr)
        for i in range(raw.shape[0]):np.testing.assert_array_equal(np.sort(a.data[a.indptr[i]:a.indptr[i+1]]),np.sort(b.data[b.indptr[i]:b.indptr[i+1]]))
    a,b=raw.tocoo(),null.tocoo();labels=np.unique(roles);blocks=np.zeros((len(labels),len(labels)),dtype=np.int64)
    for ti,post in enumerate(labels):
        for si,pre in enumerate(labels):
            count=int(np.sum((roles[a.col]==pre)&(roles[a.row]==post)))
            assert count==np.sum((roles[b.col]==pre)&(roles[b.row]==post));blocks[ti,si]=count
    kc=np.flatnonzero(roles=='KC');mbon=np.flatnonzero(roles=='MBON')
    assert (raw[mbon][:,kc]!=null[mbon][:,kc]).nnz==0
    strength=np.asarray(abs(raw).sum(axis=1)).ravel();np.testing.assert_array_equal(strength,np.asarray(abs(null).sum(axis=1)).ravel())
    w,nw=normalize_independent(raw,c['gain']),normalize_independent(null,c['gain']);assert (w[mbon][:,kc]!=nw[mbon][:,kc]).nnz==0
    mass=np.asarray(abs(nw).sum(axis=1)).ravel();np.testing.assert_allclose(mass[mass>0],c['gain'],rtol=0,atol=1e-14)
    locked=int(np.sum((roles[a.col]=='KC')&(roles[a.row]=='MBON')));delayed=raw.nnz-locked
    common=int((raw!=0).multiply(null!=0).nnz)-locked;assert delayed==log['eligible_delayed_edges']
    ready=log['accepted_swaps']==log['requested_swaps'] and 10*common<=9*delayed and matrix_hash(raw)!=matrix_hash(null)
    return dict(neurons=raw.shape[0],edges=raw.nnz,protected_current_edges=locked,delayed_edges=delayed,common_delayed_edges=common,
        delayed_overlap=common/delayed,total_edge_overlap=(common+locked)/raw.nnz,exact_degrees_and_both_raw_weight_multisets=True,
        exact_role_blocks=True,exact_incoming_strength_and_factors=True,exact_protected_raw_and_normalized_coefficients=True,no_invalid_edges=True,
        role_labels=labels.tolist(),role_block_counts=blocks.tolist(),original_raw_sha256=matrix_hash(raw),null_raw_sha256=matrix_hash(null),
        original_normalized_sha256=matrix_hash(w),null_normalized_sha256=matrix_hash(nw),protected_normalized_sha256=matrix_hash(w[mbon][:,kc]),
        construction_rule_pass=bool(ready),normalized_outgoing_multisets_constrained=False,**log)

def compare(a,b):
    if isinstance(b,dict):
        assert set(a)==set(b)
        for k in b:compare(a[k],b[k])
    elif isinstance(b,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):compare(x,y)
    elif isinstance(b,float):np.testing.assert_allclose(a,b,atol=1e-10,rtol=1e-10)
    else:assert a==b,(a,b)

def definitions(c,level,root,out):
    with attempt(out,c,'m8-independent-delay-definition'):
        check_manifest(root);raw,ids,roles,pool,obs=source(c,level);w=normalize_independent(raw,c['gain'])
        canonical=np.asarray(sorted(range(48),key=lambda j:int(ids[obs[j]])),dtype=np.int64)
        observed=np.zeros(48,dtype=np.int64)
        for k,j in enumerate(canonical):observed[j]=512+k if k<24 else 584+k-24
        rows=[];cols=[]
        for k in range(24):rows.extend([560+k,584+k]);cols.extend([536+k,560+k])
        d=sparse.csr_matrix((np.ones(48),(rows,cols)),shape=raw.shape);d.sort_indices()
        saved=sparse.load_npz(root/'delay-transition.npz').tocsr();assert matrix_hash(saved)==matrix_hash(d)
        assert d.nnz==48 and (d*d).nnz==24 and (d*d*d).nnz==0 and len(ids)>=608
        required=dict(root_ids=ids,roles=roles,input_pool_indices=pool,observed_source_indices=obs,canonical_order=canonical,
            delay_observed_indices=observed,original_incoming_strength=np.asarray(abs(raw).sum(axis=1)).ravel())
        with np.load(root/'definition.npz',allow_pickle=False) as a:
            assert set(a.files)==set(required)
            for n,v in required.items():np.testing.assert_array_equal(a[n],v)
        info=dict(level=level,source_neurons=len(ids),active_reference_coordinates=608,idle_reference_coordinates=len(ids)-608,delay_shift_edges=48,
            delay_squared_edges=24,delay_cubed_edges=0,delay_transition_sha256=matrix_hash(d),observed_coordinates=48,decoder_parameter_budget_per_lag=490,
            source_raw_sha256=matrix_hash(raw),source_normalized_sha256=matrix_hash(w),observation_root_ids=ids[obs].astype(str).tolist(),
            full_current_access_not_empirically_tested=True,new_decoder_fits=0,source_graph_neural_trajectories=0)
        compare(read(root/'definition.json'),info);cells=[]
        for seed in c['input_seeds'][level]:
            random=np.random.Generator(np.random.PCG64(seed));code=np.zeros((10,512))
            for s in range(10):code[s,random.choice(512,51,replace=False)]=.5
            prototypes=np.stack([.6*np.tanh(w[obs][:,pool]@(.6*np.tanh(row))) for row in code])
            spectra={}
            for name,indices in [('current',canonical[:24]),('past',canonical[24:])]:
                augmented=np.concatenate([prototypes[:,indices],np.ones((10,1))],axis=1)
                spectra[name]=linalg.svd(augmented,compute_uv=False,lapack_driver='gesvd')
            ranks={name:int(np.count_nonzero(s/s[0]>1e-12)) for name,s in spectra.items()}
            cell=dict(seed=seed,current_rank=ranks['current'],past_rank=ranks['past'],numerical_affine_certificate_pass=ranks['current']==10 and ranks['past']==10)
            expected=dict(prototypes=prototypes,current_singular_values=spectra['current'],past_singular_values=spectra['past'])
            with np.load(root/f'prototypes-s{seed}.npz',allow_pickle=False) as a:
                assert set(a.files)==set(expected)
                for n,v in expected.items():np.testing.assert_allclose(a[n],v,atol=1e-10,rtol=1e-10)
            np.savez_compressed(out/f'prototypes-s{seed}.npz',**expected);cells.append(cell)
        compare(read(root/'cells.json'),cells);write(out/'cells.json',cells);write(out/'checks.json',dict(all_checks_pass=True,root_mapping_rebuilt=True,
            numerical_ranks_recomputed_with_gesvd=True,delay_nilpotency_independently_verified=True,new_decoder_fits=0))
    seal(out,dict(complete=True));return cells

def graph_job(job):
    c,l,s,result,out=job['config'],job['level'],job['seed'],Path(job['result']),Path(job['out'])
    root=result/'graphs'/f'{l}_g{s}'
    with threadpool_limits(1),attempt(out,c,'m8-independent-graph') as resources:
        check_manifest(root);raw,ids,roles,pool,obs=source(c,l)
        rebuilt,log=independent_swaps(raw,roles,s,c['swaps_per_delayed_edge'][l],c['max_proposals_per_requested_swap'],resources)
        saved=sparse.load_npz(root/'raw.npz').tocsr()
        for name in ['data','indices','indptr']:np.testing.assert_array_equal(getattr(saved,name),getattr(rebuilt,name))
        report=invariants(raw,rebuilt,roles,log,c);report.update(level=l,seed=s);compare(read(root/'audit.json'),report)
        write(out/'audit.json',report);write(out/'checks.json',dict(all_checks_pass=True,exact_raw_regeneration=True,independent_both_multisets_and_degrees=True,
            exact_raw_normalized_protected_block=True,integer_overlap_gate_recomputed=True,result_manifest_sha256=sha(root/'manifest.json')))
    seal(out,dict(complete=True));return dict(identity=f'{l}/g{s}',audit=report)

def summary(graphs,maps,c,smoke):
    levels={}
    for l in c['levels']:
        members=sorted([x for x in graphs if x['level']==l],key=lambda x:x['seed']);cells=maps[l]
        assert [x['seed'] for x in members]==c['graph_seeds'][l] and [x['seed'] for x in cells]==c['input_seeds'][l]
        distinct=len({x['null_raw_sha256'] for x in members})==len(members)
        ready=None if smoke else distinct and all(x['construction_rule_pass'] for x in members) and all(x['numerical_affine_certificate_pass'] for x in cells)
        levels[l]=dict(registered_ready=ready,graphs=len(members),maps=len(cells),distinct_graphs=distinct,
            minimum_delayed_overlap=min(x['delayed_overlap'] for x in members),maximum_delayed_overlap=max(x['delayed_overlap'] for x in members),
            all_swap_budgets_complete=all(x['finite_budget_complete'] for x in members),all_delay_code_certificates_pass=all(x['numerical_affine_certificate_pass'] for x in cells))
    ready=levels[c['primary_level']]['registered_ready']
    return dict(smoke=smoke,outcome='smoke' if smoke else 'READY' if ready else 'NOT_READY',primary_ready=ready,levels=levels,graph_constructions=len(graphs),
        input_maps=sum(len(v) for v in maps.values()),new_decoder_fits=0,source_graph_neural_trajectories=0,performance_experiment_executed=False,
        real_wiring_superiority_tested=False,criterion_unchanged=True)

def verify(result,out):
    manifest=check_manifest(result);smoke=manifest['smoke'];c=config(smoke);assert read(result/'config.json')==c
    captured=source_record(c)
    with attempt(out,c,'m8-independent-validation') as resources:
        write(out/'source.json',captured);write(out/'config.json',c);write(out/'historical-preservation.json',history_preserved())
        write(out/'source-graph-audit.json',validate_whole_source(c,resources))
        maps={l:definitions(c,l,result/'definitions'/l,out/'definitions'/l) for l in c['levels']}
        jobs=[dict(config=c,level=l,seed=s,result=str(result),out=str(out/'graphs'/f'{l}_g{s}')) for l in c['levels'] for s in c['graph_seeds'][l]]
        results,usage=pooled(graph_job,jobs,1,c,'M8 independent construction')
        reports=sorted([x['audit'] for x in results],key=lambda x:(x['level'],x['seed']));recomputed=summary(reports,maps,c,smoke)
        compare(read(result/'structural-audits.json'),reports);compare(read(result/'summary.json'),recomputed)
        write(out/'recomputed-summary.json',recomputed);write(out/'process-tree-resources.json',usage)
        write(out/'checks.json',dict(all_checks_pass=True,outcome=recomputed['outcome'],primary_ready=recomputed['primary_ready'],graphs=len(reports),input_maps=sum(map(len,maps.values())),
            independent_ranks=2*sum(map(len,maps.values())),all_raw_graphs_exact=True,all_protected_coefficients_exact=True,source_graph_neural_trajectories=0,
            new_decoder_fits=0,integer_overlap_tolerance_applied=False,result_manifest_sha256=sha(result/'manifest.json')))
        source_unchanged(captured)
    seal(out,dict(complete=True,source_commit=captured['source_commit'],result_manifest_sha256=sha(result/'manifest.json')))
    print(dict(all_checks_pass=True,outcome=recomputed['outcome'],graphs=len(reports)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    with threadpool_limits(1):verify(a.result,a.out)
