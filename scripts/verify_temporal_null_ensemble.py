"""Independent null regeneration, structural checks and frozen TDC replay.

Uses P1's independent raw loaders/manual dynamics/SVD fitter; never imports
the P2 experiment, its graph generator, scalar or summary implementation.
"""
import argparse
from fractions import Fraction
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits

from verify_temporal_memory_curve import partial_source, validate_whole_source, verify_case
from tdc_support import read, write, sha, attempt, seal, check_manifest, environment, git, old_tree_unchanged
from tdc_pool import pooled


def source(c,level):
    raw,ids,roles = partial_source()
    if level == 'brain5':
        raw = sparse.load_npz(Path(c['cache'])/'brain5.npz').tocsr()
        with np.load(Path(c['cache'])/'nodes.npz',allow_pickle=False) as a:
            roles = a['roles'].copy()
    return raw,roles


def replay_swaps(raw,roles,seed,swaps):
    # Independent implementation of the registered RNG proposal/acceptance order.
    coo = raw.tocoo()
    presynaptic,postsynaptic = coo.col.copy(),coo.row.copy()
    groups = {}
    for index in range(raw.nnz):
        key = (roles[presynaptic[index]],roles[postsynaptic[index]])
        groups.setdefault(key,[]).append(index)
    groups = {key:np.asarray(indices,dtype=np.int64) for key,indices in groups.items()}
    active = {(int(u),int(v)) for u,v in zip(presynaptic,postsynaptic)}
    original = active.copy()
    random = np.random.default_rng(seed)
    requested = raw.nnz*swaps
    accepted = proposals = 0
    while accepted < requested and proposals < requested*30:
        proposals += 1
        i = int(random.integers(raw.nnz))
        key = (roles[presynaptic[i]],roles[postsynaptic[i]])
        j = int(random.choice(groups[key]))
        first_pre,first_post = int(presynaptic[i]),int(postsynaptic[i])
        second_pre,second_post = int(presynaptic[j]),int(postsynaptic[j])
        invalid = (first_pre == second_pre or first_post == second_post or
                   first_pre == second_post or second_pre == first_post or
                   (first_pre,second_post) in active or (second_pre,first_post) in active)
        if invalid:
            continue
        active.remove((first_pre,first_post))
        active.remove((second_pre,second_post))
        active.update([(first_pre,second_post),(second_pre,first_post)])
        postsynaptic[i],postsynaptic[j] = second_post,first_post
        accepted += 1
    result = sparse.csr_matrix((coo.data.copy(),(postsynaptic,presynaptic)),shape=raw.shape)
    return result,dict(requested_swaps=requested,accepted_swaps=accepted,attempts=proposals,edge_overlap=len(active&original)/raw.nnz)


def matrix_hash(a):
    a = a.tocsr()
    return hashlib.sha256(a.data.tobytes()+a.indices.tobytes()+a.indptr.tobytes()).hexdigest()


def independent_invariants(original,rewired,roles):
    assert original.shape == rewired.shape and original.nnz == rewired.nnz
    assert rewired.has_canonical_format and not rewired.diagonal().any() and (rewired.data != 0).all()
    for axis in [0,1]:
        np.testing.assert_array_equal(np.asarray((original!=0).sum(axis=axis)),np.asarray((rewired!=0).sum(axis=axis)))
    a,b = original.tocsc(),rewired.tocsc()
    for neuron in range(original.shape[1]):
        start,stop = a.indptr[neuron:neuron+2]
        new_start,new_stop = b.indptr[neuron:neuron+2]
        assert np.array_equal(np.sort(a.data[start:stop]),np.sort(b.data[new_start:new_stop])),neuron
    np.testing.assert_array_equal(np.sort(original.data),np.sort(rewired.data))
    a,b = original.tocoo(),rewired.tocoo()
    common = (original!=0).multiply(rewired!=0).tocoo()
    blocks = {}
    for pre in np.unique(roles):
        for post in np.unique(roles):
            source_count = int(np.sum((roles[a.col]==pre)&(roles[a.row]==post)))
            new_count = int(np.sum((roles[b.col]==pre)&(roles[b.row]==post)))
            assert source_count == new_count
            shared = int(np.sum((roles[common.col]==pre)&(roles[common.row]==post)))
            blocks[f'{pre}->{post}'] = dict(edges=source_count,common_edges=shared)
    return dict(neurons=original.shape[0],edges=original.nnz,role_blocks=blocks,overlap=common.nnz/original.nnz,
                original_raw_sha256=matrix_hash(original),rewired_raw_sha256=matrix_hash(rewired),
                exact_node_degrees=True,exact_presynaptic_signed_multisets=True,no_invalid_edges=True)


def structural_job(job):
    c,level,seed,graphs,out = job['config'],job['level'],job['seed'],Path(job['graphs']),Path(job['out'])
    with threadpool_limits(1):
        with attempt(out,c,'p2-independent-graph-regeneration') as resources:
            root = graphs/f'{level}_g{seed}'
            m = check_manifest(root)
            assert m['graph_seed'] == seed and m['level'] == level
            raw,roles = source(c,level)
            expected,log = replay_swaps(raw,roles,seed,c['swaps_per_edge'][level])
            saved = sparse.load_npz(root/'raw.npz').tocsr()
            assert (saved != expected).nnz == 0
            assert matrix_hash(saved) == matrix_hash(expected)
            assert log['accepted_swaps'] == log['requested_swaps'] == c['swaps_per_edge'][level]*raw.nnz
            audit = independent_invariants(raw,saved,roles)
            main = read(root/'audit.json')
            assert main['swap_log'] == log
            assert main['source_raw_weight_sha256'] == audit['original_raw_sha256']
            assert main['null_raw_weight_sha256'] == audit['rewired_raw_sha256']
            assert main['role_blocks'] == audit['role_blocks']
            assert abs(main['overlap']-audit['overlap']) < 1e-15
            resources.check()
            write(out/'audit.json',audit)
            write(out/'swap-replay.json',log)
        seal(out,dict(complete=True,graph_file_sha256=sha(root/'raw.npz')))
    return dict(identity=f'{level}/g{seed}',level=level,graph_seed=seed,audit=audit,usage=read(out/'resources.json'))


def verify_graphs(graphs,out):
    m = check_manifest(graphs)
    c = m['config']
    s = read(graphs/'source.json')
    for n,h in s['hashes'].items():
        assert sha(n) == h,n
    with attempt(out,c,'p2-independent-ensemble-audit') as resources:
        write(out/'environment.json',environment())
        full = validate_whole_source(c,resources)
        write(out/'source-graph-audit.json',full)
        jobs = [dict(config=c,level=level,seed=seed,graphs=str(graphs),out=str(out/f'{level}_g{seed}'))
                for level in c['levels'] for seed in c['null_seeds'][level]]
        results,usage = pooled(structural_job,jobs,c['graph_workers'],c,'P2 independent graph audit')
        assert len(results) == 30
        write(out/'member-checks.json',results)
        write(out/'process-tree-resources.json',usage)
        write(out/'checks.json',dict(all_checks_pass=True,graphs=30,independently_regenerated_nulls=30,
                                    graph_manifest_sha256=sha(graphs/'manifest.json'),verifier_commit=git('rev-parse','HEAD'),
                                    verifier_sha256=sha(__file__),source_graph_reconstruction=full))
    seal(out,dict(complete=True,graph_manifest_sha256=sha(graphs/'manifest.json')))


def neural_verification_job(job):
    result,out,c,block,level,graph_id = Path(job['result']),Path(job['out']),job['config'],job['block'],job['level'],job['graph_id']
    root = result/f'{level}_{graph_id}_s{block["seed"]}'
    m = check_manifest(root)
    assert m['level'] == level and m['graph_id'] == graph_id and m['block'] == block
    raw = None if graph_id == 'real' else sparse.load_npz(Path(c['graphs'])/f'{level}_{graph_id}'/'raw.npz').tocsr()
    with threadpool_limits(1):
        with attempt(out,c,'p2-independent-tdc-case') as resources:
            rows,check = verify_case(root,c,block,level,resources,raw)
            parent = Path(c['parent'])/f'{level}_s{block["seed"]}'
            pairing = read(root/'pairing.json')
            assert pairing['parent_case_sha256'] == sha(parent/'case.npz')
            with np.load(root/'case.npz',allow_pickle=False) as a,np.load(parent/'case.npz',allow_pickle=False) as p:
                for name in ['input_patterns','observed_indices']:
                    np.testing.assert_array_equal(a[name],p[name])
                for name in ['train_symbols','test_symbols','ytrain','ytest']:
                    np.testing.assert_array_equal(a[name],p[name][:len(a[name])])
                if graph_id == 'real' and not job['smoke']:
                    for name in a.files:
                        np.testing.assert_array_equal(a[name],p[name])
                elif graph_id == 'real':
                    for name in ['train_features','test_features','train_norms','test_norms']:
                        np.testing.assert_array_equal(a[name],p[name][:len(a[name])])
            write(out/'checks.json',dict(**check,graph_id=graph_id,parent_pairing_checked=True))
        seal(out,dict(complete=True,result_case_manifest_sha256=sha(root/'manifest.json')))
    return dict(identity=f'{level}/{graph_id}/s{block["seed"]}',rows=[dict(row,graph_id=graph_id) for row in rows],
                check=check,usage=read(out/'resources.json'))


def rational_score(rows,k):
    total = sum((Fraction(int(row['correct']),int(row['samples'])) for row in rows),Fraction(0))
    accuracy = total/len(rows)
    return (accuracy-Fraction(1,k))/Fraction(k-1,k)


def audit_summary(f,result,c,smoke):
    saved = pd.read_csv(result/'raw-lags.csv')
    pd.testing.assert_frame_equal(saved,f[saved.columns],check_exact=False,atol=1e-12,rtol=0)
    main = read(result/'summary.json')
    scores = {}
    history = f[f.lag.between(1,20)]
    expected_score_rows,seed_rows = [],[]
    for (level,graph_id),data in history.groupby(['level','graph_id']):
        score = rational_score(data.to_dict('records'),10)
        scores[(level,graph_id)] = score
        expected_score_rows.append(dict(level=level,graph_id=graph_id,score=float(score),numerator=score.numerator,denominator=score.denominator))
        for seed,part in data.groupby('seed'):
            seed_rows.append(dict(level=level,graph_id=graph_id,seed=seed,score=float(rational_score(part.to_dict('records'),10))))
    expected = pd.DataFrame(expected_score_rows)
    actual = pd.read_csv(result/'graph-scores.csv')
    pd.testing.assert_frame_equal(actual,expected[actual.columns],check_exact=False,atol=1e-12,rtol=0)
    seed_frame = pd.DataFrame(seed_rows)
    actual_seed = pd.read_csv(result/'seed-scores.csv')
    pd.testing.assert_frame_equal(actual_seed,seed_frame[actual_seed.columns],check_exact=False,atol=1e-12,rtol=0)
    results = {}
    for level in c['levels']:
        real = scores[(level,'real')]
        null = [score for (lv,g),score in scores.items() if lv == level and g != 'real']
        mean = sum(null,Fraction(0))/len(null)
        values = np.array(list(map(float,null)))
        sd = float(values.std(ddof=1)) if len(null)>1 else 0.
        exact_pass = None if smoke else bool(real>max(null) and real-mean >= Fraction(3,100))
        status = main['levels'][level]
        assert status['registered_rule_pass'] is exact_pass
        assert status['null_count'] == len(null)
        assert status['real_above_all_nulls'] is (real>max(null))
        assert status['real_minus_null_mean_exact'] == str(real-mean)
        assert status['fixed_input_blocks'] == len(c['blocks_by_level'][level])
        expected_values = dict(real_score=float(real),null_mean=float(mean),null_median=float(np.median(values)),
                               null_sd=sd,null_min=float(min(null)),null_max=float(max(null)),
                               real_minus_null_mean=float(real-mean),
                               empirical_percentile=100*(sum(v<real for v in null)+.5*sum(v==real for v in null))/len(null))
        for key,value in expected_values.items():
            assert abs(status[key]-value)<1e-12,key
        if sd:
            assert abs(status['standardized_effect']-float(real-mean)/sd)<1e-12
        else:
            assert status['standardized_effect'] is None
        results[level] = dict(registered_rule_pass=exact_pass,exact_real_score=str(real),exact_null_mean=str(mean),null_count=len(null))
    assert main['primary_pass'] is results['legacy5']['registered_rule_pass']
    # Check all paired input-block and lag summaries independently.
    paired = pd.read_csv(result/'paired-seed-differences.csv')
    for row in paired.itertuples():
        a = history[(history.level==row.level)&(history.graph_id==row.graph_id)&(history.seed==row.seed)]
        b = history[(history.level==row.level)&(history.graph_id=='real')&(history.seed==row.seed)]
        null_score,real_score = rational_score(a.to_dict('records'),10),rational_score(b.to_dict('records'),10)
        assert abs(row.score-float(null_score))<1e-12 and abs(row.real_score-float(real_score))<1e-12
        assert abs(row.real_minus_null-float(real_score-null_score))<1e-12
    curve = f.groupby(['level','graph_id','lag']).accuracy.mean()
    for row in pd.read_csv(result/'lag-differences.csv').itertuples():
        real = float(curve.loc[(row.level,'real',row.lag)])
        null = np.array([float(value) for (lv,g,lag),value in curve.items() if lv==row.level and g!='real' and lag==row.lag])
        assert abs(row.real_accuracy-real)<1e-12
        for key,value in {'mean':null.mean(),'median':np.median(null),'std':null.std(ddof=1) if len(null)>1 else np.nan,'min':null.min(),'max':null.max()}.items():
            actual_value = getattr(row,key)
            assert (np.isnan(value) and np.isnan(actual_value)) or abs(actual_value-value)<1e-12,key
        assert abs(row.real_minus_null_mean-(real-null.mean()))<1e-12
    assert main['cases'] == (4 if smoke else 159) and main['lag_heads'] == (84 if smoke else 3339)
    return results


def verify_run(result,out):
    m = check_manifest(result)
    c = read(result/'config.json')
    assert c == m['config']
    for n,h in read(result/'source.json')['hashes'].items():
        assert sha(n)==h,n
    graphs,structural = Path(c['graphs']),Path(c['graph_validation'])
    check_manifest(graphs)
    check_manifest(structural)
    assert sha(graphs/'manifest.json') == m['graph_manifest_sha256']
    assert sha(structural/'manifest.json') == m['graph_validation_manifest_sha256']
    assert read(structural/'checks.json')['all_checks_pass']
    with attempt(out,c,'p2-independent-main-validation') as resources:
        write(out/'environment.json',environment())
        write(out/'source-graph-audit.json',validate_whole_source(c,resources))
        jobs = []
        for level in c['levels']:
            ids = ['real']+[f'g{seed}' for seed in (c['null_seeds'][level][:1] if m['smoke'] else c['null_seeds'][level])]
            for graph_id in ids:
                for block in c['blocks']:
                    if block['seed'] in c['blocks_by_level'][level]:
                        jobs.append(dict(result=str(result),out=str(out/f'{level}_{graph_id}_s{block["seed"]}'),config=c,block=block,level=level,graph_id=graph_id,smoke=m['smoke']))
        results,usage = pooled(neural_verification_job,jobs,c['neural_workers'],c,'P2 independent TDC verification')
        f = pd.DataFrame([r for a in results for r in a['rows']]).sort_values(['level','graph_id','seed','lag']).reset_index(drop=True)
        endpoints = audit_summary(f,result,c,m['smoke'])
        f.to_csv(out/'recomputed-lags.csv',index=False)
        write(out/'process-tree-resources.json',usage)
        write(out/'job-resources.json',[dict(identity=a['identity'],usage=a['usage']) for a in results])
        exact_traces = sum(sum(a['check']['exact_full_state_trace_hash'].values()) for a in results)
        write(out/'checks.json',dict(all_checks_pass=True,cases=len(results),replayed_streams=2*len(results),
                                    independently_refit_lag_heads=len(f),exact_full_state_trace_hashes=exact_traces,
                                    endpoints=endpoints,result_manifest_sha256=sha(result/'manifest.json'),
                                    structural_validation_manifest_sha256=sha(structural/'manifest.json'),
                                    verifier_commit=git('rev-parse','HEAD'),verifier_sha256=sha(__file__),
                                    historical_preservation=old_tree_unchanged(c['baseline_commit'])))
    seal(out,dict(complete=True,result_manifest_sha256=sha(result/'manifest.json')))
    print(read(out/'checks.json'),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command',choices=['graphs','run'])
    p.add_argument('target',type=Path)
    p.add_argument('--out',type=Path,required=True)
    a = p.parse_args()
    with threadpool_limits(1):
        if a.command == 'graphs':
            verify_graphs(a.target,a.out)
        else:
            verify_run(a.target,a.out)
