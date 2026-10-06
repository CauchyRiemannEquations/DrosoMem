"""Registered matched graph ensembles applied to the frozen P1 iid TDC."""
import argparse
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits

from flying.brain.mushroom_body import role_shuffled
from flying.training import whole_brain_memory as core
from temporal_memory_curve import execute
from tdc_support import read, write, sha, attempt, seal, source_record, check_manifest, old_tree_unchanged
from tdc_pool import pooled

CONFIG = Path('configs/temporal_null_ensemble.json')


def config():
    override = read(CONFIG)
    base = read(override['base_config'])
    c = dict(base, **override)
    assert c['null_seeds']['legacy5'] == list(range(920001, 920021))
    assert c['null_seeds']['brain5'] == list(range(930001, 930011))
    assert c['blocks_by_level'] == {'legacy5': list(range(910001, 910007)), 'brain5': list(range(910001, 910004))}
    assert c['swaps_per_edge'] == {'legacy5': 5, 'brain5': 1}
    assert c['score_lags'] == list(range(1,21)) and c['minimum_score_difference'] == .03
    assert c['primary_level'] == 'legacy5' and (c['graph_workers'], c['neural_workers']) == (2,4)
    assert c['train_samples'] == 4000 and c['test_samples'] == 2000
    parent = check_manifest(Path(c['parent']))
    assert sha(Path(c['parent'])/'manifest.json') == c['parent_manifest_sha256']
    validation = check_manifest(Path(c['parent_validation']))
    checks = read(Path(c['parent_validation'])/'checks.json')
    assert checks['all_checks_pass'] and checks['result_manifest_sha256'] == c['parent_manifest_sha256']
    assert parent['config'] == base and parent['smoke'] is False
    return c


def sources(c):
    s = source_record(c, CONFIG)
    for p in [Path('scripts/tdc_pool.py'), Path(c['base_config']), Path(c['parent'])/'manifest.json', Path(c['parent_validation'])/'manifest.json']:
        s['hashes'][p.as_posix()] = sha(p)
    return s


def raw_graph(c, level):
    directory = Path('data/flywire_783_mb_left_kc512_s701')
    raw, ids, meta = core.load_connectome(directory)
    roles, _ = core.load_roles(directory, ids)
    if level == 'brain5':
        cache = Path(c['cache'])
        provenance = read(cache/'provenance.json')
        for name in ['brain5.npz', 'nodes.npz']:
            assert sha(cache/name) == provenance['files'][name]
        raw = sparse.load_npz(cache/'brain5.npz').tocsr()
        with np.load(cache/'nodes.npz', allow_pickle=False) as a:
            roles = a['roles'].copy()
    raw.sort_indices()
    return raw, roles


def outgoing_multiset(a):
    b = a.tocsc(copy=True)
    for j in range(b.shape[1]):
        b.data[b.indptr[j]:b.indptr[j+1]].sort()
    return b.indptr, b.data


def structural_audit(original, null, roles, log, swaps):
    assert original.shape == null.shape and original.nnz == null.nnz
    assert null.has_canonical_format and not null.diagonal().any() and (null.data != 0).all()
    assert log['accepted_swaps'] == log['requested_swaps'] == swaps*original.nnz
    np.testing.assert_array_equal(np.diff(original.indptr), np.diff(null.indptr))
    oc, nc = original.tocsc(), null.tocsc()
    np.testing.assert_array_equal(np.diff(oc.indptr), np.diff(nc.indptr))
    for a, b in zip(outgoing_multiset(original), outgoing_multiset(null)):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(np.sort(original.data), np.sort(null.data))
    labels, code = np.unique(roles, return_inverse=True)
    r = len(labels)
    source, rewired = original.tocoo(), null.tocoo()
    count = lambda a: np.bincount(code[a.row]*r+code[a.col], minlength=r*r).reshape(r,r)
    x, y = count(source), count(rewired)
    np.testing.assert_array_equal(x, y)
    common = (original != 0).multiply(null != 0).tocoo()
    overlap = np.bincount(code[common.row]*r+code[common.col], minlength=r*r).reshape(r,r)
    blocks = {f'{pre}->{post}': dict(edges=int(x[j,i]), common_edges=int(overlap[j,i]))
              for i, pre in enumerate(labels) for j, post in enumerate(labels)}
    actual_overlap = common.nnz/original.nnz
    assert abs(actual_overlap-log['edge_overlap']) < 1e-15
    delta = np.asarray(abs(null).sum(axis=1)-abs(original).sum(axis=1)).ravel()
    return dict(neurons=original.shape[0], edges=original.nnz, exact_in_degree=True, exact_out_degree=True,
                exact_role_blocks=True, exact_presynaptic_signed_weight_multisets=True,
                exact_global_weight_multiset=True, no_self_duplicate_zero_edges=True,
                source_raw_weight_sha256=core.weight_hash(original), null_raw_weight_sha256=core.weight_hash(null),
                overlap=actual_overlap, role_blocks=blocks, swap_log=log,
                incoming_strength_changed_nodes=int(np.count_nonzero(delta)),
                incoming_strength_mean_absolute_change=float(abs(delta).mean()),
                normalization='independent incoming-L1; normalized outgoing multisets not constrained')


def graph_job(job):
    c, level, seed, root = job['config'], job['level'], job['seed'], Path(job['root'])
    with threadpool_limits(1):
        with attempt(root, c, 'p2-null-graph') as resources:
            raw, roles = raw_graph(c, level)
            null, log = role_shuffled(raw, roles, seed, c['swaps_per_edge'][level])
            resources.check()
            audit = structural_audit(raw, null, roles, log, c['swaps_per_edge'][level])
            sparse.save_npz(root/'raw.npz', null)
            write(root/'audit.json', audit)
        seal(root, dict(complete=True, level=level, graph_seed=seed))
    return dict(identity=f'{level}/g{seed}', level=level, graph_seed=seed, audit=audit, usage=read(root/'resources.json'))


def prepare(out):
    c = config()
    c['graphs'] = out.as_posix()
    s = sources(c)
    with attempt(out, c, 'p2-graph-ensemble'):
        write(out/'config.json', c)
        write(out/'source.json', s)
        jobs = [dict(config=c, level=level, seed=seed, root=str(out/f'{level}_g{seed}'))
                for level in c['levels'] for seed in c['null_seeds'][level]]
        results, usage = pooled(graph_job, jobs, c['graph_workers'], c, 'P2 graph generation')
        assert len(results) == 30
        for level in c['levels']:
            members = [a for a in results if a['level'] == level]
            assert len({a['audit']['null_raw_weight_sha256'] for a in members}) == len(c['null_seeds'][level])
        write(out/'structural-audits.json', sorted(results, key=lambda x:(x['level'],x['graph_seed'])))
        write(out/'process-tree-resources.json', usage)
        write(out/'historical-preservation.json', old_tree_unchanged(c['baseline_commit']))
        for n,h in s['hashes'].items():
            assert sha(n) == h, n
    seal(out, dict(complete=True, config=c, source_commit=s['source_commit']))


def parent_pairing(arrays, c, block, level, real, smoke):
    parent = Path(c['parent'])/f'{level}_s{block["seed"]}'
    with np.load(parent/'case.npz', allow_pickle=False) as saved:
        for n in ['input_patterns', 'observed_indices']:
            np.testing.assert_array_equal(arrays[n], saved[n])
        for n in ['train_symbols', 'test_symbols', 'ytrain', 'ytest']:
            np.testing.assert_array_equal(arrays[n], saved[n][:len(arrays[n])])
        if real:
            if smoke:
                for n in ['train_features','test_features','train_norms','test_norms']:
                    np.testing.assert_array_equal(arrays[n], saved[n][:len(arrays[n])])
            else:
                for n in arrays:
                    np.testing.assert_array_equal(arrays[n], saved[n], err_msg=n)
    return dict(parent_case_sha256=sha(parent/'case.npz'), streams_inputs_labels_paired=True,
                real_arrays_reproduced=bool(real), smoke_prefix_only=bool(smoke))


def neural_job(job):
    c, block, level, graph_id, root = job['config'], job['block'], job['level'], job['graph_id'], Path(job['root'])
    raw = None if graph_id == 'real' else sparse.load_npz(Path(c['graphs'])/f'{level}_{graph_id}'/'raw.npz').tocsr()
    with threadpool_limits(1):
        with attempt(root, c, 'p2-tdc-case') as resources:
            arrays, rows, info = execute(c, block, level, resources, raw)
            info['graph_id'] = graph_id
            if raw is not None:
                info['raw_weight_sha256'] = core.weight_hash(raw)
            pairing = parent_pairing(arrays, c, block, level, graph_id == 'real', job['smoke'])
            np.savez_compressed(root/'case.npz', **arrays)
            write(root/'metrics.json', rows)
            write(root/'graph.json', info)
            write(root/'pairing.json', pairing)
        seal(root, dict(complete=True, block=block, level=level, graph_id=graph_id))
    return dict(identity=f'{level}/{graph_id}/s{block["seed"]}', rows=[dict(row,graph_id=graph_id) for row in rows], usage=read(root/'resources.json'))


def exact_score(data, k):
    accuracies = [Fraction(int(r.correct), int(r.samples)) for r in data.itertuples()]
    return (sum(accuracies, Fraction(0))/len(accuracies)-Fraction(1,k))/(1-Fraction(1,k))


def summarize(f, c, smoke):
    scores, blocks, summary = [], [], {}
    historical = f[f.lag.isin(c['score_lags'])]
    for (level, graph_id), data in historical.groupby(['level','graph_id']):
        score = exact_score(data, c['alphabet_size'])
        scores.append(dict(level=level, graph_id=graph_id, score=float(score), numerator=score.numerator, denominator=score.denominator))
        for seed, subset in data.groupby('seed'):
            blocks.append(dict(level=level, graph_id=graph_id, seed=int(seed), score=float(exact_score(subset,c['alphabet_size']))))
    score_frame = pd.DataFrame(scores)
    block_frame = pd.DataFrame(blocks)
    for level in c['levels']:
        a = score_frame[score_frame.level == level]
        real_row = a[a.graph_id == 'real'].iloc[0]
        real = Fraction(int(real_row.numerator), int(real_row.denominator))
        null_rows = a[a.graph_id != 'real']
        null = [Fraction(int(r.numerator), int(r.denominator)) for r in null_rows.itertuples()]
        mean = sum(null,Fraction(0))/len(null)
        sd = float(np.std([float(v) for v in null],ddof=1)) if len(null)>1 else 0.
        delta = real-mean
        gate = None if smoke else bool(all(real>v for v in null) and delta >= Fraction(str(c['minimum_score_difference'])))
        summary[level] = dict(real_score=float(real), null_count=len(null), null_mean=float(mean),
                              null_median=float(np.median([float(v) for v in null])), null_sd=sd,
                              null_min=float(min(null)), null_max=float(max(null)), real_minus_null_mean=float(delta),
                              real_minus_null_mean_exact=str(delta),
                              empirical_percentile=100*(sum(v<real for v in null)+.5*sum(v==real for v in null))/len(null),
                              standardized_effect=float(delta)/sd if sd else None, real_above_all_nulls=all(real>v for v in null),
                              registered_rule_pass=gate, fixed_input_blocks=len(c['blocks_by_level'][level]))
    real = block_frame[block_frame.graph_id == 'real'].rename(columns={'score':'real_score'})
    paired = block_frame[block_frame.graph_id != 'real'].merge(real[['level','seed','real_score']],on=['level','seed'])
    paired['real_minus_null'] = paired.real_score-paired.score
    curve = f.groupby(['level','graph_id','lag']).accuracy.mean().reset_index()
    real_curve = curve[curve.graph_id == 'real'][['level','lag','accuracy']].rename(columns={'accuracy':'real_accuracy'})
    null_curve = curve[curve.graph_id != 'real'].groupby(['level','lag']).accuracy.agg(['mean','median','std','min','max']).reset_index()
    lag_delta = real_curve.merge(null_curve,on=['level','lag'])
    lag_delta['real_minus_null_mean'] = lag_delta.real_accuracy-lag_delta['mean']
    result = dict(smoke=smoke, primary_pass=summary['legacy5']['registered_rule_pass'], levels=summary,
                  cases=int(f[['level','graph_id','seed']].drop_duplicates().shape[0]), lag_heads=len(f),
                  score_definition='unweighted mean (accuracy-.1)/.9 over historical lag1-20 and fixed paired blocks', criterion_unchanged=True)
    return result, score_frame, block_frame, paired, lag_delta


def plot(f, scores, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,2,figsize=(12,8))
    for i, level in enumerate(['legacy5','brain5']):
        curve = f[(f.level == level)&(f.lag>0)].groupby(['graph_id','lag']).accuracy.mean().reset_index()
        for graph_id, data in curve[curve.graph_id != 'real'].groupby('graph_id'):
            axes[i,0].plot(data.lag,data.accuracy,color='#a5a5a5',alpha=.45,lw=1)
        real = curve[curve.graph_id == 'real']
        axes[i,0].plot(real.lag,real.accuracy,'o-',color='#1469a1',label='real wiring',lw=2)
        axes[i,0].axhline(.1,color='#555',ls=':',label='chance')
        axes[i,0].set(title=level,xlabel='Historical lag',ylabel='Test accuracy',ylim=(0,1.02))
        axes[i,0].legend(fontsize=8)
        a = scores[scores.level == level]
        null = a[a.graph_id != 'real'].score.to_numpy()
        real_score = float(a[a.graph_id == 'real'].iloc[0].score)
        axes[i,1].scatter(null,np.linspace(-.08,.08,len(null)),color='#777',label=f'{len(null)} matched nulls')
        axes[i,1].axvline(real_score,color='#1469a1',lw=2,label='real score')
        axes[i,1].axvline(null.mean(),color='#b76126',ls='--',label='null mean')
        axes[i,1].set(title=f'{level}: finite null score distribution',xlabel='Mean chance-adjusted accuracy, lags 1–20',yticks=[],ylim=(-.2,.2))
        axes[i,1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out/'null-ensemble.png',dpi=180)
    plt.close(fig)


def run(out, graphs, graph_validation, smoke):
    c = config()
    c['graphs'], c['graph_validation'] = graphs.as_posix(), graph_validation.as_posix()
    graph_manifest = check_manifest(graphs)
    check_manifest(graph_validation)
    structural = read(graph_validation/'checks.json')
    assert structural['all_checks_pass'] and structural['graph_manifest_sha256'] == sha(graphs/'manifest.json')
    assert graph_manifest['config']['null_seeds'] == c['null_seeds']
    if smoke:
        c.update({n:c['smoke'][n] for n in ['train_samples','test_samples']})
        c['blocks_by_level'] = {level:seeds[:1] for level,seeds in c['blocks_by_level'].items()}
    s = sources(c)
    s['hashes'][(graphs/'manifest.json').as_posix()] = sha(graphs/'manifest.json')
    s['hashes'][(graph_validation/'manifest.json').as_posix()] = sha(graph_validation/'manifest.json')
    with attempt(out,c,'p2-smoke' if smoke else 'p2-main'):
        write(out/'config.json',c)
        write(out/'source.json',s)
        jobs = []
        for level in c['levels']:
            ids = ['real'] + [f'g{seed}' for seed in (c['null_seeds'][level][:1] if smoke else c['null_seeds'][level])]
            for graph_id in ids:
                for block in c['blocks']:
                    if block['seed'] in c['blocks_by_level'][level]:
                        jobs.append(dict(config=c,block=block,level=level,graph_id=graph_id,
                                         root=str(out/f'{level}_{graph_id}_s{block["seed"]}'),smoke=smoke))
        results, usage = pooled(neural_job,jobs,c['neural_workers'],c,'P2 TDC')
        f = pd.DataFrame([row for result in results for row in result['rows']]).sort_values(['level','graph_id','seed','lag']).reset_index(drop=True)
        assert len(f) == (84 if smoke else 3339)
        summary, scores, blocks, paired, lags = summarize(f,c,smoke)
        f.to_csv(out/'raw-lags.csv',index=False)
        scores.to_csv(out/'graph-scores.csv',index=False)
        blocks.to_csv(out/'seed-scores.csv',index=False)
        paired.to_csv(out/'paired-seed-differences.csv',index=False)
        lags.to_csv(out/'lag-differences.csv',index=False)
        write(out/'summary.json',summary)
        write(out/'job-resources.json',[dict(identity=a['identity'],usage=a['usage']) for a in results])
        write(out/'process-tree-resources.json',usage)
        write(out/'historical-preservation.json',old_tree_unchanged(c['baseline_commit']))
        plot(f,scores,out)
        for n,h in s['hashes'].items():
            assert sha(n) == h,n
    seal(out,dict(complete=True,smoke=smoke,config=c,source_commit=s['source_commit'],
                  graph_manifest_sha256=sha(graphs/'manifest.json'),graph_validation_manifest_sha256=sha(graph_validation/'manifest.json')))
    print(summary,flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command',choices=['prepare','run'])
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--graphs',type=Path,default=Path('results/tdc_v2/p2_graphs'))
    p.add_argument('--graph-validation',type=Path,default=Path('results/tdc_v2/p2_graph_validation'))
    p.add_argument('--smoke',action='store_true')
    a = p.parse_args()
    with threadpool_limits(1):
        if a.command == 'prepare':
            prepare(a.out)
        else:
            run(a.out,a.graphs,a.graph_validation,a.smoke)
