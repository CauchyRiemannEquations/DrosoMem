"""Preregistered disjoint KCg/non-KCg subset follow-up, same observed cohort."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import time
import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits
from flying.training import whole_brain_memory as core
from alphabet_memory import SymbolEncoder, read, check
from frozen_state_probe import Budget, labels, fit_fold, verify_fold, measures
from context_memory import estimate
from kc_ablation import AuditedReservoir, lesion, baseline, plot
import kc_ablation


def exclusive_masks(gamma, matched):
    return gamma & ~matched, matched & ~gamma


def build(c, circuit, block, arm):
    pair=int(arm[-1]);model,obs,g,raw,original=kc_ablation.build(c,circuit,block,f'matched{pair}')
    cohort='smoke' if block['seed']==61001 else 'main'
    source=Path(c['source'])/cohort/f'matched{pair}_c{circuit}_s{block["seed"]}'
    check(source)
    with np.load(source/'checkpoint.npz',allow_pickle=False) as old:
        for key in ['dead_mask','gamma_mask','intended_input_patterns','matching_strata']:
            np.testing.assert_array_equal(original[key],old[key])
    gamma,matched=original['gamma_mask'],original['dead_mask']
    gm,nm=exclusive_masks(gamma,matched);strata=original['matching_strata']
    assert gm.sum()==nm.sum() and not (gm&nm).any()
    assert Counter(map(tuple,strata[gm]))==Counter(map(tuple,strata[nm]))
    dead=gm if arm.startswith('gamma') else nm
    data=Path(f'data/flywire_783_mb_left_kc512_s{circuit}')
    _,ids,_=core.load_connectome(data);roles,_=core.load_roles(data,ids)
    bank=original['intended_input_patterns'];w=core.normalize_condition(raw,c['normalization'],c['gain'])
    w,effective=lesion(w,bank,dead);surviving,_=lesion(raw,bank,dead)
    g.pop('match_strata')
    g.update(arm=arm,removed_root_ids=[ids[i] for i in np.flatnonzero(dead)],removed_count=int(dead.sum()),
        overlap_with_gamma=int((dead&gamma).sum()),gamma_overlap_fraction=float((dead&gamma).sum()/gamma.sum()),
        removed_edges=raw.nnz-surviving.nnz,remaining_edges=surviving.nnz,
        removed_absolute_strength=float(abs(raw).sum()-abs(surviving).sum()),
        removed_input_counts=np.count_nonzero(bank[:,dead],axis=1).tolist(),
        effective_input_root_ids=[[ids[i] for i in np.flatnonzero(p)] for p in effective],
        weight_sha256=core.weight_hash(w),source_mask_manifest_sha256=core.sha256(source/'manifest.json'),
        source_mask_checkpoint_sha256=core.sha256(source/'checkpoint.npz'),source_mask_path=source.as_posix(),
        common_intersection_count=int((gamma&matched).sum()),pair=pair,exclusive_joint_strata_match=True)
    model=AuditedReservoir(w,SymbolEncoder(effective),roles,c['leak'],c['schedule']);model.dead=dead;model.checked_steps=0
    a=dict(dead_mask=dead,gamma_mask=gamma,matching_strata=strata,intended_input_patterns=bank,
        input_patterns=effective,observed_indices=obs,source_matched_mask=matched,shared_mask=gamma&matched)
    return model,obs,g,raw,a


def statistics(rows,c):
    f=pd.DataFrame(rows);metrics=['test_accuracy','frequency_excess','null_excess','r2_vs_frequency']
    cases=f[f.lag.isin(c['primary_lags'])].groupby(['seed','circuit_seed','arm'])[metrics].mean().reset_index()
    cases['pair']=cases.arm.str[-1].astype(int);cases['population']=cases.arm.str[:-1]
    blocks=cases.groupby(['seed','population'])[metrics].mean().reset_index().rename(columns={'population':'arm'})
    t=blocks.pivot(index='seed',columns='arm',values='test_accuracy')
    delta=pd.DataFrame(dict(extra_impairment=t.nongamma-t.gamma))
    access=read(Path(c['source'])/'main/summary.json')['gates']['intact_past_access']
    effect=delta.extra_impairment
    def cell_stats(x):
        r=estimate(x,c)
        for key in ['paired_dz','wins','ties','losses']:r.pop(key)
        return r
    summary=dict(gates=dict(source_intact_past_access=access,subset_specific_impairment=bool(access and effect.mean()>=.05 and (effect>0).all())),
        cells={arm:{m:cell_stats(b[m]) for m in metrics} for arm,b in blocks.groupby('arm')},
        differences={'extra_impairment':estimate(effect,c)},independent_confirmation=False)
    return cases,blocks,delta,summary


def validate(c):
    source=check(Path(c['source']))
    assert core.sha256(Path(c['source'])/'manifest.json')==c['source_manifest_sha256']
    assert c['conditions']==[f'{arm}{j}' for j in range(3) for arm in ['gamma','nongamma']]
    for key,value in source['config'].items():
        if key not in ['protocol','conditions']:assert c[key]==value,key


def execute(c, circuit, block, arm):
    model, obs, graph, raw, a = build(c, circuit, block, arm)
    for split in ['train', 'test']:
        n = c['warmup'] + c[split+'_samples']
        symbols = np.random.default_rng(block[split+'_seed']).integers(0, 4, n).astype(np.uint8)
        features, diagnostics = core.collect(model, obs, symbols, c['activity_epsilon'])
        a[split+'_symbols'], a[split+'_features'] = symbols, features
        a.update({split+'_'+key: value for key, value in diagnostics.items()})
        if split == 'train':
            a['decay_full'], a['decay_observed'] = core.decay_probe(model, obs, c['decay_steps'])
    w = c['warmup']
    a.update(fit_fold(a['train_features'][w:], a['test_features'][w:],
        labels(a['train_symbols'], np.arange(w, len(a['train_symbols'])), c['lags']),
        labels(a['test_symbols'], np.arange(w, len(a['test_symbols'])), c['lags']), c['alpha']))
    metrics = measures([a], c['lags'])
    for j, row in enumerate(metrics):
        row['training_mse'] = float(np.mean((a['train_scores'][:, j]-np.eye(4)[a['ytrain'][:, j]])**2))
    x = a['train_features'][w:]
    a['centered_singular_values'] = np.linalg.svd(x-x.mean(axis=0), compute_uv=False)
    neural = dict(**core.state_diagnostics(x), observed_sparsity=float(np.mean(abs(x) <= c['activity_epsilon'])),
        mean_mbon_norm=float(np.linalg.norm(x, axis=1).mean()), mean_active_neurons=float(a['train_active_counts'][w:].mean()),
        mean_observed_cosine=float(a['train_observed_cosine'][w-1:].mean()), clipped_features=int(np.sum(x.std(axis=0) < 1e-5)),
        observed_decay_ratio32=float(a['decay_observed'][-1]/a['decay_observed'][0]),
        zero_lesion_state_steps=model.checked_steps)
    assert core.weight_hash(model.weights) == graph['weight_sha256']
    for value in a.values():
        assert np.isfinite(value).all()
    return a, graph, metrics, neural, raw, model.weights

@threadpool_limits.wrap(limits=1)
def run(config, out):
    c = read(config); validate(c); out.mkdir(parents=True, exist_ok=False); budget = Budget(c)
    try:
        context = core.source_context(c)
        for path in [str(config), __file__, 'scripts/structural_k4.py','scripts/alphabet_memory.py','scripts/frozen_state_probe.py','scripts/context_memory.py','scripts/verify_kc_exclusive.py','scripts/kc_ablation.py']:
            path = Path(path).resolve().relative_to(Path.cwd()).as_posix(); context['source_sha256'][path] = core.sha256(path)
        prior = {p: core.sha256(p) for p in subprocess.check_output(['git','ls-files','results/'], text=True).splitlines()}
        core.write_json(out/'prior-results-sha256.json', prior); core.write_json(out/'config.json', c)
        core.write_json(out/'baseline.json', baseline()); print('Historical full checkpoint and all metrics exactly reproduced', flush=True)
        for cohort in ['smoke','main']:
            cc = dict(c)
            if cohort == 'smoke': cc.update(c['smoke'])
            root = out/cohort; root.mkdir(); rows=[]; neuralrows=[]; graphrows=[]
            for block in cc['blocks']:
                for circuit in cc['circuit_seeds']:
                    for arm in cc['conditions']:
                        budget.check(); start=time.perf_counter(); dest=root/f'{arm}_c{circuit}_s{block["seed"]}'; dest.mkdir()
                        a,g,m,n,raw,w = execute(cc,circuit,block,arm)
                        np.savez_compressed(dest/'checkpoint.npz', **a); sparse.save_npz(dest/'raw-graph.npz',raw); sparse.save_npz(dest/'weights.npz',w)
                        for name,value in [('graph',g),('metrics',m),('neural',n)]: core.write_json(dest/f'{name}.json',value)
                        b,g2,m2,n2,_,_ = execute(cc,circuit,block,arm)
                        assert g == g2 and m == m2 and n == n2
                        with np.load(dest/'checkpoint.npz',allow_pickle=False) as saved:
                            for key in b: np.testing.assert_array_equal(saved[key],b[key],err_msg=key)
                            verify_fold(dict(saved),cc['alpha'])
                        identity=dict(arm=arm,circuit_seed=circuit,**block)
                        rows.extend(dict(**identity,**r) for r in m); neuralrows.append(dict(**identity,**n))
                        graphrows.append(dict(**identity,**{k:g[k] for k in ['removed_count','overlap_with_gamma','gamma_count','gamma_overlap_fraction','removed_edges','remaining_edges','removed_absolute_strength','removed_input_counts']}))
                        core.write_json(dest/'manifest.json',dict(config=cc,identity=identity,context=context,
                            timestamp=datetime.now(timezone.utc).isoformat(),exact_replay=True,independent_lstsq=True,
                            coefficients_per_head=196,seconds_including_replay=time.perf_counter()-start,
                            artifacts={p.name:core.sha256(p) for p in dest.iterdir() if p.is_file()}))
                        print(f'{cohort} {arm} c{circuit} s{block["seed"]}: exact replay and independent refit verified',flush=True)
            pd.DataFrame(rows).to_csv(root/'raw-lag-table.csv',index=False)
            pd.DataFrame(neuralrows).to_csv(root/'neural-diagnostics.csv',index=False)
            pd.DataFrame(graphrows).to_csv(root/'lesion-audit.csv',index=False)
            if cohort == 'main':
                cases,blocks,differences,summary=statistics(rows,c)
                cases.to_csv(root/'raw-past-table.csv',index=False); blocks.to_csv(root/'seed-blocks.csv',index=False)
                differences.to_csv(root/'paired-differences.csv'); core.write_json(root/'summary.json',summary)
            plot(rows,root,c)
        for path,digest in {**prior,**context['source_sha256']}.items(): assert core.sha256(path)==digest,path
        budget.check(); usage=budget.close()
        core.write_json(out/'verification.json',dict(exact_full_replays=42,independent_lstsq=42,prior_results_unchanged=len(prior),budget=usage))
        core.write_json(out/'manifest.json',dict(config=c,context=context,timestamp=datetime.now(timezone.utc).isoformat(),
            environment=core.environment(),artifacts={p.relative_to(out).as_posix():core.sha256(p) for p in out.rglob('*') if p.is_file()}))
        print(dict(gates=summary['gates'],usage=usage),flush=True)
    finally: budget.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/kc_exclusive.json'))
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();run(args.config,args.out)
