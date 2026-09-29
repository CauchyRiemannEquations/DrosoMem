"""Preregistered annotation lesion with exact degree/input matched controls."""
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
from alphabet_memory import SymbolEncoder, symbol_bank, read, check
from frozen_state_probe import Budget, labels, fit_fold, verify_fold, measures
from context_memory import estimate
import structural_k4


def matched_masks(raw, roles, bank, gamma, seeds):
    incoming = np.asarray((raw != 0).sum(axis=1)).ravel()
    outgoing = np.asarray((raw != 0).sum(axis=0)).ravel()
    exposure = (bank.T != 0) @ (1 << np.arange(4))
    strata = list(zip(incoming.tolist(), outgoing.tolist(), exposure.tolist()))
    counts = Counter(strata[i] for i in np.flatnonzero(gamma))
    pool = {key: np.array([i for i, s in enumerate(strata) if roles[i] == 'KC' and s == key]) for key in counts}
    masks = {'intact': np.zeros(len(roles), bool), 'gamma': gamma.copy()}
    for j, seed in enumerate(seeds):
        rng = np.random.default_rng(seed); mask = np.zeros(len(roles), bool)
        for key in sorted(counts):
            mask[rng.choice(pool[key], counts[key], replace=False)] = True
        assert Counter(strata[i] for i in np.flatnonzero(mask)) == counts
        masks[f'matched{j}'] = mask
    return masks, np.asarray(strata), [dict(in_degree=k[0], out_degree=k[1], input_mask=k[2],
        target_count=counts[k], pool_count=len(pool[k])) for k in sorted(counts)]


def lesion(weights, bank, dead):
    keep = sparse.diags((~dead).astype(float))
    w = (keep @ weights @ keep).tocsr(); w.eliminate_zeros(); w.sort_indices()
    effective = bank.copy(); effective[:, dead] = 0
    return w, effective


class AuditedReservoir(core.TimedReservoir):
    def step(self, symbol):
        state = super().step(symbol)
        assert np.isfinite(state).all() and np.max(abs(state)) <= 1 + 1e-12
        assert np.all(state[self.dead] == 0), 'Ablated neurons must remain zero'
        self.checked_steps += 1
        return state


def build(c, circuit, block, arm):
    model, obs, graph, raw = structural_k4.build(c, circuit, block['seed'])
    directory = Path(f'data/flywire_783_mb_left_kc512_s{circuit}')
    _, ids, _ = core.load_connectome(directory); roles, _ = core.load_roles(directory, ids)
    annotations = pd.read_csv(directory/'annotations.csv', dtype=str).set_index('root_id').loc[list(map(str, ids))]
    gamma = np.array([role == 'KC' and str(cell).startswith('KCg') for role, cell in zip(roles, annotations.cell_type)])
    bank = model.encoder.patterns.copy()
    masks, strata, audit = matched_masks(raw, roles, bank, gamma, block['match_seeds'])
    dead = masks[arm]; weights, effective = lesion(model.weights, bank, dead)
    surviving, _ = lesion(raw, bank, dead)
    graph.update(arm=arm, prelesion_weight_sha256=graph['weight_sha256'], weight_sha256=core.weight_hash(weights),
        removed_root_ids=[ids[i] for i in np.flatnonzero(dead)], removed_count=int(dead.sum()),
        overlap_with_gamma=int((dead & gamma).sum()), gamma_count=int(gamma.sum()),
        gamma_overlap_fraction=float((dead & gamma).sum()/gamma.sum()),
        removed_edges=raw.nnz-surviving.nnz, remaining_edges=surviving.nnz,
        removed_absolute_strength=float(abs(raw).sum()-abs(surviving).sum()),
        removed_input_counts=np.count_nonzero(bank[:, dead], axis=1).tolist(),
        effective_input_root_ids=[[ids[i] for i in np.flatnonzero(p)] for p in effective],
        match_strata=audit, annotation_sha256=core.sha256(directory/'annotations.csv'),
        normalized_outgoing_weights_preserved=None, raw_outgoing_weights_preserved=None,
        surviving_weights_unchanged=True, renormalized_after_lesion=False)
    model = AuditedReservoir(weights, SymbolEncoder(effective), roles, c['leak'], c['schedule'])
    model.dead = dead; model.checked_steps = 0
    extra = dict(dead_mask=dead, gamma_mask=gamma, matching_strata=strata,
                 intended_input_patterns=bank, observed_indices=obs, input_patterns=effective)
    return model, obs, graph, raw, extra


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


def baseline():
    path = Path('results/structural_k4_main/real_c701_s34142'); m = check(path)
    a, g, metrics, neural, _, _ = structural_k4.execute(m['config'], 701, m['identity'], 'real')
    with np.load(path/'checkpoint.npz', allow_pickle=False) as saved:
        assert set(saved.files) == set(a)
        for key in a: np.testing.assert_array_equal(saved[key], a[key], err_msg=key)
    assert metrics == read(path/'metrics.json') and neural == read(path/'neural.json')
    return dict(source=path.as_posix(), manifest_sha256=core.sha256(path/'manifest.json'),
        exact_all_arrays=True, exact_all_metrics=True, expected_metrics=metrics, actual_metrics=metrics)


def statistics(rows, c):
    f = pd.DataFrame(rows); columns = ['test_accuracy','frequency_excess','null_excess','r2_vs_frequency']
    cases = f[f.lag.isin(c['primary_lags'])].groupby(['seed','circuit_seed','arm'])[columns].mean().reset_index()
    matched = cases[cases.arm.str.startswith('matched')].groupby(['seed','circuit_seed'])[columns].mean().reset_index().assign(arm='matched_mean')
    cells = pd.concat([cases[~cases.arm.str.startswith('matched')], matched])
    blocks = cells.groupby(['seed','arm'])[columns].mean().reset_index()
    t = blocks.pivot(index='seed', columns='arm', values='test_accuracy')
    differences = pd.DataFrame(dict(extra_impairment=t.matched_mean-t.gamma,
        intact_minus_gamma=t.intact-t.gamma, intact_minus_matched=t.intact-t.matched_mean))
    intact = blocks[blocks.arm == 'intact']; effect = differences.extra_impairment
    access = bool((intact.frequency_excess >= .05).all() and (intact.null_excess >= .05).all()
        and intact.frequency_excess.mean() >= .05 and intact.null_excess.mean() >= .05 and intact.r2_vs_frequency.mean() > 0)
    def cell_stats(x):
        r = estimate(x, c)
        for key in ['paired_dz','wins','ties','losses']: r.pop(key)
        return r
    summary = dict(gates=dict(intact_past_access=access, gamma_specific_impairment=bool(access and effect.mean() >= c['material_effect'] and (effect > 0).all())),
        cells={arm: {k: cell_stats(b[k]) for k in columns} for arm, b in blocks.groupby('arm')},
        differences={k: estimate(differences[k], c) for k in differences})
    return cases, blocks, differences, summary


def plot(rows, out, c):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    f = pd.DataFrame(rows); fig, ax = plt.subplots(figsize=(8, 4))
    for arm in c['conditions']:
        y = f[f.arm == arm].groupby('lag').test_accuracy.mean().reindex(c['lags'])
        ax.plot(range(len(y)), y, 'o-', label=arm, alpha=.6 if arm.startswith('matched') else 1)
    ax.set(xticks=range(len(c['lags'])), xticklabels=c['lags'], xlabel='Symbol lag', ylabel='Independent-stream refit accuracy', ylim=(0,1.03))
    ax.axhline(.25, ls=':', color='gray'); ax.legend(); fig.tight_layout(); fig.savefig(out/'lag-curve.png', dpi=180); plt.close(fig)


def validate(c):
    expected=dict(conditions=['intact','gamma','matched0','matched1','matched2'],circuit_seeds=[701,702],
        alphabet_size=4,warmup=100,train_samples=2000,test_samples=1000,gain=.9,leak=.6,
        normalization='incoming_l1',schedule='mbon_after_kc',input_fraction=.1,input_amplitude=.5,
        alpha=1.,lags=[0,1,2,3,4,5,8,12,16,24,32],primary_lags=[1,2,3,4,5,8],material_effect=.05,
        bootstrap_seed=64399,bootstrap_draws=10000)
    for key,value in expected.items():
        if c[key]!=value: raise ValueError(f'Preregistered value changed: {key}')
    blocks=[dict(seed=61142+i,train_seed=62142+i,test_seed=63142+i,match_seeds=list(range(64142+3*i,64145+3*i))) for i in range(3)]
    if c['blocks']!=blocks: raise ValueError('Preregistered blocks changed')


@threadpool_limits.wrap(limits=1)
def run(config, out):
    c = read(config); validate(c); out.mkdir(parents=True, exist_ok=False); budget = Budget(c)
    try:
        context = core.source_context(c)
        for path in [str(config), __file__, 'scripts/structural_k4.py','scripts/alphabet_memory.py','scripts/frozen_state_probe.py','scripts/context_memory.py','scripts/verify_kc_ablation.py']:
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
        core.write_json(out/'verification.json',dict(exact_full_replays=35,independent_lstsq=35,prior_results_unchanged=len(prior),budget=usage))
        core.write_json(out/'manifest.json',dict(config=c,context=context,timestamp=datetime.now(timezone.utc).isoformat(),
            environment=core.environment(),artifacts={p.relative_to(out).as_posix():core.sha256(p) for p in out.rglob('*') if p.is_file()}))
        print(dict(gates=summary['gates'],usage=usage),flush=True)
    finally: budget.close()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/kc_ablation.json'))
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();run(args.config,args.out)
