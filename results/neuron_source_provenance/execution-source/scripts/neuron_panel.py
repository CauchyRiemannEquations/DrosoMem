"""Locked ACT III-A input-only and annotated neuron-population closeout."""
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
from frozen_state_probe import Budget, labels, fit_fold, measures
from normalization_transfer import verify_transfer
from context_memory import estimate
import structural_k4
import kc_ablation
import kc_frozen


def validate(c):
    # Exact preregistered configuration, including cohorts and stopping criteria.
    if core.fingerprint(c) != CONFIG_HASH:
        raise ValueError('Configuration differs from preregistration')


def anatomy(ci):
    directory = Path(f'data/flywire_783_mb_left_kc512_s{ci}')
    raw, ids, _ = core.load_connectome(directory)
    roles, _ = core.load_roles(directory, ids)
    roles = np.asarray(roles)
    annotations = pd.read_csv(directory/'annotations.csv', dtype=str).set_index('root_id').loc[list(map(str, ids))]
    return raw, ids, roles, annotations


def panel_masks(raw, ids, roles, cells, bank, block, c):
    incoming = np.asarray((raw != 0).sum(axis=1)).ravel()
    outgoing = np.asarray((raw != 0).sum(axis=0)).ravel()
    exposure = (bank.T != 0) @ (1 << np.arange(4))
    targets = {g: (roles == 'KC') & np.array([str(x).startswith(g) for x in cells]) for g in ['KCab','KCapbp']}
    targets.update(DAN=roles == 'DAN', APL=roles == 'APL')
    candidates = np.flatnonzero(roles != 'MBON')
    order = sorted(candidates, key=lambda i: (-int(incoming[i]+outgoing[i]), int(ids[i])))
    hub = np.zeros(len(ids), bool); hub[order[:int(np.ceil(c['hub_fraction']*len(candidates)))]] = True
    targets['hub'] = hub
    masks = {'intact': np.zeros(len(ids), bool)}
    for group in c['groups']:
        target = targets[group]; masks[group] = target
        if group in ['KCab','KCapbp']:
            strata = list(zip(incoming.tolist(), outgoing.tolist(), exposure.tolist())); eligible = roles == 'KC'
        else:
            strata = [(int(x),) for x in exposure]; eligible = roles != 'MBON'
        counts = Counter(strata[i] for i in np.flatnonzero(target))
        for j, seed in enumerate(block['control_seeds'][group]):
            rng = np.random.default_rng(seed); mask = np.zeros(len(ids), bool)
            for key in sorted(counts):
                pool = np.array([i for i in range(len(ids)) if eligible[i] and strata[i] == key])
                if len(pool) < counts[key]: raise ValueError('Infeasible registered matching')
                mask[rng.choice(pool, counts[key], replace=False)] = True
            assert Counter(strata[i] for i in np.flatnonzero(mask)) == counts
            masks[f'{group}_control{j}'] = mask
    masks.update(all_KC=roles == 'KC', all_MBON=roles == 'MBON')
    return masks


class MonitoredReservoir(core.TimedReservoir):
    def step(self, symbol):
        state = super().step(symbol)
        assert np.isfinite(state).all() and np.max(abs(state)) <= 1+1e-12
        assert np.all(state[self.dead] == 0)
        self.mask_active.append(int(np.count_nonzero(abs(state[self.mask]) > self.epsilon)))
        return state


def intervene(weights, bank, mask, input_only=False):
    effective = bank.copy(); effective[:, mask] = 0
    if input_only:
        return weights.copy(), effective, np.zeros(len(mask), bool)
    w, _ = kc_ablation.lesion(weights, bank, mask)
    return w, effective, mask.copy()


def execute(c, ci, block, arm, input_source=None):
    base, obs, graph, raw = structural_k4.build(c, ci, block['seed'])
    _, ids, roles, annotations = anatomy(ci)
    bank = base.encoder.patterns.copy()
    if input_source is not None:
        with np.load(input_source/'checkpoint.npz') as old:
            mask = old['dead_mask'].copy()
            np.testing.assert_array_equal(bank, old['intended_input_patterns'])
        group = 'gamma' if arm == 'gamma' else 'matched'
        target_mask = mask
    else:
        masks = panel_masks(raw, ids, roles, annotations.cell_type, bank, block, c)
        mask = masks[arm]; group = arm.split('_control')[0]; target_mask = masks[group]
    weights, effective, dead = intervene(base.weights, bank, mask, input_source is not None)
    model = MonitoredReservoir(weights, SymbolEncoder(effective), roles, c['leak'], c['schedule'])
    model.dead = dead; model.mask = mask; model.epsilon = c['activity_epsilon']; model.mask_active = []
    incoming = np.asarray((raw != 0).sum(axis=1)).ravel(); outgoing = np.asarray((raw != 0).sum(axis=0)).ravel()
    surviving, _ = kc_ablation.lesion(raw, bank, dead)
    graph.update(arm=arm, group=group, input_only=input_source is not None,
        prelesion_weight_sha256=graph['weight_sha256'], weight_sha256=core.weight_hash(weights),
        mask_root_ids=[ids[i] for i in np.flatnonzero(mask)], mask_count=int(mask.sum()),
        dead_count=int(dead.sum()), removed_observed=int(dead[obs].sum()),
        input_counts_removed=np.count_nonzero(bank[:,mask],axis=1).tolist(),
        mask_raw_in_degree_sum=int(incoming[mask].sum()), mask_raw_out_degree_sum=int(outgoing[mask].sum()),
        mask_zero_in_degree=int(np.count_nonzero(incoming[mask] == 0)),
        target_overlap=int((mask & target_mask).sum()), target_count=int(target_mask.sum()),
        removed_edges=int(raw.nnz-surviving.nnz), remaining_edges=int(surviving.nnz),
        removed_absolute_strength=float(abs(raw).sum()-abs(surviving).sum()),
        renormalized_after_lesion=False, annotation_sha256=core.sha256(Path(f'data/flywire_783_mb_left_kc512_s{ci}/annotations.csv')))
    a = dict(intervention_mask=mask, dead_mask=dead, intended_input_patterns=bank,
        input_patterns=effective, observed_indices=obs)
    for split in ['train','test']:
        symbols = np.random.default_rng(block[split+'_seed']).integers(0,4,c['warmup']+c[split+'_samples']).astype(np.uint8)
        model.mask_active = []
        features, diagnostics = core.collect(model, obs, symbols, c['activity_epsilon'])
        a[split+'_symbols'] = symbols; a[split+'_features'] = features
        a[split+'_mask_active'] = np.asarray(model.mask_active)
        a.update({split+'_'+key:value for key,value in diagnostics.items()})
        if split == 'train': a['decay_full'], a['decay_observed'] = core.decay_probe(model, obs, c['decay_steps'])
    w = c['warmup']; x = a['train_features'][w:]
    a.update(fit_fold(x, a['test_features'][w:],
        labels(a['train_symbols'],np.arange(w,len(a['train_symbols'])),c['lags']),
        labels(a['test_symbols'],np.arange(w,len(a['test_symbols'])),c['lags']),c['alpha']))
    m = measures([a],c['lags'])
    for j,row in enumerate(m): row['training_mse'] = float(np.mean((a['train_scores'][:,j]-np.eye(4)[a['ytrain'][:,j]])**2))
    a['centered_singular_values'] = np.linalg.svd(x-x.mean(axis=0),compute_uv=False)
    n = dict(**core.state_diagnostics(x), observed_sparsity=float(np.mean(abs(x)<=c['activity_epsilon'])),
        mean_mbon_norm=float(np.linalg.norm(x,axis=1).mean()),mean_active_neurons=float(a['train_active_counts'][w:].mean()),
        mean_mask_active=float(a['train_mask_active'][w:].mean()), mean_observed_cosine=float(a['train_observed_cosine'][w-1:].mean()),
        clipped_features=int(np.count_nonzero(x.std(axis=0)<1e-5)), zero_decay_baseline=bool(a['decay_observed'][0]==0),
        observed_decay_ratio32=float(a['decay_observed'][-1]/a['decay_observed'][0]) if a['decay_observed'][0] else None)
    for value in a.values(): assert np.isfinite(value).all()
    assert core.weight_hash(model.weights) == graph['weight_sha256']
    return a,graph,m,n,raw,weights


def save_case(dest,c,ci,block,arm,context,source=None,input_source=None):
    dest.mkdir(parents=True,exist_ok=False); start=time.perf_counter()
    a,g,m,n,raw,w = execute(c,ci,block,arm,input_source)
    replay,g2,m2,n2,_,_ = execute(c,ci,block,arm,input_source)
    assert g == g2 and m == m2 and n == n2
    for key in a: np.testing.assert_array_equal(a[key],replay[key],err_msg=key)
    np.savez_compressed(dest/'checkpoint.npz',**a); sparse.save_npz(dest/'weights.npz',w)
    for name,value in [('graph',g),('metrics',m),('neural',n)]: core.write_json(dest/f'{name}.json',value)
    fm = None
    if source is not None:
        sa,_ = kc_frozen.load(source)
        for key in ['intended_input_patterns','observed_indices','train_symbols','test_symbols','ytrain','ytest']:
            np.testing.assert_array_equal(sa[key],a[key],err_msg=key)
        f,fm = kc_frozen.evaluate(sa,a,c); verify_transfer(sa,a,f)
        # Full parameter/score checkpoint; target refit remains a separate archive.
        np.savez_compressed(dest/'frozen.npz',**f); core.write_json(dest/'frozen-metrics.json',fm)
    identity=dict(circuit_seed=ci,arm=arm,**block)
    seal(dest,c,context,identity=identity,source_path=source.as_posix() if source else None,
        source_manifest_sha256=core.sha256(source/'manifest.json') if source else None,
        full_lesion_path=input_source.as_posix() if input_source else None,
        full_lesion_manifest_sha256=core.sha256(input_source/'manifest.json') if input_source else None,
        exact_trajectory_replay=True,independent_refit=True,independent_frozen_source_refit=source is not None,
        coefficients_per_lag=196,seconds=time.perf_counter()-start)
    return a,g,m,n,fm


def seal(root,c,context,**extra):
    core.write_json(root/'manifest.json',dict(config=c,context=context,timestamp=datetime.now(timezone.utc).isoformat(),
        **extra,artifacts={p.relative_to(root).as_posix():core.sha256(p) for p in root.rglob('*') if p.is_file()}))


def material(impairment, excess, access):
    return bool(access and np.mean(impairment)>=.05 and np.mean(excess)>=.05
        and np.all(np.asarray(impairment)>0) and np.all(np.asarray(excess)>0))


def summarize_panel(rows,c,out):
    f=pd.DataFrame(rows); f.to_csv(out/'raw-lag-table.csv',index=False)
    metric=['test_accuracy','train_accuracy','frequency_excess','null_excess','r2_vs_frequency','training_mse']
    cases=f[f.lag.isin(c['primary_lags'])].groupby(['cohort','seed','circuit_seed','arm','mode'])[metric].mean().reset_index()
    blocks=cases.groupby(['cohort','seed','arm','mode'])[metric].mean().reset_index()
    cases.to_csv(out/'raw-past-table.csv',index=False); blocks.to_csv(out/'seed-blocks.csv',index=False)
    stats={};gates={};pairs=[]
    for cohort in ['discovery','confirmation']:
        base=blocks[(blocks.cohort==cohort)&(blocks.arm=='intact')&(blocks['mode']=='refit')].set_index('seed')
        access=bool((base.frequency_excess>=.05).all() and (base.null_excess>=.05).all() and base.r2_vs_frequency.mean()>0)
        for mode in ['refit','frozen']:
            for group in c['groups']:
                sub=blocks[(blocks.cohort==cohort)&(blocks['mode']==mode)]
                target=sub[sub.arm==group].set_index('seed')
                control=sub[sub.arm.str.startswith(group+'_control')].groupby('seed')[metric].mean()
                impairment=base.test_accuracy-target.test_accuracy; excess=control.test_accuracy-target.test_accuracy
                key=f'{cohort}/{mode}/{group}';gates[key]=material(impairment,excess,access)
                stats[key]=dict(target=estimate(target.test_accuracy,c),control=estimate(control.test_accuracy,c),
                    impairment=estimate(impairment,c),excess_impairment=estimate(excess,c),intact_access=access)
                for seed in target.index:
                    pairs.append(dict(cohort=cohort,seed=int(seed),group=group,mode=mode,
                        intact=float(base.test_accuracy[seed]),target=float(target.test_accuracy[seed]),control=float(control.test_accuracy[seed]),
                        impairment=float(impairment[seed]),excess_impairment=float(excess[seed])))
    pd.DataFrame(pairs).to_csv(out/'paired-differences.csv',index=False)
    summary=dict(statistics=stats,gates=gates,confirmed=[f'{mode}/{g}' for mode in ['refit','frozen'] for g in c['groups']
        if all(gates[f'{cohort}/{mode}/{g}'] for cohort in ['discovery','confirmation'])],cohorts_pooled=False)
    core.write_json(out/'summary.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,5))
    for ax,cohort in zip(axes,['discovery','confirmation']):
        values=np.array([[stats[f'{cohort}/{mode}/{g}']['excess_impairment']['mean']*100 for mode in ['frozen','refit']] for g in c['groups']])
        im=ax.imshow(values,cmap='RdBu_r',vmin=-60,vmax=60,aspect='auto')
        for (i,j),value in np.ndenumerate(values):ax.text(j,i,f'{value:+.2f}',ha='center',va='center')
        ax.set(xticks=[0,1],xticklabels=['Frozen','Refit'],yticks=range(len(c['groups'])),yticklabels=c['groups'],title=cohort)
    fig.colorbar(im,ax=axes.tolist(),label='Control minus target accuracy (percentage points)',shrink=.8)
    fig.savefig(out/'sensitivity-map.png',dpi=180,bbox_inches='tight');plt.close(fig)
    return summary


def summarize_input(rows,c,out):
    f=pd.DataFrame(rows);f.to_csv(out/'raw-lag-table.csv',index=False)
    metrics=['input_accuracy','lesion_accuracy','difference','max_feature_difference']
    cases=f[f.lag.isin(c['primary_lags'])].groupby(['cohort','seed','circuit_seed','arm','mode'])[metrics].mean().reset_index()
    cases['group']=np.where(cases.arm=='gamma','gamma','matched')
    blocks=cases.groupby(['cohort','seed','circuit_seed','group','mode'])[metrics].mean().groupby(['cohort','seed','group','mode']).mean().reset_index()
    blocks.to_csv(out/'seed-blocks.csv',index=False)
    summary={}
    for (group,mode),b in blocks[blocks.cohort=='main'].groupby(['group','mode']):
        summary[f'{group}/{mode}']=dict(difference=estimate(b.difference,c),gate=bool(b.difference.mean()>=.05 and (b.difference>0).all()),
            exactly_equal_all_features=bool((b.max_feature_difference==0).all()))
    core.write_json(out/'summary.json',summary)
    return summary


@threadpool_limits.wrap(limits=1)
def run(config,out,package):
    c=read(config);validate(c)
    assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip(),'Commit implementation before execution'
    check(Path(c['source']));assert core.sha256(Path(c['source'])/'manifest.json')==c['source_manifest_sha256']
    out.mkdir(parents=True,exist_ok=False);budget=Budget(c)
    try:
        context=core.source_context(c)
        for p in list(Path('scripts').glob('*.py'))+[config,Path('docs/neuron-panel-seed-audit.json'),Path('docs/neuron-panel-annotation-audit.json')]:
            context['source_sha256'][p.as_posix()]=core.sha256(p)
        prior={p:core.sha256(p) for p in subprocess.check_output(['git','ls-files','results/'],text=True).splitlines()}
        core.write_json(out/'prior-results-sha256.json',prior);core.write_json(out/'config.json',c)
        core.write_json(out/'baseline.json',kc_ablation.baseline());print('Archived baseline exact in every array and metric',flush=True)
        rows=[];neural=[];graphs=[];count=0;transfers=0
        if package=='input':
            for cohort,settings in c['input_cohorts'].items():
                cc=dict(c,**settings)
                for block in cc['blocks']:
                    for ci in cc['circuit_seeds']:
                        sp=Path(c['source'])/'conditions'/cohort/f'intact_c{ci}_s{block["seed"]}'
                        for arm in ['gamma','matched0','matched1','matched2']:
                            budget.check();tp=sp.parent/f'{arm}_c{ci}_s{block["seed"]}'
                            dest=out/cohort/f'{arm}_c{ci}_s{block["seed"]}'
                            a,g,m,n,fm=save_case(dest,cc,ci,block,arm,context,sp,tp)
                            old,_=kc_frozen.load(tp)
                            delta=float(np.max(abs(a['test_features']-old['test_features'])))
                            oldm=read(tp/'metrics.json');oldfm=read(Path(c['source'])/'frozen'/cohort/'full'/tp.name/'metrics.json')
                            identity=dict(cohort=cohort,seed=block['seed'],circuit_seed=ci,arm=arm)
                            for mode,new,previous in [('refit',m,oldm),('frozen',fm,oldfm)]:
                                for x,y in zip(new,previous):rows.append(dict(**identity,mode=mode,lag=x['lag'],input_accuracy=x['test_accuracy'],lesion_accuracy=y['test_accuracy'],difference=x['test_accuracy']-y['test_accuracy'],max_feature_difference=delta))
                            neural.append(dict(**identity,**n));graphs.append({**identity,**g});count+=1;transfers+=1
                        print(f'input {cohort} c{ci} s{block["seed"]}:4 paired cases fully verified',flush=True)
            summary=summarize_input(rows,c,out)
            assert count==28 and transfers==28
        else:
            for cohort,settings in c['cohorts'].items():
                cc=dict(c,**settings)
                arms=['intact']+[arm for group in c['groups'] for arm in [group]+[f'{group}_control{j}' for j in range(3)]]+c['positive_controls']
                for block in cc['blocks']:
                    for ci in cc['circuit_seeds']:
                        sp=out/cohort/f'intact_c{ci}_s{block["seed"]}'
                        for arm in arms:
                            budget.check();dest=out/cohort/f'{arm}_c{ci}_s{block["seed"]}'
                            a,g,m,n,fm=save_case(dest,cc,ci,block,arm,context,None if arm=='intact' else sp)
                            identity=dict(cohort=cohort,seed=block['seed'],circuit_seed=ci,arm=arm)
                            for mode,data in [('refit',m),('frozen',fm)]:
                                if data is not None:rows.extend(dict(**identity,mode=mode,**r) for r in data)
                            neural.append(dict(**identity,**n));graphs.append({**identity,**g});count+=1;transfers+=fm is not None
                        print(f'panel {cohort} c{ci} s{block["seed"]}:23 cases,22 frozen transfers fully verified',flush=True)
            summary=summarize_panel(rows,c,out);assert count==299 and transfers==286
        pd.DataFrame(neural).to_csv(out/'neural-diagnostics.csv',index=False)
        pd.DataFrame(graphs).to_csv(out/'graph-audit.csv',index=False)
        for path,digest in {**prior,**context['source_sha256']}.items():assert core.sha256(path)==digest,path
        budget.check();usage=budget.close()
        core.write_json(out/'verification.json',dict(full_replays=count,independent_refits=count,frozen_transfers=transfers,prior_results_unchanged=len(prior),budget=usage))
        seal(out,c,context,package=package,environment=core.environment())
        print(dict(package=package,summary=summary if package=='input' else summary['confirmed'],budget=usage),flush=True)
    finally:budget.close()


CONFIG_HASH = '2dd04199e44c141c8648005dc89ae0be0649d77d820b3812c5cf11a256af41b7'

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=Path('configs/neuron_panel.json'))
    p.add_argument('--out',type=Path,required=True);p.add_argument('--package',choices=['input','panel'],required=True)
    args=p.parse_args();run(args.config,args.out,args.package)
