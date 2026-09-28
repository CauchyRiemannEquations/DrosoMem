"""Registered source-mode decomposition; no predictive parameter fitting."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from alphabet_memory import read, check
from context_memory import estimate
from frozen_state_probe import Budget
from moment_alignment import predict_aligned, measures as baseline_measures
from normalization_transfer import predict_frozen
from flying.training import whole_brain_memory as core

METRICS = ['low_state_share', 'low_signed_score_share', 'signed_minus_state',
    'total_score_energy', 'low_score_energy', 'high_score_energy', 'cross_energy',
    'diagonal_score_energy', 'low_gain', 'high_gain', 'low_train_variance_share',
    'boundary_gap', 'test_accuracy', 'r2_vs_frequency']


def source_basis(ztrain):
    _, s, vt = np.linalg.svd(ztrain, full_matrices=False)
    return s, vt.T


def decompose(ztrain, residual, weights, split=24):
    singular, basis = source_basis(ztrain)
    q = residual@basis; h = basis.T@weights
    components = np.einsum('nk,klo->nklo', q, h.reshape(len(h), -1, 4))
    total = (residual@weights).reshape(len(residual), -1, 4)
    low = (q[:, split:]@h[split:]).reshape(total.shape)
    high = (q[:, :split]@h[:split]).reshape(total.shape)
    return dict(singular=singular, basis=basis, residual=residual, weights=weights,
        projected_residual=q, projected_head=h, total=total, low=low, high=high,
        state_energy=np.mean(q*q, axis=0),
        diagonal_energy=np.mean(np.sum(components**2, axis=3), axis=0).T,
        signed_attribution=np.einsum('nklo,nlo->lk', components, total)/len(q))


def analyze(source, target, moment, parent, c):
    ztrain=(source['xtrain']-source['mean'])/source['scale']
    zs=(source['xtest']-source['mean'])/source['scale']
    zt=(target['xtest']-moment['target_mean'])/moment['target_scale']
    a=decompose(ztrain, zt-zs, source['weights'], c['split'])
    a.update(source_scores=source['scores'], aligned_scores=moment['scores'])
    return a


def verify_arrays(a, source, target, moment, c):
    v=a['basis']; z=(source['xtrain']-source['mean'])/source['scale']
    np.testing.assert_allclose(v.T@v, np.eye(48), atol=1e-12)
    np.testing.assert_allclose(v.T@(z.T@z)@v, np.diag(a['singular']**2), atol=1e-8, rtol=1e-8)
    assert np.all(np.diff(a['singular']) <= 0)
    d=(target['xtest']-moment['target_mean'])/moment['target_scale']-(source['xtest']-source['mean'])/source['scale']
    np.testing.assert_array_equal(a['residual'], d)
    np.testing.assert_array_equal(a['weights'], source['weights'])
    np.testing.assert_array_equal(a['source_scores'], source['scores'])
    np.testing.assert_array_equal(a['aligned_scores'], moment['scores'])
    low_projector=v[:, c['split']:]@v[:, c['split']:].T
    low=(d@low_projector@source['weights']).reshape(a['low'].shape)
    np.testing.assert_allclose(a['low'], low, atol=1e-9, rtol=1e-8)
    np.testing.assert_allclose(a['high'], a['total']-low, atol=1e-9, rtol=1e-8)
    np.testing.assert_allclose(a['total'], moment['scores']-source['scores'], atol=1e-9, rtol=1e-8)
    np.testing.assert_allclose(a['signed_attribution'].sum(axis=1), np.mean(np.sum(a['total']**2,axis=2),axis=0), atol=1e-9, rtol=1e-8)
    for k in range(48):
        mode=(d@np.outer(v[:,k],v[:,k])@source['weights']).reshape(a['total'].shape)
        np.testing.assert_allclose(a['diagonal_energy'][:,k], np.mean(np.sum(mode**2,axis=2),axis=0),atol=1e-9,rtol=1e-8)
        np.testing.assert_allclose(a['signed_attribution'][:,k], np.mean(np.sum(mode*a['total'],axis=2),axis=0),atol=1e-9,rtol=1e-8)
    for value in a.values(): assert np.isfinite(value).all()


def measures(a, moment, c):
    rows=[]; k=c['split']; state=a['state_energy']; train=a['singular']**2
    ratio=lambda x,y:float(x/y) if y>c['energy_floor'] else None
    gap=float((a['singular'][k-1]-a['singular'][k])/max(a['singular'][0],1e-24))
    for j, b in enumerate(baseline_measures(moment,c['lags'])):
        total=float(np.mean(np.sum(a['total'][:,j]**2,axis=1)))
        low=float(np.mean(np.sum(a['low'][:,j]**2,axis=1)))
        high=float(np.mean(np.sum(a['high'][:,j]**2,axis=1)))
        cross=float(2*np.mean(np.sum(a['low'][:,j]*a['high'][:,j],axis=1)))
        np.testing.assert_allclose(total,low+high+cross,atol=1e-9,rtol=1e-8)
        ss=ratio(state[k:].sum(),state.sum()); cs=ratio(a['signed_attribution'][j,k:].sum(),total)
        valid=ss is not None and cs is not None and gap>c['boundary_gap_min']
        rows.append(dict(lag=b['lag'], valid=valid, low_state_share=ss, low_signed_score_share=cs,
            signed_minus_state=cs-ss if ss is not None and cs is not None else None,
            total_score_energy=total, low_score_energy=low, high_score_energy=high,cross_energy=cross,
            diagonal_score_energy=float(a['diagonal_energy'][j].sum()),low_gain=ratio(low,state[k:].sum()),
            high_gain=ratio(high,state[:k].sum()),low_train_variance_share=ratio(train[k:].sum(),train.sum()),
            boundary_gap=gap,test_accuracy=b['test_accuracy'],r2_vs_frequency=b['r2_vs_frequency']))
    return rows


def decision(b, cohort, c):
    return bool(b.valid.all() and b.low_signed_score_share.mean()>=c['score_share_min']
        and b.low_state_share.mean()<=c['state_share_max']
        and (b.signed_minus_state>0).sum()>=(4 if cohort=='main' else 3))


def summarize(rows, out, c, metrics, decide):
    frame=pd.DataFrame(rows); frame.to_csv(out/'raw-lag-table.csv',index=False)
    agg={k:'mean' for k in metrics};agg['valid']='all'
    raw=frame[frame.lag.isin(c['primary_lags'])].groupby(['cohort','direction','seed','circuit_seed']).agg(agg).reset_index()
    raw.to_csv(out/'raw-past-table.csv',index=False)
    blocks=raw.groupby(['cohort','direction','seed']).agg(agg).reset_index()
    blocks.to_csv(out/'seed-blocks.csv',index=False)
    cells={};gates={}
    for (cohort,direction),b in blocks.query("cohort != 'smoke'").groupby(['cohort','direction']):
        key=f'{cohort}/{direction}'; cells[key]={}; gates[key]=decide(b,cohort,c)
        for metric in metrics:
            if b[metric].isna().any(): stats=dict(defined=False,n=int(len(b)))
            else:
                stats=estimate(b[metric],c)
                if metric not in ['signed_minus_state','log2_actual_over_reference']:
                    stats={k:v for k,v in stats.items() if k not in ['paired_dz','wins','ties','losses']}
            cells[key][metric]=stats
    directions=['renormalized_to_fixed_original','fixed_original_to_renormalized']
    result=dict(cells=cells,gates=gates,supported_directions=[d for d in directions if all(gates[f'{co}/{d}'] for co in ['main','confirmation'])])
    core.write_json(out/'summary.json',result)
    return result


def baseline(source,target,moment):
    for k in ['train_symbols','test_symbols','observed_indices','input_patterns','ytrain','ytest']:
        np.testing.assert_array_equal(source[k],target[k])
    for k in ['mean','scale','weights','bias','null_weights','null_bias']:
        np.testing.assert_array_equal(source[k],moment[k])
    np.testing.assert_array_equal(moment['target_mean'],target['xtrain'].mean(axis=0))
    np.testing.assert_array_equal(moment['target_scale'],np.maximum(target['xtrain'].std(axis=0),1e-5))
    for k,score in zip(['scores','null_scores'],predict_frozen(source,source['xtest'])):
        np.testing.assert_array_equal(source[k],score)
    for k,score in zip(['scores','null_scores'],predict_aligned(source,moment,target['xtest'])):
        np.testing.assert_array_equal(moment[k],score)


def load_inputs(path):
    meta=check(path); sp=Path(meta['source_path']);tp=Path(meta['target_path']);mp=Path(meta.get('moment_path',str(path)))
    for p in [sp,tp,mp]:check(p)
    source=dict(np.load(sp/'checkpoint.npz',allow_pickle=False))
    target=dict(np.load(tp/'checkpoint.npz',allow_pickle=False))
    moment=dict(np.load(mp/'checkpoint.npz',allow_pickle=False))
    parent=dict(np.load(path/'checkpoint.npz',allow_pickle=False))
    return meta,sp,tp,mp,source,target,moment,parent


@threadpool_limits.wrap(limits=1)
def execute(config,out,analyzer=analyze,verifier=verify_arrays,measure=measures,metrics=METRICS,decide=decision,extra_sources=()):
    c=read(config); out.mkdir(parents=True,exist_ok=False);budget=Budget(c)
    try:
        context=core.source_context(c)
        for p in [str(config),'scripts/residual_modes.py','scripts/moment_alignment.py','scripts/normalization_transfer.py',
                  'scripts/alphabet_memory.py','scripts/context_memory.py','scripts/frozen_state_probe.py',*extra_sources]:
            context['source_sha256'][p]=core.sha256(p)
        prior={p:core.sha256(p) for p in subprocess.check_output(['git','ls-files','results/'],text=True).splitlines()}
        core.write_json(out/'prior-results-sha256.json',prior);core.write_json(out/'config.json',c)
        root=Path(c['source']);check(root);rows=[];baselines=[];counts={}
        for cohort in ['smoke','main','confirmation']:
            found=set()
            for path in sorted((root/cohort).glob('*/*/manifest.json')):
                parentpath=path.parent;meta,sp,tp,mp,source,target,moment,parent=load_inputs(parentpath)
                identity=meta['identity'];assert identity['cohort']==cohort
                found.add((identity['seed'],identity['circuit_seed'],identity['direction']))
                baseline(source,target,moment);budget.check()
                a=analyzer(source,target,moment,parent,c);verifier(a,source,target,moment,c)
                dest=out/parentpath.relative_to(root);dest.mkdir(parents=True)
                np.savez_compressed(dest/'checkpoint.npz',**a)
                replay=analyzer(source,target,moment,parent,c)
                with np.load(dest/'checkpoint.npz',allow_pickle=False) as saved:
                    for k in saved.files:np.testing.assert_array_equal(saved[k],replay[k],err_msg=k)
                measured=measure(a,moment,c);rows.extend(dict(**identity,**r) for r in measured)
                core.write_json(dest/'metrics.json',measured)
                baselines.append(dict(**identity,exact_source_and_aligned_scores=True,
                    primary_accuracy=float(np.mean([r['test_accuracy'] for r in baseline_measures(moment,c['lags']) if r['lag'] in c['primary_lags']]))))
                core.write_json(dest/'manifest.json',dict(identity=identity,source_path=sp.as_posix(),target_path=tp.as_posix(),moment_path=mp.as_posix(),
                    parent_path=parentpath.as_posix(),source_hashes={p.as_posix():core.sha256(p/'manifest.json') for p in [sp,tp,mp,parentpath]},
                    new_supervised_fits=0,artifacts={p.name:core.sha256(p) for p in dest.iterdir() if p.is_file()}))
            seeds={'smoke':[34001],'main':list(range(34142,34147)),'confirmation':list(range(41142,41145))}[cohort]
            circuits=[701] if cohort=='smoke' else [701,702]
            expected={(s,ci,d) for s in seeds for ci in circuits for d in ['renormalized_to_fixed_original','fixed_original_to_renormalized']}
            assert found==expected; counts[cohort]=len(found)
            print(cohort,counts[cohort],'directions verified',flush=True)
        summary=summarize(rows,out,c,metrics,decide)
        core.write_json(out/'baselines.json',baselines)
        for p,h in {**prior,**context['source_sha256']}.items():assert core.sha256(p)==h,p
        budget.check();usage=budget.close()
        core.write_json(out/'verification.json',dict(counts=counts,exact_replays=sum(counts.values()),lag_rows=len(rows),prior_unchanged=len(prior),budget=usage))
        core.write_json(out/'manifest.json',dict(config=c,context=context,timestamp=datetime.now(timezone.utc).isoformat(),environment=core.environment(),
            parent_manifest_sha256=core.sha256(root/'manifest.json'),artifacts={p.relative_to(out).as_posix():core.sha256(p) for p in out.rglob('*') if p.is_file()}))
        print(summary['gates'],usage,flush=True)
    finally:budget.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=Path('configs/residual_modes.json'))
    p.add_argument('--out',type=Path,required=True);args=p.parse_args();execute(args.config,args.out)
