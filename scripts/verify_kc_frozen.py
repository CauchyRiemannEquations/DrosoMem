"""Independent direct-formula audit of frozen lesion scores, pairing and inference."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from alphabet_memory import read,check
from flying.training import whole_brain_memory as core


def statistics(x,c,paired=False):
    x=np.asarray(x,float);rng=np.random.default_rng(c['bootstrap_seed'])
    draws=x[rng.integers(0,len(x),size=(c['bootstrap_draws'],len(x)))].mean(axis=1)
    sd=x.std(ddof=1);q=dict(n=len(x),mean=float(x.mean()),median=float(np.median(x)),variance=float(sd**2),bootstrap95=np.quantile(draws,[.025,.975]).tolist())
    if paired:q.update(paired_dz=float(x.mean()/sd) if sd else None,wins=int((x>0).sum()),ties=int((x==0).sum()),losses=int((x<0).sum()))
    return q


def close(a,b):
    assert a.keys()==b.keys()
    for key,value in a.items():
        if value is None:assert b[key] is None
        else:np.testing.assert_allclose(value,b[key],rtol=1e-10,atol=1e-12,err_msg=key)


@threadpool_limits.wrap(limits=1)
def verify(root,out):
    top=check(root);c=top['config'];rows=[];seen=set();baselines={}
    frozen=['mean','scale','weights','bias','null_weights','null_bias']
    for file in sorted(root.glob('*/*/*/manifest.json')):
        record=check(file.parent);identity=record['identity'];cohort=identity['cohort'];family=identity['family'];arm=identity['arm'];seed=identity['seed'];ci=identity['circuit_seed']
        sp=Path(record['source_path']);tp=Path(record['target_path']);check(sp);check(tp)
        assert sp==Path(c['sources']['full'])/cohort/f'intact_c{ci}_s{seed}'
        assert tp==Path(c['sources'][family])/cohort/f'{arm}_c{ci}_s{seed}'
        assert identity['group']==('matched' if arm.startswith('matched') else ('nongamma' if arm.startswith('nongamma') else 'gamma'))
        for label,path in [('source',sp),('target',tp)]:
            assert core.sha256(path/'manifest.json')==record[label+'_manifest_sha256']
            assert core.sha256(path/'checkpoint.npz')==record[label+'_checkpoint_sha256']
        assert core.sha256(tp/'graph.json')==record['target_graph_sha256'] and core.sha256(tp/'neural.json')==record['target_neural_sha256']
        with np.load(sp/'checkpoint.npz') as source,np.load(tp/'checkpoint.npz') as target,np.load(file.parent/'checkpoint.npz') as saved:
            for key in frozen:np.testing.assert_array_equal(source[key],saved[key])
            for key in ['intended_input_patterns','observed_indices','train_symbols','test_symbols','ytrain','ytest']:
                np.testing.assert_array_equal(source[key],target[key])
            for key in ['ytrain','ytest','xtest']:np.testing.assert_array_equal(saved[key],target[key])
            for key,original in [('source_within_scores',source['scores']),('target_refit_scores',target['scores']),('target_refit_null_scores',target['null_scores'])]:
                np.testing.assert_array_equal(saved[key],original)
            n=len(source['xtrain']);x=(source['xtrain']-source['xtrain'].mean(axis=0))/np.maximum(source['xtrain'].std(axis=0),1e-5)
            design=np.vstack([x,np.eye(48)]);truth=np.eye(4)[source['ytrain']].reshape(n,44)
            z=(target['xtest']-source['mean'])/source['scale'];zt=(target['xtrain']-source['mean'])/source['scale']
            for prefix,y in [('',truth),('null_',np.roll(truth,n//2,axis=0))]:
                intercept=y.mean(axis=0);w=np.linalg.lstsq(design,np.vstack([y-intercept,np.zeros((48,44))]),rcond=None)[0]
                np.testing.assert_allclose(w,saved[prefix+'weights'],rtol=1e-9,atol=1e-9)
                np.testing.assert_array_equal(intercept,saved[prefix+'bias'])
                scores=(z@source[prefix+'weights']+intercept).reshape(len(z),11,4)
                np.testing.assert_array_equal(scores,saved[prefix+'scores'])
                np.testing.assert_array_equal(scores.argmax(axis=2),saved[prefix+'predictions'])
                independent=(z@w+intercept).reshape(scores.shape)
                np.testing.assert_allclose(independent,scores,atol=1e-9,rtol=1e-9)
                np.testing.assert_array_equal(independent.argmax(axis=2),saved[prefix+'predictions'])
            train=(zt@source['weights']+source['bias']).reshape(n,11,4)
            np.testing.assert_array_equal(train,saved['train_scores'])
            measured=read(file.parent/'metrics.json')
            for j,lag in enumerate(c['lags']):
                y=target['ytest'][:,j];oh=np.eye(4)[y];freq=np.bincount(source['ytrain'][:,j],minlength=4)/n
                acc=float(np.mean(saved['scores'][:,j].argmax(axis=1)==y));base=float(np.mean(y==freq.argmax()))
                null=float(np.mean(saved['null_scores'][:,j].argmax(axis=1)==y))
                within=float(np.mean(source['scores'][:,j].argmax(axis=1)==y));refit=float(np.mean(target['scores'][:,j].argmax(axis=1)==y))
                rn=float(np.mean(target['null_scores'][:,j].argmax(axis=1)==y));denom=np.sum((oh-freq)**2);error=np.sum((oh-saved['scores'][:,j])**2)
                expected=dict(lag=lag,test_accuracy=acc,frequency_accuracy=base,null_accuracy=null,frequency_excess=acc-base,null_excess=acc-null,
                    r2_vs_frequency=float(1-error/denom),test_mse=float(error/oh.size),source_within_accuracy=within,target_refit_accuracy=refit,
                    transfer_minus_source=acc-within,transfer_minus_target_refit=acc-refit,refit_minus_frozen=refit-acc,intact_minus_frozen=within-acc,
                    train_accuracy=float(np.mean(train[:,j].argmax(axis=1)==target['ytrain'][:,j])),training_mse=float(np.mean((train[:,j]-np.eye(4)[target['ytrain'][:,j]])**2)),
                    refit_null_accuracy=rn,refit_null_excess=refit-rn,refit_frequency_excess=refit-base,refit_r2_vs_frequency=float(1-np.sum((oh-target['scores'][:,j])**2)/denom))
                close(expected,measured[j]);rows.append(dict(**identity,**expected))
            baselines[(cohort,ci,seed)]=read(sp/'metrics.json')
        seen.add((cohort,family,arm,ci,seed))
    expected_set={(co,fam,arm,ci,b['seed']) for co,blocks,circuits in [('smoke',c['smoke']['blocks'],[701]),('main',c['blocks'],[701,702])]
        for b in blocks for ci in circuits for fam in ['full','subset'] for arm in c['conditions'][fam]}
    assert seen==expected_set and len(seen)==70 and len(baselines)==7
    saved_baselines=read(root/'baselines.json');assert len(saved_baselines)==7
    for b in saved_baselines:assert b['actual_metrics']==b['expected_metrics']==baselines[(b['cohort'],b['circuit_seed'],b['seed'])]
    f=pd.DataFrame(rows);metrics=list(expected);metrics.remove('lag');index=['cohort','family','arm','group','seed','circuit_seed','lag']
    raw=pd.read_csv(root/'raw-lag-table.csv')
    np.testing.assert_allclose(f.set_index(index)[metrics].sort_index(),raw.set_index(index)[metrics].sort_index(),rtol=1e-12,atol=1e-12)
    primary=f[f.lag.isin(c['primary_lags'])].groupby(index[:-1])[metrics].mean().reset_index()
    cells=primary.groupby(['cohort','family','group','seed','circuit_seed'])[metrics].mean().groupby(['cohort','family','group','seed']).mean()
    saved_blocks=pd.read_csv(root/'seed-blocks.csv').set_index(['cohort','family','group','seed'])
    np.testing.assert_allclose(cells.sort_index(),saved_blocks[metrics].sort_index(),rtol=1e-12,atol=1e-12)
    rawpast=pd.read_csv(root/'raw-past-table.csv').set_index(index[:-1])
    np.testing.assert_allclose(primary.set_index(index[:-1])[metrics].sort_index(),rawpast[metrics].sort_index(),rtol=1e-12,atol=1e-12)
    summary=read(root/'summary.json');gates={}
    for (family,group),b in cells.loc['main'].reset_index().groupby(['family','group']):
        key=f'{family}/{group}'
        for metric in metrics:
            paired=metric in ['refit_minus_frozen','intact_minus_frozen','transfer_minus_source','transfer_minus_target_refit']
            close(statistics(b[metric],c,paired),summary['cells'][key][metric])
        access=bool((b.frequency_excess>=.05).all() and (b.null_excess>=.05).all() and b.r2_vs_frequency.mean()>0)
        ra=bool((b.refit_frequency_excess>=.05).all() and (b.refit_null_excess>=.05).all() and b.refit_r2_vs_frequency.mean()>0)
        gain=b.refit_minus_frozen;retain=bool(gain.mean()<=.05 and (gain<=.05).all())
        gates[key]=dict(refit_past_access=ra,refit_benefit=bool(ra and gain.mean()>=.05 and (gain>0).all()),frozen_past_access=access,retention=retain,portable=access and retain)
    assert gates==summary['gates'] and summary['primary_H1']==gates['full/gamma']['refit_benefit']
    base=[]
    for (co,ci,seed),m in baselines.items():
        if co=='main':base.extend(dict(seed=seed,**r) for r in m if r['lag'] in c['primary_lags'])
    intact=pd.DataFrame(base).groupby('seed').mean();intact_access=bool((intact.frequency_excess>=.05).all() and (intact.null_excess>=.05).all() and intact.r2_vs_frequency.mean()>0)
    contrasts=[]
    for family,control in [('full','matched'),('subset','nongamma')]:
        gamma=cells.loc[('main',family,'gamma')];other=cells.loc[('main',family,control)]
        delta=other.test_accuracy-gamma.test_accuracy;refit=other.target_refit_accuracy-gamma.target_refit_accuracy
        for mode,x in [('frozen',delta),('refit',refit)]:close(statistics(x,c,True),summary['secondary_specificity'][family][mode])
        assert summary['secondary_specificity'][family]['gate']==bool(intact_access and delta.mean()>=.05 and (delta>0).all())
        contrasts.extend(dict(family=family,seed=int(seed),frozen_specificity=float(delta.loc[seed]),refit_specificity=float(refit.loc[seed])) for seed in delta.index)
    saved=pd.read_csv(root/'paired-specificity.csv').set_index(['family','seed'])
    np.testing.assert_allclose(pd.DataFrame(contrasts).set_index(['family','seed']).sort_index(),saved.sort_index(),rtol=1e-12,atol=1e-12)
    prior=read(root/'prior-results-sha256.json')
    for p,h in prior.items():assert core.sha256(p)==h,p
    out.mkdir(parents=True,exist_ok=False)
    core.write_json(out/'checks.json',dict(verified=True,transfers=len(seen),lag_rows=len(rows),main_lag_rows=len(f[f.cohort=='main']),
        source_baselines=len(baselines),source_only_lstsq=True,independent_metrics=True,independent_statistics=True,prior_results_unchanged=len(prior),
        source_manifest_sha256=core.sha256(root/'manifest.json'),verifier_sha256=core.sha256(__file__)))
    print(read(out/'checks.json'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();verify(args.root,args.out)
