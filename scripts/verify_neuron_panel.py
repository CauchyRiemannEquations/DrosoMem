"""Independent intervention, trajectory, metric and statistical audit for III-A."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits
from flying.training import whole_brain_memory as core
from alphabet_memory import read, check, symbol_bank
from frozen_state_probe import verify_fold
from normalization_transfer import verify_transfer


def independent_mask(raw,ids,roles,cells,bank,identity,c):
    arm=identity['arm'];n=len(ids)
    if arm=='intact':return np.zeros(n,bool)
    if arm.startswith('all_'):return roles==arm[4:]
    group=arm.split('_control')[0]
    indeg=np.asarray((raw!=0).sum(1)).ravel();outdeg=np.asarray((raw!=0).sum(0)).ravel()
    if group in ['KCab','KCapbp']:target=(roles=='KC')&np.asarray([str(x).startswith(group) for x in cells])
    elif group in ['DAN','APL']:target=roles==group
    else:
        pool=sorted(np.where(roles!='MBON')[0],key=lambda i:(-int(indeg[i]+outdeg[i]),int(ids[i])))
        target=np.zeros(n,bool);target[pool[:int(np.ceil(.05*len(pool)))]]=True
    if '_control' not in arm:return target
    exposure=np.array([sum(2**j for j in range(4) if bank[j,i]!=0) for i in range(n)])
    keys=[(int(indeg[i]),int(outdeg[i]),int(exposure[i])) if group.startswith('KC') else (int(exposure[i]),) for i in range(n)]
    rng=np.random.default_rng(identity['control_seeds'][group][int(arm[-1])]);result=np.zeros(n,bool)
    for key in sorted(set(keys[i] for i in np.where(target)[0])):
        count=sum(target[i] and keys[i]==key for i in range(n))
        pool=np.array([i for i in range(n) if keys[i]==key and (roles[i]=='KC' if group.startswith('KC') else roles[i]!='MBON')])
        result[rng.choice(pool,count,replace=False)]=True
    return result


def trajectory(weights,bank,roles,symbols,leak):
    state=np.zeros(len(roles));kc=np.where(roles=='KC')[0];obs=np.where(roles=='MBON')[0]
    wobs=weights[obs].copy();features=[];active=[];norms=[]
    for symbol in symbols:
        old=state.copy();u=bank[int(symbol)]
        new=(1-leak)*old+leak*np.tanh(weights@old+u)
        mixed=old.copy();mixed[kc]=new[kc]
        new[obs]=(1-leak)*old[obs]+leak*np.tanh(wobs@mixed+u[obs])
        state=new;features.append(state[obs].copy());active.append(np.count_nonzero(abs(state)>1e-8));norms.append(np.linalg.norm(state))
    return np.asarray(features),np.asarray(active),np.asarray(norms)


def audit_metrics(a,rows,frozen=False,target=None,source=None):
    for j,row in enumerate(rows):
        truth=a['ytest'][:,j];score=a['scores'][:,j];onehot=np.eye(4)[truth]
        ytrain=a['ytrain'][:,j];frequency=a['bias'].reshape(-1,4)[j]
        accuracy=float(np.mean(score.argmax(1)==truth));base=float(np.mean(truth==frequency.argmax()))
        null=float(np.mean(a['null_scores'][:,j].argmax(1)==truth))
        expected=dict(test_accuracy=accuracy,frequency_accuracy=base,null_accuracy=null,
            frequency_excess=accuracy-base,null_excess=accuracy-null,
            r2_vs_frequency=float(1-((score-onehot)**2).sum()/((onehot-frequency)**2).sum()),
            train_accuracy=float(np.mean(a['train_scores'][:,j].argmax(1)==ytrain)),
            training_mse=float(np.mean((a['train_scores'][:,j]-np.eye(4)[ytrain])**2)))
        if frozen:
            refit=float(np.mean(target['scores'][:,j].argmax(1)==truth));within=float(np.mean(source['scores'][:,j].argmax(1)==truth))
            rn=float(np.mean(target['null_scores'][:,j].argmax(1)==truth))
            expected.update(test_mse=float(np.mean((score-onehot)**2)),source_within_accuracy=within,
                target_refit_accuracy=refit,transfer_minus_source=accuracy-within,transfer_minus_target_refit=accuracy-refit,
                refit_minus_frozen=refit-accuracy,intact_minus_frozen=within-accuracy,
                refit_null_accuracy=rn,refit_null_excess=refit-rn,refit_frequency_excess=refit-base,
                refit_r2_vs_frequency=float(1-((onehot-target['scores'][:,j])**2).sum()/((onehot-frequency)**2).sum()))
        assert set(row)==set(expected)|{'lag'}
        for key,value in expected.items():np.testing.assert_allclose(row[key],value,atol=2e-12,rtol=2e-12,err_msg=key)


def audit_stats(x,saved,c):
    x=np.asarray(x);rng=np.random.default_rng(c['bootstrap_seed'])
    draws=x[rng.integers(0,len(x),(c['bootstrap_draws'],len(x)))].mean(1)
    for key,value in dict(n=len(x),mean=x.mean(),median=np.median(x),variance=x.var(ddof=1),bootstrap95=np.quantile(draws,[.025,.975]),wins=(x>0).sum(),ties=(x==0).sum(),losses=(x<0).sum()).items():
        np.testing.assert_allclose(saved[key],value,atol=1e-12,rtol=1e-12,err_msg=key)
    if x.std(ddof=1)==0:assert saved['paired_dz'] is None
    else:np.testing.assert_allclose(saved['paired_dz'],x.mean()/x.std(ddof=1),atol=1e-10,rtol=1e-10)


@threadpool_limits.wrap(limits=1)
def verify(root,out):
    manifest=check(root);c=manifest['config'];package=manifest['package'];out.mkdir(parents=True,exist_ok=False)
    counts=dict(cases=0,frozen=0,independent_trajectories=0,metric_rows=0);reconstructed=[]
    for path in sorted(root.glob('*/*/manifest.json')):
        dest=path.parent;m=check(dest);identity=m['identity'];ci=identity['circuit_seed'];cohort=dest.parent.name
        raw,ids,_=core.load_connectome(Path(f'data/flywire_783_mb_left_kc512_s{ci}'))
        roles,_=core.load_roles(Path(f'data/flywire_783_mb_left_kc512_s{ci}'),ids);roles=np.asarray(roles)
        cells=pd.read_csv(f'data/flywire_783_mb_left_kc512_s{ci}/annotations.csv',dtype=str).set_index('root_id').loc[list(map(str,ids))].cell_type
        with np.load(dest/'checkpoint.npz') as f:a=dict(f)
        g=read(dest/'graph.json');n=read(dest/'neural.json');bank=symbol_bank(roles,identity['seed'])[:4]
        np.testing.assert_array_equal(bank,a['intended_input_patterns']);np.testing.assert_array_equal(np.where(roles=='MBON')[0],a['observed_indices'])
        if package=='input':
            tp=Path(m['full_lesion_path']);check(tp);assert core.sha256(tp/'manifest.json')==m['full_lesion_manifest_sha256']
            with np.load(tp/'checkpoint.npz') as f:old=dict(f)
            mask=old['dead_mask'];dead=np.zeros(len(ids),bool)
        else:
            mask=independent_mask(raw,ids,roles,cells,bank,identity,c);dead=mask
        np.testing.assert_array_equal(mask,a['intervention_mask']);np.testing.assert_array_equal(dead,a['dead_mask'])
        effective=bank.copy();effective[:,mask]=0;np.testing.assert_array_equal(effective,a['input_patterns'])
        dense=raw.toarray();strength=abs(dense).sum(1)
        factors=np.divide(.9,strength,out=np.zeros_like(strength),where=strength>0)
        # Multiply individual entries just as the registered normalization, no post-lesion rescale.
        dense=factors[:,None]*dense;dense[dead,:]=0;dense[:,dead]=0
        weights=sparse.load_npz(dest/'weights.npz');np.testing.assert_array_equal(weights.toarray(),dense)
        assert core.weight_hash(weights)==g['weight_sha256']
        assert g['mask_root_ids']==[ids[i] for i in np.where(mask)[0]]
        assert g['mask_count']==int(mask.sum()) and g['removed_observed']==int(dead[a['observed_indices']].sum())
        surviving=raw.toarray();surviving[dead,:]=0;surviving[:,dead]=0
        assert g['remaining_edges']==np.count_nonzero(surviving)
        np.testing.assert_allclose(g['removed_absolute_strength'],abs(raw.toarray()).sum()-abs(surviving).sum())
        cc=m['config'];w=cc['warmup']
        for split in ['train','test']:
            symbols=np.random.default_rng(identity[split+'_seed']).integers(0,4,w+cc[split+'_samples']).astype(np.uint8)
            np.testing.assert_array_equal(symbols,a[split+'_symbols'])
            expected=np.column_stack([symbols[np.arange(w,len(symbols))-lag] for lag in c['lags']]);np.testing.assert_array_equal(expected,a['y'+split])
            features,active,norms=trajectory(weights,effective,roles,symbols,.6)
            np.testing.assert_array_equal(features,a[split+'_features']);np.testing.assert_array_equal(active,a[split+'_active_counts']);np.testing.assert_array_equal(norms,a[split+'_full_norm'])
            np.testing.assert_array_equal(features[w:],a['x'+split]);counts['independent_trajectories']+=1
        verify_fold(a);rows=read(dest/'metrics.json');audit_metrics(a,rows)
        for row in rows:reconstructed.append(dict(cohort=cohort,seed=identity['seed'],circuit_seed=ci,arm=identity['arm'],mode='refit',**row))
        counts['metric_rows']+=len(rows)
        if package=='panel' and identity['arm'] in ['all_KC','all_MBON']:
            assert np.count_nonzero(a['train_features'])==0 and n['effective_rank']==0
            assert n['zero_decay_baseline'] and n['observed_decay_ratio32'] is None
        if m['source_path']:
            sp=Path(m['source_path']);check(sp);assert core.sha256(sp/'manifest.json')==m['source_manifest_sha256']
            with np.load(sp/'checkpoint.npz') as f:source=dict(f)
            with np.load(dest/'frozen.npz') as f:frozen=dict(f)
            verify_transfer(source,a,frozen);fm=read(dest/'frozen-metrics.json');audit_metrics(frozen,fm,True,a,source)
            for row in fm:reconstructed.append(dict(cohort=cohort,seed=identity['seed'],circuit_seed=ci,arm=identity['arm'],mode='frozen',**row))
            counts['frozen']+=1;counts['metric_rows']+=len(fm)
        counts['cases']+=1
        if counts['cases']%25==0:print(f'Independently audited {counts["cases"]} cases',flush=True)
    summary=read(root/'summary.json');f=pd.DataFrame(reconstructed)
    if package=='panel':
        saved=pd.read_csv(root/'raw-lag-table.csv');keys=['cohort','seed','circuit_seed','arm','mode','lag']
        pd.testing.assert_frame_equal(saved.sort_values(keys).reset_index(drop=True),f[saved.columns].sort_values(keys).reset_index(drop=True),check_dtype=False,atol=1e-12,rtol=1e-12)
        past=f[f.lag.isin(c['primary_lags'])];confirmed=[]
        for group in c['groups']:
            for mode in ['frozen','refit']:
                both=[]
                for cohort in ['discovery','confirmation']:
                    q=past[past.cohort==cohort]
                    base=q[(q.arm=='intact')&(q['mode']=='refit')].groupby('seed')[['test_accuracy','frequency_excess','null_excess','r2_vs_frequency']].mean()
                    target=q[(q.arm==group)&(q['mode']==mode)].groupby('seed').test_accuracy.mean()
                    control=q[q.arm.str.startswith(group+'_control')&(q['mode']==mode)].groupby('seed').test_accuracy.mean()
                    d=base.test_accuracy-target;e=control-target;key=f'{cohort}/{mode}/{group}';stat=summary['statistics'][key]
                    for name,x in [('target',target),('control',control),('impairment',d),('excess_impairment',e)]:audit_stats(x,stat[name],c)
                    access=bool((base.frequency_excess>=.05).all() and (base.null_excess>=.05).all() and base.r2_vs_frequency.mean()>0)
                    gate=bool(access and d.mean()>=.05 and e.mean()>=.05 and (d>0).all() and (e>0).all())
                    assert gate==summary['gates'][key];both.append(gate)
                if all(both):confirmed.append(f'{mode}/{group}')
        assert set(confirmed)==set(summary['confirmed']) and not summary['cohorts_pooled']
    else:
        saved=pd.read_csv(root/'raw-lag-table.csv');deltas=[]
        for row in saved.to_dict('records'):
            query=f[(f.cohort==row['cohort'])&(f.seed==row['seed'])&(f.circuit_seed==row['circuit_seed'])&(f.arm==row['arm'])&(f['mode']==row['mode'])&(f.lag==row['lag'])]
            assert len(query)==1;np.testing.assert_allclose(query.test_accuracy.iloc[0],row['input_accuracy'],atol=1e-12)
            path=Path(c['source'])/('conditions' if row['mode']=='refit' else 'frozen')/row['cohort']
            if row['mode']=='frozen':path=path/'full'
            path=path/f'{row["arm"]}_c{row["circuit_seed"]}_s{row["seed"]}'
            previous=next(r for r in read(path/'metrics.json') if r['lag']==row['lag'])['test_accuracy']
            np.testing.assert_allclose(row['lesion_accuracy'],previous,atol=1e-12)
            np.testing.assert_allclose(row['difference'],row['input_accuracy']-previous,atol=1e-12)
        main=saved[(saved.cohort=='main')&saved.lag.isin(c['primary_lags'])].copy();main['group']=np.where(main.arm=='gamma','gamma','matched')
        for (group,mode),q in main.groupby(['group','mode']):
            x=q.groupby('seed').difference.mean();key=f'{group}/{mode}';audit_stats(x,summary[key]['difference'],c)
            assert summary[key]['gate']==bool(x.mean()>=.05 and (x>0).all())
    for p,h in read(root/'prior-results-sha256.json').items():assert core.sha256(p)==h,p
    assert counts['cases']==(28 if package=='input' else 299)
    assert counts['frozen']==(28 if package=='input' else 286)
    core.write_json(out/'checks.json',dict(**counts,all_checks_pass=True,result_manifest_sha256=core.sha256(root/'manifest.json'),prior_files_unchanged=len(read(root/'prior-results-sha256.json'))))
    print(counts,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();verify(args.root,args.out)
