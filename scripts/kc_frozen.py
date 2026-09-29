"""Frozen intact heads on saved KC lesion states; no new target training."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import time
import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits
from alphabet_memory import read, check
from frozen_state_probe import Budget, verify_fold, measures as within_measures
from normalization_transfer import FROZEN, transfer, predict_frozen, verify_transfer, measures as transfer_measures
from context_memory import estimate
from flying.training import whole_brain_memory as core


def validate(c):
    expected=dict(alphabet_size=4,warmup=100,train_samples=2000,test_samples=1000,
        lags=[0,1,2,3,4,5,8,12,16,24,32],primary_lags=[1,2,3,4,5,8],
        circuit_seeds=[701,702],alpha=1.,gain=.9,leak=.6,bootstrap_seed=65399,
        bootstrap_draws=10000,material_effect=.05,access_margin=.05,retention_tolerance=.05)
    for key,value in expected.items():
        if c[key]!=value:raise ValueError(f'Unregistered setting: {key}')
    assert c['conditions']==dict(full=['gamma','matched0','matched1','matched2'],
        subset=[f'{a}{j}' for j in range(3) for a in ['gamma','nongamma']])
    for family,path in c['sources'].items():
        m=check(Path(path));assert core.sha256(Path(path)/'manifest.json')==c['source_manifest_sha256'][family]
        for key in ['blocks','smoke','circuit_seeds','lags','primary_lags','gain','leak','normalization','schedule','input_fraction','input_amplitude']:
            assert m['config'][key]==c[key],key


def load(path):
    manifest=check(path)
    with np.load(path/'checkpoint.npz',allow_pickle=False) as a:return dict(a),manifest


def audit_pair(source,target,sp,tp):
    sg,tg=read(sp/'graph.json'),read(tp/'graph.json')
    for key in ['raw_sha256','observed_root_ids','input_root_ids','input_mapping_sha256']:
        assert sg[key]==tg[key],key
    for key in ['intended_input_patterns','observed_indices','train_symbols','test_symbols','ytrain','ytest']:
        np.testing.assert_array_equal(source[key],target[key],err_msg=key)
    assert not source['dead_mask'].any() and not target['dead_mask'][source['observed_indices']].any()
    for a in [source,target]:
        for split in ['train','test']:np.testing.assert_array_equal(a['x'+split],a[split+'_features'][100:])
    bank=source['input_patterns'].copy();bank[:,target['dead_mask']]=0
    np.testing.assert_array_equal(bank,target['input_patterns'])
    w=sparse.load_npz(sp/'weights.npz');keep=sparse.diags((~target['dead_mask']).astype(float))
    expected=(keep@w@keep).tocsr();expected.eliminate_zeros();expected.sort_indices()
    assert core.weight_hash(expected)==tg['weight_sha256']==core.weight_hash(sparse.load_npz(tp/'weights.npz'))


def evaluate(source,target,c):
    a=transfer(source,target)
    a['train_scores'],_=predict_frozen(source,target['xtrain'])
    a['ytrain']=target['ytrain'].copy()
    a['target_refit_null_scores']=target['null_scores'].copy()
    rows=transfer_measures(a,c['lags'])
    for j,row in enumerate(rows):
        y=a['ytest'][:,j];truth=np.eye(4)[y];frequency=source['bias'].reshape(-1,4)[j]
        np.testing.assert_array_equal(frequency,target['bias'].reshape(-1,4)[j])
        refit_null=float(np.mean(target['null_predictions'][:,j]==y))
        row.update(refit_minus_frozen=-row['transfer_minus_target_refit'],
            intact_minus_frozen=-row['transfer_minus_source'],
            train_accuracy=float(np.mean(a['train_scores'][:,j].argmax(axis=1)==a['ytrain'][:,j])),
            training_mse=float(np.mean((a['train_scores'][:,j]-np.eye(4)[a['ytrain'][:,j]])**2)),
            refit_null_accuracy=refit_null,refit_null_excess=row['target_refit_accuracy']-refit_null,
            refit_frequency_excess=row['target_refit_accuracy']-row['frequency_accuracy'],
            refit_r2_vs_frequency=float(1-np.sum((truth-target['scores'][:,j])**2)/np.sum((truth-frequency)**2)))
    for value in a.values():assert np.isfinite(value).all()
    return a,rows


def decisions(b,c):
    def access(prefix=''):
        return bool((b[prefix+'frequency_excess']>=c['access_margin']).all()
            and (b[prefix+'null_excess']>=c['access_margin']).all() and b[prefix+'r2_vs_frequency'].mean()>0)
    benefit=b.refit_minus_frozen
    refit_access=access('refit_');frozen_access=access()
    retention=bool(benefit.mean()<=c['retention_tolerance'] and (benefit<=c['retention_tolerance']).all())
    return dict(refit_past_access=refit_access,refit_benefit=bool(refit_access and benefit.mean()>=c['material_effect'] and (benefit>0).all()),
        frozen_past_access=frozen_access,retention=retention,portable=frozen_access and retention)


def aggregate(rows,c):
    f=pd.DataFrame(rows);metrics=[k for k in rows[0] if k not in ['cohort','family','arm','group','seed','circuit_seed','lag']]
    raw=f[f.lag.isin(c['primary_lags'])].groupby(['cohort','family','arm','group','seed','circuit_seed'])[metrics].mean().reset_index()
    grouped=raw.groupby(['cohort','family','group','seed','circuit_seed'])[metrics].mean().reset_index()
    blocks=grouped.groupby(['cohort','family','group','seed'])[metrics].mean().reset_index()
    cells={};gates={};specificity={};paired=[]
    for (family,group),b in blocks[blocks.cohort=='main'].groupby(['family','group']):
        key=f'{family}/{group}';cells[key]={}
        for metric in metrics:
            q=estimate(b[metric],c)
            if metric not in ['refit_minus_frozen','intact_minus_frozen','transfer_minus_source','transfer_minus_target_refit']:
                for k in ['paired_dz','wins','ties','losses']:q.pop(k)
            cells[key][metric]=q
        gates[key]=decisions(b,c)
    access=read(Path(c['sources']['full'])/'main/summary.json')['gates']['intact_past_access']
    for family,control in [('full','matched'),('subset','nongamma')]:
        familyrows=blocks[(blocks.cohort=='main')&(blocks.family==family)]
        a=familyrows[familyrows.group=='gamma'].set_index('seed');b=familyrows[familyrows.group==control].set_index('seed')
        delta=b.test_accuracy-a.test_accuracy;refit=b.target_refit_accuracy-a.target_refit_accuracy
        specificity[family]=dict(frozen=estimate(delta,c),refit=estimate(refit,c),
            gate=bool(access and delta.mean()>=.05 and (delta>0).all()))
        paired.extend(dict(family=family,seed=int(seed),frozen_specificity=float(delta.loc[seed]),refit_specificity=float(refit.loc[seed])) for seed in a.index)
    return raw,blocks,pd.DataFrame(paired),dict(primary_H1=gates['full/gamma']['refit_benefit'],cells=cells,gates=gates,
        secondary_specificity=specificity,independent_confirmation=False)


def plots(rows,out,c):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    f=pd.DataFrame(rows);f=f[f.cohort=='main'];fig,axes=plt.subplots(2,2,figsize=(11,8))
    for ax,(family,group) in zip(axes.ravel(),[('full','gamma'),('full','matched'),('subset','gamma'),('subset','nongamma')]):
        g=f[(f.family==family)&(f.group==group)].groupby('lag')[['test_accuracy','target_refit_accuracy','source_within_accuracy']].mean().reindex(c['lags'])
        for column,label,style in [('test_accuracy','Frozen intact head','o-'),('target_refit_accuracy','Existing target refit','s--'),('source_within_accuracy','Intact reference',':')]:
            ax.plot(range(len(g)),g[column],style,label=label)
        ax.axhline(.25,color='gray',ls=':');ax.set(title=f'{family} / {group}',xticks=range(len(g)),xticklabels=c['lags'],
            xlabel='Symbol lag',ylabel='Independent-stream accuracy',ylim=(0,1.03));ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'frozen-curve.png',dpi=180);plt.close(fig)


@threadpool_limits.wrap(limits=1)
def run(config,out):
    c=read(config);validate(c);out.mkdir(parents=True,exist_ok=False);budget=Budget(c)
    try:
        context=core.source_context(c)
        for p in [str(config),'scripts/kc_frozen.py','scripts/verify_kc_frozen.py','scripts/normalization_transfer.py','scripts/alphabet_memory.py','scripts/frozen_state_probe.py','scripts/context_memory.py']:
            context['source_sha256'][p]=core.sha256(p)
        prior={p:core.sha256(p) for p in subprocess.check_output(['git','ls-files','results/'],text=True).splitlines()}
        core.write_json(out/'prior-results-sha256.json',prior);core.write_json(out/'config.json',c)
        rows=[];baselines=[];diagnostics=[];count=0
        for cohort in ['smoke','main']:
            cc=dict(c)
            if cohort=='smoke':cc.update(c['smoke'])
            for block in cc['blocks']:
                for circuit in cc['circuit_seeds']:
                    sp=Path(c['sources']['full'])/cohort/f'intact_c{circuit}_s{block["seed"]}'
                    source,sm=load(sp);verify_fold(source)
                    scores,null=predict_frozen(source,source['xtest'])
                    np.testing.assert_array_equal(scores,source['scores']);np.testing.assert_array_equal(null,source['null_scores'])
                    old=read(sp/'metrics.json');base=within_measures([source],c['lags'])
                    for j,r in enumerate(base):
                        r['training_mse']=float(np.mean((source['train_scores'][:,j]-np.eye(4)[source['ytrain'][:,j]])**2))
                    assert base==old
                    baselines.append(dict(cohort=cohort,seed=block['seed'],circuit_seed=circuit,source=sp.as_posix(),
                        exact_refit=True,independent_lstsq=True,exact_self_prediction=True,expected_metrics=old,actual_metrics=base))
                    for family in ['full','subset']:
                        for arm in c['conditions'][family]:
                            budget.check();started=time.perf_counter()
                            tp=Path(c['sources'][family])/cohort/f'{arm}_c{circuit}_s{block["seed"]}'
                            target,tm=load(tp);audit_pair(source,target,sp,tp)
                            a,m=evaluate(source,target,c);verify_transfer(source,target,a)
                            replay,m2=evaluate(source,target,c);assert m==m2
                            dest=out/cohort/family/f'{arm}_c{circuit}_s{block["seed"]}';dest.mkdir(parents=True)
                            np.savez_compressed(dest/'checkpoint.npz',**a)
                            with np.load(dest/'checkpoint.npz',allow_pickle=False) as saved:
                                for key in replay:np.testing.assert_array_equal(saved[key],replay[key],err_msg=key)
                            for row,ref in zip(m,read(tp/'metrics.json')):assert row['target_refit_accuracy']==ref['test_accuracy']
                            group='matched' if arm.startswith('matched') else ('nongamma' if arm.startswith('nongamma') else 'gamma')
                            identity=dict(cohort=cohort,family=family,arm=arm,group=group,seed=block['seed'],circuit_seed=circuit)
                            rows.extend(dict(**identity,**r) for r in m)
                            drift=dict(mean_abs_train_shift_z=float(abs(a['train_mean_shift_z']).mean()),
                                median_train_scale_ratio=float(np.median(a['train_scale_ratio'])),
                                mean_train_feature_correlation=float(a['train_feature_correlation'].mean()),mean_paired_train_cosine=float(a['paired_train_cosine'].mean()))
                            diagnostics.append(dict(**identity,**drift));core.write_json(dest/'metrics.json',m)
                            core.write_json(dest/'manifest.json',dict(identity=identity,config=c,context=context,
                                timestamp=datetime.now(timezone.utc).isoformat(),source_path=sp.as_posix(),target_path=tp.as_posix(),
                                source_manifest_sha256=core.sha256(sp/'manifest.json'),target_manifest_sha256=core.sha256(tp/'manifest.json'),
                                source_checkpoint_sha256=core.sha256(sp/'checkpoint.npz'),target_checkpoint_sha256=core.sha256(tp/'checkpoint.npz'),
                                target_graph_sha256=core.sha256(tp/'graph.json'),target_neural_sha256=core.sha256(tp/'neural.json'),
                                new_target_fits=0,new_trajectories=0,exact_replay=True,independent_source_lstsq=True,coefficients_per_lag=196,
                                seconds_including_checks=time.perf_counter()-started,drift=drift,
                                artifacts={p.name:core.sha256(p) for p in dest.iterdir() if p.is_file()}))
                            count+=1
                    print(f'{cohort} c{circuit} s{block["seed"]}: self-check + all10 frozen transfers verified',flush=True)
        raw,blocks,paired,summary=aggregate(rows,c)
        for filename,frame in [('raw-lag-table',pd.DataFrame(rows)),('raw-past-table',raw),('seed-blocks',blocks),('paired-specificity',paired),('feature-drift',pd.DataFrame(diagnostics))]:
            frame.to_csv(out/f'{filename}.csv',index=False)
        core.write_json(out/'summary.json',summary);core.write_json(out/'baselines.json',baselines);plots(rows,out,c)
        for p,h in {**prior,**context['source_sha256']}.items():assert core.sha256(p)==h,p
        budget.check();usage=budget.close();assert count==70 and len(baselines)==7
        core.write_json(out/'verification.json',dict(transfers=count,exact_replays=count,source_self_checks=len(baselines),
            independent_source_lstsq=count,new_target_fits=0,new_trajectories=0,prior_results_unchanged=len(prior),budget=usage))
        core.write_json(out/'manifest.json',dict(config=c,context=context,timestamp=datetime.now(timezone.utc).isoformat(),environment=core.environment(),
            artifacts={p.relative_to(out).as_posix():core.sha256(p) for p in out.rglob('*') if p.is_file()}))
        print(dict(primary_H1=summary['primary_H1'],gates=summary['gates'],budget=usage),flush=True)
    finally:budget.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=Path('configs/kc_frozen.json'));p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();run(args.config,args.out)
