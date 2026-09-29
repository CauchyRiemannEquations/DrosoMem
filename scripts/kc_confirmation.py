"""Fresh input/mapping/mask cohort for the preregistered frozen/refit contrast."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import subprocess,time
import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits
from alphabet_memory import read,check
from frozen_state_probe import Budget,verify_fold,measures as within_measures
from normalization_transfer import predict_frozen,verify_transfer
from context_memory import estimate
from flying.training import whole_brain_memory as core
import kc_ablation as lesion
import kc_frozen as frozen


def validate(c):
    old=read('configs/kc_ablation.json')
    for key,value in old.items():
        if key not in ['protocol','blocks','smoke','bootstrap_seed'] and c[key]!=value:
            raise ValueError(f'Changed registered setting: {key}')
    blocks=[dict(seed=71142+i,train_seed=72142+i,test_seed=73142+i,match_seeds=list(range(74142+3*i,74145+3*i))) for i in range(3)]
    smoke=dict(circuit_seeds=[701],train_samples=200,test_samples=100,blocks=[dict(seed=71001,train_seed=72001,test_seed=73001,match_seeds=[74001,74002,74003])])
    if c['blocks']!=blocks or c['smoke']!=smoke:raise ValueError('Changed registered cohort')
    if (c['access_margin'],c['retention_tolerance'],c['bootstrap_seed'])!=(.05,.05,74399):raise ValueError('Changed registered inference')


def aggregate(rows,c,intact_access,discovery_H1):
    f=pd.DataFrame(rows);identity=['cohort','family','arm','group','seed','circuit_seed']
    metrics=[k for k in rows[0] if k not in identity+['lag']]
    raw=f[f.lag.isin(c['primary_lags'])].groupby(identity)[metrics].mean().reset_index()
    b=raw.groupby(['cohort','family','group','seed','circuit_seed'])[metrics].mean().groupby(['cohort','family','group','seed']).mean().reset_index()
    cells={};gates={}
    for group,frame in b[b.cohort=='main'].groupby('group'):
        key='full/'+group;cells[key]={};gates[key]=frozen.decisions(frame,c)
        for metric in metrics:
            q=estimate(frame[metric],c)
            if metric not in ['refit_minus_frozen','intact_minus_frozen','transfer_minus_source','transfer_minus_target_refit']:
                for k in ['paired_dz','wins','ties','losses']:q.pop(k)
            cells[key][metric]=q
    main=b[b.cohort=='main'];a=main[main.group=='gamma'].set_index('seed');m=main[main.group=='matched'].set_index('seed')
    delta=m.test_accuracy-a.test_accuracy;refit=m.target_refit_accuracy-a.target_refit_accuracy
    paired=pd.DataFrame(dict(frozen_specificity=delta,refit_specificity=refit)).reset_index();paired['family']='full'
    specificity=dict(frozen=estimate(delta,c),refit=estimate(refit,c),gate=bool(intact_access and delta.mean()>=.05 and (delta>0).all()))
    fresh=gates['full/gamma']['refit_benefit']
    summary=dict(primary_H1=fresh,discovery_H1=discovery_H1,confirmation_H1=bool(discovery_H1 and fresh),
        cells=cells,gates=gates,secondary_specificity={'full':specificity},independent_confirmation=True,
        biological_circuits_new=False,cohorts_pooled=False)
    return raw,b,paired,summary


def plot(rows,out,c):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    f=pd.DataFrame(rows);f=f[f.cohort=='main'];fig,axes=plt.subplots(1,2,figsize=(11,4.4))
    for ax,group in zip(axes,['gamma','matched']):
        b=f[f.group==group].groupby('lag')[['test_accuracy','target_refit_accuracy','source_within_accuracy']].mean().reindex(c['lags'])
        for name,label,style in [('test_accuracy','Frozen intact head','o-'),('target_refit_accuracy','Fresh target refit','s--'),('source_within_accuracy','Fresh intact reference',':')]:ax.plot(range(len(b)),b[name],style,label=label)
        ax.axhline(.25,color='gray',ls=':');ax.set(title='Fresh / '+group,xticks=range(len(b)),xticklabels=c['lags'],xlabel='Symbol lag',ylabel='Independent-stream accuracy',ylim=(0,1.03));ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'confirmation-curve.png',dpi=180);plt.close(fig)


def seal(root,c,context,**extra):
    core.write_json(root/'manifest.json',dict(config=c,context=context,timestamp=datetime.now(timezone.utc).isoformat(),
        environment=core.environment(),**extra,artifacts={p.relative_to(root).as_posix():core.sha256(p) for p in root.rglob('*') if p.is_file()}))


@threadpool_limits.wrap(limits=1)
def run(config,out):
    c=read(config);validate(c);check(Path(c['discovery']))
    assert core.sha256(Path(c['discovery'])/'manifest.json')==c['discovery_manifest_sha256']
    discovery=read(Path(c['discovery'])/'summary.json')['primary_H1'];assert discovery
    out.mkdir(parents=True,exist_ok=False);budget=Budget(c)
    try:
        context=core.source_context(c)
        for p in [str(config),'docs/kc-confirmation-seed-audit.json','scripts/kc_confirmation.py','scripts/verify_kc_confirmation.py','scripts/kc_ablation.py','scripts/kc_frozen.py','scripts/normalization_transfer.py','scripts/structural_k4.py','scripts/alphabet_memory.py','scripts/frozen_state_probe.py','scripts/context_memory.py','scripts/verify_kc_ablation.py','scripts/audit_kc_metrics.py']:
            context['source_sha256'][p]=core.sha256(p)
        prior={p:core.sha256(p) for p in subprocess.check_output(['git','ls-files','results/'],text=True).splitlines()}
        cr=out/'conditions';fr=out/'frozen';cr.mkdir();fr.mkdir()
        fc=dict(c,conditions={'full':['gamma','matched0','matched1','matched2']},sources={'full':cr.as_posix()})
        fcontext=dict(context,config_sha256=core.fingerprint(fc))
        for root,settings in [(out,c),(cr,c),(fr,fc)]:
            core.write_json(root/'config.json',settings);core.write_json(root/'prior-results-sha256.json',prior)
        core.write_json(out/'baseline.json',lesion.baseline());print('Historical complete checkpoint and metrics exactly reproduced',flush=True)
        transferrows=[];baselines=[];drifts=[];condition_count=0;transfer_count=0
        for cohort in ['smoke','main']:
            cc=dict(c)
            if cohort=='smoke':cc.update(c['smoke'])
            rows=[];neural=[];audits=[];(cr/cohort).mkdir()
            for block in cc['blocks']:
                for ci in cc['circuit_seeds']:
                    for arm in cc['conditions']:
                        budget.check();started=time.perf_counter();dest=cr/cohort/f'{arm}_c{ci}_s{block["seed"]}';dest.mkdir()
                        a,g,m,n,raw,w=lesion.execute(cc,ci,block,arm)
                        np.savez_compressed(dest/'checkpoint.npz',**a);sparse.save_npz(dest/'raw-graph.npz',raw);sparse.save_npz(dest/'weights.npz',w)
                        for name,value in [('graph',g),('metrics',m),('neural',n)]:core.write_json(dest/f'{name}.json',value)
                        replay,g2,m2,n2,_,_=lesion.execute(cc,ci,block,arm)
                        assert g==g2 and m==m2 and n==n2
                        with np.load(dest/'checkpoint.npz') as saved:
                            assert set(saved.files)==set(replay)
                            for k in replay:np.testing.assert_array_equal(saved[k],replay[k],err_msg=k)
                            verify_fold(dict(saved))
                        identity=dict(arm=arm,circuit_seed=ci,**block)
                        rows.extend(dict(**identity,**r) for r in m);neural.append(dict(**identity,**n))
                        audits.append(dict(**identity,**{k:g[k] for k in ['removed_count','overlap_with_gamma','gamma_count','gamma_overlap_fraction','removed_edges','remaining_edges','removed_absolute_strength','removed_input_counts']}))
                        core.write_json(dest/'manifest.json',dict(config=cc,identity=identity,context=context,timestamp=datetime.now(timezone.utc).isoformat(),
                            exact_replay=True,independent_lstsq=True,coefficients_per_head=196,seconds_including_replay=time.perf_counter()-started,
                            artifacts={p.name:core.sha256(p) for p in dest.iterdir() if p.is_file()}))
                        condition_count+=1
                    sp=cr/cohort/f'intact_c{ci}_s{block["seed"]}';source,_=frozen.load(sp);verify_fold(source)
                    own,null=predict_frozen(source,source['xtest']);np.testing.assert_array_equal(own,source['scores']);np.testing.assert_array_equal(null,source['null_scores'])
                    base=within_measures([source],c['lags'])
                    for j,r in enumerate(base):r['training_mse']=float(np.mean((source['train_scores'][:,j]-np.eye(4)[source['ytrain'][:,j]])**2))
                    assert base==read(sp/'metrics.json')
                    baselines.append(dict(cohort=cohort,seed=block['seed'],circuit_seed=ci,source=sp.as_posix(),expected_metrics=base,actual_metrics=base,exact_self_prediction=True,exact_refit=True))
                    for arm in fc['conditions']['full']:
                        budget.check();started=time.perf_counter();tp=cr/cohort/f'{arm}_c{ci}_s{block["seed"]}';target,_=frozen.load(tp)
                        frozen.audit_pair(source,target,sp,tp);a,m=frozen.evaluate(source,target,c);verify_transfer(source,target,a)
                        replay,m2=frozen.evaluate(source,target,c);assert m==m2
                        dest=fr/cohort/'full'/f'{arm}_c{ci}_s{block["seed"]}';dest.mkdir(parents=True)
                        np.savez_compressed(dest/'checkpoint.npz',**a)
                        with np.load(dest/'checkpoint.npz') as saved:
                            for k in replay:np.testing.assert_array_equal(saved[k],replay[k])
                        identity=dict(cohort=cohort,family='full',arm=arm,group='gamma' if arm=='gamma' else 'matched',seed=block['seed'],circuit_seed=ci)
                        transferrows.extend(dict(**identity,**r) for r in m);core.write_json(dest/'metrics.json',m)
                        drift=dict(mean_abs_train_shift_z=float(abs(a['train_mean_shift_z']).mean()),median_train_scale_ratio=float(np.median(a['train_scale_ratio'])),
                            mean_train_feature_correlation=float(a['train_feature_correlation'].mean()),mean_paired_train_cosine=float(a['paired_train_cosine'].mean()))
                        drifts.append(dict(**identity,**drift))
                        core.write_json(dest/'manifest.json',dict(identity=identity,config=fc,context=fcontext,timestamp=datetime.now(timezone.utc).isoformat(),source_path=sp.as_posix(),target_path=tp.as_posix(),
                            source_manifest_sha256=core.sha256(sp/'manifest.json'),target_manifest_sha256=core.sha256(tp/'manifest.json'),source_checkpoint_sha256=core.sha256(sp/'checkpoint.npz'),target_checkpoint_sha256=core.sha256(tp/'checkpoint.npz'),
                            target_graph_sha256=core.sha256(tp/'graph.json'),target_neural_sha256=core.sha256(tp/'neural.json'),new_transfer_target_fits=0,exact_replay=True,independent_source_lstsq=True,coefficients_per_lag=196,
                            seconds_including_checks=time.perf_counter()-started,drift=drift,artifacts={p.name:core.sha256(p) for p in dest.iterdir() if p.is_file()}))
                        transfer_count+=1
                    print(f'{cohort} c{ci} s{block["seed"]}:5conditions fully replayed +4frozen transfers verified',flush=True)
            dest=cr/cohort
            for name,data in [('raw-lag-table',rows),('neural-diagnostics',neural),('lesion-audit',audits)]:pd.DataFrame(data).to_csv(dest/f'{name}.csv',index=False)
            if cohort=='main':
                cases,blocks,differences,refit_summary=lesion.statistics(rows,c)
                cases.to_csv(dest/'raw-past-table.csv',index=False);blocks.to_csv(dest/'seed-blocks.csv',index=False);differences.to_csv(dest/'paired-differences.csv');core.write_json(dest/'summary.json',refit_summary)
            lesion.plot(rows,dest,c)
        raw,blocks,paired,summary=aggregate(transferrows,c,refit_summary['gates']['intact_past_access'],discovery)
        for name,data in [('raw-lag-table',pd.DataFrame(transferrows)),('raw-past-table',raw),('seed-blocks',blocks),('paired-specificity',paired),('feature-drift',pd.DataFrame(drifts))]:data.to_csv(fr/f'{name}.csv',index=False)
        core.write_json(fr/'summary.json',summary);core.write_json(fr/'baselines.json',baselines);plot(transferrows,fr,c)
        for p,h in {**prior,**context['source_sha256']}.items():assert core.sha256(p)==h,p
        budget.check();usage=budget.close();assert (condition_count,transfer_count,len(baselines))==(35,28,7)
        seal(cr,c,context);seal(fr,fc,fcontext)
        core.write_json(out/'verification.json',dict(condition_full_replays=35,condition_independent_refits=35,frozen_transfers=28,frozen_exact_replays=28,
            source_self_checks=7,prior_results_unchanged=len(prior),budget=usage))
        seal(out,c,context,discovery_manifest_sha256=c['discovery_manifest_sha256'])
        print(dict(confirmation_H1=summary['confirmation_H1'],gates=summary['gates'],budget=usage),flush=True)
    finally:budget.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=Path('configs/kc_confirmation.json'));p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();run(args.config,args.out)
