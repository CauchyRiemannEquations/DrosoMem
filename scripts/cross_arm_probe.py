"""Transfer the exact frozen blocked heads to the opposite paired ordering."""
import argparse,json,subprocess
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from alphabet_memory import read,check
from context_memory import estimate
from frozen_state_probe import verify_fold,measures,labels,Budget
from flying.training import whole_brain_memory as core

FROZEN=['mean','scale','weights','bias','null_weights','null_bias','xtrain','ytrain','train_indices','train_scores']


def transfer(source,x,y):
    a={k:np.array(v,copy=True) for k,v in source.items()}
    a['within_ytest']=a['ytest'].copy();a['within_scores']=a['scores'].copy()
    a['xtest']=np.asarray(x);a['ytest']=np.asarray(y)
    z=(a['xtest']-a['mean'])/a['scale']
    a['scores']=(z@a['weights']+a['bias']).reshape(len(y),y.shape[1],4)
    a['null_scores']=(z@a['null_weights']+a['null_bias']).reshape(a['scores'].shape)
    a['predictions']=a['scores'].argmax(axis=2);a['null_predictions']=a['null_scores'].argmax(axis=2)
    a['majority']=np.broadcast_to(a['bias'].reshape(-1,4).argmax(axis=1),y.shape).copy()
    a['changed_targets']=a['ytest']!=a['within_ytest']
    for key in FROZEN:
        if key in source:np.testing.assert_array_equal(a[key],source[key])
    verify_fold(a)
    return a


def summarize(rows,out,c):
    f=pd.DataFrame(rows);f.to_csv(out/'raw-task-table.csv',index=False)
    metrics=['test_accuracy','within_accuracy','transfer_difference','frequency_accuracy','null_accuracy','frequency_excess','null_excess','r2_vs_frequency']
    agg=[]
    for task,lags in [('past',c['primary_lags']),('current',[0]),('next',[-1])]:
        b=f[f.lag.isin(lags)].groupby(['cohort','level','source_arm','seed','circuit_seed'])[metrics].mean().reset_index();b['task']=task;agg.append(b)
    aggregate=pd.concat(agg);aggregate.to_csv(out/'raw-aggregate-table.csv',index=False)
    blocks=aggregate.groupby(['cohort','level','source_arm','seed','task'])[metrics].mean().reset_index();blocks.to_csv(out/'seed-blocks.csv',index=False)
    cells={};access={};retention={};retention_rows=[]
    for cohort in ['main','confirmation']:
        cells[cohort]={};access[cohort]={};retention[cohort]={};required=4 if cohort=='main' else 3
        for g in ['legacy5','brain1']:
            cells[cohort][g]={};access[cohort][g]={}
            sub=blocks[(blocks.cohort==cohort)&(blocks.level==g)]
            for arm in ['low','high']:
                b=sub[sub.source_arm==arm];cells[cohort][g][arm]={task:{m:estimate(q[m],c) for m in metrics} for task,q in b.groupby('task')}
                past=b[b.task=='past']
                access[cohort][g][arm]=bool(past.frequency_excess.mean()>=.05 and past.null_excess.mean()>=.05 and
                    ((past.frequency_excess>=.05)&(past.null_excess>=.05)).sum()>=required and past.r2_vs_frequency.mean()>0)
            # Both directions and strata are repeated conditions, not new seeds.
            paired=sub[sub.task=='past'].groupby('seed').transfer_difference.mean()
            retention[cohort][g]=dict(**estimate(paired,c),passed=bool(paired.mean()>=-.05 and (paired>=-.05).sum()>=required),
                blocks_within_tolerance=int((paired>=-.05).sum()))
            retention_rows.extend(dict(cohort=cohort,level=g,seed=int(seed),transfer_minus_within=float(value)) for seed,value in paired.items())
    pd.DataFrame(retention_rows).to_csv(out/'paired-retention.csv',index=False)
    lagblocks=f.groupby(['cohort','level','source_arm','seed','lag'])[metrics].mean().reset_index()
    lagstats={f'{cohort}/{g}/{arm}/{lag}':{m:estimate(b[m],c) for m in metrics} for (cohort,g,arm,lag),b in lagblocks.groupby(['cohort','level','source_arm','lag'])}
    result=dict(cells=cells,past_access_gates=access,retention=retention,lag_statistics=lagstats,
        supported_transfer=[f'{g}/{arm}' for g in ['legacy5','brain1'] for arm in ['low','high'] if access['main'][g][arm] and access['confirmation'][g][arm]],
        supported_retention=[g for g in ['legacy5','brain1'] if retention['main'][g]['passed'] and retention['confirmation'][g]['passed']])
    core.write_json(out/'summary.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    for ax,cohort in zip(axes,['main','confirmation']):
        for i,g in enumerate(['legacy5','brain1']):
            for j,arm in enumerate(['low','high']):
                e=cells[cohort][g][arm]['past']; x=i*2+j
                ax.plot([x-.15,x+.15],[e['within_accuracy']['mean'],e['test_accuracy']['mean']],'o-',color=['#1976b5','#e17430'][i])
                ax.annotate(f"{100*e['transfer_difference']['mean']:+.1f} pp",(x-.1,max(e['within_accuracy']['mean'],e['test_accuracy']['mean'])+.03),fontsize=9)
        ax.set(xticks=range(4),xticklabels=['partial L→H','partial H→L','whole L→H','whole H→L'],ylim=(.2,.85),ylabel='Mean past-symbol accuracy',title=f'{cohort}: within (left) → transfer (right)');ax.tick_params(axis='x',labelrotation=18);ax.axhline(.25,color='gray',ls=':')
    fig.tight_layout();fig.savefig(out/'transfer-curve.png',dpi=180);plt.close(fig)
    return result


@threadpool_limits.wrap(limits=1)
def run(config,out):
    c=read(config);out.mkdir(parents=True,exist_ok=False);budget=Budget(c)
    try:
        source_probes=Path(c['source_probes']);oldprobes=check(source_probes)
        old={p:core.sha256(p) for p in subprocess.check_output(['git','ls-files','results/'],text=True).splitlines()};core.write_json(out/'prior-results-sha256.json',old)
        core.write_json(out/'config.json',c);rows=[];verified=0
        for cohort,path in c['sources'].items():
            root=Path(path);check(root)
            for group in sorted(root.glob('*_c*_s*')):
                if not group.is_dir():continue
                low=check(group/'low');high=check(group/'high')
                for key in ['input_root_ids','observation_root_ids','input_mapping_sha256','weight_sha256']:assert low['graph'][key]==high['graph'][key]
                for source_arm,target_arm in [('low','high'),('high','low')]:
                    p=source_probes/cohort/group.name/source_arm;meta=check(p)
                    target=group/target_arm;target_meta=check(target)
                    with np.load(target/'checkpoint.npz',allow_pickle=False) as archive:x=archive['features'];s=archive['symbols']
                    dest=out/cohort/group.name/f'{source_arm}_to_{target_arm}';dest.mkdir(parents=True)
                    folds=[];sourcefolds=[];headhashes=[]
                    for i in range(3):
                        budget.check()
                        with np.load(p/f'fold{i}.npz',allow_pickle=False) as saved:oldfold=dict(saved)
                        test=oldfold['test_indices'];a=transfer(oldfold,x[test],labels(s,test,c['lags']))
                        file=dest/f'fold{i}.npz';np.savez_compressed(file,**a)
                        with np.load(file,allow_pickle=False) as saved:
                            verify_fold(dict(saved))
                            for key in FROZEN:np.testing.assert_array_equal(saved[key],oldfold[key])
                        core.write_json(dest/f'fold{i}-metrics.json',measures([a],c['lags']))
                        folds.append(a);sourcefolds.append(oldfold);verified+=1;headhashes.append(core.sha256(p/f'fold{i}.npz'))
                    measured=measures(folds,c['lags']);within=measures(sourcefolds,c['lags'])
                    r=meta['identity'];identity=dict(cohort=cohort,level=r['level'],source_arm=source_arm,target_arm=target_arm,seed=r['seed'],dataset_seed=r['dataset_seed'],circuit_seed=r['circuit_seed'])
                    for j,(v,w) in enumerate(zip(measured,within)):
                        v['within_accuracy']=w['test_accuracy'];v['transfer_difference']=v['test_accuracy']-w['test_accuracy']
                        correct=np.concatenate([a['predictions'][:,j]==a['ytest'][:,j] for a in folds]);changed=np.concatenate([a['changed_targets'][:,j] for a in folds])
                        v.update(changed_target_count=int(changed.sum()),changed_target_accuracy=float(correct[changed].mean()) if changed.any() else None,
                            same_target_count=int((~changed).sum()),same_target_accuracy=float(correct[~changed].mean()) if (~changed).any() else None,
                            current_symbol_equality_fraction=float(np.mean(np.concatenate([a['ytest'][:,0]==a['within_ytest'][:,0] for a in folds]))))
                        rows.append(dict(**identity,**v))
                    core.write_json(dest/'manifest.json',dict(identity=identity,source_probe_manifest_sha256=core.sha256(p/'manifest.json'),source_fold_hashes=headhashes,
                        target_manifest_sha256=core.sha256(target/'manifest.json'),target_checkpoint_sha256=core.sha256(target/'checkpoint.npz'),metrics=measured,
                        artifacts={q.name:core.sha256(q) for q in dest.iterdir() if q.is_file()}))
            print(f'{cohort}: {verified} frozen folds checked',flush=True)
        summary=summarize(rows,out,c)
        for p,h in old.items():assert core.sha256(p)==h,p
        budget.check();usage=budget.close()
        core.write_json(out/'verification.json',dict(frozen_folds=verified,exact_coefficient_replays=verified,source_only_verification_refits=verified,
            independent_lstsq_checks=verified,new_target_arm_fits=0,prior_files_unchanged=len(old),budget=usage))
        core.write_json(out/'manifest.json',dict(config=c,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),timestamp=datetime.now(timezone.utc).isoformat(),environment=core.environment(),
            source_probe_manifest_sha256=core.sha256(source_probes/'manifest.json'),source_hashes={p:core.sha256(p) for p in [str(Path(__file__)),c['protocol'],str(config),'scripts/frozen_state_probe.py','src/flying/models/ridge.py','scripts/context_memory.py']},
            artifacts={p.relative_to(out).as_posix():core.sha256(p) for p in out.rglob('*') if p.is_file()}))
        print(json.dumps(dict(supported_transfer=summary['supported_transfer'],supported_retention=summary['supported_retention'],usage=usage),indent=2))
    finally:budget.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=Path('configs/cross_arm_probe.json'));p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.config,a.out)
