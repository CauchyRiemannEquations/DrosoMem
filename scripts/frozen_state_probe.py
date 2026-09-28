"""Purged within-trajectory affine probes; no new reservoir trajectories."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import json,subprocess,sys,time,threading
import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits
from flying.models.ridge import RidgeDecoder
from flying.training import whole_brain_memory as core
from alphabet_memory import read,check
from context_memory import estimate


def splits(c):
    grid=np.arange(c['start'],c['stop'])
    return [(grid[np.all(np.abs(grid[:,None]-test[None,:])>c['purge'],axis=1)],test)
            for test in np.array_split(grid,c['folds'])]


def labels(symbols,times,lags):
    return np.array([[symbols[t-lag] for lag in lags] for t in times],dtype=np.int64)


def fit_fold(xtrain,xtest,ytrain,ytest,alpha=1.):
    targets=np.eye(4)[ytrain].reshape(len(ytrain),-1)
    shifted=np.roll(targets,len(targets)//2,axis=0)
    real=RidgeDecoder(alpha).fit(xtrain,targets);null=RidgeDecoder(alpha).fit(xtrain,shifted)
    shape=(len(ytest),ytest.shape[1],4)
    a=dict(xtrain=xtrain,xtest=xtest,ytrain=ytrain,ytest=ytest,mean=real.mean,scale=real.scale,
        weights=real.weights,bias=real.target_mean,null_weights=null.weights,null_bias=null.target_mean,
        scores=real.scores(xtest).reshape(shape),null_scores=null.scores(xtest).reshape(shape),
        train_scores=real.scores(xtrain).reshape(len(ytrain),ytrain.shape[1],4))
    a['predictions']=a['scores'].argmax(axis=2);a['null_predictions']=a['null_scores'].argmax(axis=2)
    a['majority']=np.broadcast_to(real.target_mean.reshape(-1,4).argmax(axis=1),ytest.shape).copy()
    verify_fold(a,alpha)
    return a


def verify_fold(a,alpha=1.):
    targets=np.eye(4)[a['ytrain']].reshape(len(a['ytrain']),-1)
    z=(a['xtrain']-a['mean'])/a['scale']; test=(a['xtest']-a['mean'])/a['scale']
    np.testing.assert_array_equal(a['mean'],a['xtrain'].mean(axis=0))
    np.testing.assert_array_equal(a['scale'],np.maximum(a['xtrain'].std(axis=0),1e-5))
    # Independent augmented least-squares solver, without normal equations.
    augmented=np.vstack([z,np.sqrt(alpha)*np.eye(z.shape[1])])
    for target,w,b,score,pred in [(targets,'weights','bias','scores','predictions'),
            (np.roll(targets,len(targets)//2,axis=0),'null_weights','null_bias','null_scores','null_predictions')]:
        np.testing.assert_array_equal(target.mean(axis=0),a[b])
        fresh=RidgeDecoder(alpha).fit(a['xtrain'],target)
        np.testing.assert_array_equal(fresh.weights,a[w]);np.testing.assert_array_equal(fresh.scores(a['xtest']).reshape(a[score].shape),a[score])
        rebuilt=(test@a[w]+a[b]).reshape(a[score].shape)
        np.testing.assert_array_equal(rebuilt,a[score]);np.testing.assert_array_equal(rebuilt.argmax(axis=2),a[pred])
        rhs=np.vstack([target-target.mean(axis=0),np.zeros((z.shape[1],target.shape[1]))])
        ref=np.linalg.lstsq(augmented,rhs,rcond=None)[0]
        reference=(test@ref+target.mean(axis=0)).reshape(a[score].shape)
        np.testing.assert_allclose(reference,a[score],atol=1e-9,rtol=1e-9)
        np.testing.assert_array_equal(reference.argmax(axis=2),a[pred])


def measures(folds,lags):
    rows=[]
    for j,lag in enumerate(lags):
        truth=np.concatenate([a['ytest'][:,j] for a in folds]);pred=np.concatenate([a['predictions'][:,j] for a in folds])
        null=np.concatenate([a['null_predictions'][:,j] for a in folds]);majority=np.concatenate([a['majority'][:,j] for a in folds])
        scores=np.concatenate([a['scores'][:,j] for a in folds]);onehot=np.eye(4)[truth]
        frequency=np.concatenate([np.broadcast_to(a['bias'].reshape(-1,4)[j],(len(a['ytest']),4)) for a in folds])
        denominator=np.sum((onehot-frequency)**2)
        traintruth=np.concatenate([a['ytrain'][:,j] for a in folds]);trainpred=np.concatenate([a['train_scores'][:,j].argmax(axis=1) for a in folds])
        accuracy=float(np.mean(truth==pred));base=float(np.mean(truth==majority));shift=float(np.mean(truth==null))
        rows.append(dict(lag=lag,test_accuracy=accuracy,train_accuracy=float(np.mean(traintruth==trainpred)),frequency_accuracy=base,
            null_accuracy=shift,frequency_excess=accuracy-base,null_excess=accuracy-shift,
            r2_vs_frequency=float(1-np.sum((onehot-scores)**2)/denominator)))
    return rows


def summarize(rows,out,c,mode='blocked'):
    f=pd.DataFrame(rows);f.to_csv(out/'raw-task-table.csv',index=False)
    metric=['test_accuracy','train_accuracy','frequency_accuracy','null_accuracy','frequency_excess','null_excess','r2_vs_frequency']
    sets={}
    for task,lags in [('past',c['primary_lags']),('current',[0]),('next',[-1])]:
        r=f[f.lag.isin(lags)].groupby(['cohort','level','arm','seed','circuit_seed'])[metric].mean().reset_index();r['task']=task;sets[task]=r
    aggregate=pd.concat(sets.values());aggregate.to_csv(out/'raw-aggregate-table.csv',index=False)
    blocks=aggregate.groupby(['cohort','level','arm','seed','task'])[metric].mean().reset_index();blocks.to_csv(out/'seed-blocks.csv',index=False)
    cells={};contrasts={};access={};nextgates={};paired=[]
    for cohort in ['main','confirmation']:
        cells[cohort]={};contrasts[cohort]={};access[cohort]={};nextgates[cohort]={}
        required=4 if cohort=='main' else 3
        for g in ['legacy5','brain1']:
            cells[cohort][g]={};access[cohort][g]={};contrasts[cohort][g]={}
            sub=blocks[(blocks.cohort==cohort)&(blocks.level==g)]
            for arm in ['low','high']:
                cells[cohort][g][arm]={}
                for task in ['past','current','next']:
                    b=sub[(sub.arm==arm)&(sub.task==task)]
                    cells[cohort][g][arm][task]={m:estimate(b[m],c) for m in metric}
                p=sub[(sub.arm==arm)&(sub.task=='past')]
                access[cohort][g][arm]=bool(p.frequency_excess.mean()>=.05 and p.null_excess.mean()>=.05 and
                    ((p.frequency_excess>=.05)&(p.null_excess>=.05)).sum()>=required and p.r2_vs_frequency.mean()>0)
            for task in ['past','current','next']:
                a=sub[(sub.arm=='low')&(sub.task==task)].set_index('seed');b=sub[(sub.arm=='high')&(sub.task==task)].set_index('seed');delta=a[metric]-b[metric]
                contrasts[cohort][g][task]={m:estimate(delta[m],c) for m in metric}
                paired.extend(dict(cohort=cohort,level=g,task=task,seed=int(seed),**r) for seed,r in delta.to_dict('index').items())
            e=contrasts[cohort][g]['next']['test_accuracy'];nextgates[cohort][g]=e['mean']>=.05 and e['wins']>=required
    pd.DataFrame(paired).to_csv(out/'paired-differences.csv',index=False)
    # Per-lag uncertainty also averages strata within seed first.
    lagblocks=f.groupby(['cohort','level','arm','seed','lag'])[metric].mean().reset_index()
    lagstats={f'{cohort}/{g}/{arm}/{lag}':{m:estimate(b[m],c) for m in metric} for (cohort,g,arm,lag),b in lagblocks.groupby(['cohort','level','arm','lag'])}
    result=dict(mode=mode,cells=cells,low_minus_high=contrasts,past_access_gates=access,next_difference_gates=nextgates,lag_statistics=lagstats,
        confirmed_past_access=[f'{g}/{arm}' for g in ['legacy5','brain1'] for arm in ['low','high'] if access['main'][g][arm] and access['confirmation'][g][arm]],
        confirmed_next_difference=[g for g in ['legacy5','brain1'] if nextgates['main'][g] and nextgates['confirmation'][g]])
    core.write_json(out/'summary.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.3))
    order=[0,1,2,3,4,5,8,-1]
    for ax,cohort in zip(axes,['main','confirmation']):
        for g,color in [('legacy5','#1976b5'),('brain1','#e17430')]:
            for arm,style in [('low','-'),('high','--')]:
                means=[lagstats[f'{cohort}/{g}/{arm}/{lag}']['test_accuracy']['mean'] for lag in order]
                ax.plot(range(8),means,'o'+style,color=color,label=f'{g} {arm}')
        ax.axhline(.25,color='gray',ls=':');ax.set(xticks=range(8),xticklabels=['now','1','2','3','4','5','8','next'],ylim=(0,1),ylabel='Held-out symbol accuracy',xlabel='Decoded delay / next symbol',title=cohort);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'probe-curve.png',dpi=180);plt.close(fig)
    return result


class Budget:
    def __init__(self,c):
        self.c=c;self.start=time.perf_counter();self.proc=psutil.Process();self.peak=self.proc.memory_info().rss;self.stop=threading.Event()
        self.thread=threading.Thread(target=self.monitor,daemon=True);self.thread.start()
    def monitor(self):
        while not self.stop.wait(.05):self.peak=max(self.peak,self.proc.memory_info().rss)
    def check(self):
        if time.perf_counter()-self.start>self.c['max_seconds'] or self.peak>self.c['max_rss_bytes']:raise RuntimeError('Budget exceeded; preserve partial artifacts')
    def close(self):
        self.stop.set();self.thread.join();return dict(seconds=time.perf_counter()-self.start,peak_sampled_rss_bytes=self.peak,sample_seconds=.05)


def baseline():
    p=Path('results/delay_main_v2/legacy5_c701_s14142');check(p)
    with np.load(p/'checkpoint.npz',allow_pickle=False) as a:
        x=a['train_features'][100:];targets=np.eye(10)[a['train_targets']].reshape(len(x),-1)
        head=RidgeDecoder(1).fit(x,targets)
        for old,new in [('mean',head.mean),('scale',head.scale),('weights',head.weights),('target_mean',head.target_mean)]:np.testing.assert_array_equal(a[old],new)
        np.testing.assert_array_equal(a['test_scores'],head.scores(a['test_features'][100:]).reshape(a['test_scores'].shape))
    return dict(source=str(p),manifest_sha256=core.sha256(p/'manifest.json'),exact_refit=True,exact_scores=True)


@threadpool_limits.wrap(limits=1)
def run(config,out):
    c=read(config);out.mkdir(parents=True,exist_ok=False);budget=Budget(c)
    try:
        core.write_json(out/'config.json',c);core.write_json(out/'baseline.json',baseline())
        old={p:core.sha256(p) for p in subprocess.check_output(['git','ls-files','results/'],text=True).splitlines()};core.write_json(out/'prior-results-sha256.json',old)
        rows=[];verified=0;sources={}
        for cohort,path in c['sources'].items():
            source=Path(path);m=check(source);sources[cohort]=core.sha256(source/'manifest.json')
            for group in sorted(source.glob('*_c*_s*')):
                if not group.is_dir():continue
                for arm in ['low','high']:
                    p=group/arm;meta=check(p);r=meta['metrics'];dest=out/cohort/group.name/arm;dest.mkdir(parents=True)
                    with np.load(p/'checkpoint.npz',allow_pickle=False) as archive:x=archive['features'];s=archive['symbols']
                    assert x.shape==(127,48) and len(s)==128 and r['alphabet_size']==4
                    folds=[]
                    for i,(train,test) in enumerate(splits(c)):
                        budget.check();a=fit_fold(x[train],x[test],labels(s,train,c['lags']),labels(s,test,c['lags']),c['alpha'])
                        a.update(train_indices=train,test_indices=test)
                        file=dest/f'fold{i}.npz';np.savez_compressed(file,**a)
                        with np.load(file,allow_pickle=False) as saved:verify_fold(dict(saved),c['alpha'])
                        foldmetrics=measures([a],c['lags']);core.write_json(dest/f'fold{i}-metrics.json',foldmetrics)
                        folds.append(a);verified+=1
                    identity=dict(cohort=cohort,level=r['level'],arm=arm,seed=r['seed'],dataset_seed=r['dataset_seed'],circuit_seed=r['circuit_seed'])
                    measured=measures(folds,c['lags']);rows.extend(dict(**identity,**v) for v in measured)
                    core.write_json(dest/'manifest.json',dict(identity=identity,source_manifest_sha256=core.sha256(p/'manifest.json'),source_checkpoint_sha256=core.sha256(p/'checkpoint.npz'),metrics=measured,
                        artifacts={q.name:core.sha256(q) for q in dest.iterdir() if q.is_file()}))
            print(f'{cohort}: {len(rows)//8} trajectories, all folds verified',flush=True)
        summary=summarize(rows,out,c)
        for p,h in old.items():assert core.sha256(p)==h,p
        budget.check();usage=budget.close()
        core.write_json(out/'verification.json',dict(exact_replayed_and_refitted_folds=verified,independent_lstsq_folds=verified,real_task_heads=verified*8,null_task_heads=verified*8,
            prior_files_unchanged=len(old),budget=usage))
        core.write_json(out/'manifest.json',dict(config=c,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),timestamp=datetime.now(timezone.utc).isoformat(),environment=core.environment(),
            source_manifests=sources,source_hashes={p:core.sha256(p) for p in [str(Path(__file__)),c['protocol'],str(config),'src/flying/models/ridge.py','scripts/context_memory.py']},
            artifacts={p.relative_to(out).as_posix():core.sha256(p) for p in out.rglob('*') if p.is_file()}))
        print(json.dumps(dict(confirmed_past_access=summary['confirmed_past_access'],confirmed_next_difference=summary['confirmed_next_difference'],usage=usage),indent=2))
    finally:budget.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=Path('configs/frozen_state_probe.json'));p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.config,a.out)
