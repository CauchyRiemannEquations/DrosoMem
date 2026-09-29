"""Supplementary metric audit using direct formulas, not the runner's measures."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from alphabet_memory import check,read
from flying.training import whole_brain_memory as core


@threadpool_limits.wrap(limits=1)
def audit(roots,out):
    records=[]
    for root in roots:
        top=check(root);c=top['config'];count=0
        for cohort in ['smoke','main']:
            rows=[]
            for file in sorted((root/cohort).glob('*/manifest.json')):
                m=check(file.parent)
                with np.load(file.parent/'checkpoint.npz') as a:
                    train_scores=(((a['xtrain']-a['mean'])/a['scale'])@a['weights']+a['bias']).reshape(a['train_scores'].shape)
                    np.testing.assert_array_equal(a['train_scores'],train_scores)
                    measured=read(file.parent/'metrics.json')
                    for j,lag in enumerate(c['lags']):
                        y=a['ytest'][:,j];yt=a['ytrain'][:,j];scores=a['scores'][:,j];train=train_scores[:,j]
                        freq=np.bincount(yt,minlength=4)/len(yt)
                        np.testing.assert_array_equal(a['majority'][:,j],np.full(len(y),freq.argmax()))
                        test_acc=float(np.mean(scores.argmax(axis=1)==y));base=float(np.mean(y==freq.argmax()))
                        null=float(np.mean(a['null_scores'][:,j].argmax(axis=1)==y));target=np.eye(4)[y]
                        expected=dict(lag=lag,test_accuracy=test_acc,train_accuracy=float(np.mean(train.argmax(axis=1)==yt)),
                            frequency_accuracy=base,null_accuracy=null,frequency_excess=test_acc-base,null_excess=test_acc-null,
                            r2_vs_frequency=float(1-np.sum((target-scores)**2)/np.sum((target-freq)**2)),
                            training_mse=float(np.mean((train-np.eye(4)[yt])**2)))
                        assert set(expected)==set(measured[j])
                        for key,value in expected.items():np.testing.assert_allclose(measured[j][key],value,rtol=0,atol=1e-13)
                        rows.append(dict(seed=m['identity']['seed'],circuit_seed=m['identity']['circuit_seed'],arm=m['identity']['arm'],**expected))
                        count+=1
            raw=pd.read_csv(root/cohort/'raw-lag-table.csv');computed=pd.DataFrame(rows);index=['seed','circuit_seed','arm','lag']
            columns=list(expected);columns.remove('lag')
            np.testing.assert_allclose(raw.set_index(index)[columns].sort_index(),computed.set_index(index)[columns].sort_index(),rtol=0,atol=1e-13)
        summary=read(root/'main/summary.json');blocks=pd.read_csv(root/'main/seed-blocks.csv')
        for arm,metrics in summary['cells'].items():
            b=blocks[blocks.arm==arm]
            for metric,q in metrics.items():
                x=b[metric].to_numpy();rng=np.random.default_rng(c['bootstrap_seed'])
                means=x[rng.integers(0,3,size=(10000,3))].mean(axis=1)
                np.testing.assert_allclose([q['mean'],q['median'],q['variance'],*q['bootstrap95']],
                    [x.mean(),np.median(x),x.var(ddof=1),*np.quantile(means,[.025,.975])],rtol=0,atol=1e-13)
        records.append(dict(root=root.as_posix(),condition_lag_rows=count,all_metric_formulas=True,raw_tables=True,cell_statistics=True,
            source_manifest_sha256=core.sha256(root/'manifest.json')))
    out.mkdir(parents=True,exist_ok=False)
    core.write_json(out/'checks.json',dict(verified=True,records=records,verifier_sha256=core.sha256(__file__)))
    print(read(out/'checks.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('roots',type=Path,nargs='+');p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();audit(args.roots,args.out)
