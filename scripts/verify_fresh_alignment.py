"""Independent graph/stream/score/paired-statistic audit of fresh confirmation."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits
from alphabet_memory import read,check
from flying.training import whole_brain_memory as core


@threadpool_limits.wrap(limits=1)
def verify(root):
    manifest=check(root);c=manifest['config'];condition_count=0;directions=set();rows=[]
    for path in sorted((root/'conditions').glob('*/manifest.json')):
        record=check(path.parent);identity=record['identity'];g=read(path.parent/'graph.json')
        original,ids,_=core.load_connectome(Path(f'data/flywire_783_mb_left_kc512_s{identity["circuit_seed"]}'))
        raw=sparse.load_npz(path.parent/'raw-graph.npz');w=sparse.load_npz(path.parent/'weights.npz')
        assert core.weight_hash(raw)==g['raw_sha256'] and core.weight_hash(w)==g['weight_sha256']
        denominator=np.asarray(abs(raw if identity['arm']=='renormalized' else original).sum(axis=1)).ravel()
        d=np.divide(.9,denominator,out=np.zeros(len(ids)),where=denominator>0)
        np.testing.assert_array_equal(w.toarray(),raw.toarray()*d[:,None])
        with np.load(path.parent/'checkpoint.npz') as a:
            np.testing.assert_array_equal(a['normalization_factors'],d)
            assert a['xtrain'].shape==(2000,48) and a['xtest'].shape==(1000,48)
            for split,ykey in [('train','ytrain'),('test','ytest')]:
                n=c[split+'_samples'];symbols=np.random.default_rng(identity[split+'_seed']).integers(0,4,100+n).astype(np.uint8)
                np.testing.assert_array_equal(a[split+'_symbols'],symbols)
                labels=np.column_stack([symbols[100-lag:100-lag+n] for lag in c['lags']])
                np.testing.assert_array_equal(a[ykey],labels)
                np.testing.assert_array_equal(a['x'+split],a[split+'_features'][100:])
            mu=a['xtrain'].mean(axis=0);scale=np.maximum(a['xtrain'].std(axis=0),1e-5)
            np.testing.assert_array_equal(a['mean'],mu);np.testing.assert_array_equal(a['scale'],scale)
            z=(a['xtrain']-mu)/scale;zt=(a['xtest']-mu)/scale
            truth=np.eye(4)[a['ytrain']].reshape(2000,44);design=np.vstack([z,np.eye(48)])
            for prefix,labels in [('',truth),('null_',np.roll(truth,1000,axis=0))]:
                bias=labels.mean(axis=0);rhs=np.vstack([labels-bias,np.zeros((48,44))])
                fitted=np.linalg.lstsq(design,rhs,rcond=None)[0]
                np.testing.assert_allclose(fitted,a[prefix+'weights'],atol=1e-9,rtol=1e-9)
                np.testing.assert_array_equal(a[prefix+'bias'],bias)
                scores=(zt@a[prefix+'weights']+bias).reshape(1000,11,4)
                np.testing.assert_array_equal(a[prefix+'scores'],scores)
                np.testing.assert_array_equal(a[prefix+'predictions'],scores.argmax(axis=2))
            assert float(a['max_abs_by_stream'].max())<=1+1e-12
        condition_count+=1
    assert condition_count==12
    for path in sorted((root/'transfers').glob('*/*/manifest.json')):
        record=check(path.parent);identity=record['identity'];sp=Path(record['source_path']);tp=Path(record['target_path'])
        check(sp);check(tp);assert core.sha256(sp/'manifest.json')==record['source_manifest_sha256']
        assert core.sha256(tp/'manifest.json')==record['target_manifest_sha256']
        sg,tg=read(sp/'graph.json'),read(tp/'graph.json')
        for key in ['raw_sha256','observed_root_ids','input_root_ids','input_mapping_sha256']:assert sg[key]==tg[key]
        directions.add((identity['seed'],identity['circuit_seed'],identity['direction']))
        with np.load(path.parent/'checkpoint.npz') as a,np.load(sp/'checkpoint.npz') as source,np.load(tp/'checkpoint.npz') as target:
            for key in ['mean','scale','weights','bias','null_weights','null_bias']:np.testing.assert_array_equal(a[key],source[key])
            for key in ['train_symbols','test_symbols','observed_indices','input_patterns','ytrain','ytest']:np.testing.assert_array_equal(source[key],target[key])
            np.testing.assert_array_equal(a['xtest'],target['xtest']);np.testing.assert_array_equal(a['ytest'],target['ytest'])
            mu=target['train_features'][100:].mean(axis=0);scale=np.maximum(target['train_features'][100:].std(axis=0),1e-5)
            np.testing.assert_array_equal(a['target_mean'],mu);np.testing.assert_array_equal(a['target_scale'],scale)
            z=(a['xtest']-mu)/scale;oldz=(a['xtest']-source['mean'])/source['scale']
            for prefix in ['', 'null_']:
                score=(z@source[prefix+'weights']+source[prefix+'bias']).reshape(1000,11,4)
                np.testing.assert_array_equal(a[prefix+'scores'],score)
                np.testing.assert_array_equal(a[prefix+'predictions'],score.argmax(axis=2))
            np.testing.assert_array_equal(a['unaligned_scores'],(oldz@source['weights']+source['bias']).reshape(1000,11,4))
            np.testing.assert_array_equal(a['target_refit_scores'],target['scores']);np.testing.assert_array_equal(a['source_within_scores'],source['scores'])
            measured=read(path.parent/'metrics.json')
            for j,row in enumerate(measured):
                y=a['ytest'][:,j];oh=np.eye(4)[y];bias=source['bias'].reshape(11,4)[j]
                acc=float(np.mean(a['predictions'][:,j]==y));null=float(np.mean(a['null_predictions'][:,j]==y))
                frequency=float(np.mean(bias.argmax()==y));within=float(np.mean(source['scores'][:,j].argmax(axis=1)==y))
                refit=float(np.mean(target['scores'][:,j].argmax(axis=1)==y));old=float(np.mean(a['unaligned_scores'][:,j].argmax(axis=1)==y))
                error=float(np.sum((a['scores'][:,j]-oh)**2));denominator=float(np.sum((oh-bias)**2))
                expected=dict(test_accuracy=acc,frequency_accuracy=frequency,null_accuracy=null,frequency_excess=acc-frequency,null_excess=acc-null,
                    r2_vs_frequency=1-error/denominator,test_mse=error/oh.size,source_within_accuracy=within,target_refit_accuracy=refit,
                    transfer_minus_source=acc-within,transfer_minus_target_refit=acc-refit,unaligned_accuracy=old,aligned_minus_unaligned=acc-old)
                for k,value in expected.items():np.testing.assert_allclose(row[k],value,atol=1e-13,rtol=1e-12)
                rows.append(dict(**identity,lag=c['lags'][j],**expected))
    assert directions=={(s,ci,d) for s in range(55142,55145) for ci in [701,702] for d in ['renormalized_to_fixed_original','fixed_original_to_renormalized']}
    metrics=list(expected);f=pd.DataFrame(rows)
    blocks=f[f.lag.isin(c['primary_lags'])].groupby(['direction','seed','circuit_seed'])[metrics].mean().groupby(['direction','seed']).mean()
    saved=pd.read_csv(root/'seed-blocks.csv').set_index(['direction','seed'])
    np.testing.assert_allclose(blocks.sort_index(),saved[metrics].sort_index(),atol=1e-13,rtol=1e-12)
    summary=read(root/'summary.json');gates={}
    for direction,b in saved.reset_index().groupby('direction'):
        access=bool(b.frequency_excess.mean()>=.05 and b.null_excess.mean()>=.05 and ((b.frequency_excess>=.05)&(b.null_excess>=.05)).all() and b.r2_vs_frequency.mean()>0)
        retention=bool(b.transfer_minus_target_refit.mean()>=-.05 and (b.transfer_minus_target_refit>=-.05).all())
        improvement=bool(b.aligned_minus_unaligned.mean()>=.05 and (b.aligned_minus_unaligned>0).all())
        gates[direction]=dict(alignment_improvement=improvement,past_access=access,retention_tolerance=retention,portable_decoding=access and retention)
        for metric in metrics:
            x=b[metric].to_numpy();s=summary['cells'][direction][metric];rng=np.random.default_rng(c['bootstrap_seed'])
            draws=np.mean(x[rng.integers(0,len(x),(10000,len(x)))],axis=1)
            for k,val in [('mean',x.mean()),('median',np.median(x)),('variance',x.var(ddof=1)),('bootstrap95',np.quantile(draws,[.025,.975]))]:
                np.testing.assert_allclose(s[k],val,atol=1e-13,rtol=1e-12)
            if metric.startswith('transfer_minus_') or metric=='aligned_minus_unaligned':
                np.testing.assert_allclose(s['paired_dz'],x.mean()/x.std(ddof=1),atol=1e-12,rtol=1e-12)
    assert gates==summary['gates']
    assert summary['confirmed_improvement']==[d for d,g in gates.items() if g['alignment_improvement']]
    assert summary['confirmed_portability']==[d for d,g in gates.items() if g['portable_decoding']]
    for p,h in {**read(root/'prior-results-sha256.json'),**manifest['context']['source_sha256']}.items():assert core.sha256(p)==h,p
    return dict(condition_graphs_and_source_refits=condition_count,directions=len(directions),transfer_lag_rows=len(rows),
        independent_streams_labels_scores_pairing_bootstrap_gates_verified=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);result=verify(args.source)
    core.write_json(args.out/'checks.json',dict(verification=result,verifier_sha256=core.sha256(__file__),environment=core.environment()));print(result)
