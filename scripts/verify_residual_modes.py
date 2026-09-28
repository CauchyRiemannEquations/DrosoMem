"""Independent projector-based saved-artifact verification."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from alphabet_memory import read,check
from flying.training import whole_brain_memory as core


@threadpool_limits.wrap(limits=1)
def verify(root):
    manifest=check(root);c=manifest['config'];rows=[];counts={}
    for path in sorted(root.glob('*/*/*/manifest.json')):
        meta=check(path.parent)
        for p,h in meta['source_hashes'].items():
            check(Path(p));assert core.sha256(Path(p)/'manifest.json')==h
        with np.load(path.parent/'checkpoint.npz') as a, np.load(Path(meta['moment_path'])/'checkpoint.npz') as m, np.load(Path(meta['source_path'])/'checkpoint.npz') as source:
            v=a['basis']; d=a['residual']; w=source['weights'];z=(source['xtrain']-source['mean'])/source['scale']
            np.testing.assert_allclose(v.T@v,np.eye(48),atol=1e-12)
            np.testing.assert_allclose(v.T@(z.T@z)@v,np.diag(a['singular']**2),atol=1e-8,rtol=1e-8)
            delta=(m['xtest']-m['target_mean'])/m['target_scale']-(source['xtest']-source['mean'])/source['scale']
            np.testing.assert_array_equal(d,delta)
            projected=d@v; rotated=v.T@w
            np.testing.assert_array_equal(projected,a['projected_residual'])
            np.testing.assert_array_equal(rotated,a['projected_head'])
            low_d=d@v[:,24:]@v[:,24:].T;high_d=d-low_d
            lows=(low_d@w).reshape(-1,11,4);highs=(high_d@w).reshape(-1,11,4);total=(d@w).reshape(-1,11,4)
            np.testing.assert_allclose(total,m['scores']-source['scores'],atol=1e-9,rtol=1e-8)
            for k,expected in [('low',lows),('high',highs),('total',total)]:
                np.testing.assert_allclose(a[k],expected,atol=1e-9,rtol=1e-8)
            state=np.mean(np.square(projected),axis=0)
            np.testing.assert_array_equal(state,a['state_energy'])
            diagonal=np.mean(projected**2,axis=0)[:,None]*np.sum(rotated.reshape(48,11,4)**2,axis=2)
            signed=np.einsum('nk,klo,nlo->lk',projected,rotated.reshape(48,11,4),total)/len(d)
            np.testing.assert_allclose(a['diagonal_energy'],diagonal.T,atol=1e-9,rtol=1e-8)
            np.testing.assert_allclose(a['signed_attribution'],signed,atol=1e-9,rtol=1e-8)
            saved=read(path.parent/'metrics.json')
            for j,row in enumerate(saved):
                te=float(np.mean(np.sum(total[:,j]**2,axis=1)))
                le=float(np.mean(np.sum(lows[:,j]**2,axis=1)))
                he=float(np.mean(np.sum(highs[:,j]**2,axis=1)))
                cross=float(2*np.mean(np.sum(lows[:,j]*highs[:,j],axis=1)))
                state_share=float(np.sum(low_d**2)/np.sum(d**2))
                signed_share=float(np.mean(np.sum(lows[:,j]*total[:,j],axis=1))/te)
                expected=dict(total_score_energy=te,low_score_energy=le,high_score_energy=he,cross_energy=cross,
                    low_state_share=state_share,low_signed_score_share=signed_share,signed_minus_state=signed_share-state_share,
                    diagonal_score_energy=float(diagonal[:,j].sum()),low_gain=le/state[24:].sum(),high_gain=he/state[:24].sum(),
                    low_train_variance_share=np.sum(a['singular'][24:]**2)/np.sum(a['singular']**2),
                    boundary_gap=(a['singular'][23]-a['singular'][24])/a['singular'][0])
                y=m['ytest'][:,j]; truth=np.eye(4)[y];score=m['scores'][:,j]
                expected.update(test_accuracy=float(np.mean(score.argmax(axis=1)==y)),
                    r2_vs_frequency=float(1-np.sum((score-truth)**2)/np.sum((truth-source['bias'].reshape(11,4)[j])**2)))
                for k,value in expected.items():np.testing.assert_allclose(row[k],value,atol=1e-8,rtol=1e-8)
                assert row['valid']==(te>1e-24 and state.sum()>1e-24 and expected['boundary_gap']>1e-8)
                rows.append(dict(**meta['identity'],**row))
        co=meta['identity']['cohort'];counts[co]=counts.get(co,0)+1
    assert counts==dict(confirmation=12,main=20,smoke=2)
    f=pd.DataFrame(rows);metrics=list(expected)
    blocks=f[f.lag.isin(c['primary_lags'])].groupby(['cohort','direction','seed','circuit_seed'])[metrics].mean().groupby(['cohort','direction','seed']).mean()
    stored=pd.read_csv(root/'seed-blocks.csv').set_index(['cohort','direction','seed'])
    np.testing.assert_allclose(blocks[metrics].sort_index(),stored[metrics].sort_index(),atol=1e-12)
    summary=read(root/'summary.json');gates={}
    for (co,direction),b in stored.reset_index().query("cohort != 'smoke'").groupby(['cohort','direction']):
        key=f'{co}/{direction}'
        gates[key]=bool(b.valid.all() and b.low_signed_score_share.mean()>=.75 and b.low_state_share.mean()<=.5 and (b.signed_minus_state>0).sum()>=(4 if co=='main' else 3))
        for metric in metrics:
            values=b[metric].to_numpy();stats=summary['cells'][key][metric]
            rng=np.random.default_rng(c['bootstrap_seed'])
            bs=np.mean(values[rng.integers(0,len(values),(10000,len(values)))],axis=1)
            for name,value in [('mean',values.mean()),('median',np.median(values)),('variance',values.var(ddof=1)),('bootstrap95',np.quantile(bs,[.025,.975]))]:
                np.testing.assert_allclose(stats[name],value,atol=1e-10,rtol=1e-10)
    assert gates==summary['gates']
    for p,h in {**read(root/'prior-results-sha256.json'),**manifest['context']['source_sha256']}.items():assert core.sha256(p)==h,p
    return dict(directions=counts,lag_rows=len(rows),projector_algebra_metrics_bootstrap_gates_verified=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);result=verify(args.source)
    core.write_json(args.out/'checks.json',dict(verification=result,verifier_sha256=core.sha256(__file__),environment=core.environment()))
    print(result)
