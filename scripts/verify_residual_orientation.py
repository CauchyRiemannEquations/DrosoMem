"""Independent covariance-trace and dense orthogonal control verification."""
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
        for p,h in meta['source_hashes'].items():check(Path(p));assert core.sha256(Path(p)/'manifest.json')==h
        with np.load(path.parent/'checkpoint.npz') as a,np.load(Path(meta['parent_path'])/'checkpoint.npz') as parent:
            for key in parent.files:np.testing.assert_array_equal(a[key],parent[key])
            q=a['projected_residual'];w=a['projected_head'];second=q.T@q/len(q)
            ref=[];draws=[]
            for j in range(11):
                h=w[:,4*j:4*j+4]
                ref.append(sum(np.trace(second[lo:hi,lo:hi])*np.trace(h[lo:hi]@h[lo:hi].T)/24 for lo,hi in [(0,24),(24,48)]))
            np.testing.assert_allclose(ref,a['reference_energy'],atol=1e-10,rtol=1e-10)
            for i,seed in enumerate(c['control_seeds']):
                rng=np.random.default_rng(seed);perm=np.r_[rng.permutation(24),24+rng.permutation(24)];sign=rng.choice([-1.,1.],48)
                np.testing.assert_array_equal(a['permutations'][i],perm);np.testing.assert_array_equal(a['signs'][i],sign)
                transform=np.eye(48)[:,perm]*sign;q2=q@transform
                for lo,hi in [(0,24),(24,48)]:
                    np.testing.assert_allclose(np.sum(q2[:,lo:hi]**2,axis=1),np.sum(q[:,lo:hi]**2,axis=1),atol=1e-11,rtol=1e-11)
                score=(q2@w).reshape(-1,11,4);draws.append(np.mean(np.sum(score**2,axis=2),axis=0))
            draws=np.array(draws);np.testing.assert_allclose(a['draw_energy'],draws,atol=1e-10,rtol=1e-10)
            actual=np.mean(np.sum(a['total']**2,axis=2),axis=0)
            saved=read(path.parent/'metrics.json')
            for j,row in enumerate(saved):
                values=dict(actual_energy=actual[j],reference_energy=ref[j],actual_over_reference=actual[j]/ref[j],
                    log2_actual_over_reference=np.log2(actual[j]/ref[j]),draw_mean_energy=draws[:,j].mean(),draw_min_energy=draws[:,j].min(),draw_max_energy=draws[:,j].max())
                for key,val in values.items():np.testing.assert_allclose(row[key],val,atol=1e-10,rtol=1e-10)
                assert row['valid']==(actual[j]>1e-24 and ref[j]>1e-24)
                rows.append(dict(**meta['identity'],**row))
        co=meta['identity']['cohort'];counts[co]=counts.get(co,0)+1
    assert counts==dict(confirmation=12,main=20,smoke=2)
    f=pd.DataFrame(rows);metrics=list(values)+['test_accuracy','r2_vs_frequency']
    blocks=f[f.lag.isin(c['primary_lags'])].groupby(['cohort','direction','seed','circuit_seed'])[metrics].mean().groupby(['cohort','direction','seed']).mean()
    stored=pd.read_csv(root/'seed-blocks.csv').set_index(['cohort','direction','seed'])
    np.testing.assert_allclose(blocks[metrics].sort_index(),stored[metrics].sort_index(),atol=1e-12)
    summary=read(root/'summary.json');gates={}
    for (co,d),b in stored.reset_index().query("cohort != 'smoke'").groupby(['cohort','direction']):
        key=f'{co}/{d}';gates[key]=bool(b.valid.all() and b.log2_actual_over_reference.mean()>=1 and (b.log2_actual_over_reference>0).sum()>=(4 if co=='main' else 3))
        for metric in metrics:
            x=b[metric].to_numpy();stats=summary['cells'][key][metric];rng=np.random.default_rng(c['bootstrap_seed'])
            means=np.mean(x[rng.integers(0,len(x),(10000,len(x)))],axis=1)
            for name,value in [('mean',x.mean()),('median',np.median(x)),('variance',x.var(ddof=1)),('bootstrap95',np.quantile(means,[.025,.975]))]:
                np.testing.assert_allclose(stats[name],value,atol=1e-10,rtol=1e-10)
    assert gates==summary['gates']
    for p,h in {**read(root/'prior-results-sha256.json'),**manifest['context']['source_sha256']}.items():assert core.sha256(p)==h,p
    return dict(directions=counts,lag_rows=len(rows),control_energy_rows=len(rows)*16,covariance_trace_controls_metrics_bootstrap_gates_verified=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);result=verify(args.source)
    core.write_json(args.out/'checks.json',dict(verification=result,verifier_sha256=core.sha256(__file__),environment=core.environment()))
    print(result)
