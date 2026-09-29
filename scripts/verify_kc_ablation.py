"""Independent saved-mask, graph, stream, ridge, metric and statistic audit."""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits
from alphabet_memory import read, check, symbol_bank
from flying.training import whole_brain_memory as core
from frozen_state_probe import verify_fold, measures


@threadpool_limits.wrap(limits=1)
def verify(root, out):
    top = check(root); c = top['config']; rows=[]; counts={}
    for cohort in ['smoke','main']:
        cc = dict(c)
        if cohort == 'smoke': cc.update(c['smoke'])
        seen=set()
        for path in sorted((root/cohort).glob('*/manifest.json')):
            m=check(path.parent); b=m['identity']; arm=b['arm']; seen.add((b['seed'],b['circuit_seed'],arm))
            assert m['config']==cc
            data=Path(f'data/flywire_783_mb_left_kc512_s{b["circuit_seed"]}')
            raw,ids,_=core.load_connectome(data); roles,_=core.load_roles(data,ids)
            annotations=pd.read_csv(data/'annotations.csv',dtype=str).set_index('root_id').loc[list(map(str,ids))]
            gamma=np.array([r=='KC' and str(t).startswith('KCg') for r,t in zip(roles,annotations.cell_type)])
            bank=symbol_bank(roles,b['seed'],.1,.5)[:4]
            degree_in=np.asarray((raw!=0).sum(axis=1)).ravel(); degree_out=np.asarray((raw!=0).sum(axis=0)).ravel()
            exposure=np.sum((bank.T!=0)*(2**np.arange(4)),axis=1)
            keys=list(zip(degree_in.tolist(),degree_out.tolist(),exposure.tolist()))
            target=Counter(keys[i] for i in np.flatnonzero(gamma))
            dead=gamma.copy() if arm=='gamma' else np.zeros(len(ids),bool)
            if arm.startswith('matched'):
                rng=np.random.default_rng(b['match_seeds'][int(arm[-1])])
                for key in sorted(target):
                    pool=[i for i,k in enumerate(keys) if roles[i]=='KC' and k==key]
                    dead[rng.choice(pool,target[key],replace=False)]=True
            if arm!='intact': assert Counter(keys[i] for i in np.flatnonzero(dead))==target
            denominator=np.asarray(abs(raw).sum(axis=1)).ravel()
            factors=np.divide(.9,denominator,out=np.zeros(len(ids)),where=denominator>0)
            expected=raw.toarray()*factors[:,None];expected[dead,:]=0;expected[:,dead]=0
            weights=sparse.load_npz(path.parent/'weights.npz');np.testing.assert_array_equal(weights.toarray(),expected)
            g=read(path.parent/'graph.json')
            assert core.weight_hash(raw)==g['raw_sha256']==core.weight_hash(sparse.load_npz(path.parent/'raw-graph.npz'))
            assert core.weight_hash(weights)==g['weight_sha256']
            assert g['removed_root_ids']==[ids[i] for i in np.flatnonzero(dead)]
            assert g['removed_count']==int(dead.sum()) and g['overlap_with_gamma']==int((dead&gamma).sum())
            assert g['removed_input_counts']==np.count_nonzero(bank[:,dead],axis=1).tolist()
            raw_dense=raw.toarray();raw_dense[dead,:]=0;raw_dense[:,dead]=0
            assert g['remaining_edges']==np.count_nonzero(raw_dense) and g['removed_edges']==raw.nnz-np.count_nonzero(raw_dense)
            assert g['removed_absolute_strength']==float(abs(raw).sum()-abs(raw_dense).sum())
            with np.load(path.parent/'checkpoint.npz',allow_pickle=False) as saved:
                a=dict(saved);np.testing.assert_array_equal(a['dead_mask'],dead);np.testing.assert_array_equal(a['gamma_mask'],gamma)
                np.testing.assert_array_equal(a['matching_strata'],np.asarray(keys))
                np.testing.assert_array_equal(a['intended_input_patterns'],bank);bank[:,dead]=0
                np.testing.assert_array_equal(a['input_patterns'],bank)
                obs=np.flatnonzero(np.array(roles)=='MBON');np.testing.assert_array_equal(a['observed_indices'],obs)
                assert len(obs)==48 and not dead[obs].any()
                assert g['observed_root_ids']==[ids[i] for i in obs]
                for split in ['train','test']:
                    n=cc[split+'_samples'];symbols=np.random.default_rng(b[split+'_seed']).integers(0,4,100+n).astype(np.uint8)
                    np.testing.assert_array_equal(a[split+'_symbols'],symbols)
                    y=np.column_stack([symbols[100-lag:100-lag+n] for lag in c['lags']])
                    np.testing.assert_array_equal(a['y'+split],y);np.testing.assert_array_equal(a['x'+split],a[split+'_features'][100:])
                verify_fold(a,1.)
                measured=measures([a],c['lags']);savedmetrics=read(path.parent/'metrics.json')
                for j,row in enumerate(measured):
                    row['training_mse']=float(np.mean((a['train_scores'][:,j]-np.eye(4)[a['ytrain'][:,j]])**2))
                assert measured==savedmetrics
                if cohort=='main':rows.extend(dict(**b,**row) for row in measured)
        assert seen=={(b['seed'],ci,arm) for b in cc['blocks'] for ci in cc['circuit_seeds'] for arm in cc['conditions']}
        counts[cohort]=len(seen)
    f=pd.DataFrame(rows);metrics=['test_accuracy','frequency_excess','null_excess','r2_vs_frequency']
    cases=f[f.lag.isin(c['primary_lags'])].groupby(['seed','circuit_seed','arm'])[metrics].mean()
    cells=[]
    for seed in [b['seed'] for b in c['blocks']]:
        for arm in ['intact','gamma','matched_mean']:
            arms=['matched0','matched1','matched2'] if arm=='matched_mean' else [arm]
            values=np.mean([cases.loc[(seed,ci,a)].values for ci in c['circuit_seeds'] for a in arms],axis=0)
            cells.append(dict(seed=seed,arm=arm,**dict(zip(metrics,values))))
    blocks=pd.DataFrame(cells).set_index(['seed','arm']).sort_index()
    saved=pd.read_csv(root/'main/seed-blocks.csv').set_index(['seed','arm']).sort_index()
    np.testing.assert_allclose(blocks,saved[metrics],atol=1e-13,rtol=0)
    t=blocks.test_accuracy.unstack();delta=dict(extra_impairment=t.matched_mean-t.gamma,intact_minus_gamma=t.intact-t.gamma,intact_minus_matched=t.intact-t.matched_mean)
    summary=read(root/'main/summary.json')
    for key,x in delta.items():
        q=summary['differences'][key];x=np.asarray(x);rng=np.random.default_rng(c['bootstrap_seed'])
        means=x[rng.integers(0,3,size=(10000,3))].mean(axis=1)
        expected=[x.mean(),np.median(x),x.var(ddof=1),*np.quantile(means,[.025,.975])]
        np.testing.assert_allclose([q['mean'],q['median'],q['variance'],*q['bootstrap95']],expected,atol=1e-13,rtol=0)
        np.testing.assert_allclose(q['paired_dz'],x.mean()/x.std(ddof=1),atol=1e-12,rtol=0)
    intact=blocks.xs('intact',level='arm');access=bool((intact.frequency_excess>=.05).all() and (intact.null_excess>=.05).all() and intact.r2_vs_frequency.mean()>0)
    effect=delta['extra_impairment'];assert summary['gates']==dict(intact_past_access=access,gamma_specific_impairment=bool(access and effect.mean()>=.05 and (effect>0).all()))
    prior=read(root/'prior-results-sha256.json')
    for path,h in prior.items():assert core.sha256(path)==h,path
    out.mkdir(parents=True,exist_ok=False)
    core.write_json(out/'checks.json',dict(verified=True,conditions=counts,main_lag_rows=len(rows),
        independently_regenerated_masks=True,independent_dense_weight_check=True,independent_lstsq=True,
        independent_pair_statistics=True,prior_results_unchanged=len(prior),source_manifest_sha256=core.sha256(root/'manifest.json')))
    print(read(out/'checks.json'),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();verify(args.root,args.out)
