"""Independent structural invariants, state trajectories and inference audit."""
import argparse,hashlib,subprocess
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits
from flying.training import whole_brain_memory as core
from alphabet_memory import read,check,symbol_bank
from frozen_state_probe import Budget,verify_fold
from normalization_transfer import verify_transfer
from verify_neuron_panel import trajectory,audit_metrics,audit_stats
from structural_controls import generate


def properties(a,b,roles,family):
    x=a.toarray();y=b.toarray();n=len(x)
    assert x.shape==y.shape and np.count_nonzero(x)==np.count_nonzero(y) and not np.diag(y).any()
    assert Counter(x[x!=0])==Counter(y[y!=0])
    if family!='random':
        for i in range(n):
            assert Counter(x[i,x[i]!=0])==Counter(y[i,y[i]!=0])
            for sign in [-1,1]:
                assert np.count_nonzero(np.sign(x[:,i])==sign)==np.count_nonzero(np.sign(y[:,i])==sign)
                assert np.count_nonzero(np.sign(x[i])==sign)==np.count_nonzero(np.sign(y[i])==sign)
    if family in ['intact','weight']:np.testing.assert_array_equal(np.sign(x),np.sign(y))
    if family=='role':
        for i in range(n):
            for role in set(roles):
                for sign in [-1,1]:
                    assert np.count_nonzero(np.sign(x[i,roles==role])==sign)==np.count_nonzero(np.sign(y[i,roles==role])==sign)
                    assert np.count_nonzero(np.sign(x[roles==role,i])==sign)==np.count_nonzero(np.sign(y[roles==role,i])==sign)
    return dict(nodes=n,edges=int(np.count_nonzero(x)),common_edges=int(np.count_nonzero((x!=0)&(y!=0))),edge_overlap=float(np.count_nonzero((x!=0)&(y!=0))/np.count_nonzero(x)),
        changed_weight_positions=int(np.count_nonzero(x!=y)),incoming_l1_changed_nodes=int(np.count_nonzero(abs(x).sum(1)!=abs(y).sum(1))),
        outgoing_l1_changed_nodes=int(np.count_nonzero(abs(x).sum(0)!=abs(y).sum(0))),in_degree_changed_nodes=int(np.count_nonzero((x!=0).sum(1)!=(y!=0).sum(1))),out_degree_changed_nodes=int(np.count_nonzero((x!=0).sum(0)!=(y!=0).sum(0))))


@threadpool_limits.wrap(limits=1)
def verify(root,out):
    m=check(root);c=m['config'];out.mkdir(parents=True,exist_ok=False);budget=Budget(c)
    try:
        graphs={}
        for ci in c['circuit_seeds']:
            directory=Path(f'data/flywire_783_mb_left_kc512_s{ci}');raw,ids,_=core.load_connectome(directory);raw.sort_indices();roles,_=core.load_roles(directory,ids)
            graphs[ci]=(raw,ids,np.asarray(roles))
        count=fcount=0;rows=[]
        for path in sorted(root.glob('*/*/manifest.json')):
            dest=path.parent;cm=check(dest);i=cm['identity'];cc=cm['config'];original,ids,roles=graphs[i['circuit_seed']]
            family='intact' if i['arm']=='intact' else i['arm'][:-1];seed=0 if family=='intact' else i['graph_seeds'][family][int(i['arm'][-1])]
            raw=sparse.load_npz(dest/'raw.npz');g=read(dest/'graph.json')
            reference,log=generate(original,roles,family,seed,c['swaps_per_edge'])
            assert log==g['generation'] and core.weight_hash(reference)==core.weight_hash(raw)==g['raw_sha256']
            assert core.weight_hash(original)==g['original_raw_sha256']
            for key,value in properties(original,raw,roles,family).items():assert g[key]==value,(key,g[key],value)
            dense=raw.toarray();strength=abs(dense).sum(1);factor=np.divide(c['gain'],strength,out=np.zeros_like(strength),where=strength>0)
            weights=sparse.load_npz(dest/'weights.npz');np.testing.assert_array_equal(weights.toarray(),factor[:,None]*dense)
            assert core.weight_hash(weights)==g['weight_sha256']
            if family!='random':np.testing.assert_array_equal(strength,np.asarray(abs(original).sum(1)).ravel())
            with np.load(dest/'checkpoint.npz') as z:a=dict(z)
            bank=symbol_bank(roles,i['seed'],c['input_fraction'],c['input_amplitude'])[:4];obs=np.flatnonzero(roles=='MBON')
            for key in ['input_patterns','intended_input_patterns']:np.testing.assert_array_equal(bank,a[key])
            np.testing.assert_array_equal(obs,a['observed_indices']);assert g['observed_root_ids']==[ids[j] for j in obs]
            assert g['input_root_ids']==[[ids[j] for j in np.flatnonzero(p)] for p in bank]
            assert g['input_mapping_sha256']==core.fingerprint(dict(ids=ids,patterns=bank.tolist()))
            w=cc['warmup']
            for split in ['train','test']:
                symbols=np.random.default_rng(i[split+'_seed']).integers(0,4,w+cc[split+'_samples']).astype(np.uint8)
                np.testing.assert_array_equal(symbols,a[split+'_symbols'])
                np.testing.assert_array_equal(np.column_stack([symbols[np.arange(w,len(symbols))-lag] for lag in c['lags']]),a['y'+split])
                features,active,norm=trajectory(weights,bank,roles,symbols,c['leak'])
                np.testing.assert_array_equal(features,a[split+'_features']);np.testing.assert_array_equal(features[w:],a['x'+split])
                np.testing.assert_array_equal(active,a[split+'_active_counts']);np.testing.assert_array_equal(norm,a[split+'_full_norm'])
            verify_fold(a);metrics=read(dest/'metrics.json');audit_metrics(a,metrics)
            ident={k:i[k] for k in ['cohort','seed','circuit_seed','arm']};rows.extend(dict(**ident,mode='refit',**r) for r in metrics)
            if cm['source_path']:
                sp=Path(cm['source_path']);check(sp);assert core.sha256(sp/'manifest.json')==cm['source_manifest_sha256']
                with np.load(sp/'checkpoint.npz') as z:source=dict(z)
                for k in ['train_symbols','test_symbols','input_patterns','observed_indices','ytrain','ytest']:np.testing.assert_array_equal(source[k],a[k])
                with np.load(dest/'frozen.npz') as z:frozen=dict(z)
                verify_transfer(source,a,frozen);fm=read(dest/'frozen-metrics.json');audit_metrics(frozen,fm,True,a,source)
                rows.extend(dict(**ident,mode='frozen',**r) for r in fm);fcount+=1
            count+=1;budget.check()
            if count%13==0:print(f'Independently verified {count}/169 cases',flush=True)
        f=pd.DataFrame(rows);saved=pd.read_csv(root/'raw-lag-table.csv');keys=['cohort','seed','circuit_seed','arm','mode','lag']
        pd.testing.assert_frame_equal(saved.sort_values(keys).reset_index(drop=True),f[saved.columns].sort_values(keys).reset_index(drop=True),check_dtype=False,atol=1e-12,rtol=1e-12)
        summary=read(root/'summary.json');past=f[f.lag.isin(c['primary_lags'])];confirmed=[];pairs=[]
        for mode in ['refit','frozen']:
            for family in c['families']:
                both=[]
                for cohort in ['discovery','confirmation']:
                    q=past[past.cohort==cohort];base=q[(q.arm=='intact')&(q['mode']=='refit')].groupby('seed')[['test_accuracy','frequency_excess','null_excess','r2_vs_frequency']].mean()
                    control=q[(q['mode']==mode)&q.arm.isin([family+str(j) for j in range(3)])].groupby('seed').test_accuracy.mean()
                    delta=base.test_accuracy-control;key=f'{cohort}/{mode}/{family}'
                    for name,x in [('intact',base.test_accuracy),('control',control),('difference',delta)]:audit_stats(x,summary['statistics'][key][name],c)
                    access=bool((base.frequency_excess>=c['access_margin']).all() and (base.null_excess>=c['access_margin']).all() and base.r2_vs_frequency.mean()>0)
                    gate=bool(access and delta.mean()>=c['material_effect'] and (delta>0).all())
                    assert gate==summary['gates'][key] and access==summary['statistics'][key]['intact_access'];both.append(gate)
                    pairs.extend(dict(cohort=cohort,mode=mode,family=family,seed=int(seed),intact=float(base.test_accuracy[seed]),control=float(control[seed]),difference=float(delta[seed])) for seed in delta.index)
                if all(both):confirmed.append(f'{mode}/{family}')
        paircsv=pd.read_csv(root/'paired-differences.csv');keys=['cohort','mode','family','seed']
        pd.testing.assert_frame_equal(paircsv.sort_values(keys).reset_index(drop=True),pd.DataFrame(pairs)[paircsv.columns].sort_values(keys).reset_index(drop=True),check_dtype=False,atol=1e-12,rtol=1e-12)
        assert confirmed==summary['confirmed'] and summary['primary_role_refit']==('refit/role' in confirmed) and not summary['cohorts_pooled']
        for p,h in read(root/'prior-results-sha256.json').items():assert core.sha256(p)==h,p
        for p,h in m['context']['source_sha256'].items():assert hashlib.sha256(subprocess.check_output(['git','show',m['context']['git_commit']+':'+p])).hexdigest()==h,p
        assert(count,fcount,len(rows))==(169,156,3575);budget.check();usage=budget.close()
        core.write_json(out/'checks.json',dict(cases=count,frozen=fcount,independent_trajectories=2*count,metric_rows=len(rows),all_checks_pass=True,source_git_bytes_verified=True,
            graph_sampler='deterministic replay; independently checked invariants, not independent sampler',result_manifest_sha256=core.sha256(root/'manifest.json'),prior_files_unchanged=len(read(root/'prior-results-sha256.json')),budget=usage,verifier_sha256=core.sha256(__file__)))
        print(dict(cases=count,frozen=fcount,metric_rows=len(rows),confirmed=confirmed),flush=True)
    except Exception as exc:
        core.write_json(out/'failure.json',dict(error=repr(exc)));raise
    finally:budget.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();verify(args.root,args.out)
