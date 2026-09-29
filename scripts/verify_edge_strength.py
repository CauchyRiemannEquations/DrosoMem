"""Independent effective-weight-stratum, trajectory and statistics audit."""
import argparse
from pathlib import Path
from collections import Counter
import hashlib,subprocess
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import shortest_path
from threadpoolctl import threadpool_limits
from flying.training import whole_brain_memory as core
from alphabet_memory import read,check,symbol_bank
from frozen_state_probe import verify_fold,Budget
from normalization_transfer import verify_transfer
from verify_neuron_panel import trajectory,audit_metrics,audit_stats


def independent_mask(raw,ids,roles,score,identity,c):
    co=raw.tocoo();n=raw.nnz;arm=identity['arm'];mask=np.zeros(n,bool)
    if arm=='intact':return mask
    dan=(roles[co.row]=='DAN')|(roles[co.col]=='DAN')
    if arm=='DAN':return dan
    assert arm in ['dan_control0','dan_control1','dan_control2']
    incoming=np.asarray(abs(raw).sum(1)).ravel()
    normalized=np.abs(co.data)*.9/incoming[co.row]
    keys=[(int(np.sign(value)),sum(abs(value)>=cut for cut in [10,20,50,100]),int(np.floor(normalized[e]*1000+1e-9))) for e,value in enumerate(co.data)]
    targets=Counter(keys[i] for i in np.flatnonzero(dan));rng=np.random.default_rng(identity['edge_seeds']['dan_control'][int(arm[-1])])
    for key in sorted(targets):
        pool=np.array([i for i,k in enumerate(keys) if k==key]);mask[rng.choice(pool,targets[key],replace=False)]=True
    assert Counter(keys[i] for i in np.where(mask)[0])==targets
    assert abs(normalized[mask].sum()-normalized[dan].sum())<dan.sum()*.001+1e-9
    return mask


def reference_family(q,family):
    if family in ['within','between']:return q[q.arm.isin([f'{family}{j}' for j in range(3)])]
    return q[q.arm==family]


@threadpool_limits.wrap(limits=1)
def verify(root,out,analysis=None):
    m=check(root);c=m['config'];out.mkdir(parents=True,exist_ok=False);budget=Budget(c)
    try:
        scores={};graphs={};ranking={}
        for ci in c['circuit_seeds']:
            raw=sparse.load_npz(root/'graphs'/str(ci)/'raw.npz');meta=read(root/'graphs'/str(ci)/'ids.json')
            directory=Path(f'data/flywire_783_mb_left_kc512_s{ci}');original,ids,_=core.load_connectome(directory);roles,_=core.load_roles(directory,ids);roles=np.asarray(roles)
            assert core.weight_hash(raw)==core.weight_hash(original)==meta['raw_sha256'];assert ids==meta['ids'];assert roles.tolist()==meta['roles']
            scores[ci]=np.zeros(raw.nnz)
            ranking[str(ci)]=dict(edges=raw.nnz,normalized_bin_width=.001)
            graphs[ci]=(raw,ids,roles);budget.check()
        rows=[];count=0;fcount=0;metriccount=0
        for path in sorted(root.glob('*/*/manifest.json')):
            dest=path.parent;cm=check(dest);i=cm['identity'];cc=cm['config'];raw,ids,roles=graphs[i['circuit_seed']];co=raw.tocoo()
            with np.load(dest/'checkpoint.npz') as z:a=dict(z)
            g=read(dest/'graph.json');mask=independent_mask(raw,ids,roles,scores[i['circuit_seed']],i,c)
            np.testing.assert_array_equal(mask,a['edge_mask']);assert g['removed_count']==int(mask.sum())
            assert g['removed_edge_ids']==[[ids[co.col[e]],ids[co.row[e]]] for e in np.flatnonzero(mask)]
            bank=symbol_bank(roles,i['seed'])[:4];np.testing.assert_array_equal(bank,a['input_patterns']);np.testing.assert_array_equal(bank,a['intended_input_patterns'])
            np.testing.assert_array_equal(np.where(roles=='MBON')[0],a['observed_indices'])
            dense=raw.toarray();strength=abs(dense).sum(1);factor=np.divide(.9,strength,out=np.zeros_like(strength),where=strength>0)
            normalized=factor[:,None]*dense;expected=normalized.copy();expected[co.row[mask],co.col[mask]]=0
            weights=sparse.load_npz(dest/'weights.npz');np.testing.assert_array_equal(weights.toarray(),expected)
            assert core.weight_hash(weights)==g['weight_sha256'] and weights.nnz==raw.nnz-mask.sum()
            np.testing.assert_allclose(g['removed_raw_l1'],abs(co.data[mask]).sum())
            np.testing.assert_allclose(g['removed_normalized_l1'],abs(normalized[co.row[mask],co.col[mask]]).sum())
            if i['arm'].startswith('dan_control'):assert abs(g['normalized_l1_difference'])<g['absolute_l1_matching_bound']
            dan=(roles[co.row]=='DAN')|(roles[co.col]=='DAN');within=roles[co.row]==roles[co.col]
            assert g['DAN_overlap']==int((mask&dan).sum()) and g['removed_within_role']==int((mask&within).sum())
            if i['arm'].startswith('dan_control'):
                keys=[(int(np.sign(x)),sum(abs(x)>=b for b in [10,20,50,100])) for x in co.data]
                assert Counter(keys[e] for e in np.where(mask)[0])==Counter(keys[e] for e in np.where(dan)[0])
            w=cc['warmup']
            for split in ['train','test']:
                symbols=np.random.default_rng(i[split+'_seed']).integers(0,4,w+cc[split+'_samples']).astype(np.uint8)
                np.testing.assert_array_equal(symbols,a[split+'_symbols'])
                truth=np.column_stack([symbols[np.arange(w,len(symbols))-lag] for lag in c['lags']]);np.testing.assert_array_equal(truth,a['y'+split])
                features,active,norm=trajectory(weights,bank,roles,symbols,.6)
                np.testing.assert_array_equal(features,a[split+'_features']);np.testing.assert_array_equal(features[w:],a['x'+split])
                np.testing.assert_array_equal(active,a[split+'_active_counts']);np.testing.assert_array_equal(norm,a[split+'_full_norm'])
            verify_fold(a);metrics=read(dest/'metrics.json');audit_metrics(a,metrics)
            ident={k:i[k] for k in ['cohort','seed','circuit_seed','arm']};rows.extend(dict(**ident,mode='refit',**r) for r in metrics);metriccount+=len(metrics)
            if cm['source_path']:
                sourcepath=Path(cm['source_path']);check(sourcepath);assert core.sha256(sourcepath/'manifest.json')==cm['source_manifest_sha256']
                with np.load(sourcepath/'checkpoint.npz') as z:source=dict(z)
                for k in ['train_symbols','test_symbols','input_patterns','observed_indices','ytrain','ytest']:np.testing.assert_array_equal(source[k],a[k])
                with np.load(dest/'frozen.npz') as z:frozen=dict(z)
                verify_transfer(source,a,frozen);fm=read(dest/'frozen-metrics.json');audit_metrics(frozen,fm,True,a,source)
                rows.extend(dict(**ident,mode='frozen',**r) for r in fm);fcount+=1;metriccount+=len(fm)
            count+=1;budget.check()
            if count%25==0:print(f'Independently audited {count} cases',flush=True)
        f=pd.DataFrame(rows);saved=pd.read_csv(root/'raw-lag-table.csv');keys=['cohort','seed','circuit_seed','arm','mode','lag']
        pd.testing.assert_frame_equal(saved.sort_values(keys).reset_index(drop=True),f[saved.columns].sort_values(keys).reset_index(drop=True),check_dtype=False,atol=1e-12,rtol=1e-12)
        analysis_root=analysis or root
        if analysis is not None:
            am=check(analysis);assert am['source_manifest_sha256']==core.sha256(root/'manifest.json')
        past=f[f.lag.isin(c['primary_lags'])];summary=read(analysis_root/'summary.json');confirmed=[]
        for mode in ['refit','frozen']:
            for family in c['families']:
                both=[]
                for cohort in ['discovery','confirmation']:
                    q=past[past.cohort==cohort];base=q[(q.arm=='intact')&(q['mode']=='refit')].groupby('seed')[['test_accuracy','frequency_excess','null_excess','r2_vs_frequency']].mean()
                    q=q[q['mode']==mode];target=reference_family(q,family).groupby('seed').test_accuracy.mean()
                    control=q[q.arm.str.startswith('dan_control' if family=='DAN' else 'uniform')].groupby('seed').test_accuracy.mean()
                    impairment=base.test_accuracy-target;excess=control-target;key=f'{cohort}/{mode}/{family}'
                    for name,x in [('target',target),('control',control),('impairment',impairment),('excess_impairment',excess)]:audit_stats(x,summary['statistics'][key][name],c)
                    access=bool((base.frequency_excess>=.05).all() and (base.null_excess>=.05).all() and base.r2_vs_frequency.mean()>0)
                    gate=bool(access and impairment.mean()>=.05 and excess.mean()>=.05 and (impairment>0).all() and (excess>0).all())
                    assert gate==summary['gates'][key];both.append(gate)
                if all(both):confirmed.append(f'{mode}/{family}')
        assert confirmed==summary['confirmed'] and summary['primary_DAN_refit']==('refit/DAN' in confirmed) and not summary['cohorts_pooled']
        for p,h in read(root/'prior-results-sha256.json').items():assert core.sha256(p)==h,p
        for p,h in m['context']['source_sha256'].items():
            assert hashlib.sha256(subprocess.check_output(['git','show',m['context']['git_commit']+':'+p])).hexdigest()==h,p
        assert(count,fcount,metriccount)==(65,52,1287);budget.check();usage=budget.close()
        core.write_json(out/'checks.json',dict(cases=count,frozen=fcount,independent_trajectories=2*count,metric_rows=metriccount,normalized_matching=ranking,
            all_checks_pass=True,source_git_bytes_verified=True,result_manifest_sha256=core.sha256(root/'manifest.json'),
            prior_files_unchanged=len(read(root/'prior-results-sha256.json')),budget=usage,
            analysis_path=analysis_root.as_posix(),analysis_manifest_sha256=core.sha256(analysis_root/'manifest.json'),
            verifier_sha256=core.sha256(__file__)))
        print(dict(cases=count,frozen=fcount,metric_rows=metriccount,confirmed=confirmed),flush=True)
    finally:budget.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--analysis',type=Path)
    args=p.parse_args();verify(args.root,args.out,args.analysis)
