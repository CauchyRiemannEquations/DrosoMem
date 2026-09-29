"""Independent pathway-mask feasibility, dynamics and inference audit."""
import argparse
from pathlib import Path
from collections import Counter
import hashlib,subprocess
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import milp,Bounds,LinearConstraint
from threadpoolctl import threadpool_limits
from flying.training import whole_brain_memory as core
from alphabet_memory import read,check,symbol_bank
from frozen_state_probe import verify_fold,Budget
from normalization_transfer import verify_transfer
from verify_neuron_panel import trajectory,audit_metrics,audit_stats


def target_reference(raw,roles,family):
    co=raw.tocoo();pre,post=family.split('_')
    return np.array([roles[u]==pre and roles[v]==post for v,u in zip(co.row,co.col)],dtype=bool)


def matching_reference(raw,target,mask,c):
    co=raw.tocoo();dense=raw.toarray();strength=np.abs(dense).sum(1)
    normalized=np.abs(co.data)*c['gain']/strength[co.row]
    assert mask.dtype==bool and mask.shape==target.shape
    assert mask.sum()==target.sum() and np.count_nonzero(co.data[mask]<0)==np.count_nonzero(co.data[target]<0)
    r=float(abs(co.data[mask]).sum());v=float(normalized[mask].sum())
    re=r/abs(co.data[target]).sum()-1;ne=v/normalized[target].sum()-1
    assert abs(re)<=c['relative_mass_tolerance']+c['numerical_guard'] and abs(ne)<=c['relative_mass_tolerance']+c['numerical_guard']
    overlap=int((mask&target).sum())
    return dict(count=int(mask.sum()),negative=int(np.count_nonzero(co.data[mask]<0)),overlap=overlap,overlap_fraction=float(overlap/target.sum()),raw_l1_error=float(re),normalized_l1_error=float(ne),raw_l1=r,normalized_l1=v)


def verify_bank(c):
    root=Path(c['mask_bank']);manifest=check(root);assert manifest['config']==c and not manifest['task_outcomes_inspected']
    mins=read(root/'minimum-overlap.json');graphs={};proofs={}
    for ci in c['circuit_seeds']:
        path=Path(f'data/flywire_783_mb_left_kc512_s{ci}');raw,ids,_=core.load_connectome(path);raw.sort_indices();roles,_=core.load_roles(path,ids);roles=np.asarray(roles);graphs[ci]=(raw,roles)
        co=raw.tocoo();strength=np.asarray(abs(raw).sum(1)).ravel();normalized=abs(co.data)*c['gain']/strength[co.row]
        for family in c['families']:
            target=target_reference(raw,roles,family);info=mins[str(ci)][family];k=info['overlap']
            with np.load(root/'minimum-masks.npz') as z:mask=z[f'c{ci}_{family}']
            measured=matching_reference(raw,target,mask,c)
            for key,value in measured.items():np.testing.assert_allclose(info[key],value,atol=1e-10,rtol=1e-10)
            assert info['solver']['status']==0 and info['solver']['gap']==0 and info['solver']['dual_bound']>k-1+1e-6
            assert info['eligible']==bool(k/target.sum()<=c['maximum_specificity_overlap'])
            if k:
                A=np.stack([np.ones(raw.nnz),(co.data<0).astype(float),abs(co.data)/abs(co.data[target]).sum(),normalized/normalized[target].sum(),target])
                e=c['relative_mass_tolerance'];lo=[target.sum(),np.count_nonzero(co.data[target]<0),1-e,1-e,0];hi=[lo[0],lo[1],1+e,1+e,k-1]
                r=milp(np.zeros(raw.nnz),integrality=np.ones(raw.nnz),bounds=Bounds(0,1),constraints=LinearConstraint(A,lo,hi),options={'time_limit':c['solver_seconds'],'mip_rel_gap':0})
                assert r.status==2,('Lower overlap not proved infeasible',ci,family,r.status)
                proofs[f'{ci}/{family}']='minimum minus one infeasible'
            else:proofs[f'{ci}/{family}']='zero overlap is lower bound'
    count=0
    for cohort,settings in c['cohorts'].items():
        for block in settings['blocks']:
            for ci in settings['circuit_seeds']:
                raw,roles=graphs[ci];name=f'c{ci}_s{block["seed"]}';record=read(root/(name+'.json'))
                assert record['cohort']==cohort and record['raw_sha256']==core.weight_hash(raw)
                with np.load(root/(name+'.npz')) as z:masks=dict(z)
                assert not masks['intact'].any() and len(masks)==21
                for family in c['families']:
                    target=target_reference(raw,roles,family);np.testing.assert_array_equal(target,masks[family]);k=mins[str(ci)][family]['overlap']
                    for j,seed in enumerate(block['edge_seeds'][family]):
                        arm=f'{family}_control{j}';mask=masks[arm];info=record['controls'][arm]
                        measured=matching_reference(raw,target,mask,c)
                        for key,value in measured.items():np.testing.assert_allclose(info[key],value,atol=1e-10,rtol=1e-10)
                        assert measured['overlap']==k and info['seed']==seed
                        cost=np.random.default_rng(seed).uniform(-1,1,raw.nnz);assert hashlib.sha256(cost.tobytes()).hexdigest()==info['cost_sha256']
                        log=info['solver'];assert log['status'] in [0,1]
                        if log['status']==0:assert log['gap']<=c['draw_mip_gap']+1e-10
                        else:assert c['accept_feasible_time_limit'] and log['time_limited_feasible']
                        np.testing.assert_allclose(cost@mask,log['objective'],atol=1e-6,rtol=1e-8)
                        assert log['dual_bound']<=log['objective']+1e-6;count+=1
    assert count==195
    for p,h in manifest['context']['source_sha256'].items():assert hashlib.sha256(subprocess.check_output(['git','show',manifest['context']['git_commit']+':'+p])).hexdigest()==h,p
    return dict(control_masks=count,minimum_overlap_checks=proofs,bank_manifest_sha256=core.sha256(root/'manifest.json'))


def independent_mask(raw,ids,roles,score,identity,c):
    name=f'c{identity["circuit_seed"]}_s{identity["seed"]}'
    with np.load(Path(c['mask_bank'])/(name+'.npz')) as z:mask=z[identity['arm']]
    if identity['arm']=='intact':assert not mask.any()
    else:
        family=identity['arm'].split('_control')[0];target=target_reference(raw,roles,family)
        matching_reference(raw,target,mask,c)
        if '_control' not in identity['arm']:np.testing.assert_array_equal(mask,target)
    return mask


def reference_family(q,family):
    return q[q.arm==family]


@threadpool_limits.wrap(limits=1)
def verify(root,out,analysis=None):
    m=check(root);c=m['config'];out.mkdir(parents=True,exist_ok=False);budget=Budget(c)
    try:
        bank_checks=verify_bank(c);print('195 masks and all minimum-overlap bounds verified',flush=True)
        minimum=read(Path(c['mask_bank'])/'minimum-overlap.json')
        scores={};graphs={};ranking={}
        for ci in c['circuit_seeds']:
            raw=sparse.load_npz(root/'graphs'/str(ci)/'raw.npz');meta=read(root/'graphs'/str(ci)/'ids.json')
            directory=Path(f'data/flywire_783_mb_left_kc512_s{ci}');original,ids,_=core.load_connectome(directory);roles,_=core.load_roles(directory,ids);roles=np.asarray(roles)
            assert core.weight_hash(raw)==core.weight_hash(original)==meta['raw_sha256'];assert ids==meta['ids'];assert roles.tolist()==meta['roles']
            scores[ci]=np.zeros(raw.nnz)
            ranking[str(ci)]=dict(edges=raw.nnz)
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
            dan=(roles[co.row]=='DAN')|(roles[co.col]=='DAN');within=roles[co.row]==roles[co.col]
            assert g['DAN_overlap']==int((mask&dan).sum()) and g['removed_within_role']==int((mask&within).sum())
            if i['arm']!='intact':
                family=i['arm'].split('_control')[0];target=target_reference(raw,roles,family);ref=matching_reference(raw,target,mask,c)
                for key,value in ref.items():np.testing.assert_allclose(g['matching'][key],value,atol=1e-10,rtol=1e-10)
                assert g['pathway']==family and g['pathway_target_count']==int(target.sum())
                assert g['minimum_overlap']==minimum[str(i['circuit_seed'])][family]['overlap']
                assert g['specificity_eligible']==all(minimum[str(ci)][family]['eligible'] for ci in c['circuit_seeds'])
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
                    control=q[q.arm.isin([family+'_control'+str(j) for j in range(3)])].groupby('seed').test_accuracy.mean()
                    impairment=base.test_accuracy-target;excess=control-target;key=f'{cohort}/{mode}/{family}'
                    for name,x in [('target',target),('control',control),('impairment',impairment),('excess_impairment',excess)]:audit_stats(x,summary['statistics'][key][name],c)
                    access=bool((base.frequency_excess>=.05).all() and (base.null_excess>=.05).all() and base.r2_vs_frequency.mean()>0)
                    gate=bool(access and impairment.mean()>=.05 and excess.mean()>=.05 and (impairment>0).all() and (excess>0).all())
                    elig=all(minimum[str(ci)][family]['eligible'] for ci in c['circuit_seeds'])
                    assert summary['statistics'][key]['numeric_specificity_gate']==gate and summary['statistics'][key]['specificity_eligible']==elig
                    sensitivity=bool(access and impairment.mean()>=.05 and (impairment>0).all())
                    assert sensitivity==summary['sensitivity_gates'][key]
                    assert (gate and elig)==summary['gates'][key];both.append(gate and elig)
                if all(both):confirmed.append(f'{mode}/{family}')
        assert confirmed==summary['confirmed'] and summary['primary_DAN_MBON_refit']==('refit/DAN_MBON' in confirmed) and not summary['cohorts_pooled']
        assert summary['refit_candidates']==[family for family in c['families'] if 'refit/'+family in confirmed]
        for p,h in read(root/'prior-results-sha256.json').items():assert core.sha256(p)==h,p
        for p,h in m['context']['source_sha256'].items():
            assert hashlib.sha256(subprocess.check_output(['git','show',m['context']['git_commit']+':'+p])).hexdigest()==h,p
        assert(count,fcount,metriccount)==(273,260,5863);budget.check();usage=budget.close()
        core.write_json(out/'checks.json',dict(cases=count,frozen=fcount,independent_trajectories=2*count,metric_rows=metriccount,graph_sizes=ranking,bank_checks=bank_checks,
            all_checks_pass=True,source_git_bytes_verified=True,result_manifest_sha256=core.sha256(root/'manifest.json'),
            prior_files_unchanged=len(read(root/'prior-results-sha256.json')),budget=usage,
            analysis_path=analysis_root.as_posix(),analysis_manifest_sha256=core.sha256(analysis_root/'manifest.json'),
            verifier_sha256=core.sha256(__file__)))
        print(dict(cases=count,frozen=fcount,metric_rows=metriccount,confirmed=confirmed),flush=True)
    except Exception as exc:
        core.write_json(out/'failure.json',dict(error=repr(exc)));raise
    finally:budget.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--analysis',type=Path)
    args=p.parse_args();verify(args.root,args.out,args.analysis)
