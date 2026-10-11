"""Independent M7 noise/score/criterion reconstruction after full M6-method replay.

Reuses only frozen independent source/manual-dynamics/old-head authentication.
No M7 main noise, scoring, endpoint or summary import.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from verify_counterfactual_tracking import case_job as clean_verify_job
from verify_temporal_memory_curve import validate_whole_source
from tdc_pool import pooled
from noise_tracking_support import config,read,write,sha,array_sha,attempt,seal,check_manifest,source_record,assert_source_unchanged,history_preserved

WORLDS=('original','counterfactual','instantaneous_original','instantaneous_counterfactual')

def independent_noise(ids,scale,seed,sigma=.1):
    order=sorted(range(48),key=lambda j:int(ids[j]))
    rng=np.random.Generator(np.random.PCG64(seed))
    canonical=np.vstack([rng.standard_normal(48) for _ in range(64)])
    z=np.zeros_like(canonical)
    for k,j in enumerate(order):z[:,j]=canonical[:,k]
    return dict(canonical_standard_normal=canonical,standard_normal=z,canonical_order=np.asarray(order,dtype=np.int64),
        amplitude=np.asarray(scale)*sigma,raw_noise=z*(np.asarray(scale)*sigma))

def reconstruction(clean,c,b,l,smoke):
    n=independent_noise(clean['root_ids'][clean['observed_indices']],clean['scale'],b['noise_seed'],c['noise_sigma'])
    for world in WORLDS:
        n['observed_'+world]=clean['trial_'+world+'_features'][:,-1]+n['raw_noise']
        norm=(n['observed_'+world]-clean['mean'])/clean['scale']
        n['scores_'+world]=norm.dot(clean['coefficients_lag2'])+clean['intercept_lag2']
        n['predictions_'+world]=np.argmax(n['scores_'+world],axis=1)
        alternative=clean['scores_'+world]+c['noise_sigma']*n['standard_normal'].dot(clean['coefficients_lag2'])
        np.testing.assert_allclose(n['scores_'+world],alternative,atol=1e-10,rtol=1e-10)
    np.testing.assert_array_equal(n['observed_instantaneous_original'],n['observed_instantaneous_counterfactual'])
    np.testing.assert_array_equal(n['scores_instantaneous_original'],n['scores_instantaneous_counterfactual'])
    np.testing.assert_array_equal(n['predictions_instantaneous_original'],n['predictions_instantaneous_counterfactual'])
    np.testing.assert_allclose(n['scores_counterfactual']-n['scores_original'],clean['scores_counterfactual']-clean['scores_original'],atol=1e-10,rtol=1e-10)
    s,r=clean['original_symbols'],clean['alternative_symbols']
    first=n['predictions_original']==s;second=n['predictions_counterfactual']==r
    joint=int(np.count_nonzero(first&second));clean_joint=int(np.sum((clean['predictions_original']==s)&(clean['predictions_counterfactual']==r)))
    for key in ('frequency_trial_predictions','current_trial_predictions'):
        pred=clean[key];assert not np.any((pred==s)&(pred==r))
    instant=n['predictions_instantaneous_original'];assert not np.any((instant==s)&(instant==r))
    cell=dict(level=l,seed=b['seed'],noise_seed=b['noise_seed'],samples=64,original_correct=int(first.sum()),replacement_correct=int(second.sum()),joint_correct=joint,
        original_accuracy=int(first.sum())/64,replacement_accuracy=int(second.sum())/64,joint_accuracy=joint/64,
        prediction_changes=int(np.sum(n['predictions_original']!=n['predictions_counterfactual'])),
        clean_joint_correct=clean_joint,clean_joint_accuracy=clean_joint/64,clean_minus_noisy_joint=(clean_joint-joint)/64,
        frequency_joint_correct=0,current_joint_correct=0,instantaneous_joint_correct=0,
        scientific_rule_pass=None if smoke else 10*joint>=9*64,technical_checks_pass=True,
        scale_floor_value_count=int(np.sum(clean['scale']==1e-5)),
        mean_raw_noise_energy=float(np.mean(np.sum(n['raw_noise']**2,axis=1))),
        mean_standardized_noise_energy=float(np.mean(np.sum((c['noise_sigma']*n['standard_normal'])**2,axis=1))))
    return n,cell

def close_tree(actual,expected):
    if isinstance(expected,dict):
        assert set(actual)==set(expected)
        for k in expected:close_tree(actual[k],expected[k])
    elif isinstance(expected,list):
        assert len(actual)==len(expected)
        for a,e in zip(actual,expected):close_tree(a,e)
    elif isinstance(expected,float):np.testing.assert_allclose(actual,expected,atol=1e-10,rtol=1e-10)
    else:assert actual==expected,(actual,expected)

def estimate(x,c):
    x=np.array(x,dtype=float);draws=np.random.Generator(np.random.PCG64(c['bootstrap_seed'])).integers(0,len(x),(10000,len(x)))
    lo,hi=np.quantile(x[draws].mean(axis=1),[.025,.975])
    return dict(mean=float(x.mean()),median=float(np.median(x)),sd=float(np.std(x,ddof=1)) if len(x)>1 else 0.,bootstrap_lo=float(lo),bootstrap_hi=float(hi))

def independent_summary(cells,c,smoke):
    levels={}
    for l in c['levels']:
        rows=sorted([x for x in cells if x['level']==l],key=lambda x:x['seed'])
        assert [x['seed'] for x in rows]==c['blocks_by_level'][l]
        passed=None if smoke else all(10*x['joint_correct']>=576 for x in rows)
        levels[l]=dict(registered_rule_pass=passed,outcome='smoke' if smoke else 'PASS' if passed else 'FAIL',cells=rows,
            **{k:estimate([x[k] for x in rows],c) for k in ['joint_accuracy','original_accuracy','replacement_accuracy','clean_joint_accuracy','clean_minus_noisy_joint']})
    return dict(smoke=smoke,outcome=levels[c['primary_level']]['outcome'],primary_pass=levels[c['primary_level']]['registered_rule_pass'],levels=levels,
        cases=len(cells),probe_pairs=64*len(cells),noise_sigma=.1,new_trained_parameters=0,decoder_parameters=490,noise_feedback=False,noise_clipping=False,
        performance_eligibility_gate=False,physiological_noise_claim=False,criterion_unchanged=True)

def case_job(job):
    c,b,l,result,out=job['config'],job['block'],job['level'],Path(job['result']),Path(job['out'])
    identity=f'{l}_s{b["seed"]}'
    verified=clean_verify_job(dict(config=c,block=b,level=l,result=str(result/'clean'),out=str(out/'clean'/identity)))
    root=result/'noisy'/identity;dest=out/'noisy'/identity
    with attempt(dest,c,'m7-independent-noise-case'):
        check_manifest(root)
        with np.load(result/'clean'/identity/'case.npz',allow_pickle=False) as a:
            clean={k:a[k] for k in a.files}
        n,cell=reconstruction(clean,c,b,l,job['smoke'])
        hashes=read(root/'array-hashes.json')
        with np.load(root/'noise.npz',allow_pickle=False) as saved:
            assert set(saved.files)==set(n)==set(hashes)
            for k,v in n.items():
                assert array_sha(saved[k])==hashes[k]
                if k.startswith('scores_'):np.testing.assert_allclose(saved[k],v,atol=1e-10,rtol=1e-10)
                else:np.testing.assert_array_equal(saved[k],v)
        close_tree(read(root/'primary-cell.json'),cell)
        rows=read(root/'trials.json');assert len(rows)==64
        for i,row in enumerate(rows):
            assert row['level']==l and row['seed']==b['seed'] and row['anchor']==int(clean['anchors'][i]) and row['evaluation_time']==int(clean['evaluation_times'][i])
            assert row['probe_index']==i and row['original_symbol']==int(clean['original_symbols'][i]) and row['replacement_symbol']==int(clean['alternative_symbols'][i])
            assert row['current_symbol']==int(clean['current_symbols'][i])
            assert row['original_correct']==bool(n['predictions_original'][i]==row['original_symbol'])
            assert row['replacement_correct']==bool(n['predictions_counterfactual'][i]==row['replacement_symbol'])
            assert row['joint_correct']==bool(n['predictions_original'][i]==row['original_symbol'] and n['predictions_counterfactual'][i]==row['replacement_symbol'])
            assert row['clean_joint_correct']==bool(clean['predictions_original'][i]==row['original_symbol'] and clean['predictions_counterfactual'][i]==row['replacement_symbol'])
            for w in WORLDS:assert row['prediction_'+w]==int(n['predictions_'+w][i])
            np.testing.assert_allclose(row['raw_noise_energy'],np.sum(n['raw_noise'][i]**2),atol=1e-10,rtol=1e-10)
            np.testing.assert_allclose(row['standardized_noise_energy'],np.sum((c['noise_sigma']*n['standard_normal'][i])**2),atol=1e-10,rtol=1e-10)
        np.savez_compressed(dest/'recomputed-noise.npz',**n)
        write(dest/'primary-cell.json',cell);write(dest/'checks.json',dict(all_checks_pass=True,noise_reconstructed=True,all_scores_predictions_match=True,shared_noise_controls_joint_zero=True,
            affine_score_difference_preserved=True,no_state_feedback=True,old_clean_replay_checks=verified['checks']))
    seal(dest,dict(complete=True,result_case_manifest_sha256=sha(root/'manifest.json')))
    return dict(identity=identity,cell=cell,clean_checks=verified['checks'])

def verify(result,out):
    manifest=check_manifest(result);smoke=manifest['smoke'];c=config(smoke);assert read(result/'config.json')==c
    source=source_record(c)
    with attempt(out,c,'m7-independent-validation') as resources:
        write(out/'source.json',source);write(out/'config.json',c);write(out/'historical-preservation.json',history_preserved(c))
        write(out/'source-graph-audit.json',validate_whole_source(c,resources))
        jobs=[dict(config=c,block=b,level=l,result=str(result),out=str(out),smoke=smoke) for l in c['levels'] for b in c['blocks'] if b['seed'] in c['blocks_by_level'][l]]
        results,usage=pooled(case_job,jobs,4,c,'M7 independent replay/noise')
        cells=sorted([x['cell'] for x in results],key=lambda x:(x['level'],x['seed']))
        summary=independent_summary(cells,c,smoke);close_tree(read(result/'summary.json'),summary)
        close_tree(read(result/'primary-cells.json'),cells)
        for b in c['blocks'][:(1 if smoke else 3)]:
            with np.load(result/'noisy'/f'legacy5_s{b["seed"]}'/'noise.npz') as p,np.load(result/'noisy'/f'brain5_s{b["seed"]}'/'noise.npz') as q:
                np.testing.assert_array_equal(p['canonical_standard_normal'],q['canonical_standard_normal'])
        write(out/'recomputed-summary.json',summary);write(out/'process-tree-resources.json',usage)
        write(out/'checks.json',dict(all_checks_pass=True,outcome=summary['outcome'],primary_pass=summary['primary_pass'],cases=len(cells),
            baseline_full_state_hashes=len(cells),prestate_hash_entries=64*len(cells),four_world_window_hash_entries=256*len(cells),old_head_svd_authentications=len(cells),
            noisy_probe_pairs=64*len(cells),exact_noise_draws_and_predictions=True,shared_noise_all_worlds=True,no_feedback_or_clipping=True,
            scientific_threshold_tolerance_applied=False,result_manifest_sha256=sha(result/'manifest.json'),source_commit=source['source_commit'],protocol_commit=source['protocol_commit']))
        assert_source_unchanged(source)
    seal(out,dict(complete=True,source_commit=source['source_commit'],result_manifest_sha256=sha(result/'manifest.json')))
    print(dict(all_checks_pass=True,outcome=summary['outcome'],cases=len(cells)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    with threadpool_limits(1):verify(a.result,a.out)
