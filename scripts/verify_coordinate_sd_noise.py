"""Independent recurrence, noise, metric and block-contrast verification."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits
from flying.data.sequences import SequenceDataset
from flying.training import whole_brain_memory as core
from alphabet_memory import read,check
from pathway_memory import seal
from frozen_state_probe import Budget
from relative_noise import read_table
from verify_allocation_noise import independent_noise,independent_head
from verify_feedback_noise import reference,first_error,additional_metrics
from verify_act5_robustness import metrics as audit_metrics
from verify_research_suite import teacher_reference,graph_invariants
from verify_neuron_panel import audit_stats
import coordinate_sd_noise as run


def exact_files(root):
    m=check(root)
    assert set(m['artifacts'])=={p.relative_to(root).as_posix() for p in root.rglob('*')
                                if p.is_file() and p!=root/'manifest.json'}
    return m


def statistics(frame,c,root):
    summary=read(root/'summary.json');blocks=[];pairs=[];doses=[];gates={}
    for co,ss in c['cohorts'].items():
        for family in c['families']:
            for seed,ds in ss:
                values={}
                for top in c['topologies']:
                    for cal in c['calibrations']:
                        z=frame[(frame.cohort==co)&(frame.family==family)&(frame.seed==seed)&
                                (frame.topology==top)&(frame.calibration==cal)]
                        assert len(z)==18 and set(z.dataset_seed)=={ds}
                        group=dict(cohort=co,family=family,topology=top,calibration=cal,seed=seed)
                        measures={m:float(np.mean(z[m].to_numpy())) for m in run.METRICS}
                        blocks.append(dict(**group,**measures));values[(top,cal)]=measures
                        for strength in c['relative_strengths']:
                            zz=z[z.strength==strength];assert len(zz)==6
                            doses.append(dict(**group,strength=strength,**{m:float(np.mean(zz[m].to_numpy())) for m in run.METRICS}))
                for control in ['degree','role']:
                    row=dict(cohort=co,family=family,seed=seed,control=control)
                    for metric in run.METRICS:
                        gc=values[(control,'common')][metric]-values[('intact','common')][metric]
                        go=values[(control,'own')][metric]-values[('intact','own')][metric]
                        gs=values[(control,'coordinate')][metric]-values[('intact','coordinate')][metric]
                        row.update({metric+'_common_gap':gc,metric+'_own_gap':go,metric+'_coordinate_gap':gs,
                            metric+'_attenuation':go-gs,metric+'_total_attenuation':gc-gs,metric+'_median_attenuation':gc-go})
                    pairs.append(row)
    p=pd.DataFrame(pairs)
    for (co,family,control),z in p.groupby(['cohort','family','control']):
        key=f'{co}/{family}/{control}'
        for metric in summary['statistics'][key]:
            x=z[metric].to_numpy();saved=summary['statistics'][key][metric]
            if np.isfinite(x).all():audit_stats(x,saved,c)
            else:assert saved is None
        a=z.exact_prefix_symbols_attenuation.to_numpy();b=z.retention_attenuation.to_numpy()
        gates[key]=None if not np.isfinite(b).all() else bool(np.mean(a)>=2 and np.mean(b)>=.1 and
            np.count_nonzero(a>0)>=(4 if co=='discovery' else 3) and np.count_nonzero(b>0)>=(4 if co=='discovery' else 3))
    for name,records,keys in [('seed-blocks.csv',blocks,run.GROUP),('dose-seed-table.csv',doses,run.GROUP+['strength']),
                              ('paired-differences.csv',pairs,['cohort','family','control','seed'])]:
        a=read_table(root/name).sort_values(keys).reset_index(drop=True)
        b=pd.DataFrame(records).sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(a[b.columns],b,check_exact=False,atol=1e-12,rtol=1e-12)
    assert summary['descriptive_gates']==gates
    keys=[f'{co}/random/degree' for co in c['cohorts']];v=[gates[k] for k in keys]
    assert summary['primary_keys']==keys and summary['primary_confirmed']==(None if any(x is None for x in v) else all(v))
    assert not summary['cohorts_pooled'] and not summary['fresh_models'] and summary['fresh_noise']
    assert summary['undefined_retention_propagated']
    assert summary['primary_contrast']=='own_gap_minus_coordinate_gap'


@threadpool_limits.wrap(limits=1)
def verify(root,out):
    m=exact_files(root);c=m['config'];out.mkdir(parents=True,exist_ok=False)
    current=run.context(c)
    for key in ['source_sha256','adapter_sha256','config_sha256','base_config_sha256']:
        assert current[key]==m['context'][key],key
    budget=Budget(dict(max_seconds=c['verification_seconds'],max_rss_bytes=c['max_rss_bytes']))
    try:
        run.verify_parents(c);v=read(root/'verification.json');smoke=v['smoke'];h=8 if smoke else 197
        seeds=c['smoke_noise_seeds'] if smoke else c['noise_seeds'];allrows=[];allcert=[]
        counts=dict(cases=0,independent_full_paths=0,independent_teacher_paths=0,graph_audits=0,
                    exact_clean_replays=0,certificates=0,intact_null_path_pairs=0)
        maxerr=0.;banks={};identities={}
        for case in run.cases(c,smoke):
            budget.check();path=root/run.suite.stem(case);exact_files(path);meta=read(path/'case.json')
            assert case==meta['case'];src=Path(meta['parent']);check(src);old=read(src/'case.json')
            assert core.sha256(src/'manifest.json')==meta['parent_manifest_sha256']
            assert meta['checkpoint_path']==(src/'checkpoint.npz').as_posix()
            assert core.sha256(src/'checkpoint.npz')==meta['checkpoint_sha256']
            paired=Path(meta['paired_intact']);check(paired)
            assert core.sha256(paired/'manifest.json')==meta['paired_manifest_sha256']
            with np.load(src/'checkpoint.npz') as a:cp=dict(a)
            for val in cp.values():assert np.isfinite(val).all()
            ds=SequenceDataset(case['family'],case['dataset_seed'],**read(c['base_config'])['dataset'])
            np.testing.assert_array_equal(cp['symbols'],ds.symbols())
            assert core.fingerprint(meta['dataset'])==core.fingerprint(ds.identity(cp['symbols']))
            head=run.feedback.load_head(cp,case['seed']);assert head.digest()==meta['head_sha256']==old['head_sha256']
            assert head.parameter_count==482 and meta['training_updates']==meta['new_fits']==0
            assert meta['graph_frozen'] and meta['head_frozen'] and meta['graph']==old['graph']
            assert meta['teacher_state_diagnostics']==old['teacher_state_diagnostics']
            model,obs,parent=run.load_model(c,case,src)
            if case['topology']!='intact':graph_invariants(src,case,c);counts['graph_audits']+=1
            features=teacher_reference(model,obs,cp['symbols'][:-1]);counts['independent_teacher_paths']+=1
            np.testing.assert_allclose(features,cp['features'],atol=1e-12,rtol=1e-10)
            loss,_,_=head.objective(cp['features'],cp['symbols'][1:],1e-5,core.sample_weights(199,3,32,4))
            assert loss==meta['parent_training_loss']
            assert np.mean(head.predict(cp['features'])==cp['symbols'][1:])==meta['parent_training_accuracy']
            target=cp['symbols'][3:3+h];teacher=cp['features'][2:2+h]
            sd=np.sqrt(np.mean((cp['features']-cp['features'].mean(0))**2,axis=0))
            own=float(np.median(sd));weight=np.sqrt(48)*sd/np.linalg.norm(sd)
            with np.load(paired/'checkpoint.npz') as a:
                f=a['features'];common=float(np.median(np.sqrt(np.mean((f-f.mean(0))**2,axis=0))))
            np.testing.assert_allclose([own,common,own/common],[meta['own_q'],meta['common_q'],meta['q_ratio']],atol=1e-18,rtol=1e-12)
            np.testing.assert_allclose([meta['training_rms'],meta['median_over_rms']],
                [np.sqrt(np.mean(sd**2)),own/np.sqrt(np.mean(sd**2))],atol=1e-18,rtol=1e-12)
            assert meta['zero_sd_coordinates']==int(np.sum(sd==0))
            ids=np.asarray(meta['graph']['observation_root_ids'],dtype=np.int64)
            noise=np.stack([independent_noise(ids,case['seed'],case['circuit_seed'],ns,h) for ns in seeds])
            pairkey=(case['seed'],case['circuit_seed'])
            if pairkey in banks:np.testing.assert_array_equal(banks[pairkey],noise)
            banks[pairkey]=noise
            ident={k:meta['graph'][k] for k in ['input_root_ids','observation_root_ids','input_mapping_sha256']}
            assert identities.setdefault(pairkey,ident)==ident
            with np.load(path/'calibration.npz') as a:bank=dict(a)
            np.testing.assert_array_equal(bank['standard_normals'],noise)
            np.testing.assert_array_equal(bank['target'],target);np.testing.assert_array_equal(bank['observed_root_ids'],ids)
            np.testing.assert_allclose(bank['training_sd'],sd,atol=1e-18,rtol=1e-12)
            np.testing.assert_allclose(bank['allocation_weights'],weight,atol=1e-12,rtol=1e-12)
            with np.load(path/'clean.npz') as a:clean=dict(a)
            with np.load(src/'clean.npz') as a:
                for key in clean:np.testing.assert_array_equal(clean[key],a[key][:h])
            counts['exact_clean_replays']+=1;cleanprefix=first_error(clean['prediction'],target)
            assert cleanprefix==meta['clean_prefix']
            rows=read_table(path/'rollouts.csv');cert=read_table(path/'certificates.csv')
            assert len(rows)==28 and len(cert)==27
            for row in rows.to_dict('records'):
                budget.check()
                for k,value in case.items():assert row[k]==value
                cal=row['calibration'];ni=seeds.index(row['noise_seed'])
                if cal=='clean':amp=np.zeros(48)
                else:
                    ai=c['calibrations'].index(cal);di=c['relative_strengths'].index(row['strength'])
                    q=common if cal=='common' else (own if cal=='own' else float(np.sqrt(np.mean(sd**2))))
                    amp=row['strength']*sd if cal=='coordinate' else row['strength']*q*weight
                    if cal=='coordinate':
                        np.testing.assert_allclose(amp[sd>0]/sd[sd>0],row['strength'],atol=1e-15,rtol=1e-12)
                        assert np.all(amp[sd==0]==0)
                    np.testing.assert_allclose(bank['amplitudes'][di,ai],amp,atol=1e-18,rtol=1e-12)
                    np.testing.assert_allclose(row['expected_energy'],48*(row['strength']*q)**2,atol=1e-18,rtol=1e-12)
                with np.load(path/row['artifact']) as a:actual=dict(a)
                for val in actual.values():assert np.isfinite(val).all()
                ref=reference(model,obs,cp,cp['symbols'][:3],h,noise[ni],amp)
                for key,value in ref.items():
                    if key in ['prediction','active_counts','clipped']:np.testing.assert_array_equal(actual[key],value)
                    else:
                        np.testing.assert_allclose(actual[key],value,atol=1e-12,rtol=1e-10)
                        maxerr=max(maxerr,float(np.max(abs(actual[key]-value))))
                np.testing.assert_array_equal(actual['position_accuracy'],actual['prediction']==target)
                proxy=dict(row,pi_memory_score=row['exact_prefix_symbols'])
                audit_metrics(actual,target,proxy,clean,head.scale,meta['graph']['neurons']);additional_metrics(actual,target,proxy)
                prefix=first_error(actual['prediction'],target);assert prefix==row['exact_prefix_symbols']
                if cleanprefix:np.testing.assert_allclose(row['retention'],min(prefix/cleanprefix,1),atol=1e-12,rtol=1e-12)
                else:assert pd.isna(row['retention'])
                x=np.maximum(-1,np.minimum(1,teacher+noise[ni]*amp));p=independent_head(x,cp);pred=p.argmax(1)
                stop=min(first_error(pred,target)+1,h)
                assert first_error(pred,target)==prefix
                np.testing.assert_array_equal(actual['prediction'][:stop],pred[:stop])
                np.testing.assert_allclose(actual['state_features'][:stop],teacher[:stop],atol=1e-12,rtol=1e-10)
                if cal!='clean':
                    np.testing.assert_allclose(bank['teacher_probabilities'][ai,ni,di],p,atol=1e-12,rtol=1e-10)
                    np.testing.assert_array_equal(bank['teacher_predictions'][ai,ni,di],pred)
                    crow=cert[(cert.calibration==cal)&(cert.strength==row['strength'])&(cert.noise_seed==row['noise_seed'])]
                    assert len(crow)==1;cr=crow.iloc[0]
                    for k,value in case.items():assert cr[k]==value
                    for k in ['exact_prefix_symbols','exact_prefix_bits','first_error_position','censored','clean_prefix','retention','expected_energy']:
                        np.testing.assert_allclose(cr[k],row[k],atol=1e-12,rtol=1e-12,equal_nan=True)
                    np.testing.assert_allclose([row['teacher_accuracy'],cr.teacher_accuracy],np.mean(pred==target),atol=1e-12,rtol=1e-12)
                    assert cr.repeated_setting==row['repeated_setting']==(cal=='own' and case['topology']=='intact')
                    counts['certificates']+=1
                if case['topology']=='intact' and cal=='own':
                    with np.load(path/row['artifact'].replace('own_','common_')) as a:
                        for key in actual:np.testing.assert_array_equal(actual[key],a[key])
                    counts['intact_null_path_pairs']+=1
                counts['independent_full_paths']+=1
            allrows.extend(rows.to_dict('records'));allcert.extend(cert.to_dict('records'))
            counts['cases']+=1;print(f'Calibration audit {counts["cases"]}/{v["cases"]}: {run.suite.stem(case)}',flush=True)
        for name,records in [('raw-rollouts.csv',allrows),('raw-certificates.csv',allcert)]:
            pd.testing.assert_frame_equal(read_table(root/name),pd.DataFrame(records),check_exact=False,atol=1e-12,rtol=1e-12)
        if not smoke:statistics(pd.DataFrame(allrows),c,root)
        assert counts['independent_full_paths']==v['actual_paths'] and counts['certificates']==v['certificates']
        assert counts['intact_null_path_pairs']==v['repeated_noisy_settings']
        assert v['unique_fresh_noisy_settings']==counts['certificates']-counts['intact_null_path_pairs']
        prior=read(root/'prior-artifacts.json')
        for p,digest in prior.items():assert core.sha256(p)==digest,p
        budget.check()
        core.write_json(out/'checks.json',dict(all_checks_pass=True,smoke=smoke,**counts,
            maximum_numeric_error=maxerr,prior_files_unchanged=len(prior),
            result_manifest_sha256=core.sha256(root/'manifest.json'),budget=budget.close()))
        seal(out,c,m['context'],purpose='Independent coordinate SD calibration audit')
    except Exception as exc:
        core.write_json(out/'failure.json',dict(error=repr(exc),budget=budget.close()))
        seal(out,c,m['context'],purpose='Preserved calibration validation failure');raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();verify(a.source,a.out)
