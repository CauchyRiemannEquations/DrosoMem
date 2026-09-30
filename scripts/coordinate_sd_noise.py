"""Frozen graph/head calibration contrast; all noisy settings run autonomously."""
import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import softmax
from threadpoolctl import threadpool_limits
from flying.training import whole_brain_memory as core
from alphabet_memory import read, check
from pathway_memory import seal
from frozen_state_probe import Budget
from context_memory import estimate
import research_suite as suite
import feedback_noise as feedback
import allocation_noise as allocation
import graph_relative_noise as previous

CONFIG='configs/coordinate_sd_noise.json'
METRICS=['exact_prefix_symbols','retention']
GROUP=['cohort','family','topology','calibration','seed']


def context(c):
    ctx=previous.context(c)
    for name in ['coordinate_sd_noise','verify_coordinate_sd_noise']:
        ctx['adapter_sha256'][f'scripts/{name}.py']=core.sha256(f'scripts/{name}.py')
    return ctx


cases=previous.cases
load_model=previous.load_model


def amplitudes(features,reference_features,strength):
    sd,own,weights=allocation.allocation(features)
    common=allocation.allocation(reference_features)[1]
    if not np.isfinite([own,common]).all() or min(own,common)<=0:
        raise ValueError('Nonpositive/invalid training scale')
    return np.stack([strength*common*weights,strength*own*weights,strength*sd]),sd,weights,common,own


def verify_parents(c):
    previous.verify_parents(c)
    for key in ['calibration_parent','calibration_validation']:
        path=Path(c[key]);assert core.sha256(path/'manifest.json')==c[key+'_sha256'];check(path)
    v=read(Path(c['calibration_validation'])/'checks.json')
    assert v['all_checks_pass'] and v['result_manifest_sha256']==c['calibration_parent_sha256']


def execute(c,case,out,ctx,budget,smoke=False):
    start=time.monotonic();out.mkdir(exist_ok=False)
    parent=Path(c['parent'])/suite.stem(case);check(parent)
    paired=Path(c['parent'])/suite.parent_stem(case);check(paired)
    with np.load(parent/'checkpoint.npz') as a:cp=dict(a)
    with np.load(paired/'checkpoint.npz') as a:reference_features=a['features']
    model,obs,meta=load_model(c,case,parent)
    head=feedback.load_head(cp,case['seed']);digest=head.digest()
    assert digest==meta['head_sha256'] and head.parameter_count==482
    h=8 if smoke else 197;target=cp['symbols'][3:3+h];teacher=cp['features'][2:2+h]
    ids=np.asarray(meta['graph']['observation_root_ids'],dtype=np.int64)
    seeds=c['smoke_noise_seeds'] if smoke else c['noise_seeds']
    noise=np.stack([allocation.fresh_bank(ids,case['seed'],case['circuit_seed'],ns,h) for ns in seeds])
    clean=feedback.autonomous(model,obs,head,cp['symbols'][:3],h,noise[0],np.zeros(48))
    with np.load(parent/'clean.npz') as old:
        for key,value in clean.items():np.testing.assert_array_equal(value,old[key][:h])
    np.savez_compressed(out/'clean.npz',**clean,position_accuracy=clean['prediction']==target)
    rows=[dict(**case,calibration='clean',strength=0.,noise_seed=seeds[0],artifact='clean.npz',
               repeated_setting=True,**suite.rollout_metrics(clean,target,head,clean,meta['graph']['neurons']))]
    cert=[];pbank=np.empty((3,3,3,h,10));ampbank=[]
    clean_prefix=core.prefix_score(target,clean['prediction'])
    pclean=softmax(head.logits(teacher),axis=1)
    feedback.verify_pre_error(clean,teacher,pclean,pclean.argmax(1),target)
    for di,dose in enumerate(c['relative_strengths']):
        amps,sd,weights,common,own=amplitudes(cp['features'],reference_features,dose);ampbank.append(amps)
        for ai,cal in enumerate(c['calibrations']):
            amp=amps[ai];q=[common,own,float(np.sqrt(np.mean(sd**2)))][ai]
            np.testing.assert_allclose(amp@amp,48*(dose*q)**2,atol=1e-18,rtol=1e-12)
            for ni,ns in enumerate(seeds):
                budget.check();x=np.clip(teacher+noise[ni]*amp,-1,1);p=softmax(head.logits(x),axis=1)
                pbank[ai,ni,di]=p;pred=p.argmax(1)
                identity=dict(**case,calibration=cal,strength=dose,noise_seed=ns,
                              repeated_setting=case['topology']=='intact' and cal=='own')
                cert.append(dict(**identity,**suite.endpoint(pred,target,clean_prefix),
                    teacher_accuracy=float(np.mean(pred==target)),expected_energy=float(amp@amp)))
                a=feedback.autonomous(model,obs,head,cp['symbols'][:3],h,noise[ni],amp)
                stop=feedback.verify_pre_error(a,x,p,pred,target)
                np.testing.assert_allclose(a['state_features'][:stop],teacher[:stop],atol=1e-12,rtol=1e-10)
                name=f'{cal}_dose{di}_noise{ni}.npz'
                np.savez_compressed(out/name,**a,position_accuracy=a['prediction']==target)
                rows.append(dict(**identity,artifact=name,teacher_accuracy=float(np.mean(pred==target)),
                    expected_energy=float(amp@amp),**suite.rollout_metrics(a,target,head,clean,meta['graph']['neurons'])))
    if case['topology']=='intact':np.testing.assert_array_equal(pbank[0],pbank[1])
    assert head.digest()==digest and core.weight_hash(model.weights)==meta['graph']['weight_sha256']
    np.savez_compressed(out/'calibration.npz',standard_normals=noise,amplitudes=np.asarray(ampbank),
        training_sd=sd,allocation_weights=weights,observed_root_ids=ids,target=target,
        teacher_probabilities=pbank,teacher_predictions=pbank.argmax(-1))
    pd.DataFrame(rows).to_csv(out/'rollouts.csv',index=False)
    pd.DataFrame(cert).to_csv(out/'certificates.csv',index=False)
    core.write_json(out/'case.json',dict(case=case,graph=meta['graph'],dataset=meta['dataset'],
        parent=parent.as_posix(),parent_manifest_sha256=core.sha256(parent/'manifest.json'),
        checkpoint_path=(parent/'checkpoint.npz').as_posix(),checkpoint_sha256=core.sha256(parent/'checkpoint.npz'),
        paired_intact=paired.as_posix(),paired_manifest_sha256=core.sha256(paired/'manifest.json'),
        head_sha256=digest,head_parameters=482,training_updates=0,new_fits=0,graph_frozen=True,head_frozen=True,
        parent_training_loss=meta['train_loss'],parent_training_accuracy=meta['training_accuracy'],
        teacher_state_diagnostics=meta['teacher_state_diagnostics'],common_q=common,own_q=own,
        q_ratio=own/common,training_rms=float(np.sqrt(np.mean(sd**2))),
        median_over_rms=own/float(np.sqrt(np.mean(sd**2))),zero_sd_coordinates=int(np.sum(sd==0)),clean_prefix=clean_prefix,horizon=h,smoke=smoke,
        actual_paths=len(rows),certificates=len(cert),seconds=time.monotonic()-start))
    seal(out,c,ctx,purpose='Frozen coordinate SD calibration case')
    return rows,cert


def contrasts(blocks):
    rows=[]
    for (co,family,seed),z in blocks.groupby(['cohort','family','seed']):
        for control in ['degree','role']:
            row=dict(cohort=co,family=family,seed=int(seed),control=control)
            for metric in METRICS:
                values={(r.topology,r.calibration):getattr(r,metric) for r in z.itertuples()}
                gc=values[(control,'common')]-values[('intact','common')]
                go=values[(control,'own')]-values[('intact','own')]
                gs=values[(control,'coordinate')]-values[('intact','coordinate')]
                row.update({metric+'_common_gap':gc,metric+'_own_gap':go,metric+'_coordinate_gap':gs,
                    metric+'_attenuation':go-gs,metric+'_total_attenuation':gc-gs,metric+'_median_attenuation':gc-go})
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(frame,c,out):
    f=frame[frame.calibration!='clean'];mean=lambda x:np.mean(x.to_numpy())
    blocks=f.groupby(GROUP)[METRICS].agg(mean).reset_index()
    blocks.to_csv(out/'seed-blocks.csv',index=False)
    f.groupby(GROUP+['strength'])[METRICS].agg(mean).reset_index().to_csv(out/'dose-seed-table.csv',index=False)
    pairs=contrasts(blocks);pairs.to_csv(out/'paired-differences.csv',index=False)
    statistics={};gates={}
    for (co,family,control),z in pairs.groupby(['cohort','family','control']):
        key=f'{co}/{family}/{control}'
        statistics[key]={m:estimate(z[m],c) if np.isfinite(z[m]).all() else None
                         for m in z.columns if m not in ['cohort','family','control','seed']}
        gates[key]=feedback.gate(z.exact_prefix_symbols_attenuation,z.retention_attenuation,co,c)
    keys=[f'{co}/random/degree' for co in c['cohorts']];values=[gates[k] for k in keys]
    core.write_json(out/'summary.json',dict(statistics=statistics,descriptive_gates=gates,primary_keys=keys,
        primary_confirmed=None if any(v is None for v in values) else all(values),
        primary_contrast='own_gap_minus_coordinate_gap',cohorts_pooled=False,fresh_models=False,fresh_noise=True,undefined_retention_propagated=True))


@threadpool_limits.wrap(limits=1)
def run(c,out,smoke=False):
    out.mkdir(parents=True,exist_ok=False);ctx=context(c);budget=Budget(c)
    core.write_json(out/'config.json',c);core.write_json(out/'started.json',ctx)
    try:
        verify_parents(c)
        prior={p.as_posix():core.sha256(p) for p in Path('results').rglob('*') if p.is_file() and out not in p.parents}
        core.write_json(out/'prior-artifacts.json',prior);rows=[];cert=[];panel=cases(c,smoke)
        for i,case in enumerate(panel,1):
            r,t=execute(c,case,out/suite.stem(case),ctx,budget,smoke);rows.extend(r);cert.extend(t)
            print(f'Calibration {i}/{len(panel)}: {suite.stem(case)}',flush=True)
        pd.DataFrame(rows).to_csv(out/'raw-rollouts.csv',index=False)
        pd.DataFrame(cert).to_csv(out/'raw-certificates.csv',index=False)
        if not smoke:summarize(pd.DataFrame(rows),c,out)
        for p,digest in prior.items():assert core.sha256(p)==digest,p
        assert context(c)==ctx;budget.check()
        repeated=sum(r['repeated_setting'] for r in rows if r['calibration']!='clean')
        core.write_json(out/'verification.json',dict(complete=True,smoke=smoke,cases=len(panel),
            actual_paths=len(rows),certificates=len(cert),unique_fresh_noisy_settings=len(cert)-repeated,
            repeated_noisy_settings=repeated,clean_parent_repeats=len(panel),new_fits=0,
            prior_files_unchanged=len(prior),environment=core.environment(),budget=budget.close()))
        seal(out,c,ctx,purpose='Coordinate SD noise main' if not smoke else 'Excluded pipeline smoke')
    except Exception as exc:
        core.write_json(out/'failure.json',dict(error=repr(exc),budget=budget.close()))
        seal(out,c,ctx,purpose='Preserved incomplete calibration attempt');raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default=CONFIG);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--smoke',action='store_true');a=p.parse_args();run(read(a.config),a.out,a.smoke)
