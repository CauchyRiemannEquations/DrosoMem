"""Fresh paired-seed confirmation using immutable numerical experiment modules."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import time
import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits
from alphabet_memory import read,check
from frozen_state_probe import Budget,verify_fold
from context_memory import estimate
from flying.training import whole_brain_memory as core
import normalization_control as simulation
import moment_alignment as alignment

ARMS=['renormalized','fixed_original']


def validate(c):
    old=read('configs/normalization_control.json')
    keys=['alphabet_size','warmup','train_samples','test_samples','lags','primary_lags','alpha','swaps_per_edge',
          'normalization','schedule','gain','leak','input_fraction','input_amplitude','activity_epsilon','decay_steps',
          'perturbation_steps','perturbation_amplitude','forgetting_ratio','circuit_seeds']
    for k in keys:
        if c[k]!=old[k]:raise ValueError('Unregistered change: '+k)
    assert c['conditions']==ARMS and c['directions']==[ARMS,ARMS[::-1]]
    assert c['alignment_margin']==c['access_margin']==c['retention_tolerance']==.05 and c['scale_floor']==1e-5
    expected=[dict(seed=55142+i,train_seed=56142+i,test_seed=57142+i,rewire_seed=58142+i,perturb_seed=59142+i) for i in range(3)]
    assert c['blocks']==expected


def archived_baselines(c):
    original=read('configs/normalization_control.json');records=[]
    for cohort in ['main','smoke']:
        setting=dict(original)
        if cohort=='smoke':setting.update(original['smoke'])
        block=setting['blocks'][0];paired={}
        for arm in ARMS:
            path=Path('results/normalization_control_'+cohort)/f'{arm}_c701_s{block["seed"]}';check(path)
            a,g,m,n,_,_=simulation.execute(setting,701,block,arm)
            with np.load(path/'checkpoint.npz',allow_pickle=False) as saved:
                for key in saved.files:np.testing.assert_array_equal(saved[key],a[key],err_msg=key)
            assert read(path/'metrics.json')==m and read(path/'neural.json')==n
            assert read(path/'graph.json')['weight_sha256']==g['weight_sha256']
            verify_fold(a);paired[arm]=a
            records.append(dict(cohort=cohort,arm=arm,path=path.as_posix(),manifest_sha256=core.sha256(path/'manifest.json'),
                arrays_exact=len(a),independent_source_refit=True,
                expected_primary_accuracy=float(np.mean([r['test_accuracy'] for r in m if r['lag'] in c['primary_lags']])),
                actual_primary_accuracy=float(np.mean([r['test_accuracy'] for r in m if r['lag'] in c['primary_lags']]))))
        if cohort=='smoke':
            for sa,ta in c['directions']:
                a=alignment.transfer(paired[sa],paired[ta]);alignment.verify_transfer(paired[sa],paired[ta],a)
                path=Path('results/moment_alignment/smoke/c701_s34001')/(sa+'_to_'+ta);check(path)
                with np.load(path/'checkpoint.npz',allow_pickle=False) as saved:
                    for key in saved.files:np.testing.assert_array_equal(saved[key],a[key],err_msg=key)
                records.append(dict(cohort='smoke_transfer',direction=sa+'_to_'+ta,path=path.as_posix(),manifest_sha256=core.sha256(path/'manifest.json'),all_arrays_exact=True))
    return records


def summarize(rows,out,c):
    frame=pd.DataFrame(rows);frame.to_csv(out/'raw-lag-table.csv',index=False)
    raw=frame[frame.lag.isin(c['primary_lags'])].groupby(['direction','seed','circuit_seed'])[alignment.METRICS].mean().reset_index()
    raw.to_csv(out/'raw-past-table.csv',index=False)
    blocks=raw.groupby(['direction','seed'])[alignment.METRICS].mean().reset_index();blocks.to_csv(out/'seed-blocks.csv',index=False)
    cells={};gates={}
    for direction,b in blocks.groupby('direction'):
        cells[direction]={};gates[direction]=alignment.decision(b,'confirmation',c)
        for metric in alignment.METRICS:
            stat=estimate(b[metric],c)
            if not (metric.startswith('transfer_minus_') or metric=='aligned_minus_unaligned'):
                stat={k:v for k,v in stat.items() if k not in ['paired_dz','wins','ties','losses']}
            cells[direction][metric]=stat
    result=dict(cohort='fresh_confirmation',n_blocks=3,cells=cells,gates=gates,
        confirmed_improvement=[d for d,g in gates.items() if g['alignment_improvement']],
        confirmed_portability=[d for d,g in gates.items() if g['portable_decoding']])
    core.write_json(out/'summary.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.4))
    for direction,color in zip(['_to_'.join(p) for p in c['directions']],['#df7126','#269568']):
        label=direction.replace('renormalized','R').replace('fixed_original','F')
        v=frame[frame.direction==direction].groupby('lag')[['test_accuracy','unaligned_accuracy','target_refit_accuracy']].mean().reindex(c['lags'])
        for col,style,name in [('test_accuracy','o-',label+' aligned'),('unaligned_accuracy',':',label+' unaligned'),('target_refit_accuracy','--',label+' target refit')]:
            axes[0].plot(range(len(v)),v[col],style,color=color,label=name)
        b=blocks[blocks.direction==direction]
        axes[1].plot(b.seed.astype(str),100*b.aligned_minus_unaligned,'o-',color=color,label=label)
    axes[0].set(xticks=range(len(c['lags'])),xticklabels=c['lags'],xlabel='Symbol lag',ylabel='Independent test accuracy',ylim=(0,1.03))
    axes[0].axhline(.25,color='gray',ls=':');axes[0].legend(fontsize=8)
    axes[1].axhline(5,color='gray',ls='--',label='Mean improvement threshold');axes[1].set(xlabel='Fresh mapping seed',ylabel='Aligned − unaligned accuracy (pp)');axes[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'fresh-alignment.png',dpi=180);plt.close(fig)
    return result


@threadpool_limits.wrap(limits=1)
def run(config,out):
    c=read(config);validate(c);out.mkdir(parents=True,exist_ok=False);budget=Budget(c)
    try:
        context=core.source_context(c)
        for p in [str(config),'docs/fresh-alignment-seed-audit.json','scripts/fresh_alignment.py','scripts/verify_fresh_alignment.py',
                  'scripts/normalization_control.py','scripts/structural_k4.py','scripts/moment_alignment.py','scripts/normalization_transfer.py',
                  'scripts/frozen_state_probe.py','scripts/alphabet_memory.py','scripts/context_memory.py']:
            context['source_sha256'][p]=core.sha256(p)
        prior={p:core.sha256(p) for p in subprocess.check_output(['git','ls-files','results/'],text=True).splitlines()}
        core.write_json(out/'prior-results-sha256.json',prior);core.write_json(out/'config.json',c)
        core.write_json(out/'baseline-smoke.json',archived_baselines(c));print('Archived full baselines and smoke transfers reproduced exactly',flush=True)
        rows=[];condition_rows=[];neural_rows=[];runs=0
        for block in c['blocks']:
            for circuit in c['circuit_seeds']:
                paired={};graphs={};paths={}
                for arm in ARMS:
                    budget.check();start=time.perf_counter();dest=out/'conditions'/f'{arm}_c{circuit}_s{block["seed"]}';dest.mkdir(parents=True)
                    a,g,m,n,raw,w=simulation.execute(c,circuit,block,arm)
                    np.savez_compressed(dest/'checkpoint.npz',**a);sparse.save_npz(dest/'raw-graph.npz',raw);sparse.save_npz(dest/'weights.npz',w)
                    for name,value in [('graph',g),('metrics',m),('neural',n)]:core.write_json(dest/(name+'.json'),value)
                    repeat,g2,m2,n2,_,_=simulation.execute(c,circuit,block,arm)
                    assert g==g2 and m==m2 and n==n2
                    with np.load(dest/'checkpoint.npz',allow_pickle=False) as saved:
                        for key in saved.files:np.testing.assert_array_equal(saved[key],repeat[key],err_msg=key)
                    verify_fold(a);paired[arm]=a;graphs[arm]=g;paths[arm]=dest
                    identity=dict(arm=arm,circuit_seed=circuit,**block)
                    condition_rows.extend(dict(**identity,**r) for r in m);neural_rows.append(dict(**identity,**n))
                    core.write_json(dest/'manifest.json',dict(identity=identity,config=c,context=context,exact_full_replay=True,independent_refit=True,
                        real_task_heads=11,null_task_heads=11,seconds_including_replay=time.perf_counter()-start,
                        artifacts={p.name:core.sha256(p) for p in dest.iterdir() if p.is_file()}))
                    runs+=1
                for sa,ta in c['directions']:
                    source,target=paired[sa],paired[ta];alignment.audit_pair(source,target,graphs[sa],graphs[ta])
                    a=alignment.transfer(source,target);alignment.verify_transfer(source,target,a)
                    direction=sa+'_to_'+ta;dest=out/'transfers'/f'c{circuit}_s{block["seed"]}'/direction;dest.mkdir(parents=True)
                    np.savez_compressed(dest/'checkpoint.npz',**a);replay=alignment.transfer(source,target)
                    with np.load(dest/'checkpoint.npz',allow_pickle=False) as saved:
                        for key in saved.files:np.testing.assert_array_equal(saved[key],replay[key],err_msg=key)
                    measured=alignment.measures(a,c['lags']);identity=dict(cohort='fresh_confirmation',direction=direction,circuit_seed=circuit,**block)
                    rows.extend(dict(**identity,**r) for r in measured);core.write_json(dest/'metrics.json',measured)
                    core.write_json(dest/'manifest.json',dict(identity=identity,source_path=paths[sa].as_posix(),target_path=paths[ta].as_posix(),
                        source_manifest_sha256=core.sha256(paths[sa]/'manifest.json'),target_manifest_sha256=core.sha256(paths[ta]/'manifest.json'),
                        transfer_target_supervised_fits=0,target_moment_parameters=96,exact_replay=True,independent_source_only_solver=True,
                        artifacts={p.name:core.sha256(p) for p in dest.iterdir() if p.is_file()}))
                print(f'fresh c{circuit} s{block["seed"]}: both conditions and directions saved/verified',flush=True)
        assert runs==12 and len(rows)==132
        summary=summarize(rows,out,c)
        pd.DataFrame(condition_rows).to_csv(out/'condition-lag-table.csv',index=False);pd.DataFrame(neural_rows).to_csv(out/'neural-diagnostics.csv',index=False)
        for p,h in {**prior,**context['source_sha256']}.items():assert core.sha256(p)==h,p
        budget.check();usage=budget.close()
        core.write_json(out/'verification.json',dict(new_condition_runs=runs,real_task_heads=132,null_task_heads=132,full_condition_replays=12,
            exact_transfer_replays=12,independent_condition_refits=12,independent_transfer_solver_checks=12,transfer_target_supervised_fits=0,
            prior_files_unchanged=len(prior),budget=usage))
        core.write_json(out/'manifest.json',dict(config=c,context=context,cohort='fresh_confirmation',timestamp=datetime.now(timezone.utc).isoformat(),environment=core.environment(),
            artifacts={p.relative_to(out).as_posix():core.sha256(p) for p in out.rglob('*') if p.is_file()}))
        print(summary['gates'],usage,flush=True)
    finally:budget.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=Path('configs/fresh_alignment.json'));p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();run(args.config,args.out)
