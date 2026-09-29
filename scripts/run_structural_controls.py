"""Preregistered III-C structural controls with matched observation budgets."""
import argparse,subprocess,hashlib,time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits
from flying.training import whole_brain_memory as core
from alphabet_memory import read,check,SymbolEncoder,symbol_bank
from frozen_state_probe import Budget,labels,fit_fold,measures
from normalization_transfer import verify_transfer
from context_memory import estimate
from edge_panel import anatomy,seal
from structural_controls import generate,audit
import kc_ablation,kc_frozen

CONFIG_HASH='3cf5b57119fe85506336e4d51648f3b8d796c5651cb361546c15e677ba1ee16f'


def execute(c,ci,block,arm):
    raw,ids,roles=anatomy(ci)
    family='intact' if arm=='intact' else arm[:-1]
    seed=0 if family=='intact' else block['graph_seeds'][family][int(arm[-1])]
    rewired,log=generate(raw,roles,family,seed,c['swaps_per_edge'])
    graphaudit=audit(raw,rewired,roles,family)
    weights=core.normalize_condition(rewired,c['normalization'],c['gain']);weights.sort_indices()
    bank=symbol_bank(roles,block['seed'],c['input_fraction'],c['input_amplitude'])[:4]
    obs=np.flatnonzero(roles=='MBON');assert len(obs)==48
    model=core.TimedReservoir(weights,SymbolEncoder(bank),roles,c['leak'],c['schedule'])
    g=dict(arm=arm,family=family,threshold=5,**graphaudit,generation=log,
        original_raw_sha256=core.weight_hash(raw),raw_sha256=core.weight_hash(rewired),weight_sha256=core.weight_hash(weights),
        observed_root_ids=[ids[i] for i in obs],input_root_ids=[[ids[i] for i in np.flatnonzero(p)] for p in bank],
        input_mapping_sha256=core.fingerprint(dict(ids=ids,patterns=bank.tolist())))
    a=dict(input_patterns=bank,intended_input_patterns=bank.copy(),observed_indices=obs)
    for split in ['train','test']:
        symbols=np.random.default_rng(block[split+'_seed']).integers(0,4,c['warmup']+c[split+'_samples']).astype(np.uint8)
        features,diag=core.collect(model,obs,symbols,c['activity_epsilon'])
        a[split+'_symbols']=symbols;a[split+'_features']=features
        a.update({split+'_'+k:v for k,v in diag.items()})
        if split=='train':a['decay_full'],a['decay_observed']=core.decay_probe(model,obs,c['decay_steps'])
    w=c['warmup'];x=a['train_features'][w:]
    a.update(fit_fold(x,a['test_features'][w:],labels(a['train_symbols'],np.arange(w,len(a['train_symbols'])),c['lags']),
        labels(a['test_symbols'],np.arange(w,len(a['test_symbols'])),c['lags']),c['alpha']))
    metrics=measures([a],c['lags'])
    for j,row in enumerate(metrics):row['training_mse']=float(np.mean((a['train_scores'][:,j]-np.eye(4)[a['ytrain'][:,j]])**2))
    a['centered_singular_values']=np.linalg.svd(x-x.mean(0),compute_uv=False)
    neural=dict(**core.state_diagnostics(x),observed_sparsity=float(np.mean(abs(x)<=c['activity_epsilon'])),
        mean_mbon_norm=float(np.linalg.norm(x,axis=1).mean()),mean_active_neurons=float(a['train_active_counts'][w:].mean()),
        mean_observed_cosine=float(a['train_observed_cosine'][w-1:].mean()),clipped_features=int(np.count_nonzero(x.std(0)<1e-5)),
        zero_decay_baseline=bool(a['decay_observed'][0]==0),
        observed_decay_ratio32=float(a['decay_observed'][-1]/a['decay_observed'][0]) if a['decay_observed'][0] else None)
    for value in a.values():assert np.isfinite(value).all()
    assert np.max(abs(x))<=1+1e-12 and core.weight_hash(model.weights)==g['weight_sha256']
    return a,g,metrics,neural,weights,rewired


def summarize(rows,c,out):
    f=pd.DataFrame(rows);f.to_csv(out/'raw-lag-table.csv',index=False)
    metrics=['test_accuracy','train_accuracy','frequency_excess','null_excess','r2_vs_frequency','training_mse']
    cases=f[f.lag.isin(c['primary_lags'])].groupby(['cohort','seed','circuit_seed','arm','mode'])[metrics].mean().reset_index()
    blocks=cases.groupby(['cohort','seed','arm','mode'])[metrics].mean().reset_index()
    cases.to_csv(out/'raw-past-table.csv',index=False);blocks.to_csv(out/'seed-blocks.csv',index=False)
    statistics={};gates={};pairs=[]
    for cohort in ['discovery','confirmation']:
        base=blocks[(blocks.cohort==cohort)&(blocks.arm=='intact')].set_index('seed')
        access=bool((base.frequency_excess>=c['access_margin']).all() and (base.null_excess>=c['access_margin']).all() and base.r2_vs_frequency.mean()>0)
        for mode in ['refit','frozen']:
            for family in c['families']:
                q=blocks[(blocks.cohort==cohort)&(blocks['mode']==mode)&blocks.arm.isin([family+str(j) for j in range(3)])]
                control=q.groupby('seed').test_accuracy.mean();effect=base.test_accuracy-control
                key=f'{cohort}/{mode}/{family}'
                statistics[key]=dict(intact=estimate(base.test_accuracy,c),control=estimate(control,c),difference=estimate(effect,c),intact_access=access)
                gates[key]=bool(access and effect.mean()>=c['material_effect'] and (effect>0).all())
                pairs.extend(dict(cohort=cohort,mode=mode,family=family,seed=int(seed),intact=float(base.test_accuracy[seed]),control=float(control[seed]),difference=float(effect[seed])) for seed in control.index)
    pd.DataFrame(pairs).to_csv(out/'paired-differences.csv',index=False)
    confirmed=[f'{mode}/{family}' for mode in ['refit','frozen'] for family in c['families'] if all(gates[f'{cohort}/{mode}/{family}'] for cohort in ['discovery','confirmation'])]
    summary=dict(statistics=statistics,gates=gates,confirmed=confirmed,primary_role_refit='refit/role' in confirmed,cohorts_pooled=False)
    core.write_json(out/'summary.json',summary);plot(f,c,pd.DataFrame(pairs),out)
    return summary


def plot(frame,c,pairs,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,cohort in zip(axes,['discovery','confirmation']):
        q=pairs[(pairs.cohort==cohort)&(pairs['mode']=='refit')]
        for j,family in enumerate(c['families']):
            x=q[q.family==family].difference.to_numpy()*100
            ax.scatter(np.full(len(x),j),x,color='steelblue');ax.plot([j-.18,j+.18],[x.mean()]*2,color='black')
        ax.axhline(5,color='red',ls='--',label='Preregistered 5pp');ax.axhline(0,color='gray',lw=.6)
        ax.set(xticks=range(4),xticklabels=c['families'],ylabel='Intact minus control accuracy (pp)',title=cohort+' / refit');ax.legend()
    fig.savefig(out/'structural-differences.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for i,cohort in enumerate(['discovery','confirmation']):
        for j,mode in enumerate(['refit','frozen']):
            ax=axes[i,j];q=frame[(frame.cohort==cohort)&(frame['mode']==mode)]
            for family in c['families']:
                y=q[q.arm.isin([family+str(k) for k in range(3)])].groupby('lag').test_accuracy.mean().reindex(c['lags'])
                ax.plot(range(len(y)),y,'o-',ms=3,label=family)
            intact=frame[(frame.cohort==cohort)&(frame.arm=='intact')].groupby('lag').test_accuracy.mean().reindex(c['lags'])
            ax.plot(range(len(intact)),intact,'k--',label='intact refit');ax.axhline(.25,color='gray',lw=.5)
            ax.set(xticks=range(len(intact)),xticklabels=c['lags'],xlabel='Past symbol lag',ylabel='Test accuracy',ylim=(0,1.02),title=f'{cohort} / {mode}');ax.legend(fontsize=8)
    fig.savefig(out/'structural-lag-curves.png',dpi=180);plt.close(fig)


@threadpool_limits.wrap(limits=1)
def run(config,out):
    c=read(config);assert core.fingerprint(c)==CONFIG_HASH,'Unregistered configuration'
    assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip(),'Commit before execution'
    check(Path(c['source']));assert core.sha256(Path(c['source'])/'manifest.json')==c['source_manifest_sha256']
    context=core.source_context(c)
    for p in [config,Path('docs/structural-controls-seed-audit.json'),Path('docs/structural-controls-graph-audit.json')]+[Path('scripts')/(name+'.py') for name in ['alphabet_memory','context_memory','frozen_state_probe','normalization_transfer','edge_masks','structural_k4','kc_ablation','kc_frozen','edge_panel','structural_controls','run_structural_controls','verify_neuron_panel','verify_structural_controls']]:
        context['source_sha256'][p.as_posix()]=core.sha256(p)
    for p,h in context['source_sha256'].items():assert hashlib.sha256(subprocess.check_output(['git','show',context['git_commit']+':'+p])).hexdigest()==h,p
    out.mkdir(parents=True,exist_ok=False);budget=Budget(c)
    try:
        prior={p:core.sha256(p) for p in subprocess.check_output(['git','ls-files','results/'],text=True).splitlines()}
        core.write_json(out/'prior-results-sha256.json',prior);core.write_json(out/'config.json',c)
        core.write_json(out/'baseline.json',kc_ablation.baseline());print('Archived baseline exactly reproduced',flush=True)
        rows=[];neural=[];audits=[];count=0;fcount=0
        for cohort,settings in c['cohorts'].items():
            cc=dict(c,**settings)
            for block in cc['blocks']:
                for ci in cc['circuit_seeds']:
                    sp=out/cohort/f'intact_c{ci}_s{block["seed"]}';source=None
                    for arm in ['intact']+[family+str(j) for family in c['families'] for j in range(3)]:
                        budget.check();start=time.perf_counter();dest=out/cohort/f'{arm}_c{ci}_s{block["seed"]}';dest.mkdir(parents=True,exist_ok=False)
                        a,g,m,n,w,raw=execute(cc,ci,block,arm);b,g2,m2,n2,w2,raw2=execute(cc,ci,block,arm)
                        assert g==g2 and m==m2 and n==n2 and core.weight_hash(w)==core.weight_hash(w2) and core.weight_hash(raw)==core.weight_hash(raw2)
                        for key in a:np.testing.assert_array_equal(a[key],b[key],err_msg=key)
                        np.savez_compressed(dest/'checkpoint.npz',**a);sparse.save_npz(dest/'weights.npz',w);sparse.save_npz(dest/'raw.npz',raw)
                        for name,value in [('graph',g),('metrics',m),('neural',n)]:core.write_json(dest/f'{name}.json',value)
                        identity=dict(cohort=cohort,circuit_seed=ci,seed=block['seed'],arm=arm)
                        rows.extend(dict(**identity,mode='refit',**r) for r in m)
                        if arm=='intact':source=a
                        else:
                            for key in ['input_patterns','observed_indices','train_symbols','test_symbols','ytrain','ytest']:np.testing.assert_array_equal(source[key],a[key])
                            f,fm=kc_frozen.evaluate(source,a,c);verify_transfer(source,a,f)
                            np.savez_compressed(dest/'frozen.npz',**f);core.write_json(dest/'frozen-metrics.json',fm)
                            rows.extend(dict(**identity,mode='frozen',**r) for r in fm);fcount+=1
                        neural.append(dict(**identity,**n));audits.append({**identity,**g});count+=1
                        seal(dest,cc,context,identity=dict(**identity,train_seed=block['train_seed'],test_seed=block['test_seed'],graph_seeds=block['graph_seeds']),
                            source_path=sp.as_posix() if arm!='intact' else None,source_manifest_sha256=core.sha256(sp/'manifest.json') if arm!='intact' else None,
                            exact_full_replay=True,independent_lstsq=True,coefficients_per_lag=196,seconds=time.perf_counter()-start)
                    print(f'{cohort} c{ci} s{block["seed"]}:13 fits/replays,12 frozen',flush=True)
        summary=summarize(rows,c,out);pd.DataFrame(neural).to_csv(out/'neural-diagnostics.csv',index=False);pd.DataFrame(audits).to_csv(out/'graph-audit.csv',index=False)
        for p,h in {**prior,**context['source_sha256']}.items():assert core.sha256(p)==h,p
        assert(count,fcount)==(169,156);budget.check();usage=budget.close()
        core.write_json(out/'verification.json',dict(full_replays=count,independent_refits=count,frozen_evaluations=fcount,prior_results_unchanged=len(prior),budget=usage,source_matches_committed_bytes=True))
        seal(out,c,context,environment=core.environment());print(dict(primary=summary['primary_role_refit'],confirmed=summary['confirmed'],budget=usage),flush=True)
    except Exception as exc:
        core.write_json(out/'failure.json',dict(error=repr(exc),git_commit=context['git_commit']));raise
    finally:budget.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=Path('configs/structural_controls.json'));p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();run(args.config,args.out)
