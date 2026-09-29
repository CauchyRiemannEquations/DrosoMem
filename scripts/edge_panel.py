"""Preregistered ACT III-B edge ablation, frozen/refit and fresh confirmation."""
import argparse
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import subprocess,time,hashlib
import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits
from flying.training import whole_brain_memory as core
from alphabet_memory import read,check,SymbolEncoder
from frozen_state_probe import Budget,labels,fit_fold,measures
from normalization_transfer import verify_transfer
from context_memory import estimate
from edge_masks import edge_betweenness,select_masks,cut_edges
import structural_k4,kc_ablation,kc_frozen


def validate(c):
    if core.fingerprint(c)!=CONFIG_HASH:raise ValueError('Changed preregistered configuration')


def anatomy(ci):
    directory=Path(f'data/flywire_783_mb_left_kc512_s{ci}')
    raw,ids,_=core.load_connectome(directory);raw.sort_indices()
    roles,_=core.load_roles(directory,ids)
    return raw,ids,np.asarray(roles)


def execute(c,ci,block,arm,centrality,mask_override=None):
    base,obs,g,raw=structural_k4.build(c,ci,block['seed'])
    _,ids,roles=anatomy(ci);bank=base.encoder.patterns.copy();co=raw.tocoo()
    mask=select_masks(raw,ids,roles,centrality,block,c)[arm] if mask_override is None else mask_override
    assert base.weights.nnz==raw.nnz
    np.testing.assert_array_equal(base.weights.indices,raw.indices);np.testing.assert_array_equal(base.weights.indptr,raw.indptr)
    weights=cut_edges(base.weights,mask);model=core.TimedReservoir(weights,SymbolEncoder(bank),roles,c['leak'],c['schedule'])
    dan=(roles[co.row]=='DAN')|(roles[co.col]=='DAN');within=roles[co.row]==roles[co.col]
    selected=np.where(mask)[0];hist=Counter(zip(np.sign(co.data[mask]).astype(int).tolist(),np.digitize(abs(co.data[mask]),c['weight_boundaries']).tolist()))
    g.update(arm=arm,prelesion_weight_sha256=g['weight_sha256'],weight_sha256=core.weight_hash(weights),
        removed_count=int(mask.sum()),remaining_edges=weights.nnz,removed_raw_l1=float(abs(co.data[mask]).sum()),
        removed_normalized_l1=float(abs(base.weights.data[mask]).sum()),DAN_overlap=int((mask&dan).sum()),DAN_count=int(dan.sum()),
        removed_within_role=int((mask&within).sum()),removed_between_role=int((mask&~within).sum()),
        signed_weight_histogram={str(k):v for k,v in sorted(hist.items())},
        endpoint_roles=dict(Counter(f'{roles[co.col[i]]}->{roles[co.row[i]]}' for i in selected)),
        removed_edge_ids=[[ids[co.col[i]],ids[co.row[i]]] for i in selected],
        input_unchanged=True,observation_unchanged=True,renormalized_after_lesion=False,neurons_removed=0)
    a=dict(edge_mask=mask,input_patterns=bank,intended_input_patterns=bank.copy(),observed_indices=obs)
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
    return a,g,metrics,neural,weights


def seal(root,c,context,**extra):
    core.write_json(root/'manifest.json',dict(config=c,context=context,timestamp=datetime.now(timezone.utc).isoformat(),**extra,
        artifacts={p.relative_to(root).as_posix():core.sha256(p) for p in root.rglob('*') if p.is_file()}))


def baseline(c,centrality):
    result=kc_ablation.baseline();path=Path(c['source'])/'confirmation/DAN_c701_s91142';m=check(path)
    raw,_,roles=anatomy(701);co=raw.tocoo();mask=(roles[co.row]=='DAN')|(roles[co.col]=='DAN')
    a,g,metrics,n,w=execute(c,701,m['identity'],'DAN',centrality,mask)
    with np.load(path/'checkpoint.npz') as saved:
        common=sorted(set(a)&set(saved.files))
        for key in common:np.testing.assert_array_equal(a[key],saved[key],err_msg=key)
    assert metrics==read(path/'metrics.json');assert core.weight_hash(w)==core.weight_hash(sparse.load_npz(path/'weights.npz'))
    return dict(intact=result,DAN=dict(source=path.as_posix(),manifest_sha256=core.sha256(path/'manifest.json'),
        matching_arrays=common,exact_common_arrays=True,exact_metrics=True,exact_weights=True,metrics=metrics))


def family_rows(frame,family):
    return frame[frame.arm.str.startswith(family)] if family in ['within','between'] else frame[frame.arm==family]


def summarize(rows,c,out):
    f=pd.DataFrame(rows);f.to_csv(out/'raw-lag-table.csv',index=False)
    metrics=['test_accuracy','train_accuracy','frequency_excess','null_excess','r2_vs_frequency','training_mse']
    cases=f[f.lag.isin(c['primary_lags'])].groupby(['cohort','seed','circuit_seed','arm','mode'])[metrics].mean().reset_index()
    blocks=cases.groupby(['cohort','seed','arm','mode'])[metrics].mean().reset_index()
    cases.to_csv(out/'raw-past-table.csv',index=False);blocks.to_csv(out/'seed-blocks.csv',index=False)
    statistics={};gates={};pairs=[];modulepairs=[]
    for cohort in ['discovery','confirmation']:
        base=blocks[(blocks.cohort==cohort)&(blocks.arm=='intact')].set_index('seed')
        access=bool((base.frequency_excess>=.05).all() and (base.null_excess>=.05).all() and base.r2_vs_frequency.mean()>0)
        for mode in ['refit','frozen']:
            sub=blocks[(blocks.cohort==cohort)&(blocks['mode']==mode)]
            for family in c['families']:
                target=family_rows(sub,family).groupby('seed').test_accuracy.mean()
                control=sub[sub.arm.str.startswith('dan_control' if family=='DAN' else 'uniform')].groupby('seed').test_accuracy.mean()
                impairment=base.test_accuracy-target;excess=control-target;key=f'{cohort}/{mode}/{family}'
                gate=bool(access and impairment.mean()>=.05 and excess.mean()>=.05 and (impairment>0).all() and (excess>0).all())
                gates[key]=gate;statistics[key]=dict(target=estimate(target,c),control=estimate(control,c),
                    impairment=estimate(impairment,c),excess_impairment=estimate(excess,c),intact_access=access)
                pairs.extend(dict(cohort=cohort,seed=int(seed),mode=mode,family=family,intact=float(base.test_accuracy[seed]),
                    target=float(target[seed]),control=float(control[seed]),impairment=float(impairment[seed]),excess_impairment=float(excess[seed])) for seed in target.index)
            a=family_rows(sub,'within').groupby('seed').test_accuracy.mean();b=family_rows(sub,'between').groupby('seed').test_accuracy.mean()
            modulepairs.extend(dict(cohort=cohort,mode=mode,seed=int(seed),within_minus_between=float(a[seed]-b[seed])) for seed in a.index)
    pd.DataFrame(pairs).to_csv(out/'paired-differences.csv',index=False);pd.DataFrame(modulepairs).to_csv(out/'within-between.csv',index=False)
    summary=dict(statistics=statistics,gates=gates,confirmed=[f'{mode}/{family}' for mode in ['refit','frozen'] for family in c['families']
        if all(gates[f'{cohort}/{mode}/{family}'] for cohort in ['discovery','confirmation'])],cohorts_pooled=False,
        primary_DAN_refit=all(gates[f'{cohort}/refit/DAN'] for cohort in ['discovery','confirmation']),
        within_minus_between={f'{cohort}/{mode}':estimate(b.within_minus_between,c) for (cohort,mode),b in pd.DataFrame(modulepairs).groupby(['cohort','mode'])})
    core.write_json(out/'summary.json',summary);plot(f,c,statistics,out)
    return summary


def plot(f,c,stats,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,5.2),layout='constrained')
    for ax,cohort in zip(axes,['discovery','confirmation']):
        values=np.array([[stats[f'{cohort}/{mode}/{g}']['excess_impairment']['mean']*100 for mode in ['frozen','refit']] for g in c['families']])
        im=ax.imshow(values,cmap='RdBu_r',vmin=-60,vmax=60,aspect='auto')
        for (i,j),v in np.ndenumerate(values):ax.text(j,i,f'{v:+.2f}',ha='center',va='center')
        ax.set(xticks=[0,1],xticklabels=['Frozen','Refit'],yticks=range(len(c['families'])),yticklabels=c['families'],title=cohort)
    fig.colorbar(im,ax=axes,label='Control minus target accuracy (pp)',shrink=.8);fig.savefig(out/'edge-sensitivity.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for i,cohort in enumerate(['discovery','confirmation']):
        for j,mode in enumerate(['refit','frozen']):
            ax=axes[i,j];q=f[(f.cohort==cohort)&(f['mode']==mode)]
            for family in c['families']:
                y=family_rows(q,family).groupby('lag').test_accuracy.mean().reindex(c['lags']);ax.plot(range(len(y)),y,'o-',label=family,ms=3)
            y=q[q.arm.str.startswith('uniform')].groupby('lag').test_accuracy.mean().reindex(c['lags']);ax.plot(range(len(y)),y,'k--',label='uniform')
            ax.set(xticks=range(len(y)),xticklabels=c['lags'],xlabel='Past symbol lag',ylabel='Test accuracy',ylim=(0,1.02),title=f'{cohort} / {mode}');ax.legend(fontsize=7,ncol=2)
    fig.savefig(out/'edge-lag-curves.png',dpi=180);plt.close(fig)


@threadpool_limits.wrap(limits=1)
def run(config,out):
    c=read(config);validate(c);assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip(),'Commit before execution'
    check(Path(c['source']));assert core.sha256(Path(c['source'])/'manifest.json')==c['source_manifest_sha256']
    context=core.source_context(c)
    for p in [config,Path('docs/edge-panel-seed-audit.json'),Path('docs/edge-panel-graph-audit.json')]+[Path('scripts')/(n+'.py') for n in ['edge_panel','edge_masks','verify_edge_panel','structural_k4','kc_ablation','kc_frozen','alphabet_memory','frozen_state_probe','normalization_transfer','context_memory','verify_neuron_panel']]:
        context['source_sha256'][p.as_posix()]=core.sha256(p)
    for p,h in context['source_sha256'].items():
        assert hashlib.sha256(subprocess.check_output(['git','show',context['git_commit']+':'+p])).hexdigest()==h,('Uncommitted source bytes',p)
    out.mkdir(parents=True,exist_ok=False);budget=Budget(c)
    try:
        prior={p:core.sha256(p) for p in subprocess.check_output(['git','ls-files','results/'],text=True).splitlines()}
        core.write_json(out/'prior-results-sha256.json',prior);core.write_json(out/'config.json',c)
        centrality={}
        for ci in c['circuit_seeds']:
            raw,ids,roles=anatomy(ci);centrality[ci]=edge_betweenness(raw);dest=out/'graphs'/str(ci);dest.mkdir(parents=True)
            sparse.save_npz(dest/'raw.npz',raw);np.savez_compressed(dest/'centrality.npz',betweenness=centrality[ci])
            core.write_json(dest/'ids.json',dict(ids=ids,roles=roles.tolist(),raw_sha256=core.weight_hash(raw)))
        core.write_json(out/'baseline.json',baseline(c,centrality[701]));print('Archived intact and DAN full-neuron baseline exactly reproduced',flush=True)
        rows=[];neural=[];audit=[];count=0;frozen_count=0
        for cohort,settings in c['cohorts'].items():
            cc=dict(c,**settings)
            for block in cc['blocks']:
                for ci in cc['circuit_seeds']:
                    raw,ids,roles=anatomy(ci);arms=list(select_masks(raw,ids,roles,centrality[ci],block,c))
                    sp=out/cohort/f'intact_c{ci}_s{block["seed"]}';source=None
                    for arm in arms:
                        budget.check();start=time.perf_counter();dest=out/cohort/f'{arm}_c{ci}_s{block["seed"]}';dest.mkdir(parents=True,exist_ok=False)
                        a,g,m,n,w=execute(cc,ci,block,arm,centrality[ci]);b,g2,m2,n2,w2=execute(cc,ci,block,arm,centrality[ci])
                        assert g==g2 and m==m2 and n==n2 and core.weight_hash(w)==core.weight_hash(w2)
                        for key in a:np.testing.assert_array_equal(a[key],b[key],err_msg=key)
                        np.savez_compressed(dest/'checkpoint.npz',**a);sparse.save_npz(dest/'weights.npz',w)
                        for name,value in [('graph',g),('metrics',m),('neural',n)]:core.write_json(dest/f'{name}.json',value)
                        identity=dict(cohort=cohort,circuit_seed=ci,seed=block['seed'],arm=arm)
                        rows.extend(dict(**identity,mode='refit',**r) for r in m)
                        if arm=='intact':source=a
                        else:
                            for key in ['input_patterns','observed_indices','train_symbols','test_symbols','ytrain','ytest']:np.testing.assert_array_equal(source[key],a[key])
                            f,fm=kc_frozen.evaluate(source,a,c);verify_transfer(source,a,f)
                            np.savez_compressed(dest/'frozen.npz',**f);core.write_json(dest/'frozen-metrics.json',fm)
                            rows.extend(dict(**identity,mode='frozen',**r) for r in fm);frozen_count+=1
                        neural.append(dict(**identity,**n));audit.append({**identity,**g});count+=1
                        seal(dest,cc,context,identity=dict(**identity,train_seed=block['train_seed'],test_seed=block['test_seed'],edge_seeds=block['edge_seeds']),
                            source_path=sp.as_posix() if arm!='intact' else None,source_manifest_sha256=core.sha256(sp/'manifest.json') if arm!='intact' else None,
                            exact_full_replay=True,independent_lstsq=True,coefficients_per_lag=196,seconds=time.perf_counter()-start)
                    print(f'{cohort} c{ci} s{block["seed"]}:17 cases fully replayed,16 frozen checks',flush=True)
        summary=summarize(rows,c,out);pd.DataFrame(neural).to_csv(out/'neural-diagnostics.csv',index=False);pd.DataFrame(audit).to_csv(out/'edge-audit.csv',index=False)
        for p,h in {**prior,**context['source_sha256']}.items():assert core.sha256(p)==h,p
        assert(count,frozen_count)==(221,208);budget.check();usage=budget.close()
        core.write_json(out/'verification.json',dict(full_replays=count,independent_refits=count,frozen_evaluations=frozen_count,prior_results_unchanged=len(prior),budget=usage,source_matches_committed_bytes=True))
        seal(out,c,context,environment=core.environment());print(dict(primary=summary['primary_DAN_refit'],confirmed=summary['confirmed'],budget=usage),flush=True)
    finally:budget.close()


CONFIG_HASH='940c158d9ff0d46d35b3b8996ba06d45798347740550cd6a5f66d6ae96a46b0e'
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=Path('configs/edge_panel.json'));p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();run(args.config,args.out)
