"""Preregistered III-D pathway sensitivity and specificity map."""
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
from edge_panel import execute as execute_base
from pathway_masks import target_mask,audit_mask


def validate(c):
    if core.fingerprint(c)!=CONFIG_HASH:raise ValueError('Changed preregistered configuration')


def anatomy(ci):
    directory=Path(f'data/flywire_783_mb_left_kc512_s{ci}')
    raw,ids,_=core.load_connectome(directory);raw.sort_indices()
    roles,_=core.load_roles(directory,ids)
    return raw,ids,np.asarray(roles)


def bank_masks(c,ci,block):
    with np.load(Path(c['mask_bank'])/f'c{ci}_s{block["seed"]}.npz') as z:return dict(z)


def eligible(c,family):
    m=read(Path(c['mask_bank'])/'minimum-overlap.json')
    return all(m[str(ci)][family]['eligible'] for ci in c['circuit_seeds'])


def execute(c,ci,block,arm,centrality,mask_override=None):
    raw,_,roles=anatomy(ci)
    mask=bank_masks(c,ci,block)[arm] if mask_override is None else mask_override
    a,g,m,n,w=execute_base(c,ci,block,arm,centrality,mask_override=mask)
    family=None if arm=='intact' else arm.split('_control')[0]
    g.update(pathway=family,minimum_overlap=None,specificity_eligible=None)
    if family:
        target=target_mask(raw,roles,family);minimum=read(Path(c['mask_bank'])/'minimum-overlap.json')[str(ci)][family]
        g.update(pathway_target_count=int(target.sum()),minimum_overlap=minimum['overlap'],specificity_eligible=eligible(c,family),
            matching=audit_mask(raw,target,mask,c,minimum['overlap'] if '_control' in arm else None))
    return a,g,m,n,w


def seal(root,c,context,**extra):
    core.write_json(root/'manifest.json',dict(config=c,context=context,timestamp=datetime.now(timezone.utc).isoformat(),**extra,
        artifacts={p.relative_to(root).as_posix():core.sha256(p) for p in root.rglob('*') if p.is_file()}))


def baseline(c,centrality):
    return kc_ablation.baseline()


def family_rows(frame,family):
    return frame[frame.arm==family]


def summarize(rows,c,out):
    f=pd.DataFrame(rows);f.to_csv(out/'raw-lag-table.csv',index=False)
    metrics=['test_accuracy','train_accuracy','frequency_excess','null_excess','r2_vs_frequency','training_mse']
    cases=f[f.lag.isin(c['primary_lags'])].groupby(['cohort','seed','circuit_seed','arm','mode'])[metrics].mean().reset_index()
    blocks=cases.groupby(['cohort','seed','arm','mode'])[metrics].mean().reset_index()
    cases.to_csv(out/'raw-past-table.csv',index=False);blocks.to_csv(out/'seed-blocks.csv',index=False)
    statistics={};gates={};sensitivity={};pairs=[];modulepairs=[]
    for cohort in ['discovery','confirmation']:
        base=blocks[(blocks.cohort==cohort)&(blocks.arm=='intact')].set_index('seed')
        access=bool((base.frequency_excess>=.05).all() and (base.null_excess>=.05).all() and base.r2_vs_frequency.mean()>0)
        for mode in ['refit','frozen']:
            sub=blocks[(blocks.cohort==cohort)&(blocks['mode']==mode)]
            for family in c['families']:
                target=family_rows(sub,family).groupby('seed').test_accuracy.mean()
                control=sub[sub.arm.isin([family+'_control'+str(j) for j in range(3)])].groupby('seed').test_accuracy.mean()
                impairment=base.test_accuracy-target;excess=control-target;key=f'{cohort}/{mode}/{family}'
                gate=bool(access and impairment.mean()>=.05 and excess.mean()>=.05 and (impairment>0).all() and (excess>0).all())
                gates[key]=bool(gate and eligible(c,family));sensitivity[key]=bool(access and impairment.mean()>=.05 and (impairment>0).all());statistics[key]=dict(target=estimate(target,c),control=estimate(control,c),
                    impairment=estimate(impairment,c),excess_impairment=estimate(excess,c),intact_access=access,specificity_eligible=eligible(c,family),numeric_specificity_gate=gate)
                pairs.extend(dict(cohort=cohort,seed=int(seed),mode=mode,family=family,intact=float(base.test_accuracy[seed]),
                    target=float(target[seed]),control=float(control[seed]),impairment=float(impairment[seed]),excess_impairment=float(excess[seed])) for seed in target.index)
    pd.DataFrame(pairs).to_csv(out/'paired-differences.csv',index=False)
    summary=dict(statistics=statistics,gates=gates,confirmed=[f'{mode}/{family}' for mode in ['refit','frozen'] for family in c['families']
        if all(gates[f'{cohort}/{mode}/{family}'] for cohort in ['discovery','confirmation'])],cohorts_pooled=False,
        primary_DAN_MBON_refit=all(gates[f'{cohort}/refit/DAN_MBON'] for cohort in ['discovery','confirmation']),
        sensitivity_gates=sensitivity,refit_candidates=[family for family in c['families'] if all(gates[f'{cohort}/refit/{family}'] for cohort in ['discovery','confirmation'])],strength_matched=True)
    core.write_json(out/'summary.json',summary);plot(f,c,statistics,out)
    return summary


def plot(f,c,stats,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,cohort in zip(axes,['discovery','confirmation']):
        values=np.array([[stats[f'{cohort}/{mode}/{g}']['excess_impairment']['mean']*100 for mode in ['frozen','refit']] for g in c['families']])
        im=ax.imshow(values,cmap='RdBu_r',vmin=-55,vmax=55,aspect='auto')
        for (i,j),v in np.ndenumerate(values):ax.text(j,i,f'{v:+.2f}',ha='center',va='center')
        ax.set(xticks=[0,1],xticklabels=['Frozen','Refit'],yticks=range(5),yticklabels=[g.replace('_',' -> ')+(' *' if not eligible(c,g) else '') for g in c['families']],title=cohort)
    fig.colorbar(im,ax=axes,label='Control minus target accuracy (pp)',shrink=.8)
    fig.suptitle('* High unavoidable overlap: not eligible for pathway specificity')
    fig.savefig(out/'pathway-sensitivity.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for ax,cohort in zip(axes,['discovery','confirmation']):
        for j,family in enumerate(c['families']):
            q=f[(f.cohort==cohort)&(f['mode']=='refit')&f.lag.isin(c['primary_lags'])]
            a=q[q.arm==family].groupby('seed').test_accuracy.mean();b=q[q.arm.isin([family+'_control'+str(k) for k in range(3)])].groupby('seed').test_accuracy.mean()
            delta=(b-a)*100;ax.scatter(np.full(3,j),delta,color='steelblue' if eligible(c,family) else 'gray');ax.plot([j-.15,j+.15],[delta.mean()]*2,color='black')
        ax.axhline(0,color='gray',lw=.6);ax.axhline(5,color='red',ls='--');ax.set(xticks=range(5),xticklabels=c['families'],ylabel='Matched control minus pathway cut (pp)',title=cohort+' / refit');ax.tick_params(axis='x',rotation=25)
    fig.savefig(out/'pathway-paired.png',dpi=180);plt.close(fig)
    positions={'KC':(-1,0),'MBON':(1,0),'DAN':(0,1),'APL':(0,-1)}
    fig,ax=plt.subplots(figsize=(10,7),layout='constrained')
    for name,pos in positions.items():ax.scatter(*pos,s=1600,color='#e5edf4',edgecolors='black',zorder=3);ax.text(*pos,name,ha='center',va='center',zorder=4)
    for j,family in enumerate(c['families']):
        pre,post=family.split('_');a=np.array(positions[pre]);b=np.array(positions[post]);d=stats[f'confirmation/refit/{family}']['excess_impairment']['mean']*100
        bend=.15 if family in ['KC_APL','APL_KC'] else 0
        arrow=FancyArrowPatch(a,b,connectionstyle=f'arc3,rad={bend}',arrowstyle='-|>',mutation_scale=18,shrinkA=23,shrinkB=23,lw=2,color='#a33b30' if eligible(c,family) else '#888888',linestyle='-' if eligible(c,family) else '--');ax.add_patch(arrow)
        mid=(a+b)/2;shift=np.array([-(b-a)[1],(b-a)[0]])*.14
        ax.text(*(mid+shift),f'{pre}->{post}\n{d:+.2f} pp',ha='center',va='center',bbox=dict(facecolor='white',edgecolor='none',alpha=.85),fontsize=9)
    ax.set(xlim=(-1.6,1.6),ylim=(-1.45,1.5),title='Five tested pathways in the computational model\nConfirmation refit: matched control minus target cut');ax.axis('off')
    fig.savefig(out/'pathway-network.png',dpi=180);plt.close(fig)


@threadpool_limits.wrap(limits=1)
def run(config,out):
    c=read(config);validate(c);assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip(),'Commit before execution'
    check(Path(c['mask_bank']));check(Path(c['source']));assert core.sha256(Path(c['source'])/'manifest.json')==c['source_manifest_sha256']
    context=core.source_context(c)
    for p in Path(c['mask_bank']).rglob('*'):
        if p.is_file():context['source_sha256'][p.as_posix()]=core.sha256(p)
    for p in [config,Path('docs/pathway-memory-seed-audit.json'),Path('docs/pathway-memory-graph-audit.json')]+[Path('scripts')/(n+'.py') for n in ['pathway_memory','pathway_masks','verify_pathway_memory','edge_panel','edge_masks','structural_k4','kc_ablation','kc_frozen','alphabet_memory','frozen_state_probe','normalization_transfer','context_memory','verify_neuron_panel']]:
        context['source_sha256'][p.as_posix()]=core.sha256(p)
    for p,h in context['source_sha256'].items():
        assert hashlib.sha256(subprocess.check_output(['git','show',context['git_commit']+':'+p])).hexdigest()==h,('Uncommitted source bytes',p)
    out.mkdir(parents=True,exist_ok=False);budget=Budget(c)
    try:
        prior={p:core.sha256(p) for p in subprocess.check_output(['git','ls-files','results/'],text=True).splitlines()}
        core.write_json(out/'prior-results-sha256.json',prior);core.write_json(out/'config.json',c)
        centrality={}
        for ci in c['circuit_seeds']:
            raw,ids,roles=anatomy(ci);centrality[ci]=np.zeros(raw.nnz);dest=out/'graphs'/str(ci);dest.mkdir(parents=True)
            sparse.save_npz(dest/'raw.npz',raw)
            core.write_json(dest/'ids.json',dict(ids=ids,roles=roles.tolist(),raw_sha256=core.weight_hash(raw)))
        core.write_json(out/'baseline.json',baseline(c,centrality[701]));print('Archived intact baseline exactly reproduced',flush=True)
        rows=[];neural=[];audit=[];count=0;frozen_count=0
        for cohort,settings in c['cohorts'].items():
            cc=dict(c,**settings)
            for block in cc['blocks']:
                for ci in cc['circuit_seeds']:
                    raw,ids,roles=anatomy(ci);arms=list(bank_masks(c,ci,block))
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
                    print(f'{cohort} c{ci} s{block["seed"]}:21 cases fully replayed,20 frozen checks',flush=True)
        summary=summarize(rows,c,out);pd.DataFrame(neural).to_csv(out/'neural-diagnostics.csv',index=False);pd.DataFrame(audit).to_csv(out/'edge-audit.csv',index=False)
        for p,h in {**prior,**context['source_sha256']}.items():assert core.sha256(p)==h,p
        assert(count,frozen_count)==(273,260);budget.check();usage=budget.close()
        core.write_json(out/'verification.json',dict(full_replays=count,independent_refits=count,frozen_evaluations=frozen_count,prior_results_unchanged=len(prior),budget=usage,source_matches_committed_bytes=True))
        seal(out,c,context,environment=core.environment());print(dict(primary=summary['primary_DAN_MBON_refit'],confirmed=summary['confirmed'],budget=usage),flush=True)
    except Exception as exc:
        core.write_json(out/'failure.json',dict(error=repr(exc)));raise
    finally:budget.close()


CONFIG_HASH='937d7260da6ef253294ea4d9c4dcd1ddbd11cf5eeecec8612928ec3272b7b86c'
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=Path('configs/pathway_memory.json'));p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();run(args.config,args.out)
