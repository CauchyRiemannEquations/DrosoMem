"""Frozen-state diagnostic of local update versus fixed-code reward direction."""
import argparse,hashlib,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits
from flying.training import whole_brain_memory as core
from flying.brain.local_reward import LocalRewardPlasticity
from alphabet_memory import read,check,symbol_bank,SymbolEncoder
from local_reward import codes,fixed_scores
from edge_panel import anatomy
from frozen_state_probe import Budget
from context_memory import estimate
from pathway_memory import seal

CONFIG_HASH='3b762ded6d494351ee9f782ae06cc8cae66f6132d3d1997ca28cf2439c5db5d8'


def local_direction(weights,roles,bank,symbols,codebook,c,seed):
    model=LocalRewardPlasticity(weights,roles,SymbolEncoder(bank),leak=c['leak'],
        learning_rate=c['plastic_learning_rate'],floor=c['plastic_floor'],
        trace_decay=c['trace_decay'],activity_rate=c['activity_rate'],baseline_rate=c['baseline_rate'])
    initial=model.weights.data.copy();total=np.zeros(len(model.pre));rng=np.random.default_rng(seed);history=[]
    logs={k:[] for k in ['actions','rewards','advantages','trace_norms','proposal_norms']}
    for epoch in range(c['plastic_epochs']):
        model.reset();values={k:[] for k in logs}
        for t,symbol in enumerate(symbols):
            state=model.forward(int(symbol),rng.normal(0,c['noise_std'],len(model.mbon)))
            if t<c['warmup']:
                model.eligibility.fill(0);model.pending=False;continue
            action=int(fixed_scores(state[model.mbon][None,:],codebook).argmax(1)[0])
            reward=float(action==symbols[t-c['target_lag']]);trace=float(np.linalg.norm(model.eligibility))
            advantage=model.reinforce(reward)
            proposal=abs(model.weights.data[model.mask])-model.original_magnitude;total+=proposal
            for k,v in zip(logs,[action,reward,advantage,trace,float(np.linalg.norm(proposal))]):values[k].append(v)
            # Never apply a proposed update to the next forward trajectory.
            model.weights.data[:]=initial
        for k in logs:logs[k].append(values[k])
        history.append(dict(epoch=epoch+1,reward_mean=float(np.mean(values['rewards'])),mean_proposal_norm=float(np.mean(values['proposal_norms']))))
    assert np.array_equal(model.weights.data,initial)
    return total/(c['plastic_epochs']*(len(symbols)-c['warmup'])),{k:np.asarray(v) for k,v in logs.items()},history,model


def tangent(vector,magnitude,post,n):
    mass=np.bincount(post,weights=magnitude,minlength=n)
    return vector-magnitude*np.bincount(post,weights=vector,minlength=n)[post]/mass[post]


def directions(mean_proposal,magnitude,post,n,c,seed):
    vectors=[tangent(mean_proposal,magnitude,post,n)];rng=np.random.default_rng(seed)
    for _ in range(c['random_directions']):vectors.append(tangent(rng.normal(size=len(magnitude))*magnitude,magnitude,post,n))
    norms=np.linalg.norm(vectors,axis=1);unit=np.asarray(vectors)/np.where(norms>0,norms,1)[:,None]
    nominal=c['nominal_relative_radius']*np.linalg.norm(magnitude)
    if norms[0]==0:unit[:]=0;radius=0.
    else:
        limits=np.divide((1-c['plastic_floor'])*magnitude[None,:],abs(unit),out=np.full_like(unit,np.inf),where=unit!=0)
        radius=min(nominal,c['boundary_fraction']*float(limits.min()))
    return unit,dict(radius=float(radius),nominal_radius=float(nominal),relative_radius=float(radius/np.linalg.norm(magnitude)),
        boundary_limited=bool(radius<nominal),resolution_limited=bool(radius<nominal/10),local_proposal_norm=float(norms[0]))


def perturb(initial,mask,post,unit,radius,orientation):
    w=initial.copy();magnitude=abs(initial.data[mask]);sign=np.sign(initial.data[mask])
    w.data[mask]=sign*(magnitude+orientation*radius*unit)
    assert np.array_equal(w.data[~mask],initial.data[~mask]) and np.array_equal(np.sign(w.data),np.sign(initial.data))
    np.testing.assert_allclose(np.bincount(post,weights=abs(w.data[mask])),np.bincount(post,weights=magnitude),atol=1e-14,rtol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(w.data-initial.data),radius if orientation else 0,atol=1e-14,rtol=1e-10)
    return w


def evaluate(weights,bank,obs,symbols,codebook,c,seed,replicas,noisy):
    noise=np.stack([np.random.default_rng(np.random.SeedSequence([seed,r])).normal(0,c['noise_std'],(len(symbols),len(obs))) for r in range(replicas)],axis=1) if noisy else np.zeros((len(symbols),replicas,len(obs)))
    state=np.zeros((weights.shape[0],replicas));scores=[];features=[]
    for t,symbol in enumerate(symbols):
        drive=weights@state+bank[int(symbol),:,None];drive[obs]+=noise[t].T
        state=(1-c['leak'])*state+c['leak']*np.tanh(drive)
        if not noisy:features.append(state[obs,0].copy())
        if t>=c['warmup']:scores.append(fixed_scores(np.ascontiguousarray(state[obs].T),codebook))
    return np.stack(scores,axis=1),np.asarray(features)


def summarize(rows,c):
    f=pd.DataFrame(rows);wide=f.pivot(index=['cohort','seed','circuit_seed','mode'],columns='variant',values='accuracy');records=[]
    for identity,r in wide.iterrows():
        random=np.asarray([r[f'random{k}_plus']-r[f'random{k}_minus'] for k in range(c['random_directions'])])
        records.append(dict(zip(['cohort','seed','circuit_seed','mode'],identity),baseline=r['baseline'],local_plus=r['local_plus'],local_minus=r['local_minus'],
            local_directional=r['local_plus']-r['local_minus'],random_absolute=float(np.mean(abs(random))),
            above_random=r['local_plus']-r['local_minus']-float(np.mean(abs(random))),forward_gain=r['local_plus']-r['baseline']))
    circuit=pd.DataFrame(records);cols=['baseline','local_plus','local_minus','local_directional','random_absolute','above_random','forward_gain']
    blocks=circuit.groupby(['cohort','seed','mode'])[cols].mean().reset_index();out={}
    for cohort in ['discovery','confirmation']:
        out[cohort]={}
        for mode in ['noisy','clean']:
            q=blocks[(blocks.cohort==cohort)&(blocks['mode']==mode)];stats={k:estimate(q[k],c) for k in cols}
            gates={name:bool(q[name].mean()>=threshold and (q[name]>0).all()) for name,threshold in [('local_directional',.0025),('above_random',.001),('forward_gain',.001)]}
            out[cohort][mode]=dict(statistics=stats,gates=gates,passes_all=all(gates.values()))
    out['confirmed_noisy_alignment']=bool(out['discovery']['noisy']['passes_all'] and out['confirmation']['noisy']['passes_all'])
    out['clean_secondary_only']=True;out['cohorts_pooled']=False
    return circuit,blocks,out


def baseline(c):
    from verify_local_reward import reference_train,trajectory
    source=Path(c['source']);check(source);assert core.sha256(source/'manifest.json')==c['source_manifest_sha256']
    dest=source/'discovery/real_contingent_c701_s241142';m=check(dest);_,_,roles=anatomy(701)
    initial=sparse.load_npz(dest/'initial-weights.npz');final=sparse.load_npz(dest/'final-weights.npz')
    with np.load(dest/'checkpoint.npz') as z:a=dict(z)
    w,h,logs=reference_train(initial,roles,a['input_patterns'],a['train_symbols'],a['codebook'],'contingent',m['config'],m['identity']['noise_seed'])
    np.testing.assert_array_equal(w.data,final.data);assert h==read(dest/'history.json')
    for k in logs:np.testing.assert_array_equal(logs[k],a[k])
    features,_,_=trajectory(final,a['input_patterns'],a['observed_indices'],a['test_symbols'],c['leak']);np.testing.assert_array_equal(features,a['test_features'])
    accuracy=float(np.mean(fixed_scores(features[m['config']['warmup']:],a['codebook']).argmax(1)==a['ytest'][:,m['config']['lags'].index(2)]))
    assert accuracy==read(dest/'fixed-metrics.json')['test_accuracy']
    return dict(source=dest.as_posix(),exact_training=True,exact_trajectory=True,expected_accuracy=read(dest/'fixed-metrics.json')['test_accuracy'],actual_accuracy=accuracy)


@threadpool_limits.wrap(limits=1)
def run(config,out):
    c=read(config);assert core.fingerprint(c)==CONFIG_HASH
    assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip(),'Commit before outcomes'
    context=core.source_context(c)
    names=['reward_direction','verify_reward_direction','plot_reward_direction','local_reward','verify_local_reward','readout_dependency','alphabet_memory','edge_panel','frozen_state_probe','context_memory','pathway_memory','verify_neuron_panel']
    for p in [config,Path('docs/reward-direction-seed-audit.json')]+[Path('scripts')/(n+'.py') for n in names]:context['source_sha256'][p.as_posix()]=core.sha256(p)
    for p,h in context['source_sha256'].items():assert hashlib.sha256(subprocess.check_output(['git','show',context['git_commit']+':'+p])).hexdigest()==h,p
    out.mkdir(parents=True,exist_ok=False);budget=Budget(c)
    try:
        prior={p:core.sha256(p) for p in subprocess.check_output(['git','ls-files','results/'],text=True).splitlines()};core.write_json(out/'prior-results-sha256.json',prior);core.write_json(out/'config.json',c)
        core.write_json(out/'baseline.json',baseline(c));print('Previous local-reward baseline exactly reproduced',flush=True)
        rows=[];radii=[];count=0
        for cohort,settings in c['cohorts'].items():
            cc=dict(c,**settings)
            for block in cc['blocks']:
                for ci in cc['circuit_seeds']:
                    budget.check();raw,ids,roles=anatomy(ci);initial=core.normalize_condition(raw,c['normalization'],c['gain']);initial.sort_indices()
                    bank=symbol_bank(roles,block['seed'],c['input_fraction'],c['input_amplitude'])[:4];codebook=codes(block['code_seed']);obs=np.flatnonzero(roles=='MBON')
                    symbols={split:np.random.default_rng(block[split+'_seed']).integers(0,4,c['warmup']+cc[split+'_samples']).astype(np.uint8) for split in ['train','test']}
                    mean,logs,history,model=local_direction(initial,roles,bank,symbols['train'],codebook,cc,block['noise_seed'])
                    unit,radius=directions(mean,model.original_magnitude,model.post,len(roles),cc,block['direction_seed'])
                    ident=dict(cohort=cohort,seed=block['seed'],circuit_seed=ci);radii.append(dict(**ident,**radius));dest=out/cohort/f'c{ci}_s{block["seed"]}';dest.mkdir(parents=True)
                    a=dict(input_patterns=bank,codebook=codebook,observed_indices=obs,train_symbols=symbols['train'],test_symbols=symbols['test'],mean_proposal=mean,unit_directions=unit,plastic_mask=model.mask,**logs)
                    truth=symbols['test'][np.arange(c['warmup'],len(symbols['test']))-c['target_lag']];a['truth']=truth
                    variants=[('baseline',0,0)]+[(name+'_'+side,j,sign) for j,name in enumerate(['local']+[f'random{k}' for k in range(c['random_directions'])]) for side,sign in [('plus',1),('minus',-1)]]
                    for name,j,sign in variants:
                        weights=perturb(initial,model.mask,model.post,unit[j],radius['radius'],sign)
                        sparse.save_npz(dest/(name+'-weights.npz'),weights)
                        for mode,replicas in [('noisy',cc['evaluation_replicates']),('clean',1)]:
                            scores,features=evaluate(weights,bank,obs,symbols['test'],codebook,cc,block['evaluation_seed'],replicas,mode=='noisy')
                            a[name+'_'+mode+'_scores']=scores;acc=(scores.argmax(2)==truth[None,:]).mean(1);a[name+'_'+mode+'_accuracy']=acc
                            rows.append(dict(**ident,variant=name,mode=mode,accuracy=float(acc.mean()),replicas=replicas))
                            if mode=='clean':a[name+'_features']=features
                            for value in [scores,features]:assert np.isfinite(value).all()
                        budget.check()
                    np.savez_compressed(dest/'checkpoint.npz',**a);sparse.save_npz(dest/'raw.npz',raw)
                    core.write_json(dest/'history.json',history);core.write_json(dest/'radius.json',radius)
                    seal(dest,cc,context,identity={**ident,**block},observation_ids=[ids[i] for i in obs],training_weights_never_accumulated=True)
                    count+=1;print(f'{cohort} c{ci} s{block["seed"]}: frozen direction +13 probes complete',flush=True)
        circuit,blocks,summary=summarize(rows,c);pd.DataFrame(rows).to_csv(out/'raw-metrics.csv',index=False);circuit.to_csv(out/'circuit-effects.csv',index=False);blocks.to_csv(out/'seed-blocks.csv',index=False);pd.DataFrame(radii).to_csv(out/'radii.csv',index=False);core.write_json(out/'summary.json',summary)
        for p,h in {**prior,**context['source_sha256']}.items():assert core.sha256(p)==h,p
        assert count==13 and len(rows)==338;budget.check();core.write_json(out/'verification.json',dict(cases=count,probe_graphs=169,metric_rows=len(rows),prior_files_unchanged=len(prior),budget=budget.close()))
        seal(out,c,context,environment=core.environment());print('Confirmed noisy alignment:',summary['confirmed_noisy_alignment'],flush=True)
    except Exception as exc:core.write_json(out/'failure.json',dict(error=repr(exc)));raise
    finally:budget.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=Path('configs/reward_direction.json'));p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.config,a.out)
