"""Paired local-update usefulness at predefined checkpoints of unchanged learning."""
import argparse,hashlib,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits
from flying.training import whole_brain_memory as core
from flying.brain.local_reward import LocalRewardPlasticity
from alphabet_memory import read,check,SymbolEncoder
from local_reward import fixed_scores
from reward_direction import tangent,perturb,evaluate
from edge_panel import anatomy
from frozen_state_probe import Budget
from context_memory import estimate
from pathway_memory import seal

CONFIG_HASH='f0a2d83e03d1970616b3224ec0fb244f07c329561b14b736e60325a474b5bd37'


def capture_train(weights,roles,bank,symbols,codebook,c,seed,epochs):
    model=LocalRewardPlasticity(weights,roles,SymbolEncoder(bank),leak=c['leak'],
        learning_rate=c['plastic_learning_rate'],floor=c['plastic_floor'],
        trace_decay=c['trace_decay'],activity_rate=c['activity_rate'],baseline_rate=c['baseline_rate'])
    rng=np.random.default_rng(seed);snapshots={0:model.weights.copy()};history=[]
    logs={k:[] for k in ['actions','actual_rewards','applied_rewards','advantages','trace_norms']}
    for epoch in range(c['plastic_epochs']):
        model.reset();values={k:[] for k in logs}
        for t,symbol in enumerate(symbols):
            state=model.forward(int(symbol),rng.normal(0,c['noise_std'],len(model.mbon)))
            if t<c['warmup']:
                model.eligibility.fill(0);model.pending=False;continue
            action=int(fixed_scores(state[model.mbon][None,:],codebook).argmax(1)[0])
            reward=float(action==symbols[t-c['target_lag']]);norm=float(np.linalg.norm(model.eligibility));adv=model.reinforce(reward)
            for k,v in zip(logs,[action,reward,reward,adv,norm]):values[k].append(v)
        for k in logs:logs[k].append(values[k])
        history.append(dict(epoch=epoch+1,online_accuracy=float(np.mean(values['actual_rewards'])),applied_reward_mean=float(np.mean(values['applied_rewards'])),mean_trace_norm=float(np.mean(values['trace_norms']))))
        if epoch+1 in epochs:snapshots[epoch+1]=model.weights.copy()
    assert set(snapshots)==set(epochs)
    return snapshots,history,{k:np.asarray(v) for k,v in logs.items()}


def local_direction(original,weights,roles,bank,symbols,codebook,c,seed):
    model=LocalRewardPlasticity(original,roles,SymbolEncoder(bank),leak=c['leak'],
        learning_rate=c['plastic_learning_rate'],floor=c['plastic_floor'],
        trace_decay=c['trace_decay'],activity_rate=c['activity_rate'],baseline_rate=c['baseline_rate'])
    model.weights.data[:]=weights.data
    current=abs(weights.data[model.mask]);initial=model.weights.data.copy();total=np.zeros(len(model.pre));rng=np.random.default_rng(seed);history=[]
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
            proposal=abs(model.weights.data[model.mask])-current;total+=proposal
            for k,v in zip(logs,[action,reward,advantage,trace,float(np.linalg.norm(proposal))]):values[k].append(v)
            # Never apply a proposed update to the next forward trajectory.
            model.weights.data[:]=initial
        for k in logs:logs[k].append(values[k])
        history.append(dict(epoch=epoch+1,reward_mean=float(np.mean(values['rewards'])),mean_proposal_norm=float(np.mean(values['proposal_norms']))))
    assert np.array_equal(model.weights.data,initial)
    return total/(c['plastic_epochs']*(len(symbols)-c['warmup'])),{k:np.asarray(v) for k,v in logs.items()},history,model


def stage_directions(mean,magnitude,post,n,c,seed):
    rng=np.random.default_rng(seed)
    vectors=[tangent(mean,magnitude,post,n)]+[tangent(rng.normal(size=len(magnitude))*magnitude,magnitude,post,n) for _ in range(c['random_directions'])]
    norms=np.linalg.norm(vectors,axis=1)
    return np.asarray(vectors)/np.where(norms>0,norms,1)[:,None],norms


def shared_radius(original_magnitude,stages,c):
    nominal=c['nominal_relative_radius']*np.linalg.norm(original_magnitude);radius=float(nominal);zero=False
    for magnitude,unit,norms in stages:
        zero=zero or bool((norms==0).any())
        limits=np.divide(magnitude[None,:],-unit,out=np.full_like(unit,np.inf),where=unit<0)
        radius=min(radius,c['boundary_fraction']*float(limits.min()))
    if zero:radius=0.
    return dict(radius=float(radius),nominal_radius=float(nominal),relative_radius=float(radius/np.linalg.norm(original_magnitude)),
        boundary_limited=bool(radius<nominal),resolution_limited=bool(radius<nominal/10),zero_direction=zero)


COLS=['baseline','local','random_mean','forward_gain','above_random']


def summarize(rows,radii,c):
    f=pd.DataFrame(rows);records=[]
    for identity,g in f.groupby(['cohort','seed','circuit_seed','epoch','mode']):
        v=g.set_index('variant').accuracy;random=float(np.mean([v[f'random{k}'] for k in range(c['random_directions'])]))
        records.append(dict(zip(['cohort','seed','circuit_seed','epoch','mode'],identity),baseline=v['baseline'],local=v['local'],random_mean=random,forward_gain=v['local']-v['baseline'],above_random=v['local']-random))
    circuit=pd.DataFrame(records);blocks=circuit.groupby(['cohort','seed','epoch','mode'])[COLS].mean().reset_index();out={};pairs=[]
    for cohort in ['discovery','confirmation']:
        out[cohort]={};identifiable=not any(r['resolution_limited'] for r in radii if r['cohort']==cohort)
        for mode in ['noisy','clean']:
            q=blocks[(blocks.cohort==cohort)&(blocks['mode']==mode)];by_epoch={}
            for epoch in c['epochs']:
                e=q[q.epoch==epoch];gates={k:bool(e[k].mean()>=.001 and (e[k]>0).all()) for k in ['forward_gain','above_random']}
                by_epoch[str(epoch)]=dict(statistics={k:estimate(e[k],c) for k in COLS},gates=gates,utility=all(gates.values()))
            start=q[q.epoch==0].set_index('seed');end=q[q.epoch==10].set_index('seed');stats={};gates={}
            for k in ['forward_gain','above_random']:
                delta=start[k]-end[k];stats[k]=estimate(delta,c);gates[k]=bool(delta.mean()>=.001 and (delta>0).all())
                pairs.extend(dict(cohort=cohort,seed=int(seed),mode=mode,contrast=k+'_attenuation',difference=float(v)) for seed,v in delta.items())
            out[cohort][mode]=dict(epochs=by_epoch,identifiable=identifiable,attenuation_statistics=stats,attenuation_gates=gates,
                attenuation=bool(identifiable and by_epoch['0']['utility'] and all(gates.values())),
                persistent_utility=bool(identifiable and by_epoch['0']['utility'] and by_epoch['10']['utility']))
    out['confirmed']={k:bool(out['discovery']['noisy'][k] and out['confirmation']['noisy'][k]) for k in ['attenuation','persistent_utility']}
    out['primary_identifiable']=bool(out['discovery']['noisy']['identifiable'] and out['confirmation']['noisy']['identifiable'])
    out['cohorts_pooled']=False;out['clean_secondary_only']=True
    return circuit,blocks,pd.DataFrame(pairs),out


@threadpool_limits.wrap(limits=1)
def run(config,out):
    c=read(config);assert core.fingerprint(c)==CONFIG_HASH
    assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip(),'Commit before outcomes'
    source=Path(c['source']);check(source);assert core.sha256(source/'manifest.json')==c['source_manifest_sha256']
    context=core.source_context(c)
    names=['reward_trajectory','verify_reward_trajectory','plot_reward_trajectory','reward_direction','verify_reward_direction','local_reward','verify_local_reward','readout_dependency','alphabet_memory','edge_panel','frozen_state_probe','context_memory','pathway_memory','verify_neuron_panel']
    for p in [config,Path('docs/reward-trajectory-seed-audit.json')]+[Path('scripts')/(n+'.py') for n in names]:context['source_sha256'][p.as_posix()]=core.sha256(p)
    for p,h in context['source_sha256'].items():assert hashlib.sha256(subprocess.check_output(['git','show',context['git_commit']+':'+p])).hexdigest()==h,p
    out.mkdir(parents=True,exist_ok=False);budget=Budget(c)
    try:
        prior={p:core.sha256(p) for p in subprocess.check_output(['git','ls-files','results/'],text=True).splitlines()};core.write_json(out/'prior-results-sha256.json',prior);core.write_json(out/'config.json',c)
        rows=[];radii=[];neural=[];baseline=[];cases=0;snapshot_count=0;trajectories=0
        paths=sorted(source.glob('*/real_contingent_*/checkpoint.npz'),key=lambda p:(p.parent.parent.name!='smoke',p.as_posix()))
        assert len(paths)==13
        for path in paths:
            parent=path.parent;cm=check(parent);cc=cm['config'];i=cm['identity'];smoke=i['cohort']=='smoke';epochs=c['smoke_epochs'] if smoke else c['epochs'];seeds=c['probe_seeds'][str(i['seed'])]
            _,ids,roles=anatomy(i['circuit_seed']);initial=sparse.load_npz(parent/'initial-weights.npz')
            with np.load(path) as z:old=dict(z)
            snapshots,history,logs=capture_train(initial,roles,old['input_patterns'],old['train_symbols'],old['codebook'],cc,i['noise_seed'],epochs)
            assert history==read(parent/'history.json');assert core.weight_hash(snapshots[epochs[-1]])==core.weight_hash(sparse.load_npz(parent/'final-weights.npz'))
            for k in logs:np.testing.assert_array_equal(logs[k],old[k])
            ident={k:i[k] for k in ['cohort','seed','circuit_seed']};dest=out/i['cohort']/f'c{i["circuit_seed"]}_s{i["seed"]}';dest.mkdir(parents=True)
            core.write_json(dest/'training-history.json',history);np.savez_compressed(dest/'training-logs.npz',**logs)
            proposal_c=dict(cc,plastic_epochs=c['smoke_proposal_epochs'] if smoke else c['proposal_epochs']);stages=[];items=[]
            for epoch in epochs:
                w=snapshots[epoch];mean,plogs,h,model=local_direction(initial,w,roles,old['input_patterns'],old['train_symbols'],old['codebook'],proposal_c,seeds['proposal_noise'])
                mag=abs(w.data[model.mask]);unit,norms=stage_directions(mean,mag,model.post,len(roles),c,seeds['direction'])
                stages.append((mag,unit,norms));items.append((epoch,w,mean,plogs,h,model,unit,norms));budget.check()
            radius=shared_radius(abs(initial.data[model.mask]),stages,c);radii.append(dict(**ident,**radius));core.write_json(dest/'radius.json',radius)
            truth=old['test_symbols'][np.arange(cc['warmup'],len(old['test_symbols']))-cc['target_lag']]
            for epoch,w,mean,plogs,h,model,unit,norms in items:
                stage=dest/f'epoch{epoch}';stage.mkdir();a=dict(mean_proposal=mean,unit_directions=unit,direction_norms=norms,plastic_mask=model.mask,truth=truth,**plogs)
                cosine=float(np.dot(unit[0],items[0][6][0]))
                for name,j,orientation in [('baseline',0,0)]+[(name,j,1) for j,name in enumerate(['local']+[f'random{k}' for k in range(c['random_directions'])])]:
                    weights=perturb(w,model.mask,model.post,unit[j],radius['radius'],orientation);sparse.save_npz(stage/(name+'-weights.npz'),weights)
                    for mode,replicas in [('noisy',c['smoke_replicates'] if smoke else c['evaluation_replicates']),('clean',1)]:
                        scores,features=evaluate(weights,old['input_patterns'],old['observed_indices'],old['test_symbols'],old['codebook'],cc,seeds['evaluation_noise'],replicas,mode=='noisy')
                        assert np.isfinite(scores).all() and np.isfinite(features).all();acc=(scores.argmax(2)==truth[None,:]).mean(1)
                        a[name+'_'+mode+'_scores']=scores;a[name+'_'+mode+'_accuracy']=acc;rows.append(dict(**ident,epoch=epoch,variant=name,mode=mode,accuracy=float(acc.mean()),replicas=replicas));trajectories+=replicas
                        if mode=='clean':
                            a[name+'_features']=features
                            if name=='baseline':
                                neural.append(dict(**ident,epoch=epoch,**core.state_diagnostics(features[cc['warmup']:]),direction_cosine_to_epoch0=cosine,local_proposal_norm=float(norms[0]),tiny_edge_fraction=float(np.mean(abs(w.data[model.mask])/abs(initial.data[model.mask])<.001))))
                                if epoch==epochs[-1]:
                                    np.testing.assert_array_equal(features,old['test_features']);expected=read(parent/'fixed-metrics.json')['test_accuracy'];assert float(acc.mean())==expected
                                    baseline.append(dict(**ident,exact_training=True,exact_trajectory=True,expected_accuracy=expected,actual_accuracy=float(acc.mean())))
                    budget.check()
                np.savez_compressed(stage/'checkpoint.npz',**a);core.write_json(stage/'proposal-history.json',h)
                seal(stage,proposal_c,context,identity=dict(**ident,epoch=epoch),source_case=parent.as_posix(),probe_seeds=seeds);snapshot_count+=1
            seal(dest,cc,context,identity=i,source_case=parent.as_posix(),source_manifest_sha256=core.sha256(parent/'manifest.json'),probe_seeds=seeds,epochs=epochs)
            cases+=1;print(f'{i["cohort"]} c{i["circuit_seed"]} s{i["seed"]}: {epochs} complete; radius/original L2={radius["relative_radius"]:.6g}',flush=True)
        circuit,blocks,pairs,summary=summarize(rows,radii,c)
        for name,data in [('raw-metrics',pd.DataFrame(rows)),('radii',pd.DataFrame(radii)),('neural',pd.DataFrame(neural)),('circuit-effects',circuit),('seed-blocks',blocks),('paired-differences',pairs)]:data.to_csv(out/(name+'.csv'),index=False)
        core.write_json(out/'baseline.json',baseline);core.write_json(out/'summary.json',summary)
        for p,h in {**prior,**context['source_sha256']}.items():assert core.sha256(p)==h,p
        assert cases==13 and snapshot_count==50 and len(rows)==700 and trajectories==11130;budget.check()
        core.write_json(out/'verification.json',dict(cases=cases,snapshots=snapshot_count,probe_graphs=350,trajectories=trajectories,metric_rows=len(rows),prior_files_unchanged=len(prior),budget=budget.close()))
        seal(out,c,context,environment=core.environment());print('Confirmed:',summary['confirmed'],'Identifiable:',summary['primary_identifiable'],flush=True)
    except Exception as exc:core.write_json(out/'failure.json',dict(error=repr(exc)));raise
    finally:budget.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=Path('configs/reward_trajectory.json'));p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.config,a.out)
