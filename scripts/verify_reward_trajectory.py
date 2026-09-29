"""Independent replay of learning snapshots, local proposals, and scalar probes."""
import argparse,hashlib,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits
from alphabet_memory import read,check
from flying.training import whole_brain_memory as core
from edge_panel import anatomy
from frozen_state_probe import Budget
from verify_local_reward import reference_train
from verify_reward_direction import reference_evaluation
from verify_neuron_panel import audit_stats


def reference_direction(original,initial,roles,bank,symbols,codes,c,seed):
    n=len(roles);row=np.repeat(np.arange(n),np.diff(initial.indptr));mask=(roles[row]=='MBON')&(roles[initial.indices]=='KC')
    post=row[mask];pre=initial.indices[mask];obs=np.flatnonzero(roles=='MBON');mag=abs(initial.data[mask]);sign=np.sign(initial.data[mask]);orig=abs(original.data[mask]);mass=np.bincount(post,weights=orig,minlength=n)
    rng=np.random.default_rng(seed);total=np.zeros(len(pre));history=[];logs={k:[] for k in ['actions','rewards','advantages','trace_norms','proposal_norms']}
    for epoch in range(c['plastic_epochs']):
        state=np.zeros(n);average=np.zeros(n);trace=np.zeros(len(pre));baseline=.25;values={k:[] for k in logs}
        for t,symbol in enumerate(symbols):
            old=state.copy();drive=initial@old+bank[int(symbol)];drive[obs]+=rng.normal(0,c['noise_std'],len(obs))
            activation=np.tanh(drive);state=(1-c['leak'])*old+c['leak']*activation
            trace=c['trace_decay']*trace+(1-c['trace_decay'])*(old[pre]*(activation[post]-average[post]));average+=c['activity_rate']*(activation-average)
            if t<c['warmup']:trace.fill(0);continue
            action=int(np.mean((state[obs][None,:]-codes)**2,axis=1).argmin());reward=float(action==symbols[t-c['target_lag']]);adv=reward-baseline
            energy=np.bincount(post,weights=old[pre]**2,minlength=n);delta=adv*trace/(1+energy[post])
            proposed=np.maximum(c['plastic_floor']*orig,mag+c['plastic_learning_rate']*sign*delta)
            sums=np.bincount(post,weights=proposed,minlength=n);proposed*=mass[post]/sums[post]
            proposal=proposed-mag;total+=proposal;baseline+=c['baseline_rate']*adv
            for k,v in zip(logs,[action,reward,adv,float(np.linalg.norm(trace)),float(np.linalg.norm(proposal))]):values[k].append(v)
        for k in logs:logs[k].append(values[k])
        history.append(dict(epoch=epoch+1,reward_mean=float(np.mean(values['rewards'])),mean_proposal_norm=float(np.mean(values['proposal_norms']))))
    return total/(c['plastic_epochs']*(len(symbols)-c['warmup'])),{k:np.asarray(v) for k,v in logs.items()},history,mask,post


def frame_check(actual,path,keys):
    expected=pd.read_csv(path)
    pd.testing.assert_frame_equal(pd.DataFrame(actual)[expected.columns].sort_values(keys).reset_index(drop=True),expected.sort_values(keys).reset_index(drop=True),check_dtype=False,atol=1e-12,rtol=1e-12)


@threadpool_limits.wrap(limits=1)
def verify(root,out):
    m=check(root);c=m['config'];source=Path(c['source']);check(source);assert core.sha256(source/'manifest.json')==c['source_manifest_sha256']
    out.mkdir(parents=True,exist_ok=False);budget=Budget(c);rows=[];radii=[];neural=[];baselines=[];cases=0;count=0;trajectories=0
    try:
        for dest in sorted(p.parent for p in root.glob('*/*/training-logs.npz')):
            cm=check(dest);i=cm['identity'];cc=cm['config'];parent=Path(cm['source_case']);check(parent);assert core.sha256(parent/'manifest.json')==cm['source_manifest_sha256']
            _,_,roles=anatomy(i['circuit_seed']);initial=sparse.load_npz(parent/'initial-weights.npz');seeds=c['probe_seeds'][str(i['seed'])];assert seeds==cm['probe_seeds']
            smoke=i['cohort']=='smoke';epochs=c['smoke_epochs'] if smoke else c['epochs'];assert epochs==cm['epochs'];ident={k:i[k] for k in ['cohort','seed','circuit_seed']}
            with np.load(parent/'checkpoint.npz') as z:old=dict(z)
            with np.load(dest/'training-logs.npz') as z:trainlogs=dict(z)
            nominal=c['nominal_relative_radius']*np.linalg.norm(abs(initial.data[(np.repeat(roles,np.diff(initial.indptr))=='MBON')&(roles[initial.indices]=='KC')]))
            radius=float(nominal);zero=False;items=[]
            for epoch in epochs:
                if epoch==0:w=initial.copy()
                else:
                    w,h,logs=reference_train(initial,roles,old['input_patterns'],old['train_symbols'],old['codebook'],'contingent',dict(cc,plastic_epochs=epoch),i['noise_seed'])
                    assert h==read(dest/'training-history.json')[:epoch] and h==read(parent/'history.json')[:epoch]
                    for k in logs:
                        np.testing.assert_array_equal(logs[k],trainlogs[k][:epoch]);np.testing.assert_array_equal(logs[k],old[k][:epoch])
                if epoch==epochs[-1]:assert core.weight_hash(w)==core.weight_hash(sparse.load_npz(parent/'final-weights.npz'))
                stage=dest/f'epoch{epoch}';sm=check(stage);assert sm['identity']==dict(**ident,epoch=epoch)
                with np.load(stage/'checkpoint.npz') as z:a=dict(z)
                pc=dict(cc,plastic_epochs=c['smoke_proposal_epochs'] if smoke else c['proposal_epochs']);assert pc==sm['config']
                mean,logs,h,mask,post=reference_direction(initial,w,roles,old['input_patterns'],old['train_symbols'],old['codebook'],pc,seeds['proposal_noise'])
                np.testing.assert_array_equal(mean,a['mean_proposal']);np.testing.assert_array_equal(mask,a['plastic_mask']);assert h==read(stage/'proposal-history.json')
                for k in logs:np.testing.assert_array_equal(logs[k],a[k])
                mag=abs(w.data[mask]);mass=np.bincount(post,weights=mag,minlength=len(roles));rng=np.random.default_rng(seeds['direction'])
                raw=[mean]+[rng.normal(size=len(mag))*mag for _ in range(c['random_directions'])]
                vectors=[v-mag*np.bincount(post,weights=v,minlength=len(roles))[post]/mass[post] for v in raw];norms=np.linalg.norm(vectors,axis=1);unit=np.asarray(vectors)/np.where(norms>0,norms,1)[:,None]
                np.testing.assert_array_equal(unit,a['unit_directions']);np.testing.assert_array_equal(norms,a['direction_norms']);zero=zero or bool((norms==0).any())
                limit=np.divide(mag[None,:],-unit,out=np.full_like(unit,np.inf),where=unit<0);radius=min(radius,c['boundary_fraction']*float(limit.min()))
                items.append((epoch,w,a,mask,post,unit,norms));budget.check()
            if zero:radius=0.
            expected=dict(radius=float(radius),nominal_radius=float(nominal),relative_radius=float(radius/np.linalg.norm(abs(initial.data[mask]))),boundary_limited=bool(radius<nominal),resolution_limited=bool(radius<nominal/10),zero_direction=zero)
            assert expected==read(dest/'radius.json');radii.append(dict(**ident,**expected))
            truth=old['test_symbols'][np.arange(cc['warmup'],len(old['test_symbols']))-cc['target_lag']]
            for epoch,w,a,mask,post,unit,norms in items:
                stage=dest/f'epoch{epoch}';np.testing.assert_array_equal(truth,a['truth']);mag=abs(w.data[mask])
                for name,j,orientation in [('baseline',0,0)]+[(name,j,1) for j,name in enumerate(['local']+[f'random{k}' for k in range(c['random_directions'])])]:
                    probe=w.copy();probe.data[mask]=np.sign(w.data[mask])*(mag+orientation*radius*unit[j]);assert core.weight_hash(probe)==core.weight_hash(sparse.load_npz(stage/(name+'-weights.npz')))
                    np.testing.assert_array_equal(probe.data[~mask],initial.data[~mask]);np.testing.assert_array_equal(np.sign(probe.data),np.sign(initial.data));assert (abs(probe.data[mask])>=.5*mag-1e-15).all()
                    np.testing.assert_allclose(np.bincount(post,weights=abs(probe.data[mask])),np.bincount(post,weights=abs(initial.data[mask])),atol=1e-14,rtol=1e-12)
                    np.testing.assert_allclose(np.linalg.norm(probe.data-w.data),radius if orientation else 0,atol=1e-14,rtol=1e-10)
                    for mode,replicas in [('noisy',c['smoke_replicates'] if smoke else c['evaluation_replicates']),('clean',1)]:
                        scores,features=reference_evaluation(probe,old['input_patterns'],old['observed_indices'],old['test_symbols'],old['codebook'],cc,seeds['evaluation_noise'],replicas,mode=='noisy')
                        np.testing.assert_array_equal(scores,a[name+'_'+mode+'_scores']);acc=(scores.argmax(2)==truth[None,:]).mean(1);np.testing.assert_array_equal(acc,a[name+'_'+mode+'_accuracy']);trajectories+=replicas
                        rows.append(dict(**ident,epoch=epoch,variant=name,mode=mode,accuracy=float(acc.mean()),replicas=replicas))
                        if mode=='clean':
                            np.testing.assert_array_equal(features,a[name+'_features'])
                            if name=='baseline':
                                neural.append(dict(**ident,epoch=epoch,**core.state_diagnostics(features[cc['warmup']:]),direction_cosine_to_epoch0=float(np.dot(unit[0],items[0][5][0])),local_proposal_norm=float(norms[0]),tiny_edge_fraction=float(np.mean(mag/abs(initial.data[mask])<.001))))
                                if epoch==epochs[-1]:
                                    np.testing.assert_array_equal(features,old['test_features']);value=read(parent/'fixed-metrics.json')['test_accuracy'];assert float(acc.mean())==value
                                    baselines.append(dict(**ident,exact_training=True,exact_trajectory=True,expected_accuracy=value,actual_accuracy=float(acc.mean())))
                    budget.check()
                count+=1
            cases+=1;print(f'Independent learning + frozen proposals + scalar scores: {cases}/13',flush=True)
        frame_check(rows,root/'raw-metrics.csv',['cohort','seed','circuit_seed','epoch','mode','variant']);frame_check(radii,root/'radii.csv',['cohort','seed','circuit_seed']);frame_check(neural,root/'neural.csv',['cohort','seed','circuit_seed','epoch'])
        assert sorted(baselines,key=lambda x:(x['cohort'],x['seed'],x['circuit_seed']))==sorted(read(root/'baseline.json'),key=lambda x:(x['cohort'],x['seed'],x['circuit_seed']))
        f=pd.DataFrame(rows);effects=[];cols=['baseline','local','random_mean','forward_gain','above_random']
        for identity,g in f.groupby(['cohort','seed','circuit_seed','epoch','mode']):
            v=g.set_index('variant').accuracy;random=sum(v[f'random{k}'] for k in range(c['random_directions']))/c['random_directions']
            effects.append(dict(zip(['cohort','seed','circuit_seed','epoch','mode'],identity),baseline=v['baseline'],local=v['local'],random_mean=random,forward_gain=v['local']-v['baseline'],above_random=v['local']-random))
        frame_check(effects,root/'circuit-effects.csv',['cohort','seed','circuit_seed','epoch','mode']);blocks=pd.DataFrame(effects).groupby(['cohort','seed','epoch','mode'])[cols].mean().reset_index();frame_check(blocks,root/'seed-blocks.csv',['cohort','seed','epoch','mode'])
        summary=read(root/'summary.json');ends={};pairs=[]
        for cohort in ['discovery','confirmation']:
            ends[cohort]={};identifiable=not any(r['resolution_limited'] for r in radii if r['cohort']==cohort)
            for mode in ['noisy','clean']:
                q=blocks[(blocks.cohort==cohort)&(blocks['mode']==mode)];ss=summary[cohort][mode];assert ss['identifiable']==identifiable;utility={}
                for epoch in c['epochs']:
                    v=q[q.epoch==epoch];es=ss['epochs'][str(epoch)]
                    for k in cols:audit_stats(v[k],es['statistics'][k],c)
                    gates={k:bool(v[k].mean()>=.001 and (v[k]>0).all()) for k in ['forward_gain','above_random']};assert gates==es['gates'] and all(gates.values())==es['utility'];utility[epoch]=all(gates.values())
                start=q[q.epoch==0].set_index('seed');end=q[q.epoch==10].set_index('seed');gates={}
                for k in ['forward_gain','above_random']:
                    d=start[k]-end[k];audit_stats(d,ss['attenuation_statistics'][k],c);gates[k]=bool(d.mean()>=.001 and (d>0).all());pairs.extend(dict(cohort=cohort,seed=int(seed),mode=mode,contrast=k+'_attenuation',difference=float(val)) for seed,val in d.items())
                assert gates==ss['attenuation_gates'];endpoints=dict(attenuation=bool(identifiable and utility[0] and all(gates.values())),persistent_utility=bool(identifiable and utility[0] and utility[10]));assert all(ss[k]==v for k,v in endpoints.items())
                if mode=='noisy':ends[cohort]=endpoints
        frame_check(pairs,root/'paired-differences.csv',['cohort','seed','mode','contrast']);assert summary['confirmed']=={k:ends['discovery'][k] and ends['confirmation'][k] for k in ends['discovery']}
        assert summary['primary_identifiable']==all(summary[x]['noisy']['identifiable'] for x in ends);assert not summary['cohorts_pooled'] and summary['clean_secondary_only']
        for p,h in read(root/'prior-results-sha256.json').items():assert core.sha256(p)==h,p
        for p,h in m['context']['source_sha256'].items():assert hashlib.sha256(subprocess.check_output(['git','show',m['context']['git_commit']+':'+p])).hexdigest()==h,p
        assert cases==13 and count==50 and trajectories==11130 and len(rows)==700
        core.write_json(out/'checks.json',dict(all_checks_pass=True,learning_runs_replayed=cases,snapshots_replayed=count,probe_graphs=350,scalar_trajectories=trajectories,exact_scores=True,metric_rows=len(rows),confirmed=summary['confirmed'],primary_identifiable=summary['primary_identifiable'],prior_files_unchanged=len(read(root/'prior-results-sha256.json')),result_manifest_sha256=core.sha256(root/'manifest.json'),verifier_sha256=core.sha256(__file__),budget=budget.close()))
    except Exception as exc:core.write_json(out/'failure.json',dict(error=repr(exc)));raise
    finally:budget.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args();verify(a.root,a.out)
