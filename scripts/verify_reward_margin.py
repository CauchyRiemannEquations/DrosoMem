"""Scalar state replay and reverse-adjoint audit independent of forward tangents."""
import argparse,hashlib,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits
from alphabet_memory import read,check,symbol_bank
from flying.training import whole_brain_memory as core
from edge_panel import anatomy
from frozen_state_probe import Budget
from verify_local_reward import reference_train
from verify_reward_trajectory import reference_direction,frame_check
from verify_neuron_panel import audit_stats


def reference(weights,bank,obs,symbols,codes,mask,unit,scale,c,seed,replicas,noisy):
    n=weights.shape[0];row=np.repeat(np.arange(n),np.diff(weights.indptr));post=row[mask];pre=weights.indices[mask];sign=np.sign(weights.data[mask]);warm=c['warmup'];length=len(symbols)-warm
    truth=symbols[np.arange(warm,len(symbols))-c['target_lag']];all_scores=[];all_margin=[];all_derivatives=[];features=[]
    for r in range(replicas):
        noise=np.random.default_rng(np.random.SeedSequence([seed,r])).normal(0,c['noise_std'],(len(symbols),len(obs))) if noisy else np.zeros((len(symbols),len(obs)))
        state=np.zeros(n);previous=[];activations=[];scores=[];observed=[]
        for t,symbol in enumerate(symbols):
            previous.append(state.copy());drive=weights@state+bank[int(symbol)];drive[obs]+=noise[t];act=np.tanh(drive);state=(1-c['leak'])*state+c['leak']*act;activations.append(act)
            if not noisy:observed.append(state[obs].copy())
            if t>=warm:scores.append(-np.mean((state[obs][None,:]-codes)**2,axis=1))
        scores=np.asarray(scores);correct=scores[np.arange(length),truth];margins=correct-(scores.sum(1)-correct)/(len(codes)-1)
        adjoint=np.zeros(n);derivative=np.zeros(len(unit));transpose=weights.T.tocsr()
        for t in range(len(symbols)-1,-1,-1):
            if t>=warm:
                y=int(symbols[t-c['target_lag']]);coefficient=2*(codes[y]-(codes.sum(0)-codes[y])/(len(codes)-1))/len(obs)
                adjoint[obs]+=coefficient/length
            local=c['leak']*(1-activations[t]**2)*adjoint
            derivative+=(unit@(sign*local[post]*previous[t][pre]))*scale
            adjoint=(1-c['leak'])*adjoint+transpose@local
        all_scores.append(scores);all_margin.append(margins);all_derivatives.append(derivative)
        if not noisy:features=observed
    return dict(scores=np.asarray(all_scores),margin=np.asarray(all_margin),mean_slopes=np.asarray(all_derivatives),features=np.asarray(features),truth=truth)


@threadpool_limits.wrap(limits=1)
def verify(root,out):
    m=check(root);c=m['config'];out.mkdir(parents=True,exist_ok=False);budget=Budget(c);rows=[];cases=0;count=0;trajectories=0;max_error=0.
    try:
        for dest in sorted(p.parent for p in root.glob('*/*/inputs.npz')):
            cm=check(dest);i=cm['identity'];cc=cm['config'];seeds=cm['probe_seeds'];_,_,roles=anatomy(i['circuit_seed']);initial=sparse.load_npz(dest/'initial-weights.npz');archive=Path(cm['archival_source']) if cm['archival_source'] else None
            with np.load(dest/'inputs.npz') as z:data=dict(z)
            if archive:
                oldcm=check(archive);assert oldcm['config']==cc and oldcm['probe_seeds']==seeds and oldcm['epochs']==cm['epochs']
                origin=Path(oldcm['source_case']);check(origin);assert core.weight_hash(initial)==core.weight_hash(sparse.load_npz(origin/'initial-weights.npz'))
                with np.load(origin/'checkpoint.npz') as z:
                    for k in data:np.testing.assert_array_equal(data[k],z[k])
            else:
                block=next(b for b in c['fresh_blocks'] if b['seed']==i['seed']);assert all(i[k]==v for k,v in block.items());assert seeds=={k:i[k] for k in seeds}
                raw,_,_=anatomy(i['circuit_seed']);expected=core.normalize_condition(raw,cc['normalization'],cc['gain']);expected.sort_indices();assert core.weight_hash(initial)==core.weight_hash(expected)
                np.testing.assert_array_equal(data['input_patterns'],symbol_bank(roles,i['seed'],cc['input_fraction'],cc['input_amplitude'])[:4]);obs=np.flatnonzero(roles=='MBON');np.testing.assert_array_equal(obs,data['observed_indices'])
                order=np.random.default_rng(i['code_seed']).permutation(48);codes=np.zeros((4,48))
                for k in range(4):codes[k,order[12*k:12*k+12]]=.25
                np.testing.assert_array_equal(data['codebook'],codes)
                for split in ['train','test']:np.testing.assert_array_equal(data[split+'_symbols'],np.random.default_rng(i[split+'_seed']).integers(0,4,cc['warmup']+cc[split+'_samples']).astype(np.uint8))
                with np.load(dest/'training-logs.npz') as z:training_logs=dict(z)
            ident={k:i[k] for k in ['cohort','seed','circuit_seed']}
            for epoch in cm['epochs']:
                stage=dest/f'epoch{epoch}';sm=check(stage);assert sm['identity']==dict(**ident,epoch=epoch);w=sparse.load_npz(stage/'weights.npz')
                with np.load(stage/'checkpoint.npz') as z:a=dict(z)
                if archive:
                    original_stage=archive/f'epoch{epoch}';check(original_stage);assert core.weight_hash(w)==core.weight_hash(sparse.load_npz(original_stage/'baseline-weights.npz'))
                    with np.load(original_stage/'checkpoint.npz') as z:
                        for k in ['mean_proposal','unit_directions','direction_norms','plastic_mask']:np.testing.assert_array_equal(a[k],z[k])
                else:
                    if epoch==0:expected=initial
                    else:
                        expected,h,logs=reference_train(initial,roles,data['input_patterns'],data['train_symbols'],data['codebook'],'contingent',dict(cc,plastic_epochs=epoch),i['noise_seed'])
                        assert h==read(dest/'training-history.json')[:epoch]
                        for k in logs:np.testing.assert_array_equal(logs[k],training_logs[k][:epoch])
                    assert core.weight_hash(w)==core.weight_hash(expected)
                    mean,logs,h,mask,post=reference_direction(initial,w,roles,data['input_patterns'],data['train_symbols'],data['codebook'],dict(cc,plastic_epochs=c['proposal_epochs']),seeds['proposal_noise'])
                    np.testing.assert_array_equal(mean,a['mean_proposal']);np.testing.assert_array_equal(mask,a['plastic_mask']);assert h==read(dest/f'proposal-history-epoch{epoch}.json')
                    for k in logs:np.testing.assert_array_equal(logs[k],a[k])
                    mag=abs(w.data[mask]);mass=np.bincount(post,weights=mag,minlength=len(roles));rng=np.random.default_rng(seeds['direction']);raw=[mean]+[rng.normal(size=len(mag))*mag for _ in range(c['random_directions'])]
                    vectors=[v-mag*np.bincount(post,weights=v,minlength=len(roles))[post]/mass[post] for v in raw];norms=np.linalg.norm(vectors,axis=1);unit=np.asarray(vectors)/np.where(norms>0,norms,1)[:,None]
                    np.testing.assert_array_equal(unit,a['unit_directions']);np.testing.assert_array_equal(norms,a['direction_norms'])
                scale=c['reference_relative_step']*np.linalg.norm(abs(initial.data[a['plastic_mask']]));assert scale==sm['reference_scale'];before=core.weight_hash(w)
                for mode,replicas in [('noisy',2 if i['cohort']=='smoke' else c['evaluation_replicates']),('clean',1)]:
                    ref=reference(w,data['input_patterns'],data['observed_indices'],data['test_symbols'],data['codebook'],a['plastic_mask'],a['unit_directions'],scale,cc,seeds['evaluation_noise'],replicas,mode=='noisy')
                    for key in ['scores','features','truth']:np.testing.assert_array_equal(ref[key],a[mode+'_'+key])
                    np.testing.assert_allclose(ref['margin'],a[mode+'_margin'],atol=1e-15,rtol=1e-13)
                    np.testing.assert_array_equal(a[mode+'_slopes'].mean(1),a[mode+'_mean_slopes'])
                    np.testing.assert_allclose(ref['mean_slopes'],a[mode+'_mean_slopes'],atol=c['derivative_atol'],rtol=c['derivative_rtol'])
                    error=float(np.max(abs(ref['mean_slopes']-a[mode+'_mean_slopes'])));assert error<=c['numerical_gate']/100;max_error=max(max_error,error)
                    accuracy=float(np.mean(ref['scores'].argmax(2)==ref['truth'][None,:]));margin=float(ref['margin'].mean())
                    for j,name in enumerate(['local']+[f'random{k}' for k in range(c['random_directions'])]):rows.append(dict(**ident,epoch=epoch,mode=mode,variant=name,slope=float(ref['mean_slopes'][:,j].mean()),margin=margin,accuracy=accuracy,replicas=replicas))
                    if archive:
                        with np.load(archive/f'epoch{epoch}/checkpoint.npz') as z:np.testing.assert_array_equal(ref['scores'],z['baseline_'+mode+'_scores'])
                    trajectories+=replicas;budget.check()
                assert core.weight_hash(w)==before;count+=1
            cases+=1;print(f'Scalar scores + reverse-adjoint means: {cases}/19 blocks',flush=True)
        frame_check(rows,root/'raw-metrics.csv',['cohort','seed','circuit_seed','epoch','mode','variant']);f=pd.DataFrame(rows);effects=[];cols=['local','random_mean','above_random','margin','accuracy'];eps=c['numerical_gate']
        for identity,g in f.groupby(['cohort','seed','circuit_seed','epoch','mode']):
            v=g.set_index('variant').slope;random=sum(v[f'random{k}'] for k in range(c['random_directions']))/c['random_directions'];effects.append(dict(zip(['cohort','seed','circuit_seed','epoch','mode'],identity),local=v['local'],random_mean=random,above_random=v['local']-random,margin=g.margin.iloc[0],accuracy=g.accuracy.iloc[0]))
        frame_check(effects,root/'circuit-effects.csv',['cohort','seed','circuit_seed','epoch','mode']);blocks=pd.DataFrame(effects).groupby(['cohort','seed','epoch','mode'])[cols].mean().reset_index();frame_check(blocks,root/'seed-blocks.csv',['cohort','seed','epoch','mode']);summary=read(root/'summary.json');ends={};pairs=[]
        for cohort in ['discovery','confirmation','fresh_confirmation']:
            ends[cohort]={}
            for mode in ['noisy','clean']:
                q=blocks[(blocks.cohort==cohort)&(blocks['mode']==mode)];ss=summary[cohort][mode];utility={}
                for epoch in c['epochs']:
                    v=q[q.epoch==epoch];es=ss['epochs'][str(epoch)]
                    for k in cols:audit_stats(v[k],es['statistics'][k],c)
                    gates={k:bool(v[k].mean()>eps and (v[k]>eps).all()) for k in ['local','above_random']};assert gates==es['gates'] and all(gates.values())==es['utility'];utility[epoch]=all(gates.values())
                start=q[q.epoch==0].set_index('seed');end=q[q.epoch==10].set_index('seed');gates={}
                for k in ['local','above_random']:
                    d=start[k]-end[k];audit_stats(d,ss['attenuation_statistics'][k],c);gates[k]=bool(d.mean()>eps and (d>eps).all());pairs.extend(dict(cohort=cohort,seed=int(seed),mode=mode,contrast=k+'_attenuation',difference=float(val)) for seed,val in d.items())
                assert gates==ss['attenuation_gates'];ep=dict(attenuation=bool(utility[0] and all(gates.values())),persistent_utility=bool(utility[0] and utility[10]));assert all(ss[k]==v for k,v in ep.items())
                if mode=='noisy':ends[cohort]=ep
        frame_check(pairs,root/'paired-differences.csv',['cohort','seed','mode','contrast']);assert summary['confirmed']=={k:all(ends[x][k] for x in ends) for k in ['attenuation','persistent_utility']};assert not summary['cohorts_pooled'] and summary['clean_secondary_only']
        for p,h in read(root/'prior-results-sha256.json').items():assert core.sha256(p)==h,p
        for p,h in m['context']['source_sha256'].items():assert hashlib.sha256(subprocess.check_output(['git','show',m['context']['git_commit']+':'+p])).hexdigest()==h,p
        assert cases==19 and count==74 and trajectories==2382 and len(rows)==888
        core.write_json(out/'checks.json',dict(all_checks_pass=True,cases=cases,snapshots=count,scalar_trajectories=trajectories,adjoint_directional_means=trajectories*6,exact_scores=True,per_time_slopes_independently_replayed=False,maximum_adjoint_error=max_error,fresh_training_runs=6,confirmed=summary['confirmed'],prior_files_unchanged=len(read(root/'prior-results-sha256.json')),result_manifest_sha256=core.sha256(root/'manifest.json'),verifier_sha256=core.sha256(__file__),budget=budget.close()))
    except Exception as exc:core.write_json(out/'failure.json',dict(error=repr(exc)));raise
    finally:budget.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args();verify(a.root,a.out)
