"""Independent scalar replay of frozen proposals and all directional probes."""
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
from verify_neuron_panel import audit_stats


def reference_direction(initial,roles,bank,symbols,codes,c,seed):
    n=len(roles);row=np.repeat(np.arange(n),np.diff(initial.indptr));mask=(roles[row]=='MBON')&(roles[initial.indices]=='KC')
    post=row[mask];pre=initial.indices[mask];obs=np.flatnonzero(roles=='MBON');mag=abs(initial.data[mask]);sign=np.sign(initial.data[mask]);mass=np.bincount(post,weights=mag,minlength=n)
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
            proposed=np.maximum(c['plastic_floor']*mag,mag+c['plastic_learning_rate']*sign*delta)
            sums=np.bincount(post,weights=proposed,minlength=n);proposed*=mass[post]/sums[post]
            proposal=proposed-mag;total+=proposal;baseline+=c['baseline_rate']*adv
            for k,v in zip(logs,[action,reward,adv,float(np.linalg.norm(trace)),float(np.linalg.norm(proposal))]):values[k].append(v)
        for k in logs:logs[k].append(values[k])
        history.append(dict(epoch=epoch+1,reward_mean=float(np.mean(values['rewards'])),mean_proposal_norm=float(np.mean(values['proposal_norms']))))
    return total/(c['plastic_epochs']*(len(symbols)-c['warmup'])),{k:np.asarray(v) for k,v in logs.items()},history,mask,post


def reference_evaluation(weights,bank,obs,symbols,codes,c,seed,replicas,noisy):
    all_scores=[];features=[]
    for r in range(replicas):
        noise=np.random.default_rng(np.random.SeedSequence([seed,r])).normal(0,c['noise_std'],(len(symbols),len(obs))) if noisy else np.zeros((len(symbols),len(obs)))
        state=np.zeros(weights.shape[0]);scores=[];observed=[]
        for t,symbol in enumerate(symbols):
            drive=weights@state+bank[int(symbol)];drive[obs]+=noise[t]
            state=(1-c['leak'])*state+c['leak']*np.tanh(drive)
            if not noisy:observed.append(state[obs].copy())
            if t>=c['warmup']:scores.append(-np.mean((state[obs][None,:]-codes)**2,axis=1))
        all_scores.append(scores)
        if not noisy:features=observed
    return np.asarray(all_scores),np.asarray(features)


@threadpool_limits.wrap(limits=1)
def verify(root,out):
    m=check(root);c=m['config'];out.mkdir(parents=True,exist_ok=False);budget=Budget(c);rows=[];counts=0;trajectories=0;radii=[]
    try:
        for p in sorted(root.glob('*/*/checkpoint.npz')):
            dest=p.parent;cm=check(dest);i=cm['identity'];cc=cm['config'];raw,ids,roles=anatomy(i['circuit_seed']);initial=core.normalize_condition(raw,c['normalization'],c['gain']);initial.sort_indices()
            assert core.weight_hash(raw)==core.weight_hash(sparse.load_npz(dest/'raw.npz'))
            with np.load(p) as z:a=dict(z)
            bank=symbol_bank(roles,i['seed'],c['input_fraction'],c['input_amplitude'])[:4];np.testing.assert_array_equal(bank,a['input_patterns'])
            order=np.random.default_rng(i['code_seed']).permutation(48);codes=np.zeros((4,48))
            for k in range(4):codes[k,order[12*k:12*k+12]]=.25
            np.testing.assert_array_equal(codes,a['codebook']);obs=np.flatnonzero(roles=='MBON');np.testing.assert_array_equal(obs,a['observed_indices'])
            for split in ['train','test']:
                symbols=np.random.default_rng(i[split+'_seed']).integers(0,4,c['warmup']+cc[split+'_samples']).astype(np.uint8);np.testing.assert_array_equal(symbols,a[split+'_symbols'])
            mean,logs,history,mask,post=reference_direction(initial,roles,bank,a['train_symbols'],codes,cc,i['noise_seed'])
            np.testing.assert_array_equal(mean,a['mean_proposal']);np.testing.assert_array_equal(mask,a['plastic_mask']);assert history==read(dest/'history.json')
            for k in logs:np.testing.assert_array_equal(logs[k],a[k])
            mag=abs(initial.data[mask]);mass=np.bincount(post,weights=mag,minlength=len(roles));rng=np.random.default_rng(i['direction_seed'])
            rawvectors=[mean]+[rng.normal(size=len(mag))*mag for _ in range(c['random_directions'])]
            vectors=[v-mag*np.bincount(post,weights=v,minlength=len(roles))[post]/mass[post] for v in rawvectors]
            norms=np.linalg.norm(vectors,axis=1);unit=np.asarray(vectors)/np.where(norms>0,norms,1)[:,None];nominal=c['nominal_relative_radius']*np.linalg.norm(mag)
            if norms[0]==0:unit[:]=0;radius=0.
            else:
                lim=np.divide((1-c['plastic_floor'])*mag[None,:],abs(unit),out=np.full_like(unit,np.inf),where=unit!=0);radius=min(nominal,c['boundary_fraction']*float(lim.min()))
            np.testing.assert_array_equal(unit,a['unit_directions']);saved_radius=read(dest/'radius.json')
            expected=dict(radius=float(radius),nominal_radius=float(nominal),relative_radius=float(radius/np.linalg.norm(mag)),boundary_limited=bool(radius<nominal),resolution_limited=bool(radius<nominal/10),local_proposal_norm=float(norms[0]))
            assert expected==saved_radius;identity={k:i[k] for k in ['cohort','seed','circuit_seed']};radii.append(dict(**identity,**expected))
            truth=a['test_symbols'][np.arange(cc['warmup'],len(a['test_symbols']))-c['target_lag']];np.testing.assert_array_equal(truth,a['truth'])
            variants=[('baseline',0,0)]+[(name+'_'+side,j,sign) for j,name in enumerate(['local']+[f'random{k}' for k in range(c['random_directions'])]) for side,sign in [('plus',1),('minus',-1)]]
            for name,j,orientation in variants:
                weights=initial.copy();weights.data[mask]=np.sign(initial.data[mask])*(mag+orientation*radius*unit[j]);actual=sparse.load_npz(dest/(name+'-weights.npz'))
                assert core.weight_hash(weights)==core.weight_hash(actual)
                np.testing.assert_array_equal(np.sign(weights.data),np.sign(initial.data));np.testing.assert_array_equal(weights.data[~mask],initial.data[~mask])
                np.testing.assert_allclose(np.bincount(post,weights=abs(weights.data[mask]),minlength=len(roles)),mass,atol=1e-14,rtol=1e-12)
                assert np.all(abs(weights.data[mask])>=c['plastic_floor']*mag)
                np.testing.assert_allclose(np.linalg.norm(weights.data-initial.data),radius if orientation else 0,atol=1e-14,rtol=1e-10)
                for mode,replicas in [('noisy',cc['evaluation_replicates']),('clean',1)]:
                    scores,features=reference_evaluation(weights,bank,obs,a['test_symbols'],codes,cc,i['evaluation_seed'],replicas,mode=='noisy')
                    np.testing.assert_array_equal(scores,a[name+'_'+mode+'_scores']);accuracy=(scores.argmax(2)==truth[None,:]).mean(1);np.testing.assert_array_equal(accuracy,a[name+'_'+mode+'_accuracy'])
                    if mode=='clean':np.testing.assert_array_equal(features,a[name+'_features'])
                    rows.append(dict(**identity,variant=name,mode=mode,accuracy=float(accuracy.mean()),replicas=replicas));trajectories+=replicas
                budget.check()
            counts+=1;print(f'Independent frozen proposals + scalar probes: {counts}/13',flush=True)
        f=pd.DataFrame(rows);saved=pd.read_csv(root/'raw-metrics.csv');keys=['cohort','seed','circuit_seed','mode','variant']
        pd.testing.assert_frame_equal(f[saved.columns].sort_values(keys).reset_index(drop=True),saved.sort_values(keys).reset_index(drop=True),check_dtype=False,atol=1e-12,rtol=1e-12)
        saved=pd.read_csv(root/'radii.csv');pd.testing.assert_frame_equal(pd.DataFrame(radii)[saved.columns].sort_values(['cohort','seed','circuit_seed']).reset_index(drop=True),saved.sort_values(['cohort','seed','circuit_seed']).reset_index(drop=True),check_dtype=False,atol=1e-12,rtol=1e-12)
        effects=[]
        for identity,g in f.groupby(['cohort','seed','circuit_seed','mode']):
            values=g.set_index('variant').accuracy;plus=values['local_plus'];minus=values['local_minus'];randomabs=sum(abs(values[f'random{k}_plus']-values[f'random{k}_minus']) for k in range(c['random_directions']))/c['random_directions']
            effects.append(dict(zip(['cohort','seed','circuit_seed','mode'],identity),baseline=values['baseline'],local_plus=plus,local_minus=minus,local_directional=plus-minus,random_absolute=randomabs,above_random=plus-minus-randomabs,forward_gain=plus-values['baseline']))
        effects=pd.DataFrame(effects);saved=pd.read_csv(root/'circuit-effects.csv');pd.testing.assert_frame_equal(effects[saved.columns],saved,check_dtype=False,atol=1e-12,rtol=1e-12)
        cols=['baseline','local_plus','local_minus','local_directional','random_absolute','above_random','forward_gain'];blocks=effects.groupby(['cohort','seed','mode'])[cols].mean().reset_index();saved=pd.read_csv(root/'seed-blocks.csv');pd.testing.assert_frame_equal(blocks[saved.columns],saved,check_dtype=False,atol=1e-12,rtol=1e-12)
        summary=read(root/'summary.json');passed=[]
        for cohort in ['discovery','confirmation']:
            for mode in ['noisy','clean']:
                q=blocks[(blocks.cohort==cohort)&(blocks['mode']==mode)];s=summary[cohort][mode]
                for key in cols:audit_stats(q[key],s['statistics'][key],c)
                gates={key:bool(q[key].mean()>=threshold and (q[key]>0).all()) for key,threshold in [('local_directional',.0025),('above_random',.001),('forward_gain',.001)]};assert gates==s['gates'] and all(gates.values())==s['passes_all']
                if mode=='noisy':passed.append(all(gates.values()))
        assert all(passed)==summary['confirmed_noisy_alignment'] and summary['clean_secondary_only'] and not summary['cohorts_pooled']
        for p,h in read(root/'prior-results-sha256.json').items():assert core.sha256(p)==h,p
        for p,h in m['context']['source_sha256'].items():assert hashlib.sha256(subprocess.check_output(['git','show',m['context']['git_commit']+':'+p])).hexdigest()==h,p
        assert counts==13 and trajectories==5187
        core.write_json(out/'checks.json',dict(all_checks_pass=True,frozen_directions_replayed=counts,probe_graphs=169,scalar_trajectories=trajectories,exact_scores=True,metric_rows=len(rows),confirmed_noisy_alignment=summary['confirmed_noisy_alignment'],prior_files_unchanged=len(read(root/'prior-results-sha256.json')),result_manifest_sha256=core.sha256(root/'manifest.json'),verifier_sha256=core.sha256(__file__),budget=budget.close()))
        print('All reward-direction audits passed',flush=True)
    except Exception as exc:core.write_json(out/'failure.json',dict(error=repr(exc)));raise
    finally:budget.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args();verify(a.root,a.out)
