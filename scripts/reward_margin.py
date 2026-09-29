"""Fixed-code margin sensitivities; analysis gradients never enter learning."""
import argparse,hashlib,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits
from flying.training import whole_brain_memory as core
from alphabet_memory import read,check,symbol_bank
from local_reward import codes,fixed_scores
from reward_trajectory import capture_train,local_direction,stage_directions
from edge_panel import anatomy
from frozen_state_probe import Budget
from context_memory import estimate
from pathway_memory import seal

CONFIG_HASH='95b1cc1bc72940c439739e487e5028be63ed47967e37672b2b79b13a44fd55f3'
COHORTS=['discovery','confirmation','fresh_confirmation']
COLS=['local','random_mean','above_random','margin','accuracy']


def margin_scores(scores,truth):
    correct=np.take_along_axis(scores,truth[None,:,None],axis=2)[:,:,0]
    return correct-(scores.sum(2)-correct)/(scores.shape[2]-1)


def evaluate(weights,bank,obs,symbols,codebook,mask,unit,scale,c,seed,replicas,noisy):
    n=weights.shape[0];d=len(unit);row=np.repeat(np.arange(n),np.diff(weights.indptr));post=row[mask]
    perturbations=[]
    for u in unit:
        v=weights.copy();v.data[:]=0;v.data[mask]=np.sign(weights.data[mask])*u*scale;v.eliminate_zeros();perturbations.append(v)
    noise=np.stack([np.random.default_rng(np.random.SeedSequence([seed,r])).normal(0,c['noise_std'],(len(symbols),len(obs))) for r in range(replicas)],axis=1) if noisy else np.zeros((len(symbols),replicas,len(obs)))
    state=np.zeros((n,replicas));tangent=np.zeros((n,replicas,d));scores=[];slopes=[];features=[]
    coefficients=2*(codebook-(codebook.sum(0)[None,:]-codebook)/(len(codebook)-1))/len(obs)
    for t,symbol in enumerate(symbols):
        drive=weights@state+bank[int(symbol),:,None];drive[obs]+=noise[t].T;activation=np.tanh(drive)
        rhs=(weights@tangent.reshape(n,-1)).reshape(n,replicas,d)+np.stack([v@state for v in perturbations],axis=2)
        tangent=(1-c['leak'])*tangent+c['leak']*(1-activation**2)[:,:,None]*rhs
        state=(1-c['leak'])*state+c['leak']*activation
        if not noisy:features.append(state[obs,0].copy())
        if t>=c['warmup']:
            scores.append(fixed_scores(np.ascontiguousarray(state[obs].T),codebook))
            slopes.append(np.einsum('j,jrd->rd',coefficients[int(symbols[t-c['target_lag']])],tangent[obs]))
    scores=np.stack(scores,axis=1);slopes=np.stack(slopes,axis=1);truth=symbols[np.arange(c['warmup'],len(symbols))-c['target_lag']]
    return dict(scores=scores,margin=margin_scores(scores,truth),slopes=slopes,features=np.asarray(features),truth=truth)


def summarize(rows,c):
    f=pd.DataFrame(rows);records=[]
    for identity,g in f.groupby(['cohort','seed','circuit_seed','epoch','mode']):
        v=g.set_index('variant').slope;random=float(np.mean([v[f'random{k}'] for k in range(c['random_directions'])]))
        records.append(dict(zip(['cohort','seed','circuit_seed','epoch','mode'],identity),local=v['local'],random_mean=random,above_random=v['local']-random,margin=g.margin.iloc[0],accuracy=g.accuracy.iloc[0]))
    circuit=pd.DataFrame(records);blocks=circuit.groupby(['cohort','seed','epoch','mode'])[COLS].mean().reset_index();out={};pairs=[];eps=c['numerical_gate']
    for cohort in COHORTS:
        out[cohort]={}
        for mode in ['noisy','clean']:
            q=blocks[(blocks.cohort==cohort)&(blocks['mode']==mode)];by_epoch={}
            for epoch in c['epochs']:
                e=q[q.epoch==epoch];gates={k:bool(e[k].mean()>eps and (e[k]>eps).all()) for k in ['local','above_random']}
                by_epoch[str(epoch)]=dict(statistics={k:estimate(e[k],c) for k in COLS},gates=gates,utility=all(gates.values()))
            start=q[q.epoch==0].set_index('seed');end=q[q.epoch==10].set_index('seed');stats={};gates={}
            for k in ['local','above_random']:
                delta=start[k]-end[k];stats[k]=estimate(delta,c);gates[k]=bool(delta.mean()>eps and (delta>eps).all())
                pairs.extend(dict(cohort=cohort,seed=int(seed),mode=mode,contrast=k+'_attenuation',difference=float(v)) for seed,v in delta.items())
            out[cohort][mode]=dict(epochs=by_epoch,attenuation_statistics=stats,attenuation_gates=gates,attenuation=bool(by_epoch['0']['utility'] and all(gates.values())),persistent_utility=bool(by_epoch['0']['utility'] and by_epoch['10']['utility']))
    out['confirmed']={k:all(out[x]['noisy'][k] for x in COHORTS) for k in ['attenuation','persistent_utility']};out['cohorts_pooled']=False;out['clean_secondary_only']=True
    return circuit,blocks,pd.DataFrame(pairs),out


@threadpool_limits.wrap(limits=1)
def run(config,out):
    c=read(config);assert core.fingerprint(c)==CONFIG_HASH;assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip(),'Commit before outcomes'
    source=Path(c['source']);oldroot=check(source);assert core.sha256(source/'manifest.json')==c['source_manifest_sha256']
    learning=Path(c['learning_source']);lc=check(learning)['config'];assert core.sha256(learning/'manifest.json')==c['learning_source_manifest_sha256']
    context=core.source_context(c)
    names=['reward_margin','verify_reward_margin','plot_reward_margin','reward_trajectory','reward_direction','verify_reward_direction','local_reward','verify_local_reward','readout_dependency','alphabet_memory','edge_panel','frozen_state_probe','context_memory','pathway_memory','verify_neuron_panel']
    for p in [config,Path('docs/reward-margin-seed-audit.json'),Path('docs/act4-completion-plan.md')]+[Path('scripts')/(n+'.py') for n in names]:context['source_sha256'][p.as_posix()]=core.sha256(p)
    for p,h in context['source_sha256'].items():assert hashlib.sha256(subprocess.check_output(['git','show',context['git_commit']+':'+p])).hexdigest()==h,p
    out.mkdir(parents=True,exist_ok=False);budget=Budget(c)
    try:
        prior={p:core.sha256(p) for p in subprocess.check_output(['git','ls-files','results/'],text=True).splitlines()};core.write_json(out/'prior-results-sha256.json',prior);core.write_json(out/'config.json',c)
        cases=[]
        for p in sorted(source.glob('*/*/training-logs.npz'),key=lambda p:(p.parent.parent.name!='smoke',p.as_posix())):
            cm=check(p.parent);i=cm['identity'];cases.append((i,cm['config'],p.parent))
        for block in c['fresh_blocks']:
            for ci in c['circuit_seeds']:cases.append((dict(cohort='fresh_confirmation',circuit_seed=ci,**block),lc,None))
        assert len(cases)==19;rows=[];snapshots=0;trajectories=0
        for i,cc,archive in cases:
            ident={k:i[k] for k in ['cohort','seed','circuit_seed']};_,ids,roles=anatomy(i['circuit_seed']);dest=out/i['cohort']/f'c{i["circuit_seed"]}_s{i["seed"]}';dest.mkdir(parents=True)
            if archive is not None:
                cm=check(archive);origin=Path(cm['source_case']);check(origin);initial=sparse.load_npz(origin/'initial-weights.npz');seeds=cm['probe_seeds'];epochs=cm['epochs']
                with np.load(origin/'checkpoint.npz') as z:data={k:z[k] for k in ['input_patterns','observed_indices','codebook','train_symbols','test_symbols']}
                stages=[]
                for epoch in epochs:
                    stage=archive/f'epoch{epoch}';check(stage)
                    with np.load(stage/'checkpoint.npz') as z:stage_data={k:z[k] for k in ['mean_proposal','unit_directions','direction_norms','plastic_mask']}
                    stages.append((epoch,sparse.load_npz(stage/'baseline-weights.npz'),stage_data))
            else:
                raw,_,_=anatomy(i['circuit_seed']);initial=core.normalize_condition(raw,cc['normalization'],cc['gain']);initial.sort_indices();epochs=c['epochs'];seeds={k:i[k] for k in ['proposal_noise','direction','evaluation_noise']}
                data=dict(input_patterns=symbol_bank(roles,i['seed'],cc['input_fraction'],cc['input_amplitude'])[:4],observed_indices=np.flatnonzero(roles=='MBON'),codebook=codes(i['code_seed']))
                for split in ['train','test']:data[split+'_symbols']=np.random.default_rng(i[split+'_seed']).integers(0,4,cc['warmup']+cc[split+'_samples']).astype(np.uint8)
                weights,h,logs=capture_train(initial,roles,data['input_patterns'],data['train_symbols'],data['codebook'],cc,i['noise_seed'],epochs);core.write_json(dest/'training-history.json',h);np.savez_compressed(dest/'training-logs.npz',**logs);stages=[]
                for epoch in epochs:
                    mean,logs,h,model=local_direction(initial,weights[epoch],roles,data['input_patterns'],data['train_symbols'],data['codebook'],dict(cc,plastic_epochs=c['proposal_epochs']),seeds['proposal_noise'])
                    unit,norms=stage_directions(mean,abs(weights[epoch].data[model.mask]),model.post,len(roles),c,seeds['direction'])
                    stages.append((epoch,weights[epoch],dict(mean_proposal=mean,unit_directions=unit,direction_norms=norms,plastic_mask=model.mask,**logs)))
                    core.write_json(dest/f'proposal-history-epoch{epoch}.json',h);budget.check()
            np.savez_compressed(dest/'inputs.npz',**data);sparse.save_npz(dest/'initial-weights.npz',initial)
            for epoch,w,d in stages:
                stage=dest/f'epoch{epoch}';stage.mkdir();a=dict(**d);before=core.weight_hash(w);scale=float(c['reference_relative_step']*np.linalg.norm(abs(initial.data[d['plastic_mask']])));assert (d['direction_norms']>0).all()
                for mode,replicas in [('noisy',2 if i['cohort']=='smoke' else c['evaluation_replicates']),('clean',1)]:
                    values=evaluate(w,data['input_patterns'],data['observed_indices'],data['test_symbols'],data['codebook'],d['plastic_mask'],d['unit_directions'],scale,cc,seeds['evaluation_noise'],replicas,mode=='noisy')
                    for key,value in values.items():assert np.isfinite(value).all();a[mode+'_'+key]=value
                    if archive is not None:
                        with np.load(archive/f'epoch{epoch}/checkpoint.npz') as old:
                            np.testing.assert_array_equal(values['scores'],old['baseline_'+mode+'_scores'])
                            if mode=='clean':np.testing.assert_array_equal(values['features'],old['baseline_features'])
                    means=values['slopes'].mean(1);a[mode+'_mean_slopes']=means;accuracy=float(np.mean(values['scores'].argmax(2)==values['truth'][None,:]));margin=float(values['margin'].mean())
                    for j,name in enumerate(['local']+[f'random{k}' for k in range(c['random_directions'])]):rows.append(dict(**ident,epoch=epoch,mode=mode,variant=name,slope=float(means[:,j].mean()),margin=margin,accuracy=accuracy,replicas=replicas))
                    trajectories+=replicas;budget.check()
                assert core.weight_hash(w)==before;sparse.save_npz(stage/'weights.npz',w);np.savez_compressed(stage/'checkpoint.npz',**a)
                seal(stage,cc,context,identity=dict(**ident,epoch=epoch),reference_scale=scale,probe_seeds=seeds);snapshots+=1
            seal(dest,cc,context,identity=i,probe_seeds=seeds,epochs=epochs,archival_source=archive.as_posix() if archive else None)
            print(f'{i["cohort"]} c{i["circuit_seed"]} s{i["seed"]}: {len(epochs)} analytic checkpoints complete',flush=True)
        circuit,blocks,pairs,summary=summarize(rows,c)
        for name,data in [('raw-metrics',pd.DataFrame(rows)),('circuit-effects',circuit),('seed-blocks',blocks),('paired-differences',pairs)]:data.to_csv(out/(name+'.csv'),index=False)
        core.write_json(out/'summary.json',summary)
        for p,h in {**prior,**context['source_sha256']}.items():assert core.sha256(p)==h,p
        assert snapshots==74 and trajectories==2382 and len(rows)==888
        core.write_json(out/'verification.json',dict(cases=len(cases),snapshots=snapshots,trajectories=trajectories,directional_sensitivities=trajectories*6,metric_rows=len(rows),prior_files_unchanged=len(prior),budget=budget.close()))
        seal(out,c,context,environment=core.environment());print('Confirmed margin endpoints:',summary['confirmed'],flush=True)
    except Exception as exc:core.write_json(out/'failure.json',dict(error=repr(exc)));raise
    finally:budget.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=Path('configs/reward_margin.json'));p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.config,a.out)
