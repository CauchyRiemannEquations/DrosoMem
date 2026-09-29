"""Scalar reward and one local covariance eligibility rule; fixed decoder."""
import argparse, hashlib, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits
from flying.training import whole_brain_memory as core
from flying.brain.plasticity import KCMBONPlasticity
from flying.brain.local_reward import LocalRewardPlasticity
from alphabet_memory import read, check, symbol_bank, SymbolEncoder
from frozen_state_probe import labels, fit_fold, measures, Budget
from structural_controls import generate, audit
from edge_panel import anatomy
from context_memory import estimate
from pathway_memory import seal
import kc_ablation

CONFIG_HASH='16cd6a282f51a18d14d3cc2eb95332743d41c2f9818d45796cfe2639d3341e25'


def codes(seed):
    order=np.random.default_rng(seed).permutation(48); result=np.zeros((4,48))
    for k in range(4): result[k,order[12*k:12*(k+1)]]=.25
    return result


def fixed_scores(features, codebook):
    return -np.mean((features[:,None,:]-codebook[None,:,:])**2,axis=2)


def train(weights, roles, bank, symbols, codebook, arm, c, noise_seed, yoked=None):
    model=LocalRewardPlasticity(weights,roles,SymbolEncoder(bank),leak=c['leak'],microsteps=1,
        learning_rate=0 if arm=='frozen' else c['plastic_learning_rate'],floor=c['plastic_floor'],
        trace_decay=c['trace_decay'],activity_rate=c['activity_rate'],baseline_rate=c['baseline_rate'])
    warm=c['warmup']; truth=symbols[np.arange(warm,len(symbols))-c['target_lag']]
    rng=np.random.default_rng(noise_seed);history=[];logs={k:[] for k in ['actions','actual_rewards','applied_rewards','advantages','trace_norms']}
    if arm=='yoked':
        assert yoked.shape==(c['plastic_epochs'],len(truth))
        yoked=np.roll(yoked,len(truth)//2,axis=1)
    for epoch in range(c['plastic_epochs']):
        model.reset(); epoch_logs={k:[] for k in logs}
        for t,digit in enumerate(symbols):
            state=model.forward(int(digit),rng.normal(0,c['noise_std'],len(model.mbon)))
            if t<warm:
                model.eligibility.fill(0); model.pending=False
                continue
            action=int(fixed_scores(state[model.mbon][None,:],codebook).argmax(1)[0])
            reward=float(action==truth[t-warm]);applied=float(yoked[epoch,t-warm]) if arm=='yoked' else reward
            norm=float(np.linalg.norm(model.eligibility));adv=model.reinforce(applied)
            for k,v in zip(logs,[action,reward,applied,adv,norm]):epoch_logs[k].append(v)
        for k in logs:logs[k].append(epoch_logs[k])
        history.append(dict(epoch=epoch+1,online_accuracy=float(np.mean(epoch_logs['actual_rewards'])),
            applied_reward_mean=float(np.mean(epoch_logs['applied_rewards'])),mean_trace_norm=float(np.mean(epoch_logs['trace_norms']))))
    return model.weights.copy(),model.audit(),history,{k:np.asarray(v) for k,v in logs.items()}


def collect(weights, roles, bank, symbols, c):
    model=KCMBONPlasticity(weights,roles,SymbolEncoder(bank),leak=c['leak'],microsteps=1,learning_rate=0,floor=c['plastic_floor'])
    return core.collect(model,model.mbon,symbols,c['activity_epsilon'])


def execute(c, ci, block, network, arm, graph, yoked=None):
    raw, ids, roles=anatomy(ci); obs=np.flatnonzero(roles=='MBON')
    control,graph_audit=graph; weights=core.normalize_condition(control,c['normalization'],c['gain']); weights.sort_indices()
    bank=symbol_bank(roles,block['seed'],c['input_fraction'],c['input_amplitude'])[:4]; codebook=codes(block['code_seed'])
    symbols={split:np.random.default_rng(block[split+'_seed']).integers(0,4,c['warmup']+c[split+'_samples']).astype(np.uint8) for split in ['train','test']}
    final,plastic_audit,history,logs=train(weights,roles,bank,symbols['train'],codebook,arm,c,block['noise_seed'],yoked)
    a=dict(input_patterns=bank,codebook=codebook,observed_indices=obs,**logs)
    for split in ['train','test']:
        x,d=collect(final,roles,bank,symbols[split],c); replay,_=collect(final,roles,bank,symbols[split],c)
        np.testing.assert_array_equal(x,replay)
        a[split+'_symbols']=symbols[split]; a[split+'_features']=x
        a.update({split+'_'+k:v for k,v in d.items()})
    w=c['warmup']; x=a['train_features'][w:]
    a.update(fit_fold(x,a['test_features'][w:],labels(symbols['train'],np.arange(w,len(symbols['train'])),c['lags']),labels(symbols['test'],np.arange(w,len(symbols['test'])),c['lags']),c['alpha']))
    metrics=measures([a],c['lags'])
    for j,r in enumerate(metrics):r['training_mse']=float(np.mean((a['train_scores'][:,j]-np.eye(4)[a['ytrain'][:,j]])**2))
    j=c['lags'].index(c['target_lag']);a['fixed_scores']=fixed_scores(a['xtest'],codebook);a['fixed_train_scores']=fixed_scores(a['xtrain'],codebook)
    y=a['ytest'][:,j];ty=a['ytrain'][:,j];mode=np.bincount(ty,minlength=4).argmax()
    direct=dict(test_accuracy=float(np.mean(a['fixed_scores'].argmax(1)==y)),train_accuracy=float(np.mean(a['fixed_train_scores'].argmax(1)==ty)),frequency_accuracy=float(np.mean(y==mode)),trainable_parameters=0,lag=c['target_lag'])
    direct['frequency_excess']=direct['test_accuracy']-direct['frequency_accuracy']
    neural=dict(**core.state_diagnostics(x),sparsity=float(np.mean(abs(x)<=c['activity_epsilon'])),mean_active=float(a['train_active_counts'][w:].mean()))
    for value in a.values():assert np.isfinite(value).all()
    return a,metrics,direct,neural,history,plastic_audit,weights,final,graph_audit


def blocks_of(rows,c):
    f=pd.DataFrame(rows);r=f[(f.decoder=='fixed')|((f.decoder=='ridge')&(f.lag==c['target_lag']))]
    cols=['test_accuracy','frequency_excess','null_excess','r2_vs_frequency']
    return r.groupby(['cohort','seed','network','arm','decoder'])[cols].mean().reset_index()


def inference(blocks,c):
    out={};differences=[]
    for cohort in ['discovery','confirmation']:
        q=blocks[blocks.cohort==cohort];cells={};series={};access={}
        for (network,arm,decoder),v in q.groupby(['network','arm','decoder']):
            key=f'{network}/{arm}/{decoder}';v=v.set_index('seed');series[key]=v.test_accuracy
            cells[key]=estimate(v.test_accuracy,c)
            access[key]=bool((v.frequency_excess>=.05).all() and ((v.null_excess>=.05).all() and v.r2_vs_frequency.mean()>0 if decoder=='ridge' else True))
        contrasts={};gates={}
        for decoder in ['fixed','ridge']:
            for control in ['frozen','yoked']:
                name=decoder+'_vs_'+control;delta=series['real/contingent/'+decoder]-series['real/'+control+'/'+decoder]
                contrasts[name]=estimate(delta,c);gates[name]=bool(delta.mean()>=.05 and (delta>0).all())
                differences.extend(dict(cohort=cohort,seed=int(seed),contrast=name,difference=float(value)) for seed,value in delta.items())
        endpoints=dict(reward_coding=access['real/contingent/fixed'] and gates['fixed_vs_frozen'] and gates['fixed_vs_yoked'],
            representation_improvement=access['real/contingent/ridge'] and gates['ridge_vs_frozen'] and gates['ridge_vs_yoked'],
            frozen_representation_access=access['real/frozen/ridge'])
        out[cohort]=dict(cells=cells,access=access,contrasts=contrasts,gates=gates,endpoints=endpoints)
    out['confirmed']={k:bool(out['discovery']['endpoints'][k] and out['confirmation']['endpoints'][k]) for k in out['discovery']['endpoints']}
    out['cohorts_pooled']=False
    return out,pd.DataFrame(differences)


def previous_positive_baseline(c):
    from verify_readout_dependency import reference_train
    source=Path(c['source']);check(source);assert core.sha256(source/'manifest.json')==c['source_manifest_sha256']
    dest=source/'discovery/real_aligned_c701_s221142';m=check(dest)
    initial=sparse.load_npz(dest/'initial-weights.npz');final=sparse.load_npz(dest/'final-weights.npz')
    _,_,roles=anatomy(701)
    with np.load(dest/'checkpoint.npz') as z:a=dict(z)
    replay,h=reference_train(initial,roles,a['input_patterns'],a['train_symbols'],a['codebook'],'aligned',m['config'])
    np.testing.assert_array_equal(replay.data,final.data);assert h==read(dest/'history.json')
    x,_=collect(final,roles,a['input_patterns'],a['test_symbols'],m['config']);np.testing.assert_array_equal(x,a['test_features'])
    acc=float(np.mean(fixed_scores(x[m['config']['warmup']:],a['codebook']).argmax(1)==a['ytest'][:,m['config']['lags'].index(2)]))
    assert acc==read(dest/'fixed-metrics.json')['test_accuracy']
    return dict(source=str(dest),exact_training_replay=True,expected_accuracy=read(dest/'fixed-metrics.json')['test_accuracy'],actual_accuracy=acc,manifest_sha256=core.sha256(dest/'manifest.json'))


@threadpool_limits.wrap(limits=1)
def run(config,out):
    c=read(config);assert core.fingerprint(c)==CONFIG_HASH
    assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip(),'Commit before outcomes'
    context=core.source_context(c)
    names=['local_reward','verify_local_reward','readout_dependency','verify_readout_dependency','structural_controls','alphabet_memory','frozen_state_probe','context_memory','edge_panel','edge_masks','pathway_memory','pathway_masks','kc_ablation','kc_frozen','structural_k4','normalization_transfer','verify_neuron_panel']
    for p in [config,Path('docs/local-reward-seed-audit.json')]+[Path('scripts')/(n+'.py') for n in names]:context['source_sha256'][p.as_posix()]=core.sha256(p)
    for p,h in context['source_sha256'].items():assert hashlib.sha256(subprocess.check_output(['git','show',context['git_commit']+':'+p])).hexdigest()==h,p
    out.mkdir(exist_ok=False);budget=Budget(c)
    try:
        prior={p:core.sha256(p) for p in subprocess.check_output(['git','ls-files','results/'],text=True).splitlines()};core.write_json(out/'prior-results-sha256.json',prior);core.write_json(out/'config.json',c)
        core.write_json(out/'baseline.json',previous_positive_baseline(c));print('Historical artificial-teacher positive baseline exactly reproduced',flush=True)
        rows=[];neural=[];count=0
        for cohort,settings in c['cohorts'].items():
            cc=dict(c,**settings)
            for block in cc['blocks']:
                for ci in cc['circuit_seeds']:
                    raw,ids,roles=anatomy(ci)
                    for network in c['networks']:
                        family='intact' if network=='real' else network
                        control=raw.copy();ga=dict(invariants=audit(raw,control,roles,'intact'));yoked=None;reward_source=None
                        for arm in c['training_arms']:
                            budget.check();a,metrics,direct,n,h,pa,initial,final,ga=execute(cc,ci,block,network,arm,(control,ga),yoked)
                            ident=dict(cohort=cohort,seed=block['seed'],circuit_seed=ci,network=network,arm=arm)
                            dest=out/cohort/f'{network}_{arm}_c{ci}_s{block["seed"]}';dest.mkdir(parents=True)
                            np.savez_compressed(dest/'checkpoint.npz',**a);sparse.save_npz(dest/'initial-weights.npz',initial);sparse.save_npz(dest/'final-weights.npz',final);sparse.save_npz(dest/'raw.npz',control)
                            for name,value in [('metrics',metrics),('fixed-metrics',direct),('neural',n),('history',h),('plastic-audit',pa),('graph-audit',ga)]:core.write_json(dest/(name+'.json'),value)
                            rows.extend(dict(**ident,decoder='ridge',**x) for x in metrics);rows.append(dict(**ident,decoder='fixed',**direct));neural.append(dict(**ident,**n));count+=1
                            seal(dest,cc,context,identity={**ident,**block},observation_ids=[ids[i] for i in a['observed_indices']],evaluation_trajectory_replayed=True,reward_source=reward_source if arm=='yoked' else None)
                            if arm=='contingent':
                                yoked=a['actual_rewards'].copy();reward_source=dict(path=dest.as_posix(),checkpoint_sha256=core.sha256(dest/'checkpoint.npz'))
                        print(f'{cohort} c{ci} s{block["seed"]} {network}:3 training arms',flush=True)
        f=pd.DataFrame(rows);f.to_csv(out/'raw-metrics.csv',index=False);b=blocks_of(rows,c);b.to_csv(out/'seed-blocks.csv',index=False)
        summary,pairs=inference(b,c);core.write_json(out/'summary.json',summary);pairs.to_csv(out/'paired-differences.csv',index=False);pd.DataFrame(neural).to_csv(out/'neural-diagnostics.csv',index=False)
        for p,h in {**prior,**context['source_sha256']}.items():assert core.sha256(p)==h,p
        assert count==39;budget.check();core.write_json(out/'verification.json',dict(cases=count,primary_decoder_evaluations=count*2,prior_files_unchanged=len(prior),budget=budget.close()))
        seal(out,c,context,environment=core.environment());print(summary['confirmed'],flush=True)
    except Exception as exc:core.write_json(out/'failure.json',dict(error=repr(exc)));raise
    finally:budget.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=Path('configs/local_reward.json'));p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.config,a.out)
