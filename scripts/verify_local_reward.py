"""Independent internal-update, dynamics and decoder audit for ACT IV-A."""
import argparse,hashlib,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits
from alphabet_memory import read,check,symbol_bank
from flying.training import whole_brain_memory as core
from frozen_state_probe import verify_fold,Budget
from verify_neuron_panel import audit_metrics,audit_stats
from edge_panel import anatomy
from structural_controls import generate,audit


def reference_train(initial,roles,bank,symbols,codebook,arm,c,noise_seed,yoked=None):
    weights=initial.copy();n=len(roles);row=np.repeat(np.arange(n),np.diff(weights.indptr))
    mask=(roles[row]=='MBON')&(roles[weights.indices]=='KC');post=row[mask];pre=weights.indices[mask]
    obs=np.flatnonzero(roles=='MBON');orig=abs(initial.data[mask]);sign=np.sign(initial.data[mask]);mass=np.bincount(post,weights=orig,minlength=n)
    truth=symbols[np.arange(c['warmup'],len(symbols))-c['target_lag']];rng=np.random.default_rng(noise_seed)
    history=[];logs={k:[] for k in ['actions','actual_rewards','applied_rewards','advantages','trace_norms']}
    for epoch in range(c['plastic_epochs']):
        state=np.zeros(n);average=np.zeros(n);trace=np.zeros(len(pre));baseline=.25;values={k:[] for k in logs}
        for t,digit in enumerate(symbols):
            old=state.copy();drive=weights@old+bank[int(digit)];drive[obs]+=rng.normal(0,c['noise_std'],len(obs))
            activation=np.tanh(drive);state=(1-c['leak'])*old+c['leak']*activation
            trace=c['trace_decay']*trace+(1-c['trace_decay'])*(old[pre]*(activation[post]-average[post]))
            average+=c['activity_rate']*(activation-average)
            if t<c['warmup']:trace.fill(0);continue
            action=int(np.mean((state[obs][None,:]-codebook)**2,axis=1).argmin());reward=float(action==truth[t-c['warmup']])
            applied=float(yoked[epoch,(t-c['warmup']-len(truth)//2)%len(truth)]) if arm=='yoked' else reward
            advantage=applied-baseline;norm=float(np.linalg.norm(trace))
            if arm!='frozen':
                energy=np.bincount(post,weights=old[pre]**2,minlength=n)
                delta=advantage*trace/(1+energy[post])
                mag=np.maximum(c['plastic_floor']*orig,abs(weights.data[mask])+c['plastic_learning_rate']*sign*delta)
                total=np.bincount(post,weights=mag,minlength=n);mag*=mass[post]/total[post];weights.data[mask]=sign*mag
            baseline+=c['baseline_rate']*advantage
            for k,v in zip(logs,[action,reward,applied,advantage,norm]):values[k].append(v)
        for k in logs:logs[k].append(values[k])
        history.append(dict(epoch=epoch+1,online_accuracy=float(np.mean(values['actual_rewards'])),applied_reward_mean=float(np.mean(values['applied_rewards'])),mean_trace_norm=float(np.mean(values['trace_norms']))))
    np.testing.assert_array_equal(initial.data[~mask],weights.data[~mask]);np.testing.assert_array_equal(np.sign(initial.data),np.sign(weights.data))
    np.testing.assert_allclose(np.bincount(post,weights=abs(weights.data[mask]),minlength=n),mass,atol=1e-14,rtol=1e-12)
    return weights,history,{k:np.asarray(v) for k,v in logs.items()}


def trajectory(weights,bank,obs,symbols,leak):
    state=np.zeros(weights.shape[0]);features=[];active=[];norm=[]
    for digit in symbols:
        previous=state.copy();activation=np.tanh(weights@previous+bank[int(digit)])
        state=(1-leak)*previous+leak*activation;features.append(state[obs].copy());active.append(np.count_nonzero(abs(state)>1e-8));norm.append(np.linalg.norm(state))
    return np.asarray(features),np.asarray(active),np.asarray(norm)


@threadpool_limits.wrap(limits=1)
def verify(root,out):
    m=check(root);c=m['config'];out.mkdir(exist_ok=False);budget=Budget(c)
    try:
        count=0;rows=[]
        for path in sorted(root.glob('*/*/checkpoint.npz')):
            dest=path.parent;cm=check(dest);i=cm['identity'];cc=cm['config'];raw,ids,roles=anatomy(i['circuit_seed'])
            control=raw.copy();assert core.weight_hash(control)==core.weight_hash(sparse.load_npz(dest/'raw.npz'))
            assert dict(invariants=audit(raw,control,roles,'intact'))==read(dest/'graph-audit.json')
            initial=core.normalize_condition(control,c['normalization'],c['gain']);initial.sort_indices()
            assert core.weight_hash(initial)==core.weight_hash(sparse.load_npz(dest/'initial-weights.npz'))
            with np.load(path) as z:a=dict(z)
            bank=symbol_bank(roles,i['seed'],c['input_fraction'],c['input_amplitude'])[:4];np.testing.assert_array_equal(bank,a['input_patterns'])
            order=np.random.default_rng(i['code_seed']).permutation(48);codes=np.zeros((4,48))
            for k in range(4):codes[k,order[12*k:12*k+12]]=.25
            np.testing.assert_array_equal(codes,a['codebook']);obs=np.flatnonzero(roles=='MBON');np.testing.assert_array_equal(obs,a['observed_indices'])
            yoked=None
            if i['arm']=='yoked':
                parent=dest.parent/f'real_contingent_c{i["circuit_seed"]}_s{i["seed"]}'
                assert cm['reward_source']==dict(path=parent.as_posix(),checkpoint_sha256=core.sha256(parent/'checkpoint.npz'))
                with np.load(parent/'checkpoint.npz') as z:yoked=z['actual_rewards'].copy()
                np.testing.assert_array_equal(a['applied_rewards'],np.roll(yoked,cc['train_samples']//2,axis=1))
                np.testing.assert_array_equal(a['applied_rewards'].sum(1),yoked.sum(1))
            trained,h,logs=reference_train(initial,roles,bank,a['train_symbols'],codes,i['arm'],cc,i['noise_seed'],yoked);final=sparse.load_npz(dest/'final-weights.npz')
            for key,value in logs.items():np.testing.assert_array_equal(value,a[key])
            np.testing.assert_array_equal(trained.data,final.data);assert h==read(dest/'history.json')
            assert read(dest/'plastic-audit.json')['final_weight_sha256']==core.weight_hash(final)
            for split in ['train','test']:
                symbols=np.random.default_rng(i[split+'_seed']).integers(0,4,cc['warmup']+cc[split+'_samples']).astype(np.uint8);np.testing.assert_array_equal(symbols,a[split+'_symbols'])
                x,active,norm=trajectory(final,bank,obs,symbols,c['leak']);np.testing.assert_array_equal(x,a[split+'_features']);np.testing.assert_array_equal(active,a[split+'_active_counts']);np.testing.assert_array_equal(norm,a[split+'_full_norm'])
                np.testing.assert_array_equal(x[cc['warmup']:],a['x'+split]);truth=np.column_stack([symbols[np.arange(cc['warmup'],len(symbols))-lag] for lag in c['lags']]);np.testing.assert_array_equal(truth,a['y'+split])
            verify_fold(a,c['alpha']);metrics=read(dest/'metrics.json');audit_metrics(a,metrics)
            direct=read(dest/'fixed-metrics.json');j=c['lags'].index(c['target_lag'])
            for split,key in [('test','fixed_scores'),('train','fixed_train_scores')]:
                scores=-np.array([[np.mean((x-code)**2) for code in codes] for x in a['x'+split]])
                np.testing.assert_array_equal(scores,a[key]);assert float(np.mean(scores.argmax(1)==a['y'+split][:,j]))==direct[split+'_accuracy']
            freq=float(np.mean(a['ytest'][:,j]==np.bincount(a['ytrain'][:,j],minlength=4).argmax()))
            assert direct['frequency_accuracy']==freq and direct['frequency_excess']==direct['test_accuracy']-freq and direct['trainable_parameters']==0
            n=read(dest/'neural.json')
            for key,value in core.state_diagnostics(a['xtrain']).items():np.testing.assert_allclose(n[key],value,atol=1e-12,rtol=1e-12)
            identity={k:i[k] for k in ['cohort','seed','circuit_seed','network','arm']};rows.extend(dict(**identity,decoder='ridge',**r) for r in metrics);rows.append(dict(**identity,decoder='fixed',**direct));count+=1;budget.check()
            if count%3==0:print(f'Independently replayed local updates and decoders: {count}/39',flush=True)
        f=pd.DataFrame(rows);saved=pd.read_csv(root/'raw-metrics.csv');keys=['cohort','seed','circuit_seed','network','arm','decoder','lag']
        pd.testing.assert_frame_equal(f[saved.columns].sort_values(keys).reset_index(drop=True),saved.sort_values(keys).reset_index(drop=True),check_dtype=False,atol=1e-12,rtol=1e-12)
        b=f[(f.decoder=='fixed')|((f.decoder=='ridge')&(f.lag==2))].groupby(['cohort','seed','network','arm','decoder'])[['test_accuracy','frequency_excess','null_excess','r2_vs_frequency']].mean().reset_index()
        saved=pd.read_csv(root/'seed-blocks.csv');pd.testing.assert_frame_equal(b[saved.columns],saved,check_dtype=False,atol=1e-12,rtol=1e-12)
        summary=read(root/'summary.json');ends={};pairrows=[]
        mapping={decoder+'_vs_'+control:('real/contingent/'+decoder,'real/'+control+'/'+decoder) for decoder in ['fixed','ridge'] for control in ['frozen','yoked']}
        for cohort in ['discovery','confirmation']:
            q=b[b.cohort==cohort];ss=summary[cohort];series={};access={};gates={}
            for (network,arm,decoder),v in q.groupby(['network','arm','decoder']):
                key=f'{network}/{arm}/{decoder}';v=v.set_index('seed');series[key]=v.test_accuracy;audit_stats(v.test_accuracy,ss['cells'][key],c)
                access[key]=bool((v.frequency_excess>=.05).all() and ((v.null_excess>=.05).all() and v.r2_vs_frequency.mean()>0 if decoder=='ridge' else True))
            assert access==ss['access']
            for name,(a,z) in mapping.items():
                d=series[a]-series[z];audit_stats(d,ss['contrasts'][name],c);gates[name]=bool(d.mean()>=.05 and (d>0).all())
                pairrows.extend(dict(cohort=cohort,seed=int(seed),contrast=name,difference=float(val)) for seed,val in d.items())
            assert gates==ss['gates']
            end=dict(reward_coding=access['real/contingent/fixed'] and gates['fixed_vs_frozen'] and gates['fixed_vs_yoked'],representation_improvement=access['real/contingent/ridge'] and gates['ridge_vs_frozen'] and gates['ridge_vs_yoked'],frozen_representation_access=access['real/frozen/ridge'])
            assert end==ss['endpoints'];ends[cohort]=end
        assert {k:bool(ends['discovery'][k] and ends['confirmation'][k]) for k in ends['discovery']}==summary['confirmed']
        saved=pd.read_csv(root/'paired-differences.csv');pd.testing.assert_frame_equal(pd.DataFrame(pairrows)[saved.columns],saved,check_dtype=False,atol=1e-12,rtol=1e-12)
        for p,h in read(root/'prior-results-sha256.json').items():assert core.sha256(p)==h,p
        for p,h in m['context']['source_sha256'].items():assert hashlib.sha256(subprocess.check_output(['git','show',m['context']['git_commit']+':'+p])).hexdigest()==h,p
        assert count==39 and len(rows)==468
        core.write_json(out/'checks.json',dict(all_checks_pass=True,internal_training_replayed=count,exact_final_weights=True,independent_trajectories=2*count,metric_rows=len(rows),confirmed=summary['confirmed'],prior_files_unchanged=len(read(root/'prior-results-sha256.json')),result_manifest_sha256=core.sha256(root/'manifest.json'),verifier_sha256=core.sha256(__file__),budget=budget.close()))
        print('All local reward audits passed',flush=True)
    except Exception as exc:core.write_json(out/'failure.json',dict(error=repr(exc)));raise
    finally:budget.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();verify(a.root,a.out)
