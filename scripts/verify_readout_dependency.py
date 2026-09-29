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


def reference_train(initial,roles,bank,symbols,codebook,arm,c):
    w=initial.copy();r=np.repeat(np.arange(len(roles)),np.diff(w.indptr));col=w.indices
    mask=(roles[r]=='MBON')&(roles[col]=='KC');post=r[mask];pre=col[mask];obs=np.flatnonzero(roles=='MBON')
    old_magnitude=abs(initial.data[mask]);sign=np.sign(initial.data[mask]);budget=np.bincount(post,weights=old_magnitude,minlength=len(roles))
    truth=symbols[np.arange(c['warmup'],len(symbols))-c['target_lag']]
    if arm=='shifted':truth=np.roll(truth,len(truth)//2)
    history=[]
    for epoch in range(c['plastic_epochs']):
        state=np.zeros(len(roles));loss=0.;correct=0
        for t,digit in enumerate(symbols):
            previous=state.copy();activation=np.tanh(w@previous+bank[int(digit)])
            state=(1-c['leak'])*previous+c['leak']*activation
            if t<c['warmup']:continue
            target=codebook[int(truth[t-c['warmup']])]
            if arm!='frozen':
                error=np.zeros(len(roles));error[obs]=target-state[obs]
                update=error[post]*c['leak']*(1-activation[post]**2)*previous[pre]
                energy=np.bincount(post,weights=previous[pre]**2,minlength=len(roles))
                magnitude=np.maximum(c['plastic_floor']*old_magnitude,abs(w.data[mask])+c['plastic_learning_rate']*sign*update/(1+energy[post]))
                total=np.bincount(post,weights=magnitude,minlength=len(roles));magnitude*=budget[post]/total[post];w.data[mask]=sign*magnitude
            distance=np.mean((state[obs][None,:]-codebook)**2,axis=1)
            loss+=float(np.mean((state[obs]-target)**2));correct+=int(distance.argmin()==truth[t-c['warmup']])
        history.append(dict(epoch=epoch+1,teacher_mse=loss/len(truth),online_teacher_accuracy=correct/len(truth)))
    np.testing.assert_array_equal(w.data[~mask],initial.data[~mask]);np.testing.assert_array_equal(np.sign(w.data),np.sign(initial.data))
    np.testing.assert_allclose(np.bincount(post,weights=abs(w.data[mask]),minlength=len(roles)),budget,atol=1e-14,rtol=1e-12)
    return w,history


def trajectory(weights,bank,obs,symbols,leak):
    state=np.zeros(weights.shape[0]);features=[];active=[];norm=[]
    for digit in symbols:
        previous=state.copy();activation=np.tanh(weights@previous+bank[int(digit)])
        state=(1-leak)*previous+leak*activation;features.append(state[obs].copy());active.append(np.count_nonzero(abs(state)>1e-8));norm.append(np.linalg.norm(state))
    return np.asarray(features),np.asarray(active),np.asarray(norm)


def audit_common(root,c):
    dest=root/'common-delays';check(dest);summary=read(dest/'summary.json');source=Path(c['source']);check(source)
    f=pd.read_csv(source/'raw-lag-table.csv');lags=[1,2,3,4,5,8];b=f[f.lag.isin(lags)].groupby(['cohort','seed','site','arm','lag'])[['test_accuracy','frequency_excess','null_excess','r2_vs_frequency']].mean().reset_index()
    saved=pd.read_csv(dest/'all-lags.csv');pd.testing.assert_frame_equal(b[saved.columns],saved,check_dtype=False,atol=1e-12,rtol=1e-12)
    eligible=[]
    for lag in lags:
        gates={}
        for site in ['MBON','KC_unstimulated']:
            q=b[(b.cohort=='discovery')&(b.arm=='intact')&(b.lag==lag)&(b.site==site)]
            gates[site]=bool((q.frequency_excess>=.05).all() and (q.null_excess>=.05).all() and q.r2_vs_frequency.mean()>0)
        assert gates==summary['eligibility'][str(lag)]
        if all(gates.values()):eligible.append(lag)
    assert eligible==summary['eligible_lags'] and summary['retrospective'] and not summary['new_confirmatory_claim']
    p=pd.read_csv(dest/'paired-lags.csv')
    for row in p.itertuples():
        losses=[]
        for site in ['MBON','KC_unstimulated']:
            q=b[(b.cohort==row.cohort)&(b.seed==row.seed)&(b.lag==row.lag)&(b.site==site)].set_index('arm')
            losses.append(q.loc['intact','test_accuracy']-q.loc['DAN_MBON','test_accuracy'])
        np.testing.assert_allclose([row.mbon_loss,row.kc_loss,row.interaction],[*losses,losses[0]-losses[1]],atol=1e-12)
        assert row.eligible==(row.lag in eligible)
    for cohort in ['discovery','confirmation']:
        q=p[(p.cohort==cohort)&p.eligible].groupby('seed')[['mbon_loss','kc_loss','interaction']].mean()
        for key in q:audit_stats(q[key],summary['statistics'][cohort][key],c)
    return dict(eligible_lags=eligible,rows=len(p),retrospective=True)


@threadpool_limits.wrap(limits=1)
def verify(root,out):
    m=check(root);c=m['config'];out.mkdir(exist_ok=False);budget=Budget(c)
    try:
        common=audit_common(root,c);count=0;rows=[]
        for path in sorted(root.glob('*/*/checkpoint.npz')):
            dest=path.parent;cm=check(dest);i=cm['identity'];cc=cm['config'];raw,ids,roles=anatomy(i['circuit_seed'])
            family='intact' if i['network']=='real' else i['network'];control,log=generate(raw,roles,family,i['graph_seed'],c['swaps_per_edge'])
            assert core.weight_hash(control)==core.weight_hash(sparse.load_npz(dest/'raw.npz'));assert dict(sampling=log,invariants=audit(raw,control,roles,family))==read(dest/'graph-audit.json')
            initial=core.normalize_condition(control,c['normalization'],c['gain']);initial.sort_indices()
            assert core.weight_hash(initial)==core.weight_hash(sparse.load_npz(dest/'initial-weights.npz'))
            with np.load(path) as z:a=dict(z)
            bank=symbol_bank(roles,i['seed'],c['input_fraction'],c['input_amplitude'])[:4];np.testing.assert_array_equal(bank,a['input_patterns'])
            order=np.random.default_rng(i['code_seed']).permutation(48);codes=np.zeros((4,48))
            for k in range(4):codes[k,order[12*k:12*k+12]]=.25
            np.testing.assert_array_equal(codes,a['codebook']);obs=np.flatnonzero(roles=='MBON');np.testing.assert_array_equal(obs,a['observed_indices'])
            trained,h=reference_train(initial,roles,bank,a['train_symbols'],codes,i['arm'],cc);final=sparse.load_npz(dest/'final-weights.npz')
            np.testing.assert_array_equal(trained.data,final.data);assert h==read(dest/'history.json')
            assert read(dest/'plastic-audit.json')['final_weight_sha256']==core.weight_hash(final)
            for split in ['train','test']:
                symbols=np.random.default_rng(i[split+'_seed']).integers(0,4,cc['warmup']+cc[split+'_samples']).astype(np.uint8);np.testing.assert_array_equal(symbols,a[split+'_symbols'])
                x,active,norm=trajectory(final,bank,obs,symbols,c['leak']);np.testing.assert_array_equal(x,a[split+'_features']);np.testing.assert_array_equal(active,a[split+'_active_counts']);np.testing.assert_array_equal(norm,a[split+'_full_norm'])
                np.testing.assert_array_equal(x[cc['warmup']:],a['x'+split]);truth=np.column_stack([symbols[np.arange(cc['warmup'],len(symbols))-lag] for lag in c['lags']]);np.testing.assert_array_equal(truth,a['y'+split])
            target=a['ytrain'][:,c['lags'].index(c['target_lag'])]
            np.testing.assert_array_equal(a['teacher_labels'],np.roll(target,len(target)//2) if i['arm']=='shifted' else target)
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
            if count%9==0:print(f'Independently replayed internal training and decoders: {count}/117',flush=True)
        f=pd.DataFrame(rows);saved=pd.read_csv(root/'raw-metrics.csv');keys=['cohort','seed','circuit_seed','network','arm','decoder','lag']
        pd.testing.assert_frame_equal(f[saved.columns].sort_values(keys).reset_index(drop=True),saved.sort_values(keys).reset_index(drop=True),check_dtype=False,atol=1e-12,rtol=1e-12)
        b=f[(f.decoder=='fixed')|((f.decoder=='ridge')&(f.lag==2))].groupby(['cohort','seed','network','arm','decoder'])[['test_accuracy','frequency_excess','null_excess','r2_vs_frequency']].mean().reset_index()
        saved=pd.read_csv(root/'seed-blocks.csv');pd.testing.assert_frame_equal(b[saved.columns],saved,check_dtype=False,atol=1e-12,rtol=1e-12)
        summary=read(root/'summary.json');ends={};pairrows=[]
        mapping={'internal_vs_frozen':('real/aligned/fixed','real/frozen/fixed'),'internal_vs_shifted':('real/aligned/fixed','real/shifted/fixed'),'real_vs_random':('real/frozen/ridge','random/frozen/ridge'),'real_vs_role':('real/frozen/ridge','role/frozen/ridge'),'representation_vs_frozen':('real/aligned/ridge','real/frozen/ridge'),'representation_vs_shifted':('real/aligned/ridge','real/shifted/ridge'),'external_head':('real/frozen/ridge','real/frozen/fixed')}
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
            end=dict(internal_coding=access['real/aligned/fixed'] and gates['internal_vs_frozen'] and gates['internal_vs_shifted'],representation_access=access['real/frozen/ridge'],original_graph_advantage=access['real/frozen/ridge'] and gates['real_vs_random'] and gates['real_vs_role'],representation_improvement=access['real/aligned/ridge'] and gates['representation_vs_frozen'] and gates['representation_vs_shifted'],external_head_dependence=access['real/frozen/ridge'] and gates['external_head'])
            assert end==ss['endpoints'];ends[cohort]=end
        assert {k:bool(ends['discovery'][k] and ends['confirmation'][k]) for k in ends['discovery']}==summary['confirmed']
        saved=pd.read_csv(root/'paired-differences.csv');pd.testing.assert_frame_equal(pd.DataFrame(pairrows)[saved.columns],saved,check_dtype=False,atol=1e-12,rtol=1e-12)
        for p,h in read(root/'prior-results-sha256.json').items():assert core.sha256(p)==h,p
        for p,h in m['context']['source_sha256'].items():assert hashlib.sha256(subprocess.check_output(['git','show',m['context']['git_commit']+':'+p])).hexdigest()==h,p
        assert count==117 and len(rows)==1404
        core.write_json(out/'checks.json',dict(all_checks_pass=True,internal_training_replayed=count,exact_final_weights=True,independent_trajectories=2*count,metric_rows=len(rows),common_delays=common,confirmed=summary['confirmed'],prior_files_unchanged=len(read(root/'prior-results-sha256.json')),result_manifest_sha256=core.sha256(root/'manifest.json'),verifier_sha256=core.sha256(__file__),budget=budget.close()))
        print('All ACT IV-A audits passed',flush=True)
    except Exception as exc:core.write_json(out/'failure.json',dict(error=repr(exc)));raise
    finally:budget.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();verify(a.root,a.out)
