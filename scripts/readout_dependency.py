"""ACT IV-A bounded computational comparison. Existing artificial teacher only."""
import argparse, hashlib, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits
from flying.training import whole_brain_memory as core
from flying.brain.plasticity import KCMBONPlasticity
from alphabet_memory import read, check, symbol_bank, SymbolEncoder
from frozen_state_probe import labels, fit_fold, measures, Budget
from structural_controls import generate, audit
from edge_panel import anatomy
from context_memory import estimate
from pathway_memory import seal
import kc_ablation

CONFIG_HASH='9e422bb0652d3d61e6070c402b08f591e23ff5ca6cca07ac4e47bdbc8027796a'


def codes(seed):
    order=np.random.default_rng(seed).permutation(48); result=np.zeros((4,48))
    for k in range(4): result[k,order[12*k:12*(k+1)]]=.25
    return result


def fixed_scores(features, codebook):
    return -np.mean((features[:,None,:]-codebook[None,:,:])**2,axis=2)


def train(weights, roles, bank, symbols, codebook, arm, c):
    model=KCMBONPlasticity(weights,roles,SymbolEncoder(bank),leak=c['leak'],microsteps=1,
                          learning_rate=0 if arm=='frozen' else c['plastic_learning_rate'],floor=c['plastic_floor'])
    w=c['warmup']; target=symbols[np.arange(w,len(symbols))-c['target_lag']]
    if arm=='shifted': target=np.roll(target,len(target)//2)
    history=[]
    for epoch in range(c['plastic_epochs']):
        model.reset()
        for digit in symbols[:w]: model.step(int(digit))
        loss=0.; correct=0
        for digit, label in zip(symbols[w:],target):
            state=model.step(int(digit),codebook[int(label)])
            loss+=float(np.mean((state[model.mbon]-codebook[int(label)])**2))
            correct+=int(fixed_scores(state[model.mbon][None,:],codebook).argmax(1)[0]==label)
        history.append(dict(epoch=epoch+1,teacher_mse=loss/len(target),online_teacher_accuracy=correct/len(target)))
    return model.weights.copy(),model.audit(),history,target


def collect(weights, roles, bank, symbols, c):
    model=KCMBONPlasticity(weights,roles,SymbolEncoder(bank),leak=c['leak'],microsteps=1,learning_rate=0,floor=c['plastic_floor'])
    return core.collect(model,model.mbon,symbols,c['activity_epsilon'])


def execute(c, ci, block, network, arm, graph):
    raw, ids, roles=anatomy(ci); obs=np.flatnonzero(roles=='MBON')
    control,graph_audit=graph; weights=core.normalize_condition(control,c['normalization'],c['gain']); weights.sort_indices()
    bank=symbol_bank(roles,block['seed'],c['input_fraction'],c['input_amplitude'])[:4]; codebook=codes(block['code_seed'])
    symbols={split:np.random.default_rng(block[split+'_seed']).integers(0,4,c['warmup']+c[split+'_samples']).astype(np.uint8) for split in ['train','test']}
    final,plastic_audit,history,teacher=train(weights,roles,bank,symbols['train'],codebook,arm,c)
    a=dict(input_patterns=bank,codebook=codebook,observed_indices=obs,teacher_labels=teacher)
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
        contrasts={}
        specs={'internal_vs_frozen':('real/aligned/fixed','real/frozen/fixed'), 'internal_vs_shifted':('real/aligned/fixed','real/shifted/fixed'),
               'real_vs_random':('real/frozen/ridge','random/frozen/ridge'),'real_vs_role':('real/frozen/ridge','role/frozen/ridge'),
               'representation_vs_frozen':('real/aligned/ridge','real/frozen/ridge'),'representation_vs_shifted':('real/aligned/ridge','real/shifted/ridge'),
               'external_head':('real/frozen/ridge','real/frozen/fixed')}
        gates={}
        for name,(a,b) in specs.items():
            delta=series[a]-series[b];contrasts[name]=estimate(delta,c);gates[name]=bool(delta.mean()>=.05 and (delta>0).all())
            differences.extend(dict(cohort=cohort,seed=int(seed),contrast=name,difference=float(value)) for seed,value in delta.items())
        endpoints=dict(internal_coding=access['real/aligned/fixed'] and gates['internal_vs_frozen'] and gates['internal_vs_shifted'],
                       representation_access=access['real/frozen/ridge'],original_graph_advantage=access['real/frozen/ridge'] and gates['real_vs_random'] and gates['real_vs_role'],
                       representation_improvement=access['real/aligned/ridge'] and gates['representation_vs_frozen'] and gates['representation_vs_shifted'],
                       external_head_dependence=access['real/frozen/ridge'] and gates['external_head'])
        out[cohort]=dict(cells=cells,access=access,contrasts=contrasts,gates=gates,endpoints=endpoints)
    out['confirmed']={k:bool(out['discovery']['endpoints'][k] and out['confirmation']['endpoints'][k]) for k in out['discovery']['endpoints']}
    out['cohorts_pooled']=False
    return out,pd.DataFrame(differences)


def common_delays(c,out):
    source=Path(c['source']);check(source);assert core.sha256(source/'manifest.json')==c['source_manifest_sha256']
    old=read(source/'config.json');f=pd.read_csv(source/'raw-lag-table.csv');cols=['test_accuracy','frequency_excess','null_excess','r2_vs_frequency']
    b=f[f.lag.isin(old['primary_lags'])].groupby(['cohort','seed','site','arm','lag'])[cols].mean().reset_index()
    eligibility={};eligible=[]
    for lag in old['primary_lags']:
        gates={}
        for site in old['sites']:
            q=b[(b.cohort=='discovery')&(b.arm=='intact')&(b.site==site)&(b.lag==lag)]
            gates[site]=bool((q.frequency_excess>=.05).all() and (q.null_excess>=.05).all() and q.r2_vs_frequency.mean()>0)
        eligibility[str(lag)]=gates
        if all(gates.values()):eligible.append(lag)
    out.mkdir(exist_ok=False);b.to_csv(out/'all-lags.csv',index=False);pairs=[]
    for cohort in ['discovery','confirmation']:
        for lag in old['primary_lags']:
            q=b[(b.cohort==cohort)&(b.lag==lag)];loss={}
            for site in old['sites']:
                t=q[q.site==site];loss[site]=t[t.arm=='intact'].set_index('seed').test_accuracy-t[t.arm=='DAN_MBON'].set_index('seed').test_accuracy
            for seed in loss['MBON'].index:pairs.append(dict(cohort=cohort,lag=lag,seed=int(seed),eligible=lag in eligible,mbon_loss=float(loss['MBON'][seed]),kc_loss=float(loss['KC_unstimulated'][seed]),interaction=float(loss['MBON'][seed]-loss['KC_unstimulated'][seed])))
    p=pd.DataFrame(pairs);p.to_csv(out/'paired-lags.csv',index=False);stats={}
    for cohort in ['discovery','confirmation']:
        q=p[(p.cohort==cohort)&p.eligible].groupby('seed')[['mbon_loss','kc_loss','interaction']].mean()
        stats[cohort]={k:estimate(q[k],c) for k in q} if len(q) else None
    core.write_json(out/'summary.json',dict(retrospective=True,previous_outcomes_seen=True,eligible_lags=eligible,eligibility=eligibility,statistics=stats,new_confirmatory_claim=False))
    core.write_json(out/'manifest.json',dict(source_manifest_sha256=c['source_manifest_sha256'],artifacts={p.name:core.sha256(p) for p in out.iterdir() if p.is_file()}))
    print('Retrospective eligible delays:',eligible,flush=True)


@threadpool_limits.wrap(limits=1)
def run(config,out):
    c=read(config);assert core.fingerprint(c)==CONFIG_HASH
    assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip(),'Commit before outcomes'
    context=core.source_context(c)
    names=['readout_dependency','verify_readout_dependency','structural_controls','alphabet_memory','frozen_state_probe','context_memory','edge_panel','edge_masks','pathway_memory','pathway_masks','kc_ablation','kc_frozen','structural_k4','normalization_transfer','verify_neuron_panel']
    for p in [config,Path('docs/readout-dependency-seed-audit.json')]+[Path('scripts')/(n+'.py') for n in names]:context['source_sha256'][p.as_posix()]=core.sha256(p)
    for p,h in context['source_sha256'].items():assert hashlib.sha256(subprocess.check_output(['git','show',context['git_commit']+':'+p])).hexdigest()==h,p
    out.mkdir(exist_ok=False);budget=Budget(c)
    try:
        prior={p:core.sha256(p) for p in subprocess.check_output(['git','ls-files','results/'],text=True).splitlines()};core.write_json(out/'prior-results-sha256.json',prior);core.write_json(out/'config.json',c)
        common_delays(c,out/'common-delays');core.write_json(out/'baseline.json',kc_ablation.baseline());print('Historical baseline reproduced',flush=True)
        rows=[];neural=[];count=0
        for cohort,settings in c['cohorts'].items():
            cc=dict(c,**settings)
            for block in cc['blocks']:
                for ci in cc['circuit_seeds']:
                    raw,ids,roles=anatomy(ci)
                    for network in c['networks']:
                        family='intact' if network=='real' else network
                        control,log=generate(raw,roles,family,block['graph_seed'],c['swaps_per_edge']);ga=dict(sampling=log,invariants=audit(raw,control,roles,family))
                        for arm in c['training_arms']:
                            budget.check();a,metrics,direct,n,h,pa,initial,final,ga=execute(cc,ci,block,network,arm,(control,ga))
                            ident=dict(cohort=cohort,seed=block['seed'],circuit_seed=ci,network=network,arm=arm)
                            dest=out/cohort/f'{network}_{arm}_c{ci}_s{block["seed"]}';dest.mkdir(parents=True)
                            np.savez_compressed(dest/'checkpoint.npz',**a);sparse.save_npz(dest/'initial-weights.npz',initial);sparse.save_npz(dest/'final-weights.npz',final);sparse.save_npz(dest/'raw.npz',control)
                            for name,value in [('metrics',metrics),('fixed-metrics',direct),('neural',n),('history',h),('plastic-audit',pa),('graph-audit',ga)]:core.write_json(dest/(name+'.json'),value)
                            rows.extend(dict(**ident,decoder='ridge',**x) for x in metrics);rows.append(dict(**ident,decoder='fixed',**direct));neural.append(dict(**ident,**n));count+=1
                            seal(dest,cc,context,identity={**ident,**block},observation_ids=[ids[i] for i in a['observed_indices']],evaluation_trajectory_replayed=True)
                        print(f'{cohort} c{ci} s{block["seed"]} {network}:3 training arms',flush=True)
        f=pd.DataFrame(rows);f.to_csv(out/'raw-metrics.csv',index=False);b=blocks_of(rows,c);b.to_csv(out/'seed-blocks.csv',index=False)
        summary,pairs=inference(b,c);core.write_json(out/'summary.json',summary);pairs.to_csv(out/'paired-differences.csv',index=False);pd.DataFrame(neural).to_csv(out/'neural-diagnostics.csv',index=False)
        for p,h in {**prior,**context['source_sha256']}.items():assert core.sha256(p)==h,p
        assert count==117;budget.check();core.write_json(out/'verification.json',dict(cases=count,primary_decoder_evaluations=count*2,prior_files_unchanged=len(prior),budget=budget.close()))
        seal(out,c,context,environment=core.environment());print(summary['confirmed'],flush=True)
    except Exception as exc:core.write_json(out/'failure.json',dict(error=repr(exc)));raise
    finally:budget.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=Path('configs/readout_dependency.json'));p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.config,a.out)
