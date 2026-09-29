"""Evaluate archived local-learning weights under matched training/test noise."""
import argparse,hashlib,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits
from flying.training import whole_brain_memory as core
from alphabet_memory import read,check
from reward_direction import evaluate
from context_memory import estimate
from frozen_state_probe import Budget
from pathway_memory import seal

CONFIG_HASH='1252c2e29e74681368f95a028940b7d56c3d7ec40cf654152bd5cf081e537454'


def summarize(rows,c):
    f=pd.DataFrame(rows);b=f.groupby(['cohort','seed','arm','mode'])[['accuracy','frequency_accuracy']].mean().reset_index();out={};pairs=[]
    for cohort in ['discovery','confirmation']:
        q=b[b.cohort==cohort];w=q.pivot(index='seed',columns=['arm','mode'],values='accuracy');cells={};contrasts={};gates={};access={}
        for (arm,mode),v in q.groupby(['arm','mode']):cells[arm+'/'+mode]=estimate(v.accuracy,c)
        for mode in ['clean','noisy']:
            z=q[(q.arm=='contingent')&(q['mode']==mode)];access[mode]=bool(((z.accuracy-z.frequency_accuracy)>=.05).all())
            for control in ['frozen','yoked']:
                key=mode+'_vs_'+control;delta=w['contingent',mode]-w[control,mode];contrasts[key]=estimate(delta,c);gates[key]=bool(delta.mean()>=.05 and (delta>0).all())
                pairs.extend(dict(cohort=cohort,seed=int(seed),contrast=key,difference=float(value)) for seed,value in delta.items())
        for control in ['frozen','yoked']:
            key='interaction_vs_'+control;delta=(w['contingent','noisy']-w[control,'noisy'])-(w['contingent','clean']-w[control,'clean']);contrasts[key]=estimate(delta,c);gates[key]=bool(delta.mean()>=.01 and (delta>0).all())
            pairs.extend(dict(cohort=cohort,seed=int(seed),contrast=key,difference=float(value)) for seed,value in delta.items())
        endpoints=dict(noisy_coding=access['noisy'] and gates['noisy_vs_frozen'] and gates['noisy_vs_yoked'],clean_coding=access['clean'] and gates['clean_vs_frozen'] and gates['clean_vs_yoked'],noise_interaction=gates['interaction_vs_frozen'] and gates['interaction_vs_yoked'])
        out[cohort]=dict(cells=cells,access=access,contrasts=contrasts,gates=gates,endpoints=endpoints)
    out['confirmed']={key:bool(out['discovery']['endpoints'][key] and out['confirmation']['endpoints'][key]) for key in out['discovery']['endpoints']}
    out['noise_removal_explanation_supported']=out['confirmed']['noisy_coding'] and out['confirmed']['noise_interaction'];out['new_training_seeds']=False;out['cohorts_pooled']=False
    return b,pd.DataFrame(pairs),out


@threadpool_limits.wrap(limits=1)
def run(config,out):
    c=read(config);assert core.fingerprint(c)==CONFIG_HASH;assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip(),'Commit before outcomes'
    context=core.source_context(c)
    names=['reward_noise','verify_reward_noise','plot_reward_noise','reward_direction','verify_reward_direction','alphabet_memory','context_memory','frozen_state_probe','pathway_memory','verify_neuron_panel']
    for p in [config,Path('docs/reward-noise-seed-audit.json')]+[Path('scripts')/(n+'.py') for n in names]:context['source_sha256'][p.as_posix()]=core.sha256(p)
    for p,h in context['source_sha256'].items():assert hashlib.sha256(subprocess.check_output(['git','show',context['git_commit']+':'+p])).hexdigest()==h,p
    out.mkdir(parents=True,exist_ok=False);budget=Budget(c)
    try:
        source=Path(c['source']);check(source);assert core.sha256(source/'manifest.json')==c['source_manifest_sha256'];check(Path(c['direction_source']));assert core.sha256(Path(c['direction_source'])/'manifest.json')==c['direction_manifest_sha256']
        prior={p:core.sha256(p) for p in subprocess.check_output(['git','ls-files','results/'],text=True).splitlines()};core.write_json(out/'prior-results-sha256.json',prior);core.write_json(out/'config.json',c);rows=[];count=0
        for p in sorted(source.glob('*/*/checkpoint.npz'),key=lambda p:('smoke' not in p.parts,p.as_posix())):
            cm=check(p.parent);i=cm['identity'];cc=cm['config'];assert cc['noise_std']==c['noise_std'];seed=c['evaluation_seeds'][str(i['seed'])];replicas=c['smoke_replicates'] if i['cohort']=='smoke' else c['evaluation_replicates']
            with np.load(p) as z:a=dict(z)
            weights=sparse.load_npz(p.parent/'final-weights.npz');j=cc['lags'].index(cc['target_lag']);truth=a['ytest'][:,j];frequency=float(np.mean(truth==np.bincount(a['ytrain'][:,j],minlength=4).argmax()))
            identity={k:i[k] for k in ['cohort','seed','circuit_seed','arm']};dest=out/i['cohort']/p.parent.name;dest.mkdir(parents=True);saved={}
            for mode,reps in [('clean',1),('noisy',replicas)]:
                scores,features=evaluate(weights,a['input_patterns'],a['observed_indices'],a['test_symbols'],a['codebook'],cc,seed,reps,mode=='noisy')
                assert np.isfinite(scores).all()
                if mode=='clean':np.testing.assert_array_equal(scores[0],a['fixed_scores']);np.testing.assert_array_equal(features,a['test_features'])
                accuracy=(scores.argmax(2)==truth[None,:]).mean(1);saved[mode+'_scores']=scores;saved[mode+'_accuracy']=accuracy
                rows.append(dict(**identity,mode=mode,accuracy=float(accuracy.mean()),frequency_accuracy=frequency,replicas=reps))
            np.savez_compressed(dest/'checkpoint.npz',**saved)
            seal(dest,cc,context,identity={**identity,'evaluation_seed':seed},source=p.parent.as_posix(),source_manifest_sha256=core.sha256(p.parent/'manifest.json'),weights_sha256=core.weight_hash(weights),noisy_replicates=replicas,clean_exact_replay=True)
            count+=1;budget.check()
            if count%3==0:print(f'Frozen-checkpoint noise evaluation {count}/39',flush=True)
        blocks,pairs,summary=summarize(rows,c);pd.DataFrame(rows).to_csv(out/'raw-metrics.csv',index=False);blocks.to_csv(out/'seed-blocks.csv',index=False);pairs.to_csv(out/'paired-differences.csv',index=False);core.write_json(out/'summary.json',summary)
        for p,h in {**prior,**context['source_sha256']}.items():assert core.sha256(p)==h,p
        assert count==39;core.write_json(out/'verification.json',dict(checkpoints=count,clean_exact_replays=count,prior_files_unchanged=len(prior),budget=budget.close()));seal(out,c,context,environment=core.environment());print(summary['confirmed'],flush=True)
    except Exception as exc:core.write_json(out/'failure.json',dict(error=repr(exc)));raise
    finally:budget.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=Path('configs/reward_noise.json'));p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.config,a.out)
