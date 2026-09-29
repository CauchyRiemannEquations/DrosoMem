"""Independent scalar evaluation and interaction statistics for frozen weights."""
import argparse,hashlib,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits
from alphabet_memory import read,check
from flying.training import whole_brain_memory as core
from verify_reward_direction import reference_evaluation
from verify_neuron_panel import audit_stats
from frozen_state_probe import Budget


@threadpool_limits.wrap(limits=1)
def verify(root,out):
    m=check(root);c=m['config'];out.mkdir(parents=True,exist_ok=False);budget=Budget(c);rows=[];count=0;trajectories=0
    try:
        for p in sorted(root.glob('*/*/checkpoint.npz')):
            cm=check(p.parent);i=cm['identity'];cc=cm['config'];source=Path(cm['source']);check(source)
            assert core.sha256(source/'manifest.json')==cm['source_manifest_sha256'];assert i['evaluation_seed']==c['evaluation_seeds'][str(i['seed'])]
            with np.load(source/'checkpoint.npz') as z:a=dict(z)
            with np.load(p) as z:saved=dict(z)
            weights=sparse.load_npz(source/'final-weights.npz');assert core.weight_hash(weights)==cm['weights_sha256'];j=cc['lags'].index(cc['target_lag']);truth=a['ytest'][:,j]
            frequency=float(np.mean(truth==np.bincount(a['ytrain'][:,j],minlength=4).argmax()));reps=c['smoke_replicates'] if i['cohort']=='smoke' else c['evaluation_replicates'];assert reps==cm['noisy_replicates']
            for mode,n in [('clean',1),('noisy',reps)]:
                scores,features=reference_evaluation(weights,a['input_patterns'],a['observed_indices'],a['test_symbols'],a['codebook'],cc,i['evaluation_seed'],n,mode=='noisy')
                assert np.isfinite(scores).all()
                np.testing.assert_array_equal(scores,saved[mode+'_scores']);acc=(scores.argmax(2)==truth[None,:]).mean(1);np.testing.assert_array_equal(acc,saved[mode+'_accuracy'])
                if mode=='clean':np.testing.assert_array_equal(scores[0],a['fixed_scores']);np.testing.assert_array_equal(features,a['test_features'])
                rows.append(dict(**{k:i[k] for k in ['cohort','seed','circuit_seed','arm']},mode=mode,accuracy=float(acc.mean()),frequency_accuracy=frequency,replicas=n));trajectories+=n
            count+=1;budget.check()
            if count%3==0:print(f'Independent frozen-checkpoint noise evaluation {count}/39',flush=True)
        f=pd.DataFrame(rows);saved=pd.read_csv(root/'raw-metrics.csv');keys=['cohort','seed','circuit_seed','arm','mode'];pd.testing.assert_frame_equal(f[saved.columns].sort_values(keys).reset_index(drop=True),saved.sort_values(keys).reset_index(drop=True),check_dtype=False,atol=1e-12,rtol=1e-12)
        b=f.groupby(['cohort','seed','arm','mode'])[['accuracy','frequency_accuracy']].mean().reset_index();saved=pd.read_csv(root/'seed-blocks.csv');pd.testing.assert_frame_equal(b[saved.columns],saved,check_dtype=False,atol=1e-12,rtol=1e-12)
        s=read(root/'summary.json');ends={};pairs=[]
        for cohort in ['discovery','confirmation']:
            q=b[b.cohort==cohort];w=q.pivot(index='seed',columns=['arm','mode'],values='accuracy');gates={};access={}
            for (arm,mode),z in q.groupby(['arm','mode']):audit_stats(z.accuracy,s[cohort]['cells'][arm+'/'+mode],c)
            for mode in ['clean','noisy']:
                z=q[(q.arm=='contingent')&(q['mode']==mode)];access[mode]=bool(((z.accuracy-z.frequency_accuracy)>=.05).all())
                for ctrl in ['frozen','yoked']:
                    key=mode+'_vs_'+ctrl;d=w['contingent',mode]-w[ctrl,mode];audit_stats(d,s[cohort]['contrasts'][key],c);gates[key]=bool(d.mean()>=.05 and (d>0).all());pairs.extend(dict(cohort=cohort,seed=int(seed),contrast=key,difference=float(val)) for seed,val in d.items())
            for ctrl in ['frozen','yoked']:
                key='interaction_vs_'+ctrl;d=(w['contingent','noisy']-w[ctrl,'noisy'])-(w['contingent','clean']-w[ctrl,'clean']);audit_stats(d,s[cohort]['contrasts'][key],c);gates[key]=bool(d.mean()>=.01 and (d>0).all());pairs.extend(dict(cohort=cohort,seed=int(seed),contrast=key,difference=float(val)) for seed,val in d.items())
            assert gates==s[cohort]['gates'] and access==s[cohort]['access']
            e=dict(noisy_coding=access['noisy'] and gates['noisy_vs_frozen'] and gates['noisy_vs_yoked'],clean_coding=access['clean'] and gates['clean_vs_frozen'] and gates['clean_vs_yoked'],noise_interaction=gates['interaction_vs_frozen'] and gates['interaction_vs_yoked']);assert e==s[cohort]['endpoints'];ends[cohort]=e
        confirmed={k:bool(ends['discovery'][k] and ends['confirmation'][k]) for k in ends['discovery']};assert confirmed==s['confirmed'] and (confirmed['noisy_coding'] and confirmed['noise_interaction'])==s['noise_removal_explanation_supported'];assert not s['new_training_seeds'] and not s['cohorts_pooled']
        saved=pd.read_csv(root/'paired-differences.csv');pd.testing.assert_frame_equal(pd.DataFrame(pairs)[saved.columns],saved,check_dtype=False,atol=1e-12,rtol=1e-12)
        for p,h in read(root/'prior-results-sha256.json').items():assert core.sha256(p)==h,p
        for p,h in m['context']['source_sha256'].items():assert hashlib.sha256(subprocess.check_output(['git','show',m['context']['git_commit']+':'+p])).hexdigest()==h,p
        assert count==39 and trajectories==1197;core.write_json(out/'checks.json',dict(all_checks_pass=True,checkpoints=count,scalar_trajectories=trajectories,exact_scores=True,confirmed=confirmed,prior_files_unchanged=len(read(root/'prior-results-sha256.json')),result_manifest_sha256=core.sha256(root/'manifest.json'),verifier_sha256=core.sha256(__file__),budget=budget.close()));print('All reward-noise audits passed',flush=True)
    except Exception as exc:core.write_json(out/'failure.json',dict(error=repr(exc)));raise
    finally:budget.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args();verify(a.root,a.out)
