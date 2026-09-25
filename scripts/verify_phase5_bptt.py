"""Replay all saved theta checkpoints and audit metrics independently."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
from flying.brain.constrained_bptt import ConstrainedBPTT
from flying.brain.diagnostics import normalize_condition
from flying.brain.mushroom_body import KCEncoder,CircuitReservoir,SelectedReadout,role_shuffled
from flying.brain.plasticity import weight_hash
from flying.brain.reward_plasticity import policy_hash
from flying.data.connectome import load_connectome,sha256
from flying.data.mushroom_body import load_roles
from flying.data.pi_digits import pi_digits
from flying.evaluation.free_recall import evaluate_recall


def verify(directory):
    root=Path(directory);cfg=json.loads((root/'config.json').read_text());manifest=json.loads((root/'manifest.json').read_text())
    for name,digest in manifest['file_sha256'].items():assert sha256(root/name)==digest,name
    frame=pd.read_csv(root/'results.csv');hist=pd.read_csv(root/'training.csv')
    recalls=[json.loads(s) for s in (root/'recalls.jsonl').read_text().splitlines()]
    keys=['circuit','seed','normalization','model','condition','view']
    assert not frame.duplicated(keys).any()
    expected=np.prod([len(cfg[k]) for k in ['circuits','seeds','normalizations','models','conditions','views']])
    assert len(frame)==len(recalls)==manifest['evaluations']==expected
    bykey={tuple(r[k] for k in keys):r for r in recalls}
    for row in frame.to_dict('records'):
        r=bykey[tuple(row[k] for k in keys)]
        assert len(r['prediction'])==len(r['target'])==cfg['pi_length']-cfg['prompt_length']
        score=next((i for i,(a,b) in enumerate(zip(r['target'],r['prediction'])) if a!=b),len(r['target']))
        assert score==r['pi_memory_score']==row['pi_memory_score']
        assert r['censored']==(score==r['horizon'])
        assert r['first_error_digit_index']==(None if score==r['horizon'] else cfg['prompt_length']+score)
    archive=np.load(root/'checkpoints.npz');digits=pi_digits(cfg['pi_length']);labels=digits[1:]
    replayed=0;selected_checked=0
    with threadpool_limits(limits=1):
        for directory in cfg['circuits']:
            raw,ids,_=load_connectome(directory);roles,_=load_roles(directory,ids);circuit=Path(directory).name
            for seed in cfg['seeds']:
                encoder=KCEncoder(roles,seed,cfg['input_fraction'],cfg['input_amplitude'])
                shuffled,_=role_shuffled(raw,roles,seed);graphs={'fly':raw,'role_shuffled':shuffled}
                for norm in cfg['normalizations']:
                    for model in cfg['models']:
                        initial=normalize_condition(graphs[model],norm,cfg['gain']);initial.sort_indices()
                        base=CircuitReservoir(initial,encoder,cfg['leak'],1);policy=SelectedReadout(np.flatnonzero(roles=='MBON'))
                        kwargs=dict(epochs=cfg['readout_epochs'],learning_rate=cfg['readout_learning_rate'],l2=cfg['readout_l2'])
                        policy.fit(base.states(digits[:-1]),labels,**kwargs)
                        group=frame[(frame.circuit==circuit)&(frame.seed==seed)&(frame.normalization==norm)&(frame.model==model)]
                        for condition,part in group.groupby('condition'):
                            row=part.iloc[0];trainer=ConstrainedBPTT(initial,roles,encoder,policy,cfg['leak'],cfg['floor'])
                            trainer.theta=archive[f'theta_{int(row.checkpoint_index)}'];assert np.max(abs(trainer.theta))<=cfg['theta_cap']
                            audit=trainer.audit();assert audit['final_weight_sha256']==row.final_weight_sha256
                            weights,_,_=trainer.materialize(trainer.theta)
                            reservoir=CircuitReservoir(weights,encoder,cfg['leak'],1);states=reservoir.states(digits[:-1])
                            refit=SelectedReadout(policy.indices);refit.fit(states,labels,**kwargs)
                            targets=labels
                            if condition=='permuted_bptt':targets=np.random.default_rng(np.random.SeedSequence([seed,9701])).permutation(labels)
                            objective=trainer.objective(digits[:-1],targets,gradient=False)[0]
                            h=hist[(hist.circuit==circuit)&(hist.seed==seed)&(hist.normalization==norm)&(hist.model==model)&(hist.condition==condition)]
                            assert np.isclose(objective,row.selected_loss,rtol=0,atol=1e-12)
                            assert np.isclose(row.selected_loss,h.loss.min(),rtol=0,atol=1e-12)
                            assert int(h.loc[h.loss.idxmin(),'epoch'])==int(row.selected_epoch)
                            selected_checked+=1
                            for row in part.to_dict('records'):
                                readout=policy if row['view']=='fixed_policy' else refit
                                assert policy_hash(readout)==row['readout_sha256']
                                r=evaluate_recall(reservoir,readout,digits,cfg['prompt_length'],len(digits)-cfg['prompt_length'])
                                assert r['prediction']==bykey[tuple(row[k] for k in keys)]['prediction']
                                logits=readout.model.features(states[:,policy.indices])@readout.model.weights
                                loss=np.mean(logsumexp(logits,axis=1)-logits[np.arange(len(labels)),labels])
                                assert np.isclose(loss,row['true_train_cross_entropy'],rtol=0,atol=1e-12)
                                assert np.isclose(np.mean(logits.argmax(axis=1)==labels),row['train_accuracy'],rtol=0,atol=1e-14)
                                assert weight_hash(reservoir.weights)==audit['final_weight_sha256'];replayed+=1
    report=dict(evaluations=int(expected),all_recall_scores_checked=True,all_checkpoint_readouts_refitted_and_replayed=replayed,
                all_sequences_exact=True,training_objective_selections_checked=selected_checked,all_edge_sign_budget_invariants_passed=True,
                full_experiment_repeated=False)
    (root/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');a=p.parse_args();verify(a.directory)
