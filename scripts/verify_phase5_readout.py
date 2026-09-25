"""Rebuild all states and replay all saved heads; refit a prespecified subset."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
from flying.training.phase5_readout import (KEYS,conditions,nonlinear_seed,head_arrays,digest,logits,SelectedReadout,
    NonlinearReadout,weight_hash,sha256,pi_digits,evaluate_recall)

def verify(directory):
    root=Path(directory);cfg=json.loads((root/'config.json').read_text());manifest=json.loads((root/'manifest.json').read_text())
    for name,h in manifest['file_sha256'].items():assert sha256(root/name)==h,name
    for name,h in manifest['code_sha256'].items():assert sha256(name)==h,name
    frame=pd.read_csv(root/'evaluations.csv');saved=np.load(root/'readouts.npz')
    recalls={r['evaluation_index']:r for r in map(json.loads,(root/'recalls.jsonl').read_text().splitlines())}
    assert len(frame)==len(recalls)==384 and not frame.duplicated(KEYS+['readout_type','initialization','epochs']).any()
    digits=pi_digits(cfg['pi_length']);labels=digits[1:];replayed=refitted=state_count=0
    with threadpool_limits(limits=1):
        for key,reservoir,mbon in conditions(cfg):
            states=reservoir.states(digits[:-1]);state_count+=1;subset=frame
            for k,v in key.items():subset=subset[subset[k]==v]
            assert len(subset)==8
            mean=states[:,mbon].mean(axis=0);scale=np.maximum(states[:,mbon].std(axis=0),1e-5)
            refit=(key['circuit']==Path(cfg['circuits'][0]).name and key['seed']==cfg['seeds'][0]
                   and key['normalization']=='incoming_l1' and key['schedule']=='sync_one' and key['model']=='fly')
            nonlinear_refits={}
            if refit:
                for initialization in cfg['initializations']:
                    head=NonlinearReadout(mbon,cfg['hidden_units'],nonlinear_seed(key['seed'],initialization))
                    nonlinear_refits[initialization]=head.fit(states,labels,epochs=max(cfg['checkpoints']),learning_rate=cfg['learning_rate'],l2=cfg['l2'],checkpoints=cfg['checkpoints'])[0]
            for row in subset.to_dict('records'):
                ix=row['evaluation_index'];kind=row['readout_type'];epoch=row['epochs'];initialization=row['initialization']
                assert row['state_sha256']==hashlib.sha256(states.tobytes()).hexdigest()
                assert row['weight_sha256']==weight_hash(reservoir.weights)
                if kind=='affine':
                    head=SelectedReadout(mbon)
                    for name in ['mean','scale','weights']:setattr(head.model,name,saved[f'{name}_{ix}'].copy())
                    assert row['parameter_count']==head.model.weights.size==490
                else:
                    head=NonlinearReadout(mbon,cfg['hidden_units'],nonlinear_seed(key['seed'],initialization))
                    head.mean=saved[f'mean_{ix}'].copy();head.scale=saved[f'scale_{ix}'].copy()
                    head.parameters={name:saved[f'{name}_{ix}'].copy() for name in ['w1','b1','w2','b2']}
                    assert row['parameter_count']==head.parameter_count==482
                arrays=head_arrays(head)
                assert np.array_equal(arrays['mean'],mean) and np.array_equal(arrays['scale'],scale)
                assert row['readout_sha256']==digest(head)
                pred=head.predict(states);raw=logits(head,states)
                assert np.array_equal(pred,saved[f'teacher_prediction_{ix}'])
                assert np.isclose(row['train_accuracy'],np.mean(pred==labels),atol=1e-14,rtol=0)
                ce=np.mean(logsumexp(raw,axis=1)-raw[np.arange(len(labels)),labels])
                assert np.isclose(row['cross_entropy'],ce,atol=1e-12,rtol=0)
                got=evaluate_recall(reservoir,head,digits,cfg['prompt_length'],len(digits)-cfg['prompt_length'])
                for name,value in got.items():assert recalls[ix][name]==value,(ix,name)
                target=digits[cfg['prompt_length']:];teacher=pred[cfg['prompt_length']-1:]
                prefix=next((i for i,(a,b) in enumerate(zip(target,teacher)) if a!=b),len(target))
                prefix_string=next((i for i,(a,b) in enumerate(zip(got['target'],got['prediction'])) if a!=b),got['horizon'])
                assert prefix==prefix_string==got['pi_memory_score']==row['pi_memory_score']==row['teacher_forced_prefix']
                assert got['censored']==(prefix==got['horizon'])
                assert got['first_error_digit_index']==(None if got['censored'] else cfg['prompt_length']+prefix)
                assert digest(head)==row['readout_sha256'] and weight_hash(reservoir.weights)==row['weight_sha256']
                replayed+=1
                if refit:
                    if kind=='affine':
                        fitted=SelectedReadout(mbon);fitted.fit(states,labels,epochs=epoch,learning_rate=cfg['learning_rate'],l2=cfg['l2'])
                    else:fitted=nonlinear_refits[initialization][epoch]
                    for name,array in head_arrays(fitted).items():assert np.array_equal(array,arrays[name]),(ix,name)
                    assert digest(fitted)==digest(head);refitted+=1
    result=dict(fixed_state_conditions_rebuilt=state_count,all_saved_heads_replayed=replayed,all_recall_strings_exact=True,
        teacher_forced_first_error_equivalence_checked=replayed,all_weight_and_readout_hashes_exact=True,
        same_training_states_and_statistics_verified=True,prespecified_checkpoints_independently_refitted=refitted,
        refit_subset='first circuit, first seed, fly, incoming_l1, sync_one: both affine budgets and all nonlinear initializations/checkpoints')
    assert state_count==48 and replayed==384 and refitted==8
    (root/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');a=p.parse_args();verify(a.directory)
