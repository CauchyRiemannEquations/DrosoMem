"""Independent score/event audit plus predetermined checkpoint replay/refits."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits
from flying.brain.mushroom_body import KCEncoder, CircuitReservoir, SelectedReadout
from flying.brain.reward_plasticity import policy_hash
from flying.brain.plasticity import weight_hash
from flying.data.connectome import load_connectome, sha256
from flying.data.mushroom_body import load_roles
from flying.data.pi_digits import pi_digits
from flying.evaluation.free_recall import evaluate_recall
from flying.training.phase5_diagnostic import interpolate, readout_views


def verify(directory):
    root=Path(directory);cfg=json.loads((root/'config.json').read_text())
    manifest=json.loads((root/'manifest.json').read_text())
    for name, digest in manifest['file_sha256'].items():
        assert sha256(root/name)==digest, name
    frame=pd.read_csv(root/'results.csv')
    recalls=[json.loads(s) for s in (root/'recalls.jsonl').read_text().splitlines()]
    keys=['circuit','seed','normalization','model','reward_delay','direction','alpha','view']
    assert not frame.duplicated(keys).any()
    assert len(recalls)==len(frame)==manifest['evaluations']
    expected=0
    selection=json.loads((root/'locked_selection.json').read_text()) if manifest['stage']=='confirmation' else None
    for norm in cfg['normalizations']:
        for delay in cfg['reward_delays']:
            alphas=cfg['alphas'] if selection is None else sorted(set([0.]+cfg['confirmation_anchors']+
                [r['alpha'] for r in selection['candidates'] if r['normalization']==norm and r['reward_delay']==delay and r['alpha'] is not None]))
            expected+=len(cfg['circuits'])*len(cfg[manifest['stage']+'_seeds'])*len(cfg['models'])*2*len(alphas)*len(cfg['views'])
    assert len(frame)==expected
    for r,row in zip(recalls,frame.to_dict('records')):
        assert all(r[k]==row[k] for k in keys)
        score=next((i for i,(a,b) in enumerate(zip(r['target'],r['prediction'])) if a!=b),len(r['target']))
        assert len(r['target'])==len(r['prediction'])==cfg['pi_length']-cfg['prompt_length']
        assert score==r['pi_memory_score']==row['pi_memory_score']
        assert r['first_error_digit_index']==(None if score==r['horizon'] else cfg['prompt_length']+score)
        assert r['censored']==(score==r['horizon'])
    zero=frame[frame.alpha==0]
    assert zero.groupby(['circuit','seed','normalization','model'])['pi_memory_score'].nunique().eq(1).all()
    assert zero.groupby(['circuit','seed','normalization','model'])['readout_sha256'].nunique().eq(1).all()
    digits=pi_digits(cfg['pi_length'])
    events=np.load(root/'reward_events.npz');event_keys=json.loads((root/'event_keys.json').read_text())
    assert len(event_keys)==manifest['endpoint_training_runs']
    assert np.array_equal(events['true_rewards'],events['actions']==digits[1:])
    count=0
    for trace,yoked in zip(event_keys[::2],event_keys[1::2]):
        assert trace['direction']=='reward_trace' and yoked['direction']=='yoked_reward'
        t=trace['event_index'];y=yoked['event_index']
        assert np.array_equal(events['applied_rewards'][t],events['true_rewards'][t])
        assert np.array_equal(events['applied_rewards'][y].sum(axis=1),events['true_rewards'][t].sum(axis=1))
        count+=events['true_rewards'][t].size+events['true_rewards'][y].size
    # First circuit/first seed, incoming-L1/delay 3/real: specified before seeing outcomes.
    chosen=frame[(frame.circuit==Path(cfg['circuits'][0]).name)&(frame.seed==cfg[manifest['stage']+'_seeds'][0])&
                 (frame.normalization=='incoming_l1')&(frame.reward_delay==3)&(frame.model=='fly')]
    _,ids,_=load_connectome(cfg['circuits'][0]);roles,_=load_roles(cfg['circuits'][0],ids)
    archive=np.load(root/'endpoints.npz');replayed=0
    indexed={tuple(r[k] for k in keys):r for r in recalls}
    with threadpool_limits(limits=1):
        for event_index, group in chosen.groupby('event_index'):
            i=int(event_index)
            initial=sparse.csr_matrix((archive[f'initial_{i}'],archive[f'indices_{i}'],archive[f'indptr_{i}']),shape=(len(ids),len(ids)))
            endpoint=initial.copy();endpoint.data=archive[f'endpoint_{i}'].copy()
            encoder=KCEncoder(roles,cfg[manifest['stage']+'_seeds'][0],cfg['input_fraction'],cfg['input_amplitude'])
            assert np.array_equal(encoder.patterns,archive[f'encoder_{i}'])
            policy=SelectedReadout(np.flatnonzero(roles=='MBON'))
            for field in ('weights','mean','scale'):setattr(policy.model,field,archive[f'policy_{field}_{i}'].copy())
            for alpha, subset in group.groupby('alpha'):
                weights,_=interpolate(initial,endpoint,alpha,roles)
                reservoir=CircuitReservoir(weights,encoder,cfg['leak'],cfg['microsteps'])
                views=readout_views(policy,reservoir.states(digits[:-1]),digits[1:],cfg)
                for row in subset.to_dict('records'):
                    ro=views[row['view']]
                    assert policy_hash(ro)==row['readout_sha256']
                    assert weight_hash(weights)==row['weight_sha256']
                    r=evaluate_recall(reservoir,ro,digits,cfg['prompt_length'],len(digits)-cfg['prompt_length'])
                    logged=indexed[tuple(row[k] for k in keys)]
                    assert r['prediction']==logged['prediction'] and r['pi_memory_score']==logged['pi_memory_score']
                    replayed+=1
    result=dict(evaluations_checked=len(frame),recall_scores_exact=True,reward_events_checked=count,
                yoked_counts_match=True,zero_alpha_exact=True,expected_design_count=expected,
                predetermined_readouts_refitted_and_replayed=replayed,all_selected_sequences_exact=True,
                legacy_endpoint_checks=manifest['legacy_endpoint_checks'])
    (root/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory');args=parser.parse_args();verify(args.directory)
