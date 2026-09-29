import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from test_readout_dependency import fixture
from local_reward import codes
from reward_direction import local_direction,directions,perturb,evaluate,summarize
from verify_reward_direction import reference_direction,reference_evaluation


def config():
    return dict(leak=.6,plastic_learning_rate=.05,plastic_floor=1e-4,warmup=3,target_lag=2,
        plastic_epochs=2,trace_decay=.8,activity_rate=.05,baseline_rate=.05,noise_std=.02,
        random_directions=5,nominal_relative_radius=.01,boundary_fraction=.5)


def test_frozen_proposals_independent_replay_and_symmetric_constraints():
    weights,roles,bank=fixture();original=weights.data.copy();symbols=np.random.default_rng(42).integers(0,4,43);c=config()
    mean,logs,h,model=local_direction(weights,roles,bank,symbols,codes(7),c,18)
    other,logs2,h2,mask,post=reference_direction(weights,roles,bank,symbols,codes(7),c,18)
    np.testing.assert_array_equal(mean,other);assert h==h2
    for key in logs:np.testing.assert_array_equal(logs[key],logs2[key])
    np.testing.assert_array_equal(weights.data,original);np.testing.assert_array_equal(model.weights.data,original)
    unit,r=directions(mean,abs(original[mask]),post,len(roles),c,31)
    assert r['radius']>0
    for u in unit:
        plus=perturb(weights,mask,post,u,r['radius'],1);minus=perturb(weights,mask,post,u,r['radius'],-1)
        np.testing.assert_allclose((plus.data+minus.data)/2,original,rtol=1e-14,atol=1e-15)
        assert np.all(abs(plus.data[mask])>=c['plastic_floor']*abs(original[mask]))
        assert np.all(abs(minus.data[mask])>=c['plastic_floor']*abs(original[mask]))
    zero,rz=directions(np.zeros_like(mean),abs(original[mask]),post,len(roles),c,31)
    assert rz['radius']==0 and rz['resolution_limited'] and not zero.any()


def test_batch_scores_match_independent_scalar_trajectories():
    weights,roles,bank=fixture();symbols=np.random.default_rng(13).integers(0,4,43);obs=np.flatnonzero(roles=='MBON');c=config()
    for noisy,replicas in [(True,3),(False,1)]:
        scores,features=evaluate(weights,bank,obs,symbols,codes(7),c,94,replicas,noisy)
        expected,expected_features=reference_evaluation(weights,bank,obs,symbols,codes(7),c,94,replicas,noisy)
        np.testing.assert_array_equal(scores,expected);np.testing.assert_array_equal(features,expected_features)


def test_inference_uses_seed_blocks_and_requires_every_seed():
    c=dict(config(),bootstrap_seed=17,bootstrap_draws=1000);rows=[]
    variants=['baseline','local_plus','local_minus']+[f'random{k}_{side}' for k in range(5) for side in ['plus','minus']]
    for cohort in ['smoke','discovery','confirmation']:
        for seed in [1,2,3]:
            for ci in [701,702]:
                for mode in ['noisy','clean']:
                    for variant in variants:
                        value={'local_plus':.205,'local_minus':.195}.get(variant,.2)
                        rows.append(dict(cohort=cohort,seed=seed,circuit_seed=ci,mode=mode,variant=variant,accuracy=value,replicas=32))
    _,_,s=summarize(rows,c);assert s['confirmed_noisy_alignment']
    assert s['confirmation']['noisy']['statistics']['local_directional']['n']==3
    for r in rows:
        if r['cohort']=='confirmation' and r['seed']==2 and r['variant']=='local_plus':r['accuracy']=.199
    _,_,s=summarize(rows,c);assert not s['confirmed_noisy_alignment']
    assert s['confirmation']['noisy']['statistics']['forward_gain']['mean']>.001
    assert not s['confirmation']['noisy']['gates']['forward_gain']
