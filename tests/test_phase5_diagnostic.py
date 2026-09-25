import copy
import json
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from flying.brain.mushroom_body import SelectedReadout
from flying.brain.reward_plasticity import policy_hash
from flying.brain.plasticity import weight_hash
from flying.models.readout import Readout
from flying.training.phase5_diagnostic import interpolate, readout_views, select_candidates, save_archive


def test_atomic_archive_roundtrip(tmp_path):
    arrays={'matrix':np.arange(12000,dtype=float).reshape(120,100),'indices':np.arange(10)}
    path=tmp_path/'checkpoints.npz'
    save_archive(path,arrays)
    with np.load(path) as saved:
        for key,value in arrays.items():assert np.array_equal(saved[key],value)
    assert not list(tmp_path.glob('*.part'))


def test_interpolation_preserves_invariants_and_exact_endpoints():
    roles = np.array(['KC','KC','MBON','DAN'])
    a = sparse.csr_matrix(([.2,.4,-.1], ([2,2,0],[0,1,3])),shape=(4,4))
    b = sparse.csr_matrix(([.3,.3,-.1], ([2,2,0],[0,1,3])),shape=(4,4))
    assert weight_hash(interpolate(a,b,0,roles)[0]) == weight_hash(a)
    assert weight_hash(interpolate(a,b,1,roles)[0]) == weight_hash(b)
    w, audit = interpolate(a,b,.3,roles)
    assert np.allclose(w.toarray(), .7*a.toarray()+.3*b.toarray())
    assert np.array_equal(np.sign(w.data), np.sign(a.data))
    assert np.allclose(abs(w).sum(axis=1),abs(a).sum(axis=1))
    assert audit['relative_plastic_change'] > 0
    for alpha in (-.1,1.1,float('nan')):
        with pytest.raises(ValueError): interpolate(a,b,alpha,roles)
    changed = b.copy(); changed[0,3] = -.2
    with pytest.raises(ValueError): interpolate(a,changed,.1,roles)


def test_fixed_statistics_fit_matches_original_when_statistics_equal():
    rng = np.random.default_rng(1); x=rng.normal(size=(30,4)); y=np.arange(30)%10
    original=Readout(); original.fit(x,y,epochs=10)
    fixed=Readout(); fixed.fit(x,y,epochs=10,feature_stats=(original.mean,original.scale))
    assert np.array_equal(original.weights,fixed.weights)
    with pytest.raises(ValueError): fixed.fit(x,y,feature_stats=(np.zeros(4),np.zeros(4)))


def test_four_views_separate_statistics_from_coefficients_and_do_not_mutate_policy():
    rng=np.random.default_rng(2); x=rng.normal(size=(50,6)); y=np.arange(50)%10
    cfg=dict(readout_epochs=20,readout_learning_rate=.03,readout_l2=1e-5)
    policy=SelectedReadout([1,3,5]);policy.fit(x,y,epochs=20)
    old=policy_hash(policy);changed=x*2+3
    views=readout_views(policy,changed,y,cfg)
    assert old==policy_hash(policy)==policy_hash(views['fixed'])
    assert np.array_equal(views['stats_only'].model.weights,policy.model.weights)
    assert np.array_equal(views['stats_only'].model.mean,changed[:,policy.indices].mean(axis=0))
    assert np.array_equal(views['coefficients_only'].model.mean,policy.model.mean)
    assert np.array_equal(views['coefficients_only'].model.scale,policy.model.scale)
    assert not np.array_equal(views['coefficients_only'].model.weights,policy.model.weights)
    # A pure positive per-feature affine change can be corrected without any labels.
    assert np.allclose(policy.model.features(x[:,policy.indices]),views['stats_only'].model.features(changed[:,policy.indices]))
    changed_labels=readout_views(policy,changed,(y+1)%10,cfg)
    assert policy_hash(changed_labels['stats_only'])==policy_hash(views['stats_only'])
    assert policy_hash(changed_labels['fixed'])==policy_hash(views['fixed'])
    outside=changed.copy();outside[:,[0,2,4]]+=1e9
    external=readout_views(policy,outside,y,cfg)
    for key in views: assert policy_hash(external[key])==policy_hash(views[key])


def test_zero_interpolation_views_are_all_exact_baseline():
    rng=np.random.default_rng(3);x=rng.normal(size=(30,5));y=np.arange(30)%10
    policy=SelectedReadout([0,2]);policy.fit(x,y,epochs=10)
    cfg=dict(readout_epochs=10,readout_learning_rate=.03,readout_l2=1e-5)
    assert all(policy_hash(v)==policy_hash(policy) for v in readout_views(policy,x,y,cfg).values())


def test_candidate_requires_both_controls_and_smaller_alpha_breaks_ties():
    rows=[]
    for seed in [1,2]:
        for alpha,t,y in [(0,5,5),(.01,6,6),(.03,7,5),(.1,7,5),(1,8,9)]:
            for direction,score in [('reward_trace',t),('yoked_reward',y)]:
                rows.append(dict(circuit='fixture',seed=seed,normalization='spectral',model='fly',reward_delay=0,
                                 alpha=alpha,view='fixed',direction=direction,pi_memory_score=score))
    frame=pd.DataFrame(rows)
    candidate=select_candidates(frame)[0]
    json.dumps(candidate)
    assert candidate['alpha']==.03 and candidate['discovery_gain_frozen']==2
    frame.loc[(frame.direction=='reward_trace') & (frame.alpha>0),'pi_memory_score']=4
    assert select_candidates(frame)[0]['alpha'] is None
