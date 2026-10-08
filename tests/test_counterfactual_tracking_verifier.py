"""Toy-only independent verifier safeguards before source-graph outcomes."""
import hashlib

import numpy as np
import pytest
from scipy import sparse

from verify_counterfactual_tracking import (manual_step,frozen_scores,joint_endpoint,
    aggregate_endpoint,window_symbols,trajectory_window,state_sha)


def test_direct_replacement_keeps_future_and_lag2_target():
    stream = np.asarray([9,3,7,1,2,8],dtype=np.int64)
    first,second = window_symbols(stream,0)
    np.testing.assert_array_equal(first,[9,3,7])
    np.testing.assert_array_equal(second,[0,3,7])
    assert stream[2]==first[-1]==second[-1]
    assert first[0]!=second[0] and np.array_equal(stream,[9,3,7,1,2,8])


def test_joint_credits_both_worlds_without_filtering():
    s,r = np.asarray([0,0,0,0]),np.asarray([1,1,1,1])
    cell = joint_endpoint(np.asarray([0,0,2,2]),np.asarray([1,2,1,2]),s,r)
    assert cell['samples']==4 and cell['original_correct']==cell['replacement_correct']==2
    assert cell['joint_correct']==1 and cell['scientific_rule_pass'] is False


def test_exact_boundary_57fails_58passes_and_equality():
    source,changed = np.zeros(64,dtype=int),np.ones(64,dtype=int)
    original = source.copy()
    prediction = np.full(64,2)
    prediction[:57]=1
    assert joint_endpoint(original,prediction,source,changed)['scientific_rule_pass'] is False
    prediction[57]=1
    assert joint_endpoint(original,prediction,source,changed)['scientific_rule_pass'] is True
    assert joint_endpoint(np.zeros(10),np.asarray([1]*9+[2]),np.zeros(10),np.ones(10))['scientific_rule_pass'] is True


def test_all_block_conjunction_smoke_null_and_low_original_failure():
    base = dict(samples=64,joint_correct=64,scientific_rule_pass=True,technical_checks_pass=True)
    cells = [base.copy() for _ in range(6)]
    assert aggregate_endpoint(cells)['outcome']=='PASS'
    cells[3]['joint_correct']=57
    cells[3]['scientific_rule_pass']=False
    assert aggregate_endpoint(cells)['outcome']=='FAIL'
    assert aggregate_endpoint(cells,smoke=True)==dict(outcome='smoke',registered_rule_pass=None)
    raw = joint_endpoint(np.full(64,2),np.ones(64),np.zeros(64),np.ones(64))
    assert raw['replacement_correct']==64 and raw['joint_correct']==0
    assert raw['scientific_rule_pass'] is False


def test_unchanged_prediction_has_exact_zero_joint_count():
    source = np.arange(64)%10
    changed = (source+1)%10
    for prediction in (source,changed,np.full(64,3),np.arange(64)%7):
        assert joint_endpoint(prediction,prediction,source,changed)['joint_correct']==0


def test_frozen_affine_score_and_lowest_class_tie():
    features = np.asarray([[3.,7.],[4.,5.]])
    mean,scale = np.asarray([1.,3.]),np.asarray([2.,4.])
    coefficient = np.asarray([[2.,2.,-1.],[3.,3.,1.]])
    scores = frozen_scores(features,mean,scale,coefficient,np.asarray([1.,1.,0.]))
    np.testing.assert_array_equal(scores[0],[6.,6.,0.])
    assert scores[0].argmax()==0
    np.testing.assert_array_equal(mean,[1.,3.])
    np.testing.assert_array_equal(scale,[2.,4.])
    with pytest.raises(AssertionError):
        frozen_scores(features,mean,np.asarray([2.,0.]),coefficient,np.zeros(3))


def test_manual_same_step_current_and_instantaneous_flush():
    weights = sparse.csr_matrix(([.9,.8,.4],([1,2,2],[0,1,0])),shape=(3,3))
    kc,mb = np.asarray([0]),np.asarray([2])
    pre = np.asarray([.1,.2,.3])
    pattern = np.asarray([.5,0.,0.])
    following = manual_step(weights,kc,mb,pre,pattern)
    assert following[0]==pytest.approx(.6*np.tanh(.5))
    assert following[1]==pytest.approx(.6*np.tanh(.9*.1))
    # MBON receives previous OTHER, not the new OTHER coordinate.
    assert following[2]==pytest.approx(.6*np.tanh(.8*.2+.4*following[0]))
    patterns = np.asarray([[0.,0.,0.],[.5,0.,0.]])
    a,final_a,hash_a = trajectory_window(weights,patterns,kc,mb,pre,[0,1,1],[0,1,2],history=False)
    b,final_b,hash_b = trajectory_window(weights,patterns,kc,mb,pre,[1,1,1],[0,1,2],history=False)
    np.testing.assert_array_equal(final_a,final_b)
    assert final_a[0]>0 and final_a[2]>0
    assert state_sha(final_a)==state_sha(final_b) and hash_a!=hash_b
    assert hash_a==hashlib.sha256(a.tobytes()).hexdigest()


def test_margin_change_equals_frozen_state_projection():
    old,new = np.asarray([[.2,.4],[.1,.8]]),np.asarray([[.3,.1],[.5,.4]])
    mean,scale = np.asarray([.1,.2]),np.asarray([.03,.07])
    beta = np.asarray([[1.,3.,2.],[2.,1.,-1.]])
    intercept = np.asarray([.1,.2,.3])
    before,after = [frozen_scores(x,mean,scale,beta,intercept) for x in (old,new)]
    source,replacement = np.asarray([0,1]),np.asarray([1,2])
    index = np.arange(2)
    observed = (after[index,replacement]-after[index,source])-(before[index,replacement]-before[index,source])
    projected = np.sum((new-old)/scale*(beta[:,replacement]-beta[:,source]).T,axis=1)
    np.testing.assert_allclose(observed,projected,atol=1e-12,rtol=1e-12)
