import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from test_readout_dependency import fixture
from readout_dependency import codes
from local_reward import train
from verify_local_reward import reference_train
from flying.brain.local_reward import LocalRewardPlasticity
from alphabet_memory import SymbolEncoder


def test_local_rule_exact_reference_and_yoked_marginals():
    w,roles,bank=fixture();symbols=np.random.default_rng(83).integers(0,4,43)
    c=dict(leak=.6,plastic_learning_rate=.05,plastic_floor=1e-4,warmup=3,target_lag=2,
           plastic_epochs=2,trace_decay=.8,activity_rate=.05,baseline_rate=.05,noise_std=.02)
    yoked=None
    for arm in ['contingent','frozen','yoked']:
        actual,audit,h,logs=train(w,roles,bank,symbols,codes(7),arm,c,19,yoked)
        reference,h2,logs2=reference_train(w,roles,bank,symbols,codes(7),arm,c,19,yoked)
        np.testing.assert_array_equal(actual.data,reference.data);assert h==h2
        for key in logs:np.testing.assert_array_equal(logs[key],logs2[key])
        assert audit['nonplastic_edges_unchanged'] and audit['max_budget_error']<1e-12
        assert (audit['changed_edges']==0) if arm=='frozen' else (audit['changed_edges']>0)
        if arm=='contingent':yoked=logs['actual_rewards']
        if arm=='yoked':
            np.testing.assert_array_equal(logs['applied_rewards'],np.roll(yoked,20,axis=1))
            np.testing.assert_array_equal(logs['applied_rewards'].sum(1),yoked.sum(1))


def test_scalar_reward_changes_weights_only_after_forward_and_rejects_targets():
    w,roles,bank=fixture()
    a=LocalRewardPlasticity(w,roles,SymbolEncoder(bank));b=LocalRewardPlasticity(w,roles,SymbolEncoder(bank))
    for m in [a,b]:m.forward(0,np.zeros(48));m.reinforce(.25)
    np.testing.assert_array_equal(a.forward(1,np.zeros(48)),b.forward(1,np.zeros(48)))
    before=a.state.copy();a.reinforce(1);b.reinforce(0)
    np.testing.assert_array_equal(before,a.state);assert not np.array_equal(a.weights.data,b.weights.data)
    with pytest.raises(ValueError):a.reinforce(1)
    a.forward(2,np.zeros(48))
    with pytest.raises(ValueError):a.reinforce(np.ones(48))
    with pytest.raises(TypeError):a.step(2,np.ones(48))
