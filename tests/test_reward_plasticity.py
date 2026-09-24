import numpy as np
from scipy import sparse
from scipy.special import softmax
from flying.brain.mushroom_body import KCEncoder,SelectedReadout,CircuitReservoir
from flying.brain.reward_plasticity import RewardPlasticity,policy_hash
from flying.brain.plasticity import weight_hash

def fixture(**kwargs):
    roles=np.array(['KC']*10+['MBON']*2)
    w=sparse.csr_matrix(([.2,.2,.2,.2],([10,10,11,11],[0,1,0,2])),shape=(12,12))
    encoder=KCEncoder(roles,7,fraction=.5);policy=SelectedReadout([10,11])
    policy.model.mean=np.zeros(2);policy.model.scale=np.array([.3,.5])
    policy.model.weights=np.random.default_rng(8).normal(size=(3,10))
    return RewardPlasticity(w,roles,encoder,policy,**kwargs)

def test_policy_score_gradient_matches_one_step_finite_difference():
    t=fixture(temperature=2.);prior=np.linspace(0,.5,12);t.state=prior.copy()
    prob,previous,activation=t.forward_policy(3);action=4
    gradient=t.score_gradient(previous,activation,prob,action)
    for k,position in enumerate(t.positions):
        def objective(offset):
            w=t.weights.copy();w.data[position]+=offset
            state=(1-t.leak)*prior+t.leak*np.tanh(w@prior+t.encoder(3))
            logits=t.policy.model.features(state[t.mbon])@t.policy.model.weights/t.temperature
            return np.log(softmax(logits,axis=1)[0,action])
        numeric=(objective(1e-6)-objective(-1e-6))/(2e-6)
        assert np.isclose(gradient[k],numeric,rtol=1e-5,atol=1e-8)

def test_delayed_delivery_flush_and_reward_identity():
    t=fixture(reward_delay=3,trace_decay=.8)
    labels=np.array([1,4,1,5,9,2]);digits=np.array([3,1,4,1,5,9]);policy_before=policy_hash(t.policy)
    captured=[];original=t.apply_reward
    def capture(reward):captured.append(reward);original(reward)
    t.apply_reward=capture
    history,events=t.fit_reward(digits,labels,epochs=2,seed=11)
    assert captured==events['applied_rewards'].ravel().tolist()
    assert all(h['delivered_rewards']==len(labels) for h in history)
    assert np.array_equal(events['true_rewards'],events['actions']==labels)
    assert np.array_equal(events['true_rewards'],events['applied_rewards'])
    assert policy_before==policy_hash(t.policy)
    t.audit()
    frozen=CircuitReservoir(t.weights,t.encoder);before=weight_hash(frozen.weights)
    frozen.states(digits);assert before==weight_hash(frozen.weights)

def test_trace_decay_and_yoked_reward_ignore_label_for_updates():
    a=fixture(reward_delay=2,trace_decay=.8);b=fixture(reward_delay=2,trace_decay=.8)
    digits=np.array([3,1,4,1,5]);signals=np.array([[0,1,0,1,0]],dtype=np.uint8)
    _,ea=a.fit_reward(digits,np.zeros(5,dtype=int),epochs=1,seed=22,yoked_rewards=signals)
    _,eb=b.fit_reward(digits,np.ones(5,dtype=int),epochs=1,seed=22,yoked_rewards=signals)
    assert np.array_equal(a.weights.data,b.weights.data)
    assert np.array_equal(ea['actions'],eb['actions'])
    assert np.array_equal(ea['applied_rewards'],signals)
    # No-trace cannot retain the last gradient during terminal delayed delivery.
    c=fixture(reward_delay=2,trace_decay=0)
    c.fit_reward(digits,np.zeros(5,dtype=int),epochs=1,seed=22)
    assert np.array_equal(c.eligibility,np.zeros_like(c.eligibility))

def test_zero_learning_is_exact_and_seeded_replay_is_exact():
    a=fixture(learning_rate=0,reward_delay=3);before=weight_hash(a.weights)
    digits=np.array([3,1,4,1,5,9]);labels=np.array([1,4,1,5,9,2])
    a.fit_reward(digits,labels,epochs=2,seed=12);assert before==weight_hash(a.weights)
    a=fixture(reward_delay=3);b=fixture(reward_delay=3)
    ha,ea=a.fit_reward(digits,labels,epochs=2,seed=12)
    hb,eb=b.fit_reward(digits,labels,epochs=2,seed=12)
    assert ha==hb and weight_hash(a.weights)==weight_hash(b.weights)
    for key in ea:assert np.array_equal(ea[key],eb[key])

def test_delay_longer_than_sequence_preserves_delivery_time():
    digits=np.array([3,1]);labels=np.array([1,4]);decay=.8;delay=5
    reference=fixture(learning_rate=0,reward_delay=delay,trace_decay=decay)
    rng=np.random.default_rng(12);trace=np.zeros(len(reference.positions))
    for digit in digits:
        prob,previous,activation=reference.forward_policy(digit)
        action=min(9,int(np.searchsorted(np.cumsum(prob),rng.random())))
        trace=decay*trace+reference.score_gradient(previous,activation,prob,action)
    trained=fixture(learning_rate=0,reward_delay=delay,trace_decay=decay)
    trained.fit_reward(digits,labels,epochs=1,seed=12)
    assert np.linalg.norm(trace)>0
    assert np.allclose(trained.eligibility,trace*decay**delay,rtol=1e-12,atol=1e-14)
