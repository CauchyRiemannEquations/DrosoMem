import numpy as np
from scipy import sparse
from flying.brain.constrained_bptt import ConstrainedBPTT
from flying.brain.mushroom_body import KCEncoder,SelectedReadout,CircuitReservoir
from flying.brain.reward_plasticity import policy_hash
from flying.brain.plasticity import weight_hash


def fixture():
    roles=np.array(['KC']*10+['MBON']*2)
    # Include MBON→KC feedback so recurrent derivative genuinely matters.
    w=sparse.csr_matrix(([.3,.2,.15,.25,.4,.3],([10,10,11,11,0,2],[0,1,2,3,10,11])),shape=(12,12))
    encoder=KCEncoder(roles,7,fraction=.5);p=SelectedReadout([10,11])
    p.model.mean=np.zeros(2);p.model.scale=np.array([.2,.3])
    p.model.weights=np.random.default_rng(8).normal(size=(3,10))
    return ConstrainedBPTT(w,roles,encoder,p)


def test_full_temporal_derivative_matches_finite_differences():
    t=fixture();digits=np.array([3,1,4,1,5,9,2]);labels=np.array([1,4,1,5,9,2,6])
    theta=np.array([.2,-.1,.3,-.4]);loss,grad,_=t.objective(digits,labels,theta)
    numeric=[]
    for j in range(len(theta)):
        step=np.zeros_like(theta);step[j]=1e-5
        numeric.append((t.objective(digits,labels,theta+step,gradient=False)[0]-t.objective(digits,labels,theta-step,gradient=False)[0])/2e-5)
    assert np.allclose(grad,numeric,rtol=2e-5,atol=1e-9)
    local=t.objective(digits,labels,theta,temporal=False)[1]
    assert np.linalg.norm(grad-local)>1e-6
    # At zero logits the exact frozen special case still has the same derivative.
    zero=np.zeros_like(theta);g0=t.objective(digits,labels,zero)[1]
    for j in range(len(theta)):
        step=np.zeros_like(theta);step[j]=1e-5
        derivative=(t.objective(digits,labels,zero+step,gradient=False)[0]-t.objective(digits,labels,zero-step,gradient=False)[0])/2e-5
        assert np.isclose(g0[j],derivative,rtol=2e-5,atol=1e-9)


def test_zero_weights_and_forward_match_reservoir():
    t=fixture();w,_,_=t.materialize(t.theta)
    assert weight_hash(w)==weight_hash(t.initial)
    digits=np.array([3,1,4,1,5]);labels=np.array([1,4,1,5,9])
    states=CircuitReservoir(w,t.encoder,leak=t.leak,microsteps=1).states(digits)
    logits=t.policy.model.features(states[:,t.mbon])@t.policy.model.weights
    from scipy.special import logsumexp
    expected=np.mean(logsumexp(logits,axis=1)-logits[np.arange(len(labels)),labels])
    assert t.objective(digits,labels,gradient=False)[0]==expected
    t.fit(digits,labels,epochs=3,learning_rate=0)
    assert weight_hash(t.materialize(t.theta)[0])==weight_hash(t.initial)


def test_optimization_reduces_true_objective_preserving_anatomical_constraints():
    t=fixture();digits=np.tile([3,1,4,1,5,9,2],3);labels=np.roll(digits,-1)
    before=policy_hash(t.policy)
    history,selection=t.fit(digits,labels,epochs=30,learning_rate=.03)
    assert selection['selected_loss']<selection['initial_loss']-1e-4
    assert selection['selected_loss']==min(h['loss'] for h in history)
    assert policy_hash(t.policy)==before
    audit=t.audit();assert audit['relative_plastic_change']>0 and audit['max_budget_error']<1e-12
    t2=fixture();h2,s2=t2.fit(digits,labels,epochs=30,learning_rate=.03)
    assert history==h2 and selection==s2 and np.array_equal(t.theta,t2.theta)


def test_labels_never_enter_forward_weights_or_state():
    t=fixture();digits=np.array([3,1,4,1,5]);labels=np.array([1,4,1,5,9])
    before=weight_hash(t.materialize(t.theta)[0]);policy=policy_hash(t.policy)
    a=t.objective(digits,labels);b=t.objective(digits,(labels+1)%10)
    assert a[0]!=b[0]
    assert before==weight_hash(t.materialize(t.theta)[0]) and policy==policy_hash(t.policy)
