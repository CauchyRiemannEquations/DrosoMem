import numpy as np
from scipy import sparse
from flying.brain.plasticity import KCMBONPlasticity,teacher_codes,weight_hash
from flying.brain.mushroom_body import KCEncoder,CircuitReservoir
from flying.brain.role_random import role_random
from flying.data.connectome import load_connectome
from flying.data.mushroom_body import load_roles
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def real_graph():
    path=ROOT/'data/flywire_783_mb_left_kc512_s701';w,ids,_=load_connectome(path);roles,_=load_roles(path,ids)
    return w,roles

def test_role_random_matches_plastic_budget_and_role_blocks():
    a,roles=real_graph();b=role_random(a,roles,42)
    assert a.nnz==b.nnz and not b.diagonal().any()
    for source in np.unique(roles):
        for target in np.unique(roles):
            pre=np.flatnonzero(roles==source);post=np.flatnonzero(roles==target)
            x=a[post][:,pre];y=b[post][:,pre]
            assert x.nnz==y.nnz
            assert np.array_equal(np.sort(abs(x.data)),np.sort(abs(y.data)))
    assert weight_hash(b)==weight_hash(role_random(a,roles,42))

def test_zero_learning_rate_equals_fixed_dynamics():
    roles=np.array(['KC']*10+['MBON']*2);w=sparse.csr_matrix(([.3,.2,-.1],([10,11,0],[0,1,11])),shape=(12,12))
    encoder=KCEncoder(roles,7);r=CircuitReservoir(w,encoder,microsteps=2)
    p=KCMBONPlasticity(w,roles,encoder,microsteps=2,learning_rate=0)
    for d in [3,1,4,1,5,9]:assert np.array_equal(r.step(d),p.step(d,np.array([.2,.1])))
    assert p.audit()['changed_edges']==0

def test_teacher_not_in_forward_state_and_only_allowed_weights_change():
    roles=np.array(['KC']*10+['MBON']*2)
    w=sparse.csr_matrix(([.2,.2,.3,.1,-.1],([10,10,11,11,0],[0,1,0,2,11])),shape=(12,12))
    encoder=KCEncoder(roles,5);p=KCMBONPlasticity(w,roles,encoder,learning_rate=.2)
    q=KCMBONPlasticity(w,roles,encoder,learning_rate=.2)
    p.state[0]=q.state[0]=.5
    assert np.array_equal(p.step(3,np.array([.8,.8])),q.step(3,np.zeros(2)))
    assert not np.array_equal(p.weights.data,q.weights.data)
    audit=p.audit();assert audit['changed_edges']>0 and audit['nonplastic_edges_unchanged']
    assert np.array_equal(p.weights.indices,w.indices) and np.array_equal(p.weights.indptr,w.indptr)
    frozen=CircuitReservoir(p.weights,encoder);before=weight_hash(frozen.weights)
    frozen.states([3,1,4,1,5,9]);assert before==weight_hash(frozen.weights)
    assert not frozen.weights.data.flags.writeable

def test_simple_teacher_step_reduces_local_error():
    roles=np.array(['KC']*10+['MBON']*2)
    w=sparse.csr_matrix(([.2,.2,.2,.2],([10,10,11,11],[0,1,0,1])),shape=(12,12))
    encoder=KCEncoder(roles,4);p=KCMBONPlasticity(w,roles,encoder,learning_rate=.1)
    prior=np.zeros(12);prior[0]=1;p.state=prior.copy();target=np.array([.5,.5])
    old=p.step(2,target)[p.mbon]
    activation=np.tanh(p.weights@prior+encoder(2));new=((1-p.leak)*prior+p.leak*activation)[p.mbon]
    assert np.mean((new-target)**2)<np.mean((old-target)**2)
    p.audit()
    assert np.array_equal(teacher_codes(48,42),teacher_codes(48,42))
