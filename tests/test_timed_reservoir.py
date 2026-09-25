import numpy as np
from scipy import sparse
from flying.brain.mushroom_body import KCEncoder,CircuitReservoir,SelectedReadout
from flying.brain.timed_reservoir import TimedReservoir
from flying.brain.plasticity import weight_hash
from flying.evaluation.free_recall import evaluate_recall
from flying.evaluation.metrics import pi_memory_score


def fixture():
    roles=np.array(['KC']*10+['MBON','DAN','APL'])
    w=sparse.csr_matrix(([.7,.2,-.1,.3],([10,10,0,11],[0,11,12,10])),shape=(13,13))
    return roles,w,KCEncoder(roles,19,fraction=.5)


def test_synchronous_schedules_exactly_match_existing_implementation():
    roles,w,e=fixture()
    for schedule,count in [('sync_one',1),('sync_two',2)]:
        a=TimedReservoir(w,e,roles,schedule=schedule);b=CircuitReservoir(w,e,microsteps=count)
        assert np.array_equal(a.states([3,1,4,1,5]),b.states([3,1,4,1,5]))
        assert weight_hash(a.weights)==weight_hash(w)


def test_mbon_schedule_integrates_once_and_uses_new_kc_only():
    roles,w,e=fixture();r=TimedReservoir(w,e,roles,schedule='mbon_after_kc')
    old=np.linspace(-.2,.3,13);r.state=old.copy();stim=e(3)
    sync=(1-r.leak)*old+r.leak*np.tanh(w@old+stim)
    mixed=old.copy();mixed[:10]=sync[:10]
    expected=sync.copy();expected[10]=(1-r.leak)*old[10]+r.leak*np.tanh(w[10]@mixed+stim[10])[0]
    assert np.allclose(r.step(3),expected,rtol=0,atol=1e-15)
    assert np.array_equal(r.state[np.arange(13)!=10],sync[np.arange(13)!=10])


def test_current_input_is_unavailable_then_available_without_new_edges():
    roles=np.array(['KC']*10+['MBON'])
    w=sparse.csr_matrix(([.8],([10],[0])),shape=(11,11))
    class Encoder:
        def __call__(self,d):
            x=np.zeros(11);x[0]=d/10;return x
    e=Encoder()
    for schedule in ['sync_one','mbon_after_kc','sync_two']:
        a=TimedReservoir(w,e,roles,schedule=schedule);b=TimedReservoir(w,e,roles,schedule=schedule)
        xa=a.step(1)[10];xb=b.step(9)[10]
        if schedule=='sync_one':assert xa==xb==0
        else:assert xb>xa>0
        assert weight_hash(a.weights)==weight_hash(w)


def test_first_free_recall_error_equals_teacher_forced_error_from_same_reset():
    roles,w,e=fixture();digits=np.array([3,1,4,1,5,9,2,6,5,3,5,8,9,7,9])
    for schedule in ['sync_one','mbon_after_kc','sync_two']:
        r=TimedReservoir(w,e,roles,schedule=schedule);states=r.states(digits[:-1])
        readout=SelectedReadout([10]);readout.fit(states,digits[1:],epochs=4)
        expected=pi_memory_score(digits[3:],readout.predict(states[2:]))
        got=evaluate_recall(r,readout,digits,3,len(digits)-3)
        assert got['pi_memory_score']==expected
