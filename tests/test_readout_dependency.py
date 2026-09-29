import sys
from pathlib import Path
import numpy as np
from scipy import sparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from readout_dependency import codes,train,fixed_scores
from verify_readout_dependency import reference_train,trajectory
from flying.brain.plasticity import KCMBONPlasticity
from alphabet_memory import SymbolEncoder


def fixture():
    roles=np.array(['KC']*16+['MBON']*48);rng=np.random.default_rng(9)
    a=rng.uniform(.001,.03,(64,64));a[rng.random(a.shape)<.7]=0;np.fill_diagonal(a,0)
    a[:,16:]*=-1
    weights=sparse.csr_matrix(a);bank=np.zeros((4,64))
    for k in range(4):bank[k,k*4:k*4+4]=.5
    return weights,roles,bank


def test_internal_update_has_independent_exact_replay_and_preserves_constraints():
    w,roles,bank=fixture();symbols=np.random.default_rng(23).integers(0,4,43)
    c=dict(leak=.6,plastic_learning_rate=.05,plastic_floor=1e-4,warmup=3,target_lag=2,plastic_epochs=2)
    for arm in ['frozen','aligned','shifted']:
        actual,audit,h,target=train(w,roles,bank,symbols,codes(7),arm,c)
        reference,h2=reference_train(w,roles,bank,symbols,codes(7),arm,c)
        np.testing.assert_array_equal(actual.data,reference.data);assert h==h2
        assert audit['nonplastic_edges_unchanged'] and audit['max_budget_error']<1e-12
        assert (audit['changed_edges']==0) if arm=='frozen' else (audit['changed_edges']>0)


def test_target_affects_only_future_dynamics_not_current_forward_state():
    w,roles,bank=fixture()
    a=KCMBONPlasticity(w,roles,SymbolEncoder(bank));b=KCMBONPlasticity(w,roles,SymbolEncoder(bank))
    a.step(0);b.step(0)
    np.testing.assert_array_equal(a.step(1,np.full(48,.25)),b.step(1,np.zeros(48)))
    assert not np.array_equal(a.weights.data,b.weights.data)


def test_fixed_code_decoder_requires_no_fitting_and_round_trips_prototypes():
    code=codes(44);assert np.all(np.sum(code!=0,axis=0)==1) and np.all(np.sum(code!=0,axis=1)==12)
    np.testing.assert_array_equal(fixed_scores(code,code).argmax(1),np.arange(4))
