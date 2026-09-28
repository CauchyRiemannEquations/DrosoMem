import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path('scripts').resolve()))
from alphabet_memory import SymbolReadout,SymbolEncoder,symbol_bank,dataset,read
from flying.brain.mushroom_body import KCEncoder
from flying.models.nonlinear_readout import NonlinearReadout
from flying.training.whole_brain_memory import rollout


def test_nested_symbol_bank_preserves_legacy_and_cardinality():
    roles=np.array(['KC']*512+['MBON']*48);bank=symbol_bank(roles,42)
    np.testing.assert_array_equal(bank[:10],KCEncoder(roles,42).patterns)
    assert np.all(np.count_nonzero(bank,axis=1)==51) and not bank[:,512:].any()
    for k in [2,4,10,16]:
        np.testing.assert_array_equal(bank[:k],symbol_bank(roles,42,k=k))
        encoder=SymbolEncoder(bank[:k]);np.testing.assert_array_equal(encoder(k-1),bank[k-1])
        with pytest.raises(ValueError):encoder(k)
        with pytest.raises(ValueError):encoder(-1)
        with pytest.raises(ValueError):encoder(.5)


def test_k10_head_exact_legacy_fit_and_k_parameter_counts():
    rng=np.random.default_rng(41);x=rng.normal(size=(30,48));y=rng.integers(0,10,30)
    legacy=NonlinearReadout(np.arange(48),8,32);new=SymbolReadout(np.arange(48),8,32,10)
    _,lh=legacy.fit(x,y,epochs=7,checkpoints=(7,));_,nh=new.fit(x,y,epochs=7,checkpoints=(7,))
    assert legacy.digest()==new.digest() and lh==nh
    ref=None
    for k,expected in [(2,410),(4,428),(10,482),(16,536)]:
        head=SymbolReadout(np.arange(48),8,32,k);head.initialize(x)
        assert head.parameter_count==expected
        if ref is not None:np.testing.assert_array_equal(ref,head.parameters['w1'])
        ref=head.parameters['w1']


@pytest.mark.parametrize('k',[2,4,16])
def test_generalized_objective_gradients_training_and_label_bounds(k):
    rng=np.random.default_rng(55);x=rng.normal(size=(20,5));y=np.arange(20)%k
    head=SymbolReadout(np.arange(5),3,33,k);head.initialize(x);weights=np.linspace(1,4,len(y))
    _,grad,_=head.objective(x,y,.01,weights);eps=1e-6
    for name in head.parameters:
        index=(0,)*(head.parameters[name].ndim);old=head.parameters[name][index]
        head.parameters[name][index]=old+eps;plus=head.objective(x,y,.01,weights)[0]
        head.parameters[name][index]=old-eps;minus=head.objective(x,y,.01,weights)[0]
        head.parameters[name][index]=old
        assert grad[name][index]==pytest.approx((plus-minus)/(2*eps),rel=1e-4,abs=1e-7)
    initial=head.objective(x,y)[0];head.fit(x,y,epochs=20,learning_rate=.03,checkpoints=(20,))
    assert head.objective(x,y)[0]<initial
    for bad in [np.full(20,k),np.full(20,-1),np.zeros(20,dtype=float)]:
        with pytest.raises(ValueError):head.objective(x,bad)


def test_common_prompt_and_rollout_keeps_symbol15_integer():
    c=read('configs/alphabet_memory.json')
    for k in c['alphabet_sizes']:
        symbols=dataset(c,k,12).symbols();assert len(symbols)==128
        np.testing.assert_array_equal(symbols[:3],[0,1,0]);assert symbols.min()>=0 and symbols.max()<k
    class Model:
        def reset(self):self.inputs=[]
        def step(self,symbol):self.inputs.append(symbol);return np.zeros(2)
    head=SymbolReadout(np.arange(2),2,42,16);head.initialize(np.zeros((5,2)))
    head.parameters['w2'][:]=0;head.parameters['b2'][15]=10
    model=Model();pred,prob,_=rollout(model,np.arange(2),head,[0,1,0],5)
    np.testing.assert_array_equal(pred,[15]*5);assert prob.shape==(5,16)
    assert model.inputs==[0,1,0]+[15]*5
