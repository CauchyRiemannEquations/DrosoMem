import numpy as np
import pytest
from flying.evaluation.delayed_memory import delayed_labels,decode_delays
from flying.models.ridge import RidgeDecoder,RidgeReadout

def test_delay_alignment_and_validation():
    digits=np.arange(10)
    assert np.array_equal(delayed_labels(digits,[0,1,3],3)[0],[3,2,0])
    assert np.array_equal(delayed_labels(digits,[0,1,3],3)[-1],[9,8,6])
    with pytest.raises(ValueError): delayed_labels(digits,[4],3)

def test_known_shift_register_recovers_past_but_memoryless_does_not():
    rng=np.random.default_rng(11); train=rng.integers(0,10,1600);test=rng.integers(0,10,1000)
    def register(d):
        x=np.zeros((len(d),30))
        for lag in range(3): x[lag:,lag*10:(lag+1)*10]=np.eye(10)[d[:len(d)-lag]]
        return x
    a=register(train);b=register(test)
    rows,pred,truth=decode_delays(a,b,train,test,[0,1,2],20)
    assert np.array_equal(pred,truth)
    assert all(r['r2_vs_training_frequency']>.99 for r in rows)
    rows,_,_=decode_delays(a[:,:10],b[:,:10],train,test,[0,1,2],20)
    assert rows[0]['accuracy']==1
    assert rows[1]['accuracy']<.15 and rows[2]['accuracy']<.15

def test_ridge_training_statistics_and_observation_mask():
    rng=np.random.default_rng(31);x=rng.normal(size=(100,8));labels=np.arange(100)%10
    r=RidgeReadout([1,3],.1);r.fit(x,labels)
    mean=r.model.mean.copy();weights=r.model.weights.copy()
    changed=x.copy();changed[:,[0,2,4,5,6,7]]+=1e9
    assert np.array_equal(r.predict(x),r.predict(changed))
    r.predict(x*100)
    assert np.array_equal(mean,r.model.mean) and np.array_equal(weights,r.model.weights)
    assert np.allclose(mean,x[:,[1,3]].mean(axis=0))
    with pytest.raises(ValueError): RidgeDecoder(0)
