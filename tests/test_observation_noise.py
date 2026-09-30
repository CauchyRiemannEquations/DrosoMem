import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import observation_noise as study
from verify_observation_noise import independent_bank,independent_head
from flying.models.nonlinear_readout import NonlinearReadout


def test_zero_clipping_and_input_immutability():
    x=np.array([[.9,-.9],[.1,.2]]);z=np.array([[2.,-2.],[1.,-1.]])
    before=x.copy();noise_before=z.copy()
    np.testing.assert_array_equal(study.instantaneous(x,z,0),x)
    np.testing.assert_allclose(study.instantaneous(x,z,.2),[[1.,-1.],[.3,0.]])
    np.testing.assert_array_equal(x,before);np.testing.assert_array_equal(z,noise_before)
    with pytest.raises(ValueError):study.instantaneous(x,z,-.1)
    with pytest.raises(ValueError):study.instantaneous(x,np.full_like(x,np.nan),.1)


def test_canonical_time_stream_and_shared_neuron_pairing():
    ids=np.array([17,22,39,44,51]);observed=np.array([44,22]);z=study.noise_bank(ids,observed,31,701,7,372001)
    rng=np.random.default_rng(np.random.SeedSequence([372001,31,701,1]))
    expected=np.stack([rng.standard_normal(5)[[3,1]] for _ in range(7)])
    np.testing.assert_array_equal(z,expected)
    np.testing.assert_array_equal(z,independent_bank(ids,observed,31,701,7,372001))
    other=study.noise_bank(ids,np.array([22,39]),31,701,7,372001)
    np.testing.assert_array_equal(z[:,1],other[:,0])


def test_independent_decoder_and_frozen_parameters():
    head=NonlinearReadout([0,1,2],4,5);x=np.array([[-.3,.1,.7],[.3,.2,-.7]])
    head.initialize(x);before=head.digest();cp=dict(mean=head.mean,scale=head.scale,**head.parameters)
    pred,probs=independent_head(x,cp)
    np.testing.assert_array_equal(pred,head.predict(x))
    logits=head.logits(x);ex=np.exp(logits-logits.max(1,keepdims=True))
    np.testing.assert_allclose(probs,ex/ex.sum(1,keepdims=True),atol=1e-14)
    assert head.digest()==before


def test_material_cost_requires_registered_magnitude_and_seed_consistency():
    c={'history_cost_threshold':.05}
    assert study.history_gate([.06,.06,.06],'confirmation',c)
    assert not study.history_gate([.04,.04,.04],'confirmation',c)
    assert not study.history_gate([.2,.2,-.01],'confirmation',c)
    assert study.history_gate([.1,.1,.1,.1,-.01],'discovery',c)
