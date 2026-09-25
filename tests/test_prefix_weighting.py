import numpy as np
import pytest
from flying.models.nonlinear_readout import NonlinearReadout
from flying.training.phase5_prefix import sample_weights


def fixture():
    rng = np.random.default_rng(25)
    states = rng.normal(size=(7,4)); labels = rng.integers(0,10,7)
    head = NonlinearReadout(np.arange(4),hidden=3,seed=42); head.initialize(states)
    return head, states, labels


def test_weighted_gradient_finite_difference():
    head, states, labels = fixture(); weights = np.array([1,1,4,4,4,1,1.])
    _, gradients, _ = head.objective(states,labels,.01,weights)
    for name, parameter in head.parameters.items():
        for index in np.ndindex(parameter.shape):
            original = parameter[index]; eps = 1e-6
            parameter[index] = original + eps; plus = head.objective(states,labels,.01,weights)[0]
            parameter[index] = original - eps; minus = head.objective(states,labels,.01,weights)[0]
            parameter[index] = original
            assert np.isclose((plus-minus)/(2*eps),gradients[name][index],atol=1e-7,rtol=1e-5)


def test_uniform_and_weight_scale_equivalence():
    head, states, labels = fixture()
    plain = head.objective(states,labels)
    for weights in [np.ones(7),np.full(7,4.)]:
        weighted = head.objective(states,labels,sample_weight=weights)
        assert np.isclose(plain[0],weighted[0],atol=1e-14)
        for name in plain[1]: np.testing.assert_allclose(plain[1][name],weighted[1][name],atol=1e-14)


@pytest.mark.parametrize('weights',[np.zeros(7),np.ones(6),[-1]*7,[float('nan')]*7,[float('inf')]*7])
def test_invalid_sample_weights(weights):
    head, states, labels = fixture()
    with pytest.raises(ValueError): head.objective(states,labels,sample_weight=weights)


def test_exact_scored_digit_window():
    weights = sample_weights(199,3,32,4)
    np.testing.assert_array_equal(np.flatnonzero(weights==4)+1,np.arange(3,35))
    assert weights.sum() == 295


def test_weighting_keeps_statistics_and_initialization_fixed():
    _,states,labels = fixture()
    heads = [NonlinearReadout(np.arange(4),hidden=3,seed=42) for _ in range(2)]
    for head,weights in zip(heads,[np.ones(7),np.array([1,1,4,4,4,1,1])]):
        head.fit(states,labels,epochs=2,checkpoints=(2,),sample_weight=weights)
    np.testing.assert_array_equal(heads[0].mean,heads[1].mean)
    np.testing.assert_array_equal(heads[0].scale,heads[1].scale)
    assert heads[0].digest()!=heads[1].digest()
