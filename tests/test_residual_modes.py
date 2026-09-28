from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path('scripts').resolve()))
from residual_modes import decompose


def test_cancellation_is_not_additive_diagonal_energy():
    z=np.diag([3.,2.]);d=np.array([[1.,1.],[-1.,-1.]])
    w=np.array([[1.,0,0,0],[-.9,0,0,0]])
    a=decompose(z,d,w,1)
    assert a['diagonal_energy'].sum()>100*np.mean(np.sum(a['total']**2,axis=2))
    assert (a['signed_attribution']<0).any()
    np.testing.assert_allclose(a['signed_attribution'].sum(axis=1),np.mean(np.sum(a['total']**2,axis=2),axis=0))


def test_train_only_basis_and_mode_sign_invariance():
    rng=np.random.default_rng(513);z=rng.normal(size=(100,8));d=rng.normal(size=(20,8));w=rng.normal(size=(8,12))
    a=decompose(z,d,w,4);b=decompose(z,10*d,2*w,4)
    np.testing.assert_array_equal(a['basis'],b['basis'])
    signs=rng.choice([-1,1],8);v=a['basis']*signs
    np.testing.assert_allclose((d@v)@(v.T@w),a['total'].reshape(20,12),atol=1e-12)
    np.testing.assert_allclose(a['total'],a['low']+a['high'],atol=1e-12)


def test_zero_residual_has_zero_score_attributions():
    a=decompose(np.diag([2.,1.]),np.zeros((3,2)),np.ones((2,4)),1)
    for key in ['total','low','high','signed_attribution','diagonal_energy']:
        np.testing.assert_array_equal(a[key],np.zeros_like(a[key]))
