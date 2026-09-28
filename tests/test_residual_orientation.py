from pathlib import Path
import itertools
import sys
import numpy as np
sys.path.insert(0,str(Path('scripts').resolve()))
from residual_orientation import orientation_reference,controls


def test_analytic_expectation_matches_exhaustive_two_dimensional_groups():
    q=np.array([[1.,2.,3.,-1.],[-2.,1.,4.,2.]])
    h=np.arange(16,dtype=float).reshape(4,4)/5
    energies=[]
    for p1,p2 in itertools.product(itertools.permutations(range(2)),repeat=2):
        p=list(p1)+[x+2 for x in p2]
        for signs in itertools.product([-1.,1.],repeat=4):
            score=(q[:,p]*signs)@h
            energies.append(np.mean(np.sum(score**2,axis=1)))
    np.testing.assert_allclose(orientation_reference(q,h,2),[np.mean(energies)],atol=1e-12)


def test_control_preserves_row_group_energy_and_source_arrays():
    rng=np.random.default_rng(99);q=rng.normal(size=(10,8));h=rng.normal(size=(8,12));old=q.copy()
    a=controls(q,h,4,[1,2,3]);b=controls(q,h,4,[1,2,3])
    for k in a:np.testing.assert_array_equal(a[k],b[k])
    np.testing.assert_array_equal(q,old)
    for p,s in zip(a['permutations'],a['signs']):
        np.testing.assert_allclose(np.linalg.eigvalsh(np.cov(q[:,p]*s,rowvar=False)),np.linalg.eigvalsh(np.cov(q,rowvar=False)),atol=1e-12)


def test_isotropic_head_has_constant_energy_under_orientation():
    rng=np.random.default_rng(101);q=rng.normal(size=(15,4));h=np.eye(4)
    a=controls(q,h,2,[7,8,9])
    np.testing.assert_allclose(a['draw_energy'][:,0],a['reference_energy'][0],atol=1e-12)
