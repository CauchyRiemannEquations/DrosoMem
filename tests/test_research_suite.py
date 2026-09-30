import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import research_suite as run


def test_registered_case_and_trajectory_counts():
    c=run.read(run.CONFIG)
    for stage,count,paths,certs,independent in [('sequence',96,420,1728,350),('structure',96,672,1728,672),('correlation',96,52,5184,52)]:
        cases=run.cases(c,stage);assert len(cases)==count and len({run.stem(x) for x in cases})==count
        path=lambda x:(13 if run.selected(c,x,stage) else 0) if stage=='correlation' else (7 if run.selected(c,x,stage) else 1)
        assert sum(path(x) for x in cases)==paths
        assert sum(path(x) for x in cases if run.independently_selected(c,x,stage))==independent
        assert count*(54 if stage=='correlation' else 18)==certs
    assert [len(run.cases(c,s,True)) for s in ['sequence','structure','correlation']]==[4,3,2]


def test_common_noise_preserves_marginal_variance_and_expected_energy():
    # Exact orthogonal sample basis, so covariance is checked without Monte Carlo tolerance.
    z=np.sqrt(3)*np.array([[1,0],[0,1],[0,0]],float);g=np.sqrt(3)*np.array([0,0,1.])
    for rho in [0,.25,.75,1]:
        x=run.correlated(z,g,rho);np.testing.assert_allclose(x.T@x/3,[[1,rho],[rho,1]],atol=1e-15)
        for amplitude in [np.ones(2),np.sqrt(2)*np.array([.6,.8])]:
            np.testing.assert_allclose(np.mean(np.sum((x*amplitude)**2,axis=1)),2,atol=1e-15)
    np.testing.assert_array_equal(run.correlated(z,g,0),z)
    with pytest.raises(ValueError):run.correlated(z,g,1.1)


def test_common_and_independent_draws_are_paired_and_canonical():
    c=run.read(run.CONFIG);case=run.cases(c,'sequence')[0];ids=np.array([17,4,9])
    z,g=run.noise_bank(c,case,ids,7);p,h=run.noise_bank(c,dict(case,level='brain1',family='random'),ids[[2,0,1]],7)
    np.testing.assert_array_equal(z[:,:, [2,0,1]],p);np.testing.assert_array_equal(g,h)


def test_generic_prefix_does_not_mislabel_random_as_pi():
    x=run.endpoint(np.array([1,2,4]),np.array([1,2,3]),3)
    assert x['exact_prefix_symbols']==2 and 'pi_memory_score' not in x
    assert x['first_error_position']==3 and x['retention']==2/3
    assert run.endpoint([1],[1],0)['retention'] is None


def test_undefined_retention_cannot_be_ignored_by_gate():
    c=run.read(run.CONFIG)
    assert run.feedback.gate([10,10,10],[.3,np.nan,.3],'confirmation',c) is None
    assert not run.feedback.gate([10,10,10],[.09,.09,.09],'confirmation',c)


def test_structural_graph_draw_reused_across_sequence_families():
    c=run.read(run.CONFIG);cases=[x for x in run.cases(c,'structure') if x['topology']!='intact']
    draws={}
    for x in cases:
        key=(x['cohort'],x['seed'],x['circuit_seed'],x['topology']);s=run.graph_seed(c,x)
        assert draws.setdefault(key,s)==s
    assert len(set(draws.values()))==32
