import sys
from pathlib import Path
import numpy as np
import pytest
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import relative_noise as study


def test_small_sigma_csv_round_trip(tmp_path):
    values=np.array([.00011182421773467538,1.2345678912345678e-6,1.2345678912345678e-4])
    path=tmp_path/'sigma.csv';pd.DataFrame({'sigma':values}).to_csv(path,index=False)
    np.testing.assert_array_equal(study.read_table(path).sigma,values)


def test_raw_population_scale_without_head_floor_and_multiplicative_equivariance():
    x=np.array([[-2.,0.,-4.],[2.,2.,4.]])*1e-9
    q,sigma=study.calibration(x,.3)
    assert q==2e-9 and sigma==.3*q and q<1e-5
    q2,sigma2=study.calibration(7*x+2,.3)
    np.testing.assert_allclose([q2,sigma2],7*np.array([q,sigma]),rtol=1e-7)
    assert study.calibration(x,0)[1]==0


@pytest.mark.parametrize('x,r',[(np.ones((5,48)),.3),(np.ones(4),.3),(np.array([[0,np.nan],[1,2]]),.3),(np.array([[0,0],[1,2]]),-1)])
def test_invalid_calibration_is_not_replaced_by_epsilon(x,r):
    with pytest.raises(ValueError):study.calibration(x,r)


def test_both_absolute_and_relative_gain_and_seed_consistency_are_required():
    c=study.config()
    assert study.gate(np.array([.2]*3),np.array([3]*3),3,c)
    assert not study.gate(np.array([.2]*3),np.array([1]*3),3,c)
    assert not study.gate(np.array([.4,.4,-.01]),np.array([4,4,4]),3,c)
    assert study.gate(np.array([.2,np.nan,.2]),np.array([3]*3),3,c) is None


def test_prespecified_scope_and_fresh_seeds():
    c=study.config();assert len(study.old.case_list(c))==48
    assert c['relative_strengths']==[.0003,.03,.3]
    audit=study.read('docs/relative-noise-seed-audit.json')
    assert audit['fresh_model_seeds']==c['cohorts']['confirmation']
    assert not audit['prior_occurrences']
