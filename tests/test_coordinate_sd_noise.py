import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import coordinate_sd_noise as run


def test_coordinate_noise_matches_each_nonzero_training_sd():
    x=np.random.default_rng(19).normal(size=(35,48))*np.linspace(.1,2,48);x[:,0]=0
    amp,sd,w,q,own=run.amplitudes(x,x,.03)
    np.testing.assert_array_equal(amp[2],.03*sd)
    np.testing.assert_allclose(amp[2,sd>0]/sd[sd>0],.03,atol=1e-16)
    assert amp[2,0]==0
    np.testing.assert_allclose(amp[2]@amp[2],.03**2*np.sum(sd**2),atol=1e-16)
    np.testing.assert_array_equal(amp[0],amp[1])


def test_rescaling_invariance_and_expected_energy():
    x=np.random.default_rng(12).normal(size=(25,48))*np.linspace(.03,3,48)
    a,*_=run.amplitudes(x,x,.3);b,*_=run.amplitudes(4*x,x,.3)
    np.testing.assert_allclose(b[0],a[0]);np.testing.assert_allclose(b[1:],4*a[1:])
    np.testing.assert_allclose(b[2]@b[2],16*(a[2]@a[2]))


def panel(degree_coordinate=15):
    rows=[]
    for top,values in [('intact',[10,10,4]),('degree',[18,17,degree_coordinate]),('role',[9,9,6])]:
        for cal,value in zip(['common','own','coordinate'],values):
            rows.append(dict(cohort='discovery',family='random',seed=1,topology=top,
                calibration=cal,exact_prefix_symbols=value,retention=value/20))
    return pd.DataFrame(rows)


def test_primary_must_subtract_changed_intact_not_just_control_drop():
    x=run.contrasts(panel()).set_index('control').loc['degree']
    assert x.exact_prefix_symbols_own_gap==7 and x.exact_prefix_symbols_coordinate_gap==11
    assert x.exact_prefix_symbols_attenuation==-4 # control drops2, intact drops6
    assert x.exact_prefix_symbols_total_attenuation==-3
    assert x.exact_prefix_symbols_median_attenuation==1


def test_incremental_positive_does_not_require_intact_victory():
    x=run.contrasts(panel(8)).set_index('control').loc['degree']
    assert x.exact_prefix_symbols_coordinate_gap==4 and x.exact_prefix_symbols_attenuation==3


def test_undefined_retention_propagates_without_excluding_seed():
    f=panel();f.loc[(f.topology=='degree')&(f.calibration=='coordinate'),'retention']=np.nan
    x=run.contrasts(f).set_index('control').loc['degree']
    assert np.isnan(x.retention_attenuation)
    assert run.feedback.gate([3,3,3],[.2,np.nan,.2],'confirmation',run.read(run.CONFIG)) is None


def test_fixed_scope_counts_and_noise_disjointness():
    c=run.read(run.CONFIG);p=run.cases(c)
    assert c['calibrations']==['common','own','coordinate'] and len(p)==96
    assert len(p)*28==2688 and len(p)*27==2592
    assert sum(x['topology']=='intact' for x in p)*9==288
    assert len(run.cases(c,True))*28==84
    old=run.read('configs/graph_relative_noise.json')
    assert not set(c['noise_seeds']) & set(old['noise_seeds']+old['smoke_noise_seeds']+c['smoke_noise_seeds'])
