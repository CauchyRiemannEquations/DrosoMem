import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import graph_relative_noise as run


def test_signal_rescaling_changes_own_energy_but_not_common_energy():
    x=np.random.default_rng(29).normal(size=(31,48))*np.linspace(.1,2,48)
    a,_,w,q,own=run.amplitudes(x,x,.03)
    b,_,ww,qq,other=run.amplitudes(3*x,x,.03)
    np.testing.assert_allclose(a[0],a[1],atol=1e-16)
    np.testing.assert_allclose(b[0],a[0],atol=1e-16)
    np.testing.assert_allclose(b[1],3*a[1],atol=1e-16)
    np.testing.assert_allclose(b[1]@b[1],9*(b[0]@b[0]),atol=1e-15)
    np.testing.assert_allclose(w,ww);assert qq==q
    np.testing.assert_allclose(other,3*own)
    with pytest.raises(ValueError):run.amplitudes(np.zeros_like(x),x,.03)


def test_attenuation_sign_is_gap_reduction_not_intact_victory():
    rows=[]
    # Control still wins after calibration, but its gap reduces by 3.
    for top,values in [('intact',[10,10]),('degree',[18,15]),('role',[8,9])]:
        for cal,value in zip(['common','own'],values):
            rows.append(dict(cohort='discovery',family='random',seed=1,topology=top,
                             calibration=cal,exact_prefix_symbols=value,retention=value/20))
    x=run.contrasts(pd.DataFrame(rows)).set_index('control')
    assert x.loc['degree','exact_prefix_symbols_attenuation']==3
    assert x.loc['degree','exact_prefix_symbols_own_gap']==5
    assert x.loc['role','exact_prefix_symbols_attenuation']==-1
    rows[2]['retention']=np.nan
    assert np.isnan(run.contrasts(pd.DataFrame(rows)).set_index('control').loc['degree','retention_attenuation'])


def test_joint_gate_keeps_negative_and_undefined_blocks():
    c=run.read(run.CONFIG)
    assert run.feedback.gate([3,3,3],[.2,.2,.2],'confirmation',c)
    assert not run.feedback.gate([3,3,0],[.2,.2,.2],'confirmation',c)
    assert not run.feedback.gate([3,3,3],[.2,.2,0],'confirmation',c)
    assert run.feedback.gate([3,3,3],[.2,np.nan,.2],'confirmation',c) is None


def test_preregistered_counts_fresh_noise_and_frozen_scope():
    c=run.read(run.CONFIG);panel=run.cases(c);old=run.read('configs/research_suite.json')
    assert len(panel)==96 and len({run.suite.stem(x) for x in panel})==96
    assert all(x['level']=='legacy5' for x in panel)
    assert len(panel)*19==1824 and len(panel)*18==1728
    assert sum(x['topology']=='intact' for x in panel)*9==288
    assert len(run.cases(c,True))*19==57
    assert not set(c['noise_seeds']) & set(old['noise_seeds']+old['common_noise_seeds']+c['smoke_noise_seeds'])
