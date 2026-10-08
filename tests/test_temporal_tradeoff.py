"""Exact prospective conjunction and diagnostic/reference safeguards."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import pandas as pd
from temporal_tradeoff import summarize,current_diagnostics
from tradeoff_support import config

def table(c):
    rows=[]
    for level,seeds in c['blocks_by_level'].items():
        for seed in seeds:
            for arm in c['arms']:
                for lag in c['lags']:
                    correct=(1900 if arm=='carry_only' else 2000) if lag==0 else (254 if arm=='carry_only' else 200)
                    accuracy=correct/2000
                    rows.append(dict(level=level,arm=arm,seed=seed,lag=lag,samples=2000,correct=correct,accuracy=accuracy,
                        chance_adjusted=(accuracy-.1)/.9,baseline_excess=accuracy-.1,frequency_accuracy=.1,current_accuracy=.1))
    return pd.DataFrame(rows)

def test_exact_both_boundaries_and_carry_accuracy_is_outcome():
    c=config();f=table(c);s=summarize(f,c,False)[0]
    assert s['outcome']=='PASS' and s['primary_pass'] is True
    p=s['levels']['legacy5']['primary']
    assert p['historical_gain']['mean_exact']=='3/100'
    assert p['current_difference']['mean_exact']=='-1/20'
    f.loc[(f.arm=='carry_only')&(f.lag==0),'correct']=1700
    assert summarize(f,c,False)[0]['primary_pass'] is True

def test_historical_direction_is_required_even_when_mean_passes():
    c=config();f=table(c)
    f.loc[(f.arm=='carry_only')&(f.lag>0),'correct']=320
    f.loc[(f.level=='legacy5')&(f.arm=='carry_only')&(f.seed==1200006)&(f.lag>0),'correct']=180
    s=summarize(f,c,False)[0];p=s['levels']['legacy5']['primary']
    assert p['historical_mean_pass'] is True and p['all_history_positive'] is False
    assert s['outcome']=='FAIL'
    f=table(c);f.loc[(f.arm=='carry_only')&(f.lag>0),'correct']=253
    assert summarize(f,c,False)[0]['primary_pass'] is False

def test_current_direction_and_mean_are_both_required():
    c=config();f=table(c)
    f.loc[(f.arm=='carry_only')&(f.lag==0),'correct']=1700
    f.loc[(f.level=='legacy5')&(f.arm=='carry_only')&(f.seed==1200006)&(f.lag==0),'correct']=2000
    s=summarize(f,c,False)[0];p=s['levels']['legacy5']['primary']
    assert p['current_mean_pass'] is True and p['all_current_negative'] is False
    assert s['outcome']=='FAIL'
    f=table(c);f.loc[(f.arm=='carry_only')&(f.lag==0),'correct']=1901
    assert summarize(f,c,False)[0]['primary_pass'] is False

def test_reference_failure_nulls_endpoint_and_smoke_is_not_confirmation():
    c=config();f=table(c)
    assert summarize(f,c,True)[0]['outcome']=='smoke'
    assert summarize(f,c,True)[0]['primary_pass'] is None
    f.loc[(f.level=='legacy5')&(f.arm=='instantaneous')&(f.seed==1200006)&(f.lag==0),'correct']=1979
    s=summarize(f,c,False)[0]
    assert s['outcome']=='assay-invalid' and s['primary_pass'] is None

def test_error_strata_and_undefined_conditional_fraction():
    a=dict(ytest=np.array([[1,1],[2,9],[3,9],[4,4]]),predictions=np.array([[1,0],[9,0],[8,0],[4,0]]))
    d=current_diagnostics(a)
    assert d['repeat']==dict(samples=2,correct=2,accuracy=1.)
    assert d['switch']==dict(samples=2,correct=0,accuracy=0.)
    assert d['errors_total']==2 and d['errors_equal_previous_symbol']==1 and d['error_equals_previous_fraction']==.5
    assert d['current_confusion'][2][9]==1
    a['predictions'][:,0]=a['ytest'][:,0];a['ytest'][:,1]=0
    d=current_diagnostics(a)
    assert d['error_equals_previous_fraction'] is None and d['repeat']['accuracy'] is None
