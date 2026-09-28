"""Known temporal representations and leakage checks for the current runner."""
import importlib.util
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location('delayed_runner',Path(__file__).parents[1]/'scripts/run_delayed_symbol.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


def synthetic():
    rng=np.random.default_rng(43);train=rng.integers(0,10,1600);test=rng.integers(0,10,1000)
    def register(d):
        x=np.zeros((len(d),30))
        for lag in range(3):x[lag:,lag*10:(lag+1)*10]=np.eye(10)[d[:len(d)-lag]]
        return x
    return train,test,register(train),register(test)


def test_shift_register_and_misaligned_head():
    train,test,x,y=synthetic();c=dict(warmup=100,lags=[0,1,2],alpha=1.,null_shift=750)
    a=runner.fit_probe(x,y,train,test,c);runner.independent_checks(a,c)
    rows=runner.measures(a,c)
    assert all(r['test_accuracy']==1 and r['null_accuracy']<.15 for r in rows)
    assert all(r['r2_vs_training_frequency']>.99 for r in rows)
    # Exact target alignment includes the current symbol and the last two inputs.
    np.testing.assert_array_equal(a['test_targets'][0],test[[100,99,98]])


def test_test_stream_cannot_change_fitted_parameters():
    train,test,x,y=synthetic();c=dict(warmup=100,lags=[0,1,2],alpha=1.,null_shift=750)
    a=runner.fit_probe(x,y,train,test,c)
    b=runner.fit_probe(x,y*1e6,train,(test+3)%10,c)
    for name in ['weights','null_weights','mean','scale','target_mean','null_target_mean']:
        np.testing.assert_array_equal(a[name],b[name])
    memoryless=runner.fit_probe(x[:,:10],y[:,:10],train,test,c)
    rows=runner.measures(memoryless,c)
    assert rows[0]['test_accuracy']==1 and all(r['test_accuracy']<.15 for r in rows[1:])
