from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path('scripts').resolve()))
from frozen_state_probe import splits,labels,fit_fold,measures


def test_alignment_and_purged_disjoint_support():
    c=dict(start=8,stop=127,folds=3,purge=9)
    s=np.arange(128); grid=np.arange(8,127); y=labels(s,grid,[0,1,8,-1])
    np.testing.assert_array_equal(y[0],[8,7,0,9]);np.testing.assert_array_equal(y[-1],[126,125,118,127])
    pairs=splits(c);np.testing.assert_array_equal(np.concatenate([test for _,test in pairs]),grid)
    assert [len(train) for train,_ in pairs]==[70,61,71]
    for train,test in pairs:
        assert set(t for i in train for t in range(i-8,i+2)).isdisjoint(t for i in test for t in range(i-8,i+2))


def test_known_delay_and_train_only_standardization():
    rng=np.random.default_rng(381);s=rng.integers(0,4,300)
    x=np.eye(4)[s[:-1]]; y=labels(s,np.arange(1,300),[1])
    a=fit_fold(x[:180],x[180:],y[:180],y[180:])
    assert measures([a],[1])[0]['test_accuracy']==1.
    # A single finite circular null need not score near chance; test the
    # intended operation and preserved marginals, not a probabilistic promise.
    np.testing.assert_array_equal(a['bias'],a['null_bias'])
    assert not np.array_equal(a['weights'],a['null_weights'])
    # Large held-out translation cannot change training statistics or coefficients.
    b=fit_fold(x[:180],x[180:]+7,y[:180],y[180:])
    for key in ['mean','scale','weights','bias']:np.testing.assert_array_equal(a[key],b[key])


def test_unequal_folds_are_concatenated_not_equally_weighted():
    rng=np.random.default_rng(83);x=rng.normal(size=(130,6));s=rng.integers(0,4,(130,2))
    a=fit_fold(x[:80],x[80:90],s[:80],s[80:90]);b=fit_fold(x[:80],x[90:],s[:80],s[90:])
    metrics=measures([a,b],[0,1])
    for j in [0,1]:
        expected=np.mean(np.r_[a['predictions'][:,j]==a['ytest'][:,j],b['predictions'][:,j]==b['ytest'][:,j]])
        assert metrics[j]['test_accuracy']==expected
