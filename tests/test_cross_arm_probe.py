from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path('scripts').resolve()))
from frozen_state_probe import fit_fold
from cross_arm_probe import transfer,FROZEN


def test_target_labels_cannot_adapt_frozen_probe():
    rng=np.random.default_rng(22);x=rng.normal(size=(100,8));y=rng.integers(0,4,(100,2))
    source=fit_fold(x[:70],x[70:],y[:70],y[70:]);source['train_indices']=np.arange(70)
    target=rng.normal(size=(30,8));a=transfer(source,target,y[70:]);b=transfer(source,target,(y[70:]+1)%4)
    for key in FROZEN:
        np.testing.assert_array_equal(a[key],source[key]);np.testing.assert_array_equal(b[key],source[key])
    np.testing.assert_array_equal(a['scores'],b['scores'])
    np.testing.assert_array_equal(a['predictions'],b['predictions'])
    assert not np.array_equal(a['changed_targets'],b['changed_targets'])


def test_identical_transfer_has_zero_score_change():
    rng=np.random.default_rng(31);x=rng.normal(size=(100,5));y=rng.integers(0,4,(100,1))
    source=fit_fold(x[:70],x[70:],y[:70],y[70:]);a=transfer(source,x[70:],y[70:])
    np.testing.assert_array_equal(a['scores'],source['scores'])
    assert not a['changed_targets'].any()
