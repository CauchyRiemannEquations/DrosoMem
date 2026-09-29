from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path('scripts').resolve()))
from kc_frozen import decisions,evaluate
from frozen_state_probe import fit_fold


def test_refit_gain_is_not_access_or_portability_and_requires_all_blocks():
    c=dict(access_margin=.05,material_effect=.05,retention_tolerance=.05)
    b=pd.DataFrame(dict(frequency_excess=[.2]*3,null_excess=[.2]*3,r2_vs_frequency=[-.1]*3,
        refit_frequency_excess=[.5]*3,refit_null_excess=[.5]*3,refit_r2_vs_frequency=[.4]*3,refit_minus_frozen=[.3]*3))
    q=decisions(b,c);assert q['refit_benefit'] and not q['frozen_past_access'] and not q['portable']
    b['refit_minus_frozen']=[.3,.3,0.];assert not decisions(b,c)['refit_benefit']
    b['refit_minus_frozen']=[.3]*3;b['refit_r2_vs_frequency']=[-.1]*3
    assert not decisions(b,c)['refit_benefit']


def test_frozen_inference_does_not_use_target_head_or_target_standardization():
    rng=np.random.default_rng(7);x=rng.normal(size=(100,6));y=rng.integers(0,4,size=(100,2))
    source=fit_fold(x[:70],x[70:],y[:70],y[70:]);target={k:v.copy() for k,v in source.items()}
    target['xtrain']+=1;target['xtest']+=1
    first,_=evaluate(source,target,dict(lags=[0,1]))
    target['mean']+=999;target['scale']*=999;target['weights']+=999;target['scores']+=999
    second,_=evaluate(source,target,dict(lags=[0,1]))
    np.testing.assert_array_equal(first['scores'],second['scores'])
    assert not np.array_equal(first['scores'],source['scores'])
    expected=(((target['xtest']-source['mean'])/source['scale'])@source['weights']+source['bias']).reshape(first['scores'].shape)
    np.testing.assert_array_equal(first['scores'],expected)
