import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path('scripts').resolve()))
from context_intervention import construct,verify_construction,error_count
from context_memory import ambiguity


def test_paired_generator_preserves_interface_and_strict_direction():
    r=construct(812,100)
    verify_construction(r)
    assert r==construct(812,100)
    assert np.array(r['proposals']).min()>=3 and np.array(r['proposals']).max()<128
    for arm,sign in [('low',-1),('high',1)]:
        a=r['arms'][arm]; s=np.array(a['symbols'])
        assert s[:3].tolist()==[0,1,0] and np.bincount(s).tolist()==[32]*4
        delta=np.diff(a['error_counts']); assert np.all(sign*delta>=0)
        assert np.array_equal(delta!=0,a['accepted'])
        assert abs(error_count(s)/125-ambiguity(s,3)['ambiguity_error_floor'])<1e-15


def test_conflict_counter_includes_boundary_and_detects_aliasing():
    periodic=np.tile(np.arange(4),32)
    assert error_count(periodic)==0
    rng=np.random.default_rng(900)
    for _ in range(10):
        s=rng.integers(0,4,128)
        assert abs(error_count(s)/125-ambiguity(s,3)['ambiguity_error_floor'])<1e-15
