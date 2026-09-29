from collections import Counter
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path('scripts').resolve()))
from kc_exclusive import exclusive_masks, statistics


def test_shared_subtraction_keeps_exact_joint_histograms_and_disjoint_membership():
    keys=[(1,2,3),(1,2,3),(1,2,3),(4,5,6),(4,5,6),(4,5,6)]
    g=np.array([1,1,0,1,0,0],bool);m=np.array([0,1,1,0,1,0],bool)
    a,b=exclusive_masks(g,m)
    assert not (a&b).any() and a.sum()==b.sum()==2
    assert (g[a]).all() and not g[b].any()
    assert Counter(keys[i] for i in np.flatnonzero(a))==Counter(keys[i] for i in np.flatnonzero(b))
    assert not a[g&m].any() and not b[g&m].any()
