"""Structural replay and scalar boundary tests using synthetic fixtures only."""
from fractions import Fraction
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from flying.brain.mushroom_body import role_shuffled
from temporal_null_ensemble import exact_score
from verify_temporal_null_ensemble import replay_swaps,independent_invariants


def fixture():
    n = 12
    pre = np.repeat(np.arange(n),2)
    post = np.array([(i+shift)%n for i in range(n) for shift in [1,3]])
    values = np.array([(5+shift)*(1 if i%2 else -1) for i in range(n) for shift in [1,3]],float)
    raw = sparse.csr_matrix((values,(post,pre)),shape=(n,n))
    roles = np.array(['KC']*6+['MBON']*6)
    return raw,roles


def test_independent_swap_order_and_structural_invariants():
    raw,roles = fixture()
    a,log = role_shuffled(raw,roles,4242,5)
    b,second = replay_swaps(raw,roles,4242,5)
    assert (a!=b).nnz == 0 and log == second
    assert log['accepted_swaps'] == 5*raw.nnz
    audit = independent_invariants(raw,b,roles)
    assert audit['exact_node_degrees'] and audit['exact_presynaptic_signed_multisets']


def test_global_degree_match_does_not_hide_changed_weights():
    raw,roles = fixture()
    altered = raw.copy()
    altered.data[0] += 1
    with pytest.raises(AssertionError):
        independent_invariants(raw,altered,roles)


def test_chance_adjustment_and_exact_material_effect_boundary():
    real = pd.DataFrame([dict(correct=600,samples=2000)]*20)
    null = pd.DataFrame([dict(correct=546,samples=2000)]*20)
    assert exact_score(real,10)-exact_score(null,10) == Fraction(3,100)
    assert exact_score(pd.DataFrame([dict(correct=200,samples=2000)]),10) == 0
    assert exact_score(pd.DataFrame([dict(correct=100,samples=2000)]),10) < 0
