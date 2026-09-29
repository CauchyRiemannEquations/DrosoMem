import sys
from pathlib import Path
import numpy as np
import pytest
from scipy import sparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from structural_controls import generate,audit
from verify_structural_controls import properties
from edge_panel import anatomy

@pytest.mark.parametrize('family',['random','degree','role','weight'])
def test_registered_graph_controls_and_exact_replay(family):
    raw,ids,roles=anatomy(701)
    a,log=generate(raw,roles,family,169002)
    b,log2=generate(raw,roles,family,169002)
    assert log==log2 and (a!=b).nnz==0
    assert audit(raw,a,roles,family)==properties(raw,a,roles,family)
    assert (a!=raw).nnz>0
    if family in ['degree','role']:assert log['accepted']==10*raw.nnz
    if family!='random':assert audit(raw,a,roles,family)['incoming_l1_changed_nodes']==0

def test_incomplete_swap_sampler_fails_instead_of_relaxing():
    raw=sparse.csr_matrix(np.ones((3,3))-np.eye(3))
    with pytest.raises(RuntimeError,match='Incomplete swaps'):generate(raw,np.array(['KC']*3),'role',123)

def test_property_audit_rejects_degree_matched_but_wrong_incoming_strength():
    raw,ids,roles=anatomy(701);wrong=raw.copy();wrong.data[[0,-1]]=wrong.data[[-1,0]]
    with pytest.raises(AssertionError):properties(raw,wrong,roles,'role')
