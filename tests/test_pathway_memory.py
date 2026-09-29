import sys,json
from pathlib import Path
import numpy as np
import pytest
from scipy import sparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from pathway_masks import target_mask,constraints,solve,audit_mask
from verify_pathway_memory import target_reference,matching_reference
from edge_panel import anatomy
from alphabet_memory import read


def test_pathway_orientation_is_pre_to_post():
    raw=sparse.csr_matrix(([5.,7.],([1,0],[0,1])),shape=(2,2));roles=np.array(['KC','MBON'])
    mask=target_mask(raw,roles,'KC_MBON')
    np.testing.assert_array_equal(mask,[False,True]);np.testing.assert_array_equal(mask,target_reference(raw,roles,'KC_MBON'))


def test_disjoint_dan_control_count_sign_and_both_masses():
    c=read(Path('configs/pathway_memory.json'));raw,ids,roles=anatomy(701)
    target=target_mask(raw,roles,'DAN_MBON');A,lo,hi=constraints(raw,target,c)
    mask,log=solve(A,lo,hi,target.astype(float),c,0)
    assert not (mask&target).any() and log['gap']==0
    a=audit_mask(raw,target,mask,c,0);b=matching_reference(raw,target,mask,c)
    for k in a:np.testing.assert_allclose(a[k],b[k],atol=1e-10)
    A=np.vstack([A,target]);lo=np.append(lo,0);hi=np.append(hi,0)
    cost=np.random.default_rng(189002).uniform(-1,1,raw.nnz)
    sample,log=solve(A,lo,hi,cost,c,c['draw_mip_gap'])
    matching_reference(raw,target,sample,c);assert not (sample&target).any()


def test_mass_audit_rejects_equal_count_wrong_weight():
    c=read(Path('configs/pathway_memory.json'))
    raw=sparse.csr_matrix(([5.,50.],([1,2],[0,0])),shape=(3,3));target=np.array([True,False]);wrong=~target
    with pytest.raises(AssertionError):matching_reference(raw,target,wrong,c)


def test_apl_specificity_is_graph_limited_before_outcomes():
    c=read(Path('configs/pathway_memory.json'));raw,ids,roles=anatomy(701)
    target=target_mask(raw,roles,'APL_KC');A,lo,hi=constraints(raw,target,c)
    mask,log=solve(A,lo,hi,target.astype(float),c,0)
    fraction=float((mask&target).sum()/target.sum())
    assert fraction>c['maximum_specificity_overlap'] and (mask&target).sum()==460
    json.dumps(dict(eligible=bool(fraction<=c['maximum_specificity_overlap']),solver=log))
