"""Pre-outcome synthetic M8 feasibility and independent-replay safeguards."""
import ast
from pathlib import Path
import numpy as np
import pytest
from scipy import sparse
from structural_specificity_design import rewire,audit_graph,delay_layout,spectra,summarize
from verify_structural_specificity_design import independent_swaps,invariants

def toy():
    roles=np.asarray(['KC','KC','DAN','DAN','MBON','MBON'])
    pre=[0,1,2,3,0,1,2,3];post=[4,5,0,1,2,3,4,5];weights=[7,9,-5,-5,6,6,8,11]
    return sparse.csr_matrix((weights,(post,pre)),shape=(6,6),dtype=float),roles

def test_independent_equal_weight_replay_preserves_both_multisets_and_current_factors():
    raw,roles=toy();saved=raw.copy()
    for seed in [1,2]:
        a,log=rewire(raw,roles,seed,5);b,other=independent_swaps(raw,roles,seed,5)
        assert log==other and log['accepted_swaps']==log['requested_swaps']
        for name in ['data','indices','indptr']:np.testing.assert_array_equal(getattr(a,name),getattr(b,name))
        assert audit_graph(raw,a,roles,log,{'gain':.9})==invariants(raw,b,roles,other,{'gain':.9})
    for name in ['data','indices','indptr']:np.testing.assert_array_equal(getattr(raw,name),getattr(saved,name))

def test_unequal_signed_weights_and_protected_edges_never_swap():
    raw,roles=toy();null,log=rewire(raw,roles,8,5)
    for target,source in [(4,0),(5,1),(4,2),(5,3)]:assert null[target,source]==raw[target,source]
    assert log['eligible_delayed_edges']==6 and log['proposal_pool_edges']==4
    assert not null.diagonal().any() and null.has_canonical_format

def test_empty_proposal_pool_is_bounded_not_ready_not_impossibility():
    raw=sparse.csr_matrix(([5.,5.],([1,2],[0,0])),shape=(3,3));roles=np.asarray(['APL','DAN','DAN'])
    a,log=rewire(raw,roles,1,5);b,other=independent_swaps(raw,roles,1,5)
    assert log==other and log['attempts']==log['accepted_swaps']==0 and not log['finite_budget_complete']
    assert (a!=raw).nnz==(b!=raw).nnz==0
    assert audit_graph(raw,a,roles,log,{'gain':.9})['construction_rule_pass'] is False

def test_exact_integer_overlap_boundary_at_ninety_percent():
    pre=np.repeat(np.arange(10),2);post=np.ravel(np.column_stack([(np.arange(10)+1)%10,(np.arange(10)+2)%10]))
    raw=sparse.csr_matrix((np.full(20,5.),(post,pre)),shape=(10,10))
    changed=post.copy();changed[0],changed[10]=changed[10],changed[0]
    null=sparse.csr_matrix((np.full(20,5.),(changed,pre)),shape=(10,10));roles=np.full(10,'OTHER')
    log=dict(eligible_delayed_edges=20,finite_budget_complete=True)
    report=audit_graph(raw,null,roles,log,{'gain':.9});assert report['common_delayed_edges']==18 and report['construction_rule_pass']
    assert not audit_graph(raw,raw.copy(),roles,log,{'gain':.9})['construction_rule_pass']

def test_delay_layout_preserves_current_and_exact_two_step_past_within_state_budget():
    roots=np.arange(48,dtype=np.int64)[::-1]+1000;d,order,observed=delay_layout(686,roots)
    assert d.nnz==48 and (d@d).nnz==24 and (d@d@d).nnz==0 and len(set(observed))==48
    codes=np.random.default_rng(11).normal(size=(10,48));stream=[1,2,3,4,5];state=np.zeros(686)
    for t,s in enumerate(stream):
        state=d@state;state[512:536]=codes[s,order[:24]];state[536:560]=codes[s,order[24:]]
        expected=np.zeros(48);expected[order[:24]]=codes[s,order[:24]]
        if t>=2:expected[order[24:]]=codes[stream[t-2],order[24:]]
        np.testing.assert_array_equal(state[observed],expected)
    with pytest.raises(AssertionError):delay_layout(607,roots)

def test_fixed_half_rank_failure_is_retained_without_feature_selection():
    p=np.zeros((10,48));p[:,24:34]=np.eye(10)
    singular,ranks=spectra(p,np.arange(48),1e-12)
    assert ranks=={'current':1,'past':10}
    np.testing.assert_array_equal(p[:,:24],0)

def test_missing_or_failed_construction_cannot_be_promoted_and_smoke_has_null_gate():
    c=dict(levels=['legacy5'],primary_level='legacy5',graph_seeds={'legacy5':[1,2]},input_seeds={'legacy5':[3]})
    rows=[dict(level='legacy5',seed=i,null_raw_sha256=str(i),construction_rule_pass=True,delayed_overlap=.8,finite_budget_complete=True) for i in [1,2]]
    maps={'legacy5':[dict(seed=3,numerical_affine_certificate_pass=True)]}
    assert summarize(rows,maps,c,False)['primary_ready'] is True
    rows[1]['construction_rule_pass']=False;assert summarize(rows,maps,c,False)['outcome']=='NOT_READY'
    assert summarize(rows,maps,c,True)['primary_ready'] is None
    with pytest.raises(AssertionError):summarize(rows[:1],maps,c,False)

def test_independent_verifier_has_no_main_algorithm_import():
    tree=ast.parse(Path('scripts/verify_structural_specificity_design.py').read_text())
    modules=[node.module for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)]
    assert 'structural_specificity_design' not in modules
