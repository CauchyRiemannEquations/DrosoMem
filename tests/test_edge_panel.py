from pathlib import Path
import sys,copy
import numpy as np
import pytest
from scipy import sparse
sys.path.insert(0,str(Path('scripts').resolve()))
from edge_masks import edge_betweenness,cut_edges,select_masks
from verify_edge_panel import independent_centrality
from edge_panel import validate,family_rows
from alphabet_memory import read


def test_directed_diamond_ties_and_isolate_centrality():
    a=sparse.csr_matrix((np.ones(4),([1,2,3,3],[0,0,1,2])),shape=(5,5))
    np.testing.assert_array_equal(edge_betweenness(a),[1.5,1.5,1.5,1.5])
    np.testing.assert_array_equal(edge_betweenness(a),independent_centrality(a))
    cycle=sparse.csr_matrix((np.ones(3),([1,2,0],[0,1,2])),shape=(4,4))
    np.testing.assert_array_equal(edge_betweenness(cycle),[3.,3.,3.])
    np.testing.assert_array_equal(edge_betweenness(cycle),independent_centrality(cycle))


def test_edge_cut_preserves_other_weights_and_all_nodes():
    a=sparse.csr_matrix([[0.,-.4,.2],[.1,0.,0],[0,.9,0]])
    w=cut_edges(a,np.array([True,False,False,True]))
    np.testing.assert_array_equal(w.toarray(),[[0,0,.2],[.1,0,0],[0,0,0]])
    np.testing.assert_array_equal(a.data,[-.4,.2,.1,.9]);assert w.shape==a.shape
    with pytest.raises(ValueError):cut_edges(a,np.array([True]))


def test_preregistration_rejects_changed_criteria_and_seed():
    c=read('configs/edge_panel.json');validate(c)
    for key,value in [('edge_fraction',.1),('material_effect',.01),('weight_boundaries',[20,50])]:
        changed=copy.deepcopy(c);changed[key]=value
        with pytest.raises(ValueError):validate(changed)
    changed=copy.deepcopy(c);changed['cohorts']['confirmation']['blocks'][0]['seed']=121142
    with pytest.raises(ValueError):validate(changed)


def test_between_family_never_includes_betweenness():
    import pandas as pd
    f=pd.DataFrame(dict(arm=['between0','between1','between2','betweenness','within0']))
    assert family_rows(f,'between').arm.tolist()==['between0','between1','between2']
