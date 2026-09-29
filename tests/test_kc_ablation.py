from pathlib import Path
import sys
import numpy as np
from scipy import sparse
sys.path.insert(0,str(Path('scripts').resolve()))
from kc_ablation import matched_masks, lesion, statistics


def test_matching_preserves_joint_strata_and_uses_deterministic_all_kc_pool():
    raw=sparse.csr_matrix(np.zeros((8,8)));bank=np.zeros((4,8));bank[0,:4]=.5
    gamma=np.array([1,1,0,0,1,0,0,0],bool);roles=['KC']*8
    masks,strata,audit=matched_masks(raw,roles,bank,gamma,[40,41,42])
    again,_,_=matched_masks(raw,roles,bank,gamma,[40,41,42])
    for name in ['matched0','matched1','matched2']:
        mask=masks[name];assert mask.sum()==3 and mask[:4].sum()==2
        np.testing.assert_array_equal(mask,again[name])
    assert all(row['pool_count']==4 for row in audit)


def test_lesion_keeps_surviving_weights_exact_without_renormalizing():
    raw=sparse.csr_matrix([[0,.2,.7],[.4,0,.1],[.3,.6,0]])
    bank=np.ones((4,3));dead=np.array([False,True,False])
    w,effective=lesion(raw,bank,dead)
    np.testing.assert_array_equal(w.toarray(),[[0,0,.7],[0,0,0],[.3,0,0]])
    assert effective[:,1].sum()==0 and np.all(bank==1)


def test_large_gamma_deficit_alone_is_not_population_specificity():
    c=dict(primary_lags=[1],material_effect=.05,bootstrap_seed=1,bootstrap_draws=100)
    rows=[dict(seed=s,circuit_seed=ci,arm=arm,lag=1,test_accuracy=a,frequency_excess=.4,null_excess=.4,r2_vs_frequency=.2)
          for s in [1,2,3] for ci in [701,702] for arm,a in [('intact',.9),('gamma',.5),('matched0',.49),('matched1',.5),('matched2',.51)]]
    _,blocks,d,s=statistics(rows,c)
    assert len(blocks)==9 and (d.intact_minus_gamma>.39).all()
    assert s['gates']['intact_past_access'] and not s['gates']['gamma_specific_impairment']
