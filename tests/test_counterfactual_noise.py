"""Synthetic M7 safeguards; no source graph or prior replacement outcomes."""
import numpy as np
import pytest
from counterfactual_noise import draw_noise,panel,counts,summarize
from verify_counterfactual_noise import independent_noise,reconstruction

def fixture():
    ids=np.arange(48,dtype=np.int64)[::-1]+1000
    a=dict(root_ids=ids,observed_indices=np.arange(48),scale=np.ones(48),mean=np.zeros(48),coefficients_lag2=np.eye(48,10),intercept_lag2=np.zeros(10),
           original_symbols=np.zeros(64,dtype=np.int64),alternative_symbols=np.ones(64,dtype=np.int64),current_symbols=np.full(64,2),
           frequency_trial_predictions=np.full(64,3),current_trial_predictions=np.full(64,4),anchors=np.arange(64),evaluation_times=np.arange(64)+2)
    for world,label in [('original',0),('counterfactual',1),('instantaneous_original',2),('instantaneous_counterfactual',2)]:
        x=np.zeros((64,3,48));x[:,:,label]=1
        a['trial_'+world+'_features']=x;a['scores_'+world]=x[:,-1]@a['coefficients_lag2'];a['predictions_'+world]=a['scores_'+world].argmax(axis=1)
    return a

def cfg():
    return dict(noise_sigma=.1,score_atol=1e-10,score_rtol=1e-10,levels=['legacy5','brain5'],primary_level='legacy5',
                blocks_by_level={'legacy5':list(range(6)),'brain5':list(range(3))},bootstrap_seed=1730001,bootstrap_draws=10000)

def test_canonical_draws_independent_generation_and_graph_pairing():
    a=fixture();n=draw_noise(a['root_ids'],a['scale'],7);v=independent_noise(a['root_ids'],a['scale'],7)
    for k in n:np.testing.assert_array_equal(n[k],v[k])
    shuffled=np.roll(np.arange(48),5);q=draw_noise(a['root_ids'][shuffled],a['scale'][shuffled],7)
    np.testing.assert_array_equal(n['canonical_standard_normal'],q['canonical_standard_normal'])
    np.testing.assert_array_equal(n['standard_normal'][:,shuffled],q['standard_normal'])

def test_shared_noise_controls_no_mutation_and_affine_intervention_difference():
    a=fixture();before={k:v.copy() for k,v in a.items()};c=cfg();b={'seed':1,'noise_seed':9}
    n,cell,rows=panel(a,c,b,'legacy5');v,other=reconstruction(a,c,b,'legacy5',False)
    assert cell==other and cell['instantaneous_joint_correct']==0
    for k in a:np.testing.assert_array_equal(a[k],before[k])
    for k in n:np.testing.assert_allclose(n[k],v[k],atol=1e-12)
    np.testing.assert_allclose(n['scores_counterfactual']-n['scores_original'],a['scores_counterfactual']-a['scores_original'],atol=1e-12)
    assert len(rows)==64

def test_frozen_scale_floor_receives_noise_without_clipping():
    scale=np.full(48,1e-5);n=draw_noise(np.arange(48),scale,5)
    np.testing.assert_array_equal(n['amplitude'],.1*scale)
    assert np.any(n['raw_noise']!=0)
    a=fixture();a['trial_original_features'][:,-1,0]=100
    a['scores_original']=a['trial_original_features'][:,-1]@a['coefficients_lag2']
    a['predictions_original']=a['scores_original'].argmax(axis=1)
    n,_,_=panel(a,cfg(),{'seed':1,'noise_seed':1},'legacy5')
    assert n['observed_original'][:,0].min()>99
    from counterfactual_noise import score
    assert score(np.full((1,48),2.),np.zeros(48),np.ones(48),np.eye(48,10),np.zeros(10)).max()==2

def test_integer_gate_all_anchors_no_clean_success_filter_smoke_null():
    c=cfg();cells=[]
    for l,seeds in c['blocks_by_level'].items():
        for s in seeds:
            cells.append(dict(level=l,seed=s,samples=64,joint_correct=58,joint_accuracy=58/64,original_accuracy=58/64,replacement_accuracy=58/64,clean_joint_accuracy=1.,clean_minus_noisy_joint=6/64))
    assert summarize(cells,c,False)['primary_pass'] is True
    cells[0].update(joint_correct=57,joint_accuracy=57/64)
    assert summarize(cells,c,False)['primary_pass'] is False
    assert summarize(cells,c,True)['primary_pass'] is None
    assert counts(np.zeros(64,dtype=int),np.ones(64,dtype=int),np.zeros(64,dtype=int),np.ones(64,dtype=int))['joint_correct']==64
    bad=np.full(64,5);assert counts(bad,np.ones(64,dtype=int),np.zeros(64,dtype=int),np.ones(64,dtype=int))['joint_correct']==0

def test_missing_block_rejected_and_noise_seed_does_not_change_head():
    with pytest.raises(AssertionError):summarize([],cfg(),False)
    a=fixture();beta=a['coefficients_lag2'].copy()
    for seed in [1,2]:panel(a,cfg(),{'seed':1,'noise_seed':seed},'legacy5')
    np.testing.assert_array_equal(a['coefficients_lag2'],beta)
