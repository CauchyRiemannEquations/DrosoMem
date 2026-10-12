"""Synthetic pre-outcome M9 dynamics/training/paired/validity safeguards."""
import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from scipy import sparse
import structural_tracking as main
from verify_structural_tracking import manual_step,fit_svd,summary as independent_summary
from temporal_mechanism import FactorReservoir
from flying.models.ridge import RidgeDecoder

def layout():
    rows=np.r_[np.arange(560,584),np.arange(584,608)];cols=np.r_[np.arange(536,560),np.arange(560,584)]
    d=sparse.csr_matrix((np.ones(48),(rows,cols)),shape=(686,686));order=np.arange(48)[::-1]
    observed=np.empty(48,dtype=np.int64);observed[order[:24]]=np.arange(512,536);observed[order[24:]]=np.arange(584,608)
    rng=np.random.default_rng(14);code=np.zeros((10,512))
    for s in range(10):code[s,rng.choice(512,51,False)]=.5
    p=rng.normal(size=(10,48));p[3,order[24]]=-0.
    model=main.DelayReference(d,code,p,order);data=dict(kind='delay',n=686,code=code,templates=p,order=order)
    return model,data,observed

def test_delay_manual_replay_current_past_and_signed_zero_are_exact():
    model,m,obs=layout();state=np.zeros(686);stream=[1,3,4,2,5]
    for t,s in enumerate(stream):
        state=manual_step(m,state,s);actual=model.step(s);assert actual.tobytes()==state.tobytes()
        np.testing.assert_array_equal(actual[obs][model.order[:24]],model.prototypes[s,model.order[:24]])
        if t>=2:np.testing.assert_array_equal(actual[obs][model.order[24:]],0.+model.prototypes[stream[t-2],model.order[24:]])

def test_actual_replacement_has_same_future_and_instantaneous_endpoint_pair():
    model,m,obs=layout();previous=np.random.default_rng(12).normal(size=686);original=np.array([2,3,4]);changed=np.array([3,3,4])
    a,fa,_=main.window(None,model,obs,previous,original,True);b,fb,_=main.window(None,model,obs,previous,changed,True)
    np.testing.assert_array_equal(a[-1,model.order[:24]],b[-1,model.order[:24]])
    assert not np.array_equal(a[-1,model.order[24:]],b[-1,model.order[24:]])
    _,fi,_=main.window(None,model,obs,previous,original,False);_,fj,_=main.window(None,model,obs,previous,changed,False);assert fi.tobytes()==fj.tobytes()

def test_source_zero_carry_manual_update_matches_existing_factor_model():
    rng=np.random.default_rng(16);w=sparse.csr_matrix(rng.normal(size=(8,8))*.05);patterns=np.zeros((10,8));patterns[:,:2]=rng.random((10,2))*.5
    encoder=SimpleNamespace(patterns=patterns);encoder_call=lambda s:patterns[s]
    class Encoder:
        def __call__(self,s):return encoder_call(s)
    base=SimpleNamespace(weights=w,encoder=Encoder(),kc=np.array([0,1]),mbon=np.array([6,7]),mbon_weights=w[[6,7]],leak=.6)
    model=FactorReservoir(base,False,True);model.state=rng.normal(size=8)
    m=dict(kind='real',n=8,weights=w,patterns=patterns,kc=base.kc,mb=base.mbon,mb_weights=base.mbon_weights)
    expected=manual_step(m,model.state,3);np.testing.assert_array_equal(model.step(3),expected)

def test_equal_budget_current_lag2_heads_use_train_only_scaler_and_svd_refit():
    rng=np.random.default_rng(17);stream=rng.integers(10,size=420);times=np.arange(20,420);y=np.column_stack([stream[times],stream[times-2]])
    x=np.zeros((400,48));x[:,:10]=np.eye(10)[y[:,0]];x[:,10:20]=np.eye(10)[y[:,1]]
    fitted=RidgeDecoder(1.).fit(x,np.eye(10)[y].reshape(400,20));mean,scale,beta,bias=fit_svd(x,y)
    np.testing.assert_array_equal(scale[20:],1e-5);np.testing.assert_allclose(beta,fitted.weights,atol=1e-9)
    np.testing.assert_array_equal(mean,fitted.mean);np.testing.assert_array_equal(bias,fitted.target_mean)
    assert beta.shape==(48,20) and beta.size+bias.size==980
    before=fitted.mean.copy();fitted.scores(np.full((3,48),100.));np.testing.assert_array_equal(fitted.mean,before)

def c_and_cells(blocks=2):
    c=dict(levels=['legacy5'],primary_level='legacy5',graph_ids={'legacy5':['real','g1','delay']},blocks_by_level={'legacy5':list(range(blocks))},bootstrap_seed=1930001)
    cells=[dict(level='legacy5',graph_id=g,seed=s,samples=64,joint_correct=64 if g!='g1' else 60,current_correct=90,test_samples=100) for g in c['graph_ids']['legacy5'] for s in range(blocks)]
    return c,cells

def test_exact_three_percentage_point_boundary_not_rounded_and_all_null_order():
    c,cells=c_and_cells(25)
    for x in cells:
        if x['graph_id']=='g1':x['joint_correct']=60 if x['seed']<12 else 64
    a,_,_=main.summarize(cells,c,False);b,_,_=independent_summary(cells,c,False)
    assert a==b and a['primary_pass'] is True and a['levels']['legacy5']['real_minus_null_mean_exact']=='3/100'
    next(x for x in cells if x['graph_id']=='g1' and x['seed']==0)['joint_correct']=61
    assert main.summarize(cells,c,False)[0]['primary_pass'] is False
    for x in cells:x['joint_correct']=64
    a=main.summarize(cells,c,False)[0];assert not a['primary_pass'] and a['levels']['legacy5']['empirical_percentile']==50 and a['levels']['legacy5']['standardized_effect'] is None

def test_current_access_failure_is_assay_invalid_no_graph_or_anchor_filter():
    c,cells=c_and_cells();cells[2]['current_correct']=89
    a=main.summarize(cells,c,False)[0];assert a['outcome']=='assay-invalid' and a['primary_pass'] is None and a['cases']==6
    assert main.summarize(cells,c,True)[0]['outcome']=='smoke'

def test_delay_benchmark_cannot_rescue_or_invalidate_primary():
    c,cells=c_and_cells()
    for x in cells:
        if x['graph_id']=='delay':x['current_correct']=0;x['joint_correct']=0
    a=main.summarize(cells,c,False)[0];assert a['primary_pass'] is True and a['levels']['legacy5']['delay_outcome']=='assay-invalid'

def test_missing_duplicate_or_unknown_case_is_rejected():
    c,cells=c_and_cells()
    for modified in [cells[:-1],cells+[cells[0]],cells+[dict(cells[0],graph_id='unknown')]]:
        with pytest.raises(AssertionError):main.summarize(modified,c,False)

def test_all_pairs_count_even_when_original_prediction_is_wrong():
    s=np.zeros(64,dtype=int);r=np.ones(64,dtype=int);p=s.copy();p[:8]=3
    assert main.count_pair(p,r,s,r)['joint_correct']==56
    assert main.count_pair(s,s,s,r)['joint_correct']==0

def test_synthetic_end_to_end_train_freeze_test_and_probe_replay(monkeypatch):
    model,m,obs=layout();patterns=np.zeros((10,686));patterns[:,:512]=model.code
    base=SimpleNamespace(encoder=SimpleNamespace(patterns=patterns))
    monkeypatch.setattr(main,'setup',lambda c,b,l,g:(base,model,obs,np.arange(48),dict(weight_sha256=main.core.weight_hash(model.weights))))
    c=dict(warmup=20,train_samples=400,test_samples=200);b=dict(seed=1,train_seed=23,test_seed=24)
    a,cell,rows,metrics,events,info=main.execute(c,b,'legacy5','delay',SimpleNamespace(check=lambda:None),True)
    assert cell['samples']==64 and len(rows)==len(events)==64 and cell['tracking90_pass'] is None
    for j,lag in enumerate([0,2]):np.testing.assert_array_equal(a['ytest'][:,j],a['test_symbols'][a['test_times']-lag])
    assert info['heads_frozen_before_test'] and all(not a[n].flags.writeable for n in ['mean','scale','coefficients','intercept'])
    mean,scale,beta,bias=fit_svd(a['train_features'][20:],a['ytrain']);np.testing.assert_allclose(a['coefficients'],beta,atol=1e-9)

def test_independent_verifier_imports_no_main_m9_algorithms():
    tree=ast.parse(Path('scripts/verify_structural_tracking.py').read_text());modules=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    assert 'structural_tracking' not in modules
