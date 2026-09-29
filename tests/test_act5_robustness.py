import sys
from pathlib import Path
import numpy as np
import pytest
from scipy import sparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import act5_robustness as study
from verify_act5_robustness import reference
from flying.brain.timed_reservoir import TimedReservoir
from flying.models.nonlinear_readout import NonlinearReadout
from flying.training import whole_brain_memory as core


def tiny():
    roles=np.array(['KC','MBON','APL']);weights=sparse.csr_matrix([[.1,.2,0],[.8,.1,-.2],[.2,0,.1]])
    patterns=np.zeros((10,3));patterns[:,0]=np.linspace(-.4,.4,10)
    model=TimedReservoir(weights,core.MappedEncoder(patterns),roles,.6,'mbon_after_kc')
    head=NonlinearReadout([0],2,17);head.initialize(np.array([[-.1],[.1]]))
    mapping=dict(roles=roles,node_ix=np.array([1,3,4]),universe_size=5)
    return model,np.array([1]),head,mapping


def test_zero_all_families_and_inputs_are_unchanged():
    model,obs,head,mp=tiny();draws=dict(edge=np.arange(model.weights.nnz)/model.weights.nnz,neuron=np.array([.1,.4,.8]),weight=np.ones(model.weights.nnz))
    wh,hh=core.weight_hash(model.weights),head.digest();expected=None
    for kind in ['clean',*study.FAMILIES]:
        w,dead,n=study.alter(model.weights,draws,kind,0)
        a=study.autonomous(model,obs,head,[3,1,4],8,mp,w,dead,kind,0,np.random.default_rng(12))
        if expected is None:expected=a
        for k in a:np.testing.assert_array_equal(a[k],expected[k])
        assert n==0 and not dead.any()
    assert core.weight_hash(model.weights)==wh and head.digest()==hh
    np.testing.assert_array_equal(model.state,np.zeros(3))


@pytest.mark.parametrize('kind',study.FAMILIES)
def test_scalar_reference_and_target_free_vs_teacher_modes(kind):
    model,obs,head,mp=tiny();draws=dict(edge=np.linspace(0,.9,model.weights.nnz),neuron=np.array([.9,.9,.01]),weight=np.linspace(-2,2,model.weights.nnz))
    strength=.1;w,dead,_=study.alter(model.weights,draws,kind,strength)
    symbols=np.array([3,1,4,1,5,9,2,6,5,3,5]);target= symbols[3:]
    for mode in ['autonomous','teacher']:
        f=study.autonomous if mode=='autonomous' else study.teacher_forced
        a=f(model,obs,head,symbols[:3] if mode=='autonomous' else symbols,8,mp,w,dead,kind,strength,np.random.default_rng(42))
        b=reference(model,obs,head,symbols[:3],8,mp,w,dead,kind,strength,np.random.default_rng(42),target if mode=='teacher' else None)
        for k in a:np.testing.assert_allclose(a[k],b[k],atol=1e-13,rtol=1e-13)
    import inspect
    assert 'target' not in inspect.signature(study.autonomous).parameters
    assert 'symbols' not in inspect.signature(study.autonomous).parameters


def test_dead_kc_cannot_leak_into_interim_mbon_update():
    model,obs,head,mp=tiny();dead=np.array([True,False,False])
    result=study.advance(model,9,dead)
    np.testing.assert_array_equal(result,np.zeros(3))
    model,obs,head,mp=tiny();assert model.step(9)[1]>0


def test_nested_dropout_sign_preservation_and_input_immutability():
    model,*_=tiny();w=model.weights;before=core.weight_hash(w)
    draws=dict(edge=np.linspace(0,.9,w.nnz),neuron=np.array([.05,.3,.7]),weight=np.linspace(-1,1,w.nnz))
    low,dl,nl=study.alter(w,draws,'edge_dropout',.2);hi,dh,nh=study.alter(w,draws,'edge_dropout',.6)
    assert nh>nl and np.all((hi.toarray()!=0)<=(low.toarray()!=0))
    modified,_,_=study.alter(w,draws,'weight_noise',.05)
    np.testing.assert_array_equal(np.sign(modified.data),np.sign(w.data))
    np.testing.assert_array_equal(modified.indices,w.indices)
    np.testing.assert_allclose(modified.data/w.data,np.exp(.05*draws['weight']-.05**2/2))
    _,a,_=study.alter(w,draws,'neuron_dropout',.1);_,b,_=study.alter(w,draws,'neuron_dropout',.5)
    assert np.all(a<=b) and core.weight_hash(w)==before


def test_canonical_noise_and_pulse_timing_clipping():
    model,obs,head,mp=tiny();model.weights=sparse.csr_matrix((3,3));model.mbon_weights=model.weights[model.mbon];model.leak=1
    model.encoder=core.MappedEncoder(np.zeros((10,3)))
    a=study.autonomous(model,obs,head,[3,1,4],4,mp,model.weights,np.zeros(3,bool),'ongoing',10,np.random.default_rng(5))
    expected=10*np.random.default_rng(5).standard_normal((4,5))[:,mp['node_ix']]
    np.testing.assert_array_equal(a['features'][:,0],np.clip(expected,-1,1)[:,1])
    np.testing.assert_array_equal(a['clipped'],(np.abs(expected)>1).sum(1))
    pulse=study.autonomous(model,obs,head,[3,1,4],4,mp,model.weights,np.zeros(3,bool),'pulse',10,np.random.default_rng(5))
    assert pulse['features'][0,0]==a['features'][0,0]
    np.testing.assert_array_equal(pulse['features'][1:],0)


def rows(c):
    r=[]
    for co,seeds in c['cohorts'].items():
        for seed in seeds:
            for ci in c['circuit_seeds']:
                for g in c['conditions']:
                    common=dict(cohort=co,seed=seed,circuit_seed=ci,level=g,mode='autonomous',reused_clean=False,accuracy=.5)
                    r.append(dict(**common,kind='clean',strength=0,pi_memory_score=40))
                    for kind in study.FAMILIES:
                        for strength in c['strengths'][kind]:r.append(dict(**common,kind=kind,strength=strength,pi_memory_score=36 if g=='brain1' else 24))
    return r


def test_confirmation_required_and_undefined_baseline_inconclusive(tmp_path):
    c=study.config();a=rows(c);s=study.summarize(a,c,tmp_path)
    assert s['primary_confirmed'] and 'brain1/ongoing' in s['confirmed_absolute_robustness']
    for row in a:
        if row['cohort']=='confirmation' and row['level']=='brain1' and row['kind']=='ongoing':row['pi_memory_score']=0
    (tmp_path/'failed').mkdir();s=study.summarize(a,c,tmp_path/'failed');assert not s['primary_confirmed']
    for row in a:
        if row['cohort']=='discovery' and row['level']=='brain1' and row['kind']=='clean':row['pi_memory_score']=0
    (tmp_path/'undefined').mkdir();s=study.summarize(a,c,tmp_path/'undefined')
    assert s['comparisons']['discovery/brain1-legacy5/ongoing']['passed'] is None
    assert not s['primary_confirmed']


def test_no_vacuous_prefix32_robustness(tmp_path):
    c=study.config();a=rows(c)
    for row in a:row['pi_memory_score']=4
    s=study.summarize(a,c,tmp_path)
    assert not s['confirmed_absolute_robustness']


def test_fixed_scope():
    c=study.config();assert len(study.case_list(c))==48
    assert sum(len(x) for x in c['strengths'].values())==15
    assert set(c['cohorts']['discovery']).isdisjoint(c['cohorts']['confirmation'])
    assert core.sha256(Path(c['source'])/'manifest.json')==c['source_manifest_sha256']
