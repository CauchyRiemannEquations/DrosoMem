import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from test_readout_dependency import fixture
from test_reward_direction import config
from local_reward import codes
from verify_local_reward import reference_train
from reward_trajectory import capture_train,local_direction,stage_directions,shared_radius,summarize
from verify_reward_trajectory import reference_direction
from reward_direction import perturb


def test_training_snapshots_and_frozen_proposals_preserve_original_rule():
    initial,roles,bank=fixture();symbols=np.random.default_rng(46).integers(0,4,43);c=config();code=codes(7)
    snapshots,h,logs=capture_train(initial,roles,bank,symbols,code,c,18,[0,1,2])
    for epoch in [1,2]:
        w,hh,ll=reference_train(initial,roles,bank,symbols,code,'contingent',dict(c,plastic_epochs=epoch),18)
        np.testing.assert_array_equal(w.data,snapshots[epoch].data);assert h[:epoch]==hh
        for k in ll:np.testing.assert_array_equal(logs[k][:epoch],ll[k])
    current=snapshots[2];mean,logs,h,model=local_direction(initial,current,roles,bank,symbols,code,c,33)
    other,ll,hh,mask,post=reference_direction(initial,current,roles,bank,symbols,code,c,33)
    np.testing.assert_array_equal(mean,other);assert h==hh
    for k in ll:np.testing.assert_array_equal(logs[k],ll[k])
    np.testing.assert_array_equal(model.original_magnitude,abs(initial.data[mask]));np.testing.assert_array_equal(model.weights.data,current.data)
    assert np.linalg.norm(current.data-initial.data)>0
    stages=[]
    for w in snapshots.values():
        unit,norms=stage_directions(mean,abs(w.data[mask]),post,len(roles),c,42);stages.append((abs(w.data[mask]),unit,norms))
    r=shared_radius(abs(initial.data[mask]),stages,c)
    assert r['radius']>0
    for w,(_,unit,_) in zip(snapshots.values(),stages):
        for u in unit:perturb(w,mask,post,u,r['radius'],1)


def test_one_sided_radius_handles_floor_and_locks_shared_temporal_budget():
    c=config();m=np.array([1e-9,1.]);u=np.array([[1.,-1.]])/np.sqrt(2);norm=np.array([1.])
    forward=shared_radius(np.ones(2),[(m,u,norm)],c);assert not forward['boundary_limited']
    reverse=shared_radius(np.ones(2),[(m,u,norm),(m,-u,norm)],c);assert reverse['resolution_limited'] and reverse['radius']<1e-8
    zero=shared_radius(np.ones(2),[(m,u,np.array([0.]))],c);assert zero['radius']==0 and zero['zero_direction']


def test_temporal_gates_exclude_smoke_and_require_identifiability_all_seeds():
    c=dict(config(),epochs=[0,1,5,10],bootstrap_seed=18,bootstrap_draws=1000);rows=[];radii=[]
    for cohort in ['smoke','discovery','confirmation']:
        for seed in [1,2,3]:
            for ci in [701,702]:
                radii.append(dict(cohort=cohort,seed=seed,circuit_seed=ci,resolution_limited=False))
                for epoch in c['epochs']:
                    for mode in ['noisy','clean']:
                        for variant in ['baseline','local']+[f'random{k}' for k in range(5)]:
                            rows.append(dict(cohort=cohort,seed=seed,circuit_seed=ci,epoch=epoch,mode=mode,variant=variant,accuracy=.2+(.005 if epoch==0 else .002) if variant=='local' else .2))
    _,_,_,s=summarize(rows,radii,c);assert all(s['confirmed'].values());assert s['discovery']['noisy']['epochs']['0']['statistics']['forward_gain']['n']==3
    radii[-1]['resolution_limited']=True
    _,_,_,s=summarize(rows,radii,c);assert not s['primary_identifiable'] and not any(s['confirmed'].values())
    radii[-1]['resolution_limited']=False
    for r in rows:
        if r['cohort']=='confirmation' and r['seed']==2 and r['epoch']==10 and r['variant']=='local':r['accuracy']=.21
    _,_,_,s=summarize(rows,radii,c);assert not s['confirmed']['attenuation'] and s['confirmed']['persistent_utility']


def test_checkpoint_floor_uses_original_weights_not_checkpoint_as_new_origin():
    original,roles,bank=fixture();current=original.copy();c=dict(config(),plastic_learning_rate=1e-20)
    row=np.repeat(np.arange(len(roles)),np.diff(original.indptr));indices=np.flatnonzero((row==16)&(roles[original.indices]=='KC'))
    assert len(indices)>1
    removed=current.data[indices[0]]*(1-1e-8);current.data[indices[0]]-=removed;current.data[indices[1]]+=removed
    symbols=np.random.default_rng(91).integers(0,4,43)
    mean,logs,h,model=local_direction(original,current,roles,bank,symbols,codes(7),c,23)
    other,_,hh,_,_=reference_direction(original,current,roles,bank,symbols,codes(7),c,23)
    np.testing.assert_array_equal(mean,other);assert h==hh and np.linalg.norm(mean)>0
    # With a negligible nonzero learning rate, displacement comes from restoring
    # the ORIGINAL floor, which a mistakenly reconstructed checkpoint would miss.
    np.testing.assert_array_equal(model.weights.data,current.data)
