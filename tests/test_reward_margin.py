import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from test_readout_dependency import fixture
from test_reward_direction import config
from local_reward import codes
from reward_trajectory import local_direction,stage_directions
from reward_direction import evaluate as primal
from reward_margin import evaluate,margin_scores,summarize
from verify_reward_margin import reference


def test_forward_tangents_equal_reverse_adjoint_and_central_differences():
 w,roles,bank=fixture();old=w.data.copy();c=config();symbols=np.random.default_rng(55).integers(0,4,43);code=codes(12);obs=np.flatnonzero(roles=='MBON')
 mean,_,_,model=local_direction(w,w,roles,bank,symbols,code,c,23);unit,_=stage_directions(mean,abs(w.data[model.mask]),model.post,len(roles),c,77);scale=.01*np.linalg.norm(abs(w.data[model.mask]))
 for noisy,reps in [(True,3),(False,1)]:
  a=evaluate(w,bank,obs,symbols,code,model.mask,unit,scale,c,41,reps,noisy);b=reference(w,bank,obs,symbols,code,model.mask,unit,scale,c,41,reps,noisy)
  np.testing.assert_array_equal(a['scores'],b['scores']);np.testing.assert_array_equal(a['features'],b['features']);np.testing.assert_array_equal(a['margin'],b['margin'])
  np.testing.assert_allclose(a['slopes'].mean(1),b['mean_slopes'],atol=1e-14,rtol=1e-10)
  for epsilon in [1e-3,3e-4,1e-4]:
   for j in range(len(unit)):
    plus=w.copy();minus=w.copy();delta=scale*np.sign(w.data[model.mask])*unit[j]*epsilon;plus.data[model.mask]+=delta;minus.data[model.mask]-=delta
    sp,_=primal(plus,bank,obs,symbols,code,c,41,reps,noisy);sm,_=primal(minus,bank,obs,symbols,code,c,41,reps,noisy)
    fd=(margin_scores(sp,a['truth'])-margin_scores(sm,a['truth']))/(2*epsilon)
    np.testing.assert_allclose(fd,a['slopes'][:,:,j],atol=1e-12,rtol=1e-4)
 np.testing.assert_array_equal(w.data,old)


def test_margin_derivative_has_no_spurious_common_squared_state_term():
 code=codes(31);x=np.random.default_rng(77).normal(size=(3,48));dx=np.random.default_rng(78).normal(size=x.shape);truth=np.array([0,1,3]);epsilon=1e-5
 def margins(v):
  scores=-np.mean((v[:,None,:]-code[None,:,:])**2,axis=2);return scores[np.arange(3),truth]-(scores.sum(1)-scores[np.arange(3),truth])/3
 coefficient=2*(code[truth]-(code.sum(0)-code[truth])/3)/48
 np.testing.assert_allclose((margins(x+epsilon*dx)-margins(x-epsilon*dx))/(2*epsilon),(coefficient*dx).sum(1),atol=1e-10,rtol=1e-8)


def test_margin_inference_requires_fresh_seed_cohort_and_all_seed_signs():
 c=dict(epochs=[0,1,5,10],random_directions=5,numerical_gate=1e-10,bootstrap_seed=92,bootstrap_draws=1000);rows=[]
 for cohort in ['smoke','discovery','confirmation','fresh_confirmation']:
  for seed in [1,2,3]:
   for ci in [701,702]:
    for epoch in c['epochs']:
     for mode in ['noisy','clean']:
      for variant in ['local']+[f'random{k}' for k in range(5)]:rows.append(dict(cohort=cohort,seed=seed,circuit_seed=ci,epoch=epoch,mode=mode,variant=variant,slope=(3e-6 if epoch==0 else 1e-6) if variant=='local' else 0.,margin=.001,accuracy=.3))
 _,_,_,s=summarize(rows,c);assert all(s['confirmed'].values());assert s['fresh_confirmation']['noisy']['epochs']['0']['statistics']['local']['n']==3
 for r in rows:
  if r['cohort']=='fresh_confirmation' and r['seed']==2 and r['epoch']==10 and r['variant']=='local':r['slope']=-1e-6
 _,_,_,s=summarize(rows,c);assert s['confirmed']['attenuation'] and not s['confirmed']['persistent_utility']
 for r in rows:
  if r['cohort']=='fresh_confirmation' and r['seed']==2 and r['epoch']==0 and r['variant']=='local':r['slope']=0.
 _,_,_,s=summarize(rows,c);assert not any(s['confirmed'].values())
