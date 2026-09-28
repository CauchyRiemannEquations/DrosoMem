"""Fixed-energy signed-permutation control; analytic orientation expectation."""
import argparse
from pathlib import Path
import numpy as np
from residual_modes import execute, analyze as decompose_archive, verify_arrays as verify_modes
from moment_alignment import measures as baseline_measures

METRICS=['actual_energy','reference_energy','actual_over_reference','log2_actual_over_reference',
         'draw_mean_energy','draw_min_energy','draw_max_energy','test_accuracy','r2_vs_frequency']


def orientation_reference(q,h,split):
    shaped=h.reshape(len(h),-1,4);reference=np.zeros(shaped.shape[1])
    for start,end in [(0,split),(split,len(h))]:
        state=float(np.mean(np.sum(q[:,start:end]**2,axis=1)))
        reference+=state*np.sum(shaped[start:end]**2,axis=(0,2))/(end-start)
    return reference


def controls(q,h,split,seeds):
    permutations=[];signs=[];energies=[]
    for seed in seeds:
        rng=np.random.default_rng(seed)
        perm=np.concatenate([rng.permutation(split),split+rng.permutation(q.shape[1]-split)])
        sign=rng.choice([-1.,1.],size=q.shape[1]);transformed=q[:,perm]*sign
        for lo,hi in [(0,split),(split,q.shape[1])]:
            np.testing.assert_allclose(np.sum(transformed[:,lo:hi]**2,axis=1),np.sum(q[:,lo:hi]**2,axis=1),atol=1e-12,rtol=1e-12)
        score=(transformed@h).reshape(len(q),-1,4)
        permutations.append(perm);signs.append(sign);energies.append(np.mean(np.sum(score**2,axis=2),axis=0))
    return dict(permutations=np.array(permutations),signs=np.array(signs),draw_energy=np.array(energies),
        reference_energy=orientation_reference(q,h,split))


def analyze(source,target,moment,parent,c):
    replay=decompose_archive(source,target,moment,parent,c)
    for k in replay:np.testing.assert_array_equal(replay[k],parent[k],err_msg=k)
    a={k:np.array(v,copy=True) for k,v in parent.items()}
    a.update(controls(a['projected_residual'],a['projected_head'],c['split'],c['control_seeds']))
    return a


def verify_arrays(a,source,target,moment,c):
    verify_modes(a,source,target,moment,c)
    q=a['projected_residual'];h=a['projected_head'];k=c['split']
    second=q.T@q/len(q);shaped=h.reshape(48,11,4)
    ref=np.zeros(11)
    for lo,hi in [(0,k),(k,48)]:
        ref+=np.trace(second[lo:hi,lo:hi])*np.einsum('klo,klo->l',shaped[lo:hi],shaped[lo:hi])/(hi-lo)
    np.testing.assert_allclose(a['reference_energy'],ref,atol=1e-10,rtol=1e-10)
    for i,(p,s) in enumerate(zip(a['permutations'],a['signs'])):
        assert sorted(p[:k])==list(range(k)) and sorted(p[k:])==list(range(k,48))
        matrix=np.eye(48)[:,p]*s
        np.testing.assert_array_equal(matrix.T@matrix,np.eye(48))
        transformed=q@matrix
        # Orthogonal coordinate change preserves covariance spectrum including mean subtraction.
        np.testing.assert_allclose(np.linalg.eigvalsh(np.cov(transformed,rowvar=False)),np.linalg.eigvalsh(np.cov(q,rowvar=False)),atol=1e-10,rtol=1e-8)
        scores=(transformed@h).reshape(len(q),11,4)
        np.testing.assert_allclose(a['draw_energy'][i],np.mean(np.sum(scores**2,axis=2),axis=0),atol=1e-10,rtol=1e-10)


def measures(a,moment,c):
    rows=[];energy=np.mean(np.sum(a['total']**2,axis=2),axis=0)
    for j,b in enumerate(baseline_measures(moment,c['lags'])):
        ref=float(a['reference_energy'][j]);actual=float(energy[j]);valid=ref>c['energy_floor'] and actual>c['energy_floor']
        rows.append(dict(lag=b['lag'],valid=valid,actual_energy=actual,reference_energy=ref,
            actual_over_reference=actual/ref if valid else None,log2_actual_over_reference=float(np.log2(actual/ref)) if valid else None,
            draw_mean_energy=float(a['draw_energy'][:,j].mean()),draw_min_energy=float(a['draw_energy'][:,j].min()),draw_max_energy=float(a['draw_energy'][:,j].max()),
            test_accuracy=b['test_accuracy'],r2_vs_frequency=b['r2_vs_frequency']))
    return rows


def decision(b,cohort,c):
    return bool(b.valid.all() and b.log2_actual_over_reference.mean()>=c['log2_ratio_min'] and
        (b.log2_actual_over_reference>0).sum()>=(4 if cohort=='main' else 3))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=Path('configs/residual_orientation.json'))
    p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    execute(args.config,args.out,analyze,verify_arrays,measures,METRICS,decision,
        extra_sources=['scripts/residual_orientation.py','scripts/verify_residual_orientation.py'])
