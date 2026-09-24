"""Descriptive graph/state diagnostics; no claim of causal explanations."""
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from flying.brain.reservoir import normalize


def normalize_condition(a, method, gain=.9):
    if method == 'spectral':
        return normalize(a,gain)[0]
    if method == 'incoming_l1':
        # Bound each neuron's summed absolute recurrent input; rows are postsynaptic.
        if not 0 < gain < 1:
            raise ValueError('incoming_l1 requires gain in (0,1) for contraction')
        strength=np.asarray(abs(a).sum(axis=1)).ravel()
        factor=np.divide(gain,strength,out=np.zeros_like(strength,dtype=float),where=strength>0)
        return (sparse.diags(factor)@a).tocsr()
    raise ValueError('Unknown normalization')


def graph_diagnostics(a):
    a=sparse.csr_matrix(a)
    indeg=np.asarray((a!=0).sum(axis=1)).ravel()
    outdeg=np.asarray((a!=0).sum(axis=0)).ravel()
    _,weak=connected_components(a,directed=True,connection='weak')
    _,strong=connected_components(a,directed=True,connection='strong')
    return dict(neurons=a.shape[0],edges=a.nnz,isolates=int(((indeg+outdeg)==0).sum()),
                largest_weak=int(np.bincount(weak).max()),largest_strong=int(np.bincount(strong).max()),
                weak_components=int(weak.max()+1),strong_components=int(strong.max()+1),
                inhibitory_edge_fraction=float(np.mean(a.data<0)) if a.nnz else 0.)


def state_diagnostics(states):
    s=np.linalg.svd(states-states.mean(axis=0),compute_uv=False)
    if s.sum()==0:
        rank=0.
    else:
        p=s[s>0]/s.sum();rank=float(np.exp(-np.sum(p*np.log(p))))
    return dict(effective_rank=rank,mean_neuron_std=float(states.std(axis=0).mean()),
                saturation_fraction=float(np.mean(abs(states)>.95)))
