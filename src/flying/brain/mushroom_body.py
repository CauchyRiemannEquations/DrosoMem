"""Role-restricted stimulation, observation and structural controls."""
import numpy as np
from scipy import sparse
from flying.encoding.digit_encoder import DigitEncoder
from flying.brain.reservoir import Reservoir
from flying.models.readout import Readout

class KCEncoder(DigitEncoder):
    def __init__(self, roles, seed=42, fraction=.1, amplitude=.5):
        indices = np.flatnonzero(np.asarray(roles) == 'KC')
        local = DigitEncoder(len(indices), seed, fraction, amplitude)
        self.patterns = np.zeros((10, len(roles)))
        self.patterns[:, indices] = local.patterns
        self.patterns.flags.writeable = False

class CircuitReservoir(Reservoir):
    def __init__(self, w, encoder, leak=.6, microsteps=4):
        super().__init__(w, encoder, leak)
        if not isinstance(microsteps, int) or microsteps < 1:
            raise ValueError('microsteps must be a positive integer')
        self.microsteps = microsteps
    def step(self, digit):
        for _ in range(self.microsteps):
            state = super().step(digit)
        return state

class SelectedReadout:
    def __init__(self, indices):
        self.indices = np.asarray(indices, dtype=int)
        self.model = Readout()
    def fit(self, states, labels, **kwargs):
        return self.model.fit(np.atleast_2d(states)[:, self.indices], labels, **kwargs)
    def predict(self, states):
        return self.model.predict(np.atleast_2d(states)[:, self.indices])

def ablate(w, roles, condition):
    """Operate AFTER normalization: retain surviving weights exactly."""
    c = w.tocoo(); roles = np.asarray(roles)
    if condition == 'no_dan':
        keep = (roles[c.col] != 'DAN') & (roles[c.row] != 'DAN')
    elif condition == 'no_feedback':
        keep = ~((roles[c.col] != 'KC') & (roles[c.row] == 'KC'))
    elif condition == 'leaky_only':
        keep = np.zeros(c.nnz, dtype=bool)
    else:
        raise ValueError(condition)
    return sparse.csr_matrix((c.data[keep], (c.row[keep], c.col[keep])), shape=w.shape)

def role_shuffled(a, roles, seed, swaps_per_edge=5):
    """Double swaps within pre/post role blocks; exact node in/out degrees."""
    c = a.tocoo(); pre=c.col.copy(); post=c.row.copy(); roles=np.asarray(roles)
    groups={}
    for i,(u,v) in enumerate(zip(pre,post)):
        groups.setdefault((roles[u],roles[v]),[]).append(i)
    groups={k:np.array(v) for k,v in groups.items()}
    rng=np.random.default_rng(seed); edges=set(zip(pre.tolist(),post.tolist())); original=edges.copy()
    requested=swaps_per_edge*len(pre); accepted=attempts=0
    while accepted < requested and attempts < requested*30:
        attempts+=1; i=int(rng.integers(len(pre)))
        candidates=groups[(roles[pre[i]],roles[post[i]])]
        j=int(rng.choice(candidates)); u,v,x,y=map(int,(pre[i],post[i],pre[j],post[j]))
        if u==x or v==y or u==y or x==v or (u,y) in edges or (x,v) in edges:
            continue
        edges.remove((u,v)); edges.remove((x,y)); edges.add((u,y)); edges.add((x,v))
        post[i],post[j]=y,v; accepted+=1
    out=sparse.csr_matrix((c.data.copy(),(post,pre)),shape=a.shape)
    return out,dict(requested_swaps=requested,accepted_swaps=accepted,attempts=attempts,
                    edge_overlap=len(edges & original)/len(edges))
