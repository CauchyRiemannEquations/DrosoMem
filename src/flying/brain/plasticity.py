"""Supervised KC→MBON delta updates with a fixed anatomical mask.

This is a computational teaching rule, not a dopamine model. Incoming plastic
absolute strength is conserved by row competition after each local update.
"""
import hashlib
import numpy as np
from scipy import sparse

def weight_hash(w):
    w=w.tocsr()
    return hashlib.sha256(w.data.tobytes()+w.indices.tobytes()+w.indptr.tobytes()).hexdigest()

class KCMBONPlasticity:
    def __init__(self, weights, roles, encoder, leak=.6, microsteps=1, learning_rate=.05, floor=1e-4):
        if not 0<leak<=1 or not isinstance(microsteps,int) or microsteps<1 or learning_rate<0 or not 0<floor<1:
            raise ValueError('Invalid plasticity settings')
        self.initial=sparse.csr_matrix(weights,dtype=float).copy();self.initial.sort_indices()
        self.weights=self.initial.copy();self.roles=np.asarray(roles);self.encoder=encoder
        self.leak=leak;self.microsteps=microsteps;self.learning_rate=learning_rate;self.floor=floor
        row=np.repeat(np.arange(len(roles)),np.diff(self.weights.indptr));col=self.weights.indices
        self.mask=(self.roles[row]=='MBON')&(self.roles[col]=='KC')
        if not self.mask.any(): raise ValueError('No existing KC→MBON edges')
        self.positions=np.flatnonzero(self.mask);self.post=row[self.mask];self.pre=col[self.mask]
        self.sign=np.sign(self.initial.data[self.mask]);self.original_magnitude=abs(self.initial.data[self.mask])
        self.budget=np.bincount(self.post,weights=self.original_magnitude,minlength=len(roles))
        self.mbon=np.flatnonzero(self.roles=='MBON');self.reset()
    def reset(self):self.state=np.zeros(self.weights.shape[0])
    def step(self,digit,target=None):
        # The target is absent from forward dynamics; it is used only afterwards.
        for _ in range(self.microsteps):
            previous=self.state.copy();activation=np.tanh(self.weights@previous+self.encoder(int(digit)))
            self.state=(1-self.leak)*previous+self.leak*activation
        if target is not None:
            target=np.asarray(target)
            if target.shape!=(len(self.mbon),):raise ValueError('Target must address MBON only')
            error=np.zeros_like(self.state);error[self.mbon]=target-self.state[self.mbon]
            if self.learning_rate:
                # One-step semi-gradient: stop gradients through previous state and earlier microsteps.
                delta=error[self.post]*self.leak*(1-activation[self.post]**2)*previous[self.pre]
                energy=np.bincount(self.post,weights=previous[self.pre]**2,minlength=len(self.state))
                magnitude=np.maximum(self.floor*self.original_magnitude,
                    abs(self.weights.data[self.mask])+self.learning_rate*self.sign*delta/(1+energy[self.post]))
                total=np.bincount(self.post,weights=magnitude,minlength=len(self.state))
                magnitude*=self.budget[self.post]/total[self.post]
                self.weights.data[self.mask]=self.sign*magnitude
        return self.state.copy()
    def fit(self,digits,labels,teacher_codes,epochs=30):
        if len(digits)!=len(labels) or not len(digits) or epochs<1:raise ValueError('Invalid training sequence')
        history=[]
        for epoch in range(1,epochs+1):
            self.reset();loss=0.
            for digit,label in zip(digits,labels):
                target=teacher_codes[int(label)];state=self.step(digit,target)
                loss+=float(np.mean((target-state[self.mbon])**2))
            history.append(dict(epoch=epoch,teacher_mse=loss/len(digits)))
        return history
    def audit(self):
        before=self.initial.data;after=self.weights.data
        assert np.array_equal(before[~self.mask],after[~self.mask])
        assert np.array_equal(np.sign(before),np.sign(after))
        mass=np.bincount(self.post,weights=abs(after[self.mask]),minlength=len(self.state))
        assert np.allclose(mass,self.budget,rtol=1e-12,atol=1e-14)
        return dict(plastic_edges=int(self.mask.sum()),changed_edges=int(np.sum(before!=after)),
            relative_plastic_change=float(np.linalg.norm(after[self.mask]-before[self.mask])/np.linalg.norm(before[self.mask])),
            max_budget_error=float(np.max(abs(mass-self.budget))),nonplastic_edges_unchanged=True,
            initial_weight_sha256=weight_hash(self.initial),final_weight_sha256=weight_hash(self.weights))

def teacher_codes(mbon_count,seed,fraction=.25,amplitude=.25):
    """Fixed artificial positive sparse target patterns, independent of input code."""
    if mbon_count<10 or not 0<fraction<=1 or amplitude<=0:raise ValueError('Invalid teacher code')
    rng=np.random.default_rng(np.random.SeedSequence([seed,9401]));codes=np.zeros((10,mbon_count))
    for d in range(10):codes[d,rng.choice(mbon_count,max(1,int(mbon_count*fraction)),replace=False)]=amplitude
    return codes
