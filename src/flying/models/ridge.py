"""Train-only standardization and affine ridge decoder for diagnostic tasks."""
import numpy as np

class RidgeDecoder:
    def __init__(self, alpha=1.):
        if alpha <= 0: raise ValueError('alpha must be positive')
        self.alpha=alpha
    def fit(self, x, targets):
        x=np.atleast_2d(x); targets=np.asarray(targets,dtype=float)
        if targets.ndim!=2 or len(x)!=len(targets) or not len(x):
            raise ValueError('Expected aligned, nonempty 2D arrays')
        self.mean=x.mean(axis=0); self.scale=np.maximum(x.std(axis=0),1e-5)
        z=(x-self.mean)/self.scale; self.target_mean=targets.mean(axis=0)
        self.weights=np.linalg.solve(z.T@z+self.alpha*np.eye(z.shape[1]),z.T@(targets-self.target_mean))
        return self
    def scores(self,x):
        return (np.atleast_2d(x)-self.mean)/self.scale@self.weights+self.target_mean

class RidgeReadout:
    def __init__(self, indices, alpha=.001):
        self.indices=np.asarray(indices,dtype=int); self.model=RidgeDecoder(alpha)
    def fit(self,states,labels):
        self.model.fit(np.atleast_2d(states)[:,self.indices],np.eye(10)[labels])
    def predict(self,states):
        return self.model.scores(np.atleast_2d(states)[:,self.indices]).argmax(axis=1)
