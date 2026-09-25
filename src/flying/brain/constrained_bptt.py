"""Exact sparse BPTT positive control on existing KC→MBON edges only.

Computational supervised optimization, NOT local/dopamine learning. One update
per digit. The softmax policy and its feature statistics stay fixed throughout.
"""
import numpy as np
from scipy import sparse
from scipy.special import logsumexp, softmax
from flying.brain.plasticity import weight_hash
from flying.brain.reward_plasticity import policy_hash


class ConstrainedBPTT:
    def __init__(self, weights, roles, encoder, policy, leak=.6, floor=1e-4):
        if not 0 < leak <= 1 or not 0 <= floor < 1:
            raise ValueError('Invalid dynamics or floor')
        self.initial = sparse.csr_matrix(weights, dtype=float).copy()
        self.initial.sort_indices()
        self.roles = np.asarray(roles); self.encoder = encoder; self.policy = policy
        self.leak = leak; self.floor = floor; self.n = weights.shape[0]
        row = np.repeat(np.arange(self.n), np.diff(self.initial.indptr))
        self.mask = (self.roles[row] == 'MBON') & (self.roles[self.initial.indices] == 'KC')
        self.post = row[self.mask]; self.pre = self.initial.indices[self.mask]
        self.magnitude = abs(self.initial.data[self.mask]); self.sign = np.sign(self.initial.data[self.mask])
        if not len(self.magnitude) or np.any(self.magnitude <= 0):
            raise ValueError('Need nonzero plastic edges')
        self.budget = np.bincount(self.post, weights=self.magnitude, minlength=self.n)
        self.mbon = np.flatnonzero(self.roles == 'MBON')
        if not np.array_equal(policy.indices, self.mbon):
            raise ValueError('Readout must observe MBON only')
        self.log_magnitude = np.log(self.magnitude)
        self.theta = np.zeros(len(self.magnitude))

    def materialize(self, theta):
        theta = np.asarray(theta, dtype=float)
        if theta.shape != self.theta.shape or not np.isfinite(theta).all():
            raise ValueError('Invalid edge logits')
        logits = self.log_magnitude + theta
        maximum = np.full(self.n, -np.inf); np.maximum.at(maximum, self.post, logits)
        e = np.exp(logits - maximum[self.post])
        total = np.bincount(self.post, weights=e, minlength=self.n)
        probabilities = e / total[self.post]
        allocations = self.budget[self.post] * probabilities
        weights = self.initial.copy()
        # Preserve exact frozen numerical baseline at theta=0.
        if np.any(theta):
            weights.data[self.mask] = self.sign * (self.floor*self.magnitude + (1-self.floor)*allocations)
        return weights, probabilities, allocations

    def objective(self, digits, labels, theta=None, temporal=True, gradient=True):
        theta = self.theta if theta is None else theta
        digits=np.asarray(digits); labels=np.asarray(labels)
        if len(digits) != len(labels) or not len(digits):
            raise ValueError('Invalid sequence')
        weights, probabilities, allocations = self.materialize(theta)
        states = np.zeros((len(digits)+1, self.n)); activations = np.empty((len(digits), self.n))
        for t, digit in enumerate(digits):
            activations[t] = np.tanh(weights @ states[t] + self.encoder(int(digit)))
            states[t+1] = (1-self.leak)*states[t] + self.leak*activations[t]
        model=self.policy.model
        features=model.features(states[1:,self.mbon]); logits=features @ model.weights
        loss=float(np.mean(logsumexp(logits,axis=1)-logits[np.arange(len(labels)),labels]))
        accuracy=float(np.mean(logits.argmax(axis=1)==labels))
        if not gradient:
            return loss, accuracy
        error=softmax(logits,axis=1);error[np.arange(len(labels)),labels]-=1;error/=len(labels)
        direct=np.zeros_like(activations)
        direct[:,self.mbon]=(error @ model.weights[:-1].T)/model.scale
        delta=np.empty_like(activations); carried=np.zeros(self.n)
        transposed=weights.T.tocsr()
        for t in range(len(digits)-1,-1,-1):
            adjoint=direct[t] + carried if temporal else direct[t]
            delta[t]=self.leak*(1-activations[t]**2)*adjoint
            if temporal:
                carried=(1-self.leak)*adjoint + transposed @ delta[t]
        edge_gradient=np.einsum('ij,ij->j',delta[:,self.post],states[:-1,self.pre])
        signed=edge_gradient*self.sign
        row_mean=np.bincount(self.post,weights=probabilities*signed,minlength=self.n)
        theta_gradient=(1-self.floor)*allocations*(signed-row_mean[self.post])
        return loss, theta_gradient, accuracy

    def fit(self, digits, labels, epochs=100, learning_rate=.01, temporal=True, theta_cap=2., gradient_clip=5.):
        if epochs < 1 or learning_rate < 0 or theta_cap <= 0 or gradient_clip <= 0:
            raise ValueError('Invalid optimization settings')
        original_policy=policy_hash(self.policy)
        moment=np.zeros_like(self.theta); variance=np.zeros_like(self.theta)
        initial_loss, initial_accuracy=self.objective(digits,labels,gradient=False)
        best_loss=initial_loss; best_theta=self.theta.copy(); best_epoch=0
        history=[dict(epoch=0,loss=initial_loss,accuracy=initial_accuracy,best_loss=best_loss)]
        for epoch in range(1,epochs+1):
            _, grad, _=self.objective(digits,labels,temporal=temporal)
            grad*=min(1.,gradient_clip/max(float(np.linalg.norm(grad)),1e-300))
            moment=.9*moment+.1*grad;variance=.999*variance+.001*grad**2
            self.theta-=learning_rate*(moment/(1-.9**epoch))/(np.sqrt(variance/(1-.999**epoch))+1e-8)
            self.theta=np.clip(self.theta,-theta_cap,theta_cap)
            loss,accuracy=self.objective(digits,labels,gradient=False)
            if loss < best_loss:
                best_loss=loss;best_theta=self.theta.copy();best_epoch=epoch
            history.append(dict(epoch=epoch,loss=loss,accuracy=accuracy,best_loss=best_loss))
        self.theta=best_theta
        assert policy_hash(self.policy)==original_policy
        return history,dict(selected_epoch=best_epoch,initial_loss=initial_loss,selected_loss=best_loss,
                            selection='minimum training objective, including epoch zero; never free recall')

    def audit(self):
        weights,_,_=self.materialize(self.theta)
        assert np.array_equal(weights.data[~self.mask],self.initial.data[~self.mask])
        assert np.array_equal(np.sign(weights.data),np.sign(self.initial.data))
        assert np.array_equal(weights.indices,self.initial.indices) and np.array_equal(weights.indptr,self.initial.indptr)
        err=float(np.max(abs(np.asarray(abs(weights).sum(axis=1)-abs(self.initial).sum(axis=1)))))
        assert err < 1e-12
        return dict(max_budget_error=err,relative_plastic_change=float(np.linalg.norm(weights.data[self.mask]-self.initial.data[self.mask])/np.linalg.norm(self.magnitude)),
                    initial_weight_sha256=weight_hash(self.initial),final_weight_sha256=weight_hash(weights),plastic_edges=int(self.mask.sum()))
