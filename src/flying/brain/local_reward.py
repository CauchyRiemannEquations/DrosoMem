"""Reward-gated rate covariance, not STDP or a physiological dopamine model.

The learner receives input, local exploration, and a scalar reward only.
No label, codebook, classifier gradient or target activity enters this class.
"""
import numpy as np
from .plasticity import KCMBONPlasticity


class LocalRewardPlasticity(KCMBONPlasticity):
    def __init__(self, *args, trace_decay=.8, activity_rate=.05, baseline_rate=.05, **kwargs):
        if not 0 <= trace_decay < 1 or not 0 < activity_rate <= 1 or not 0 < baseline_rate <= 1:
            raise ValueError('Invalid trace settings')
        self.trace_decay=trace_decay; self.activity_rate=activity_rate; self.baseline_rate=baseline_rate
        super().__init__(*args, **kwargs)

    def reset(self):
        super().reset()
        self.post_mean=np.zeros(len(self.state)); self.eligibility=np.zeros(len(self.pre))
        self.energy=np.zeros(len(self.state)); self.baseline=.25; self.pending=False

    def step(self, *args, **kwargs):
        raise TypeError('Use forward(symbol, perturbation) then reinforce(scalar_reward)')

    def forward(self, symbol, perturbation):
        noise=np.asarray(perturbation)
        if noise.shape != (len(self.mbon),) or not np.isfinite(noise).all():
            raise ValueError('One finite perturbation per MBON required')
        previous=self.state.copy(); drive=self.weights@previous+self.encoder(int(symbol))
        drive[self.mbon]+=noise; activation=np.tanh(drive)
        self.state=(1-self.leak)*previous+self.leak*activation
        local=previous[self.pre]*(activation[self.post]-self.post_mean[self.post])
        self.eligibility=self.trace_decay*self.eligibility+(1-self.trace_decay)*local
        self.post_mean+=(self.activity_rate*(activation-self.post_mean))
        self.energy=np.bincount(self.post,weights=previous[self.pre]**2,minlength=len(self.state))
        self.pending=True
        return self.state.copy()

    def reinforce(self, reward):
        if not self.pending or not np.isscalar(reward) or not np.isfinite(reward) or not 0 <= reward <= 1:
            raise ValueError('Exactly one scalar reward in [0,1] after forward required')
        advantage=float(reward)-self.baseline
        if self.learning_rate:
            update=advantage*self.eligibility/(1+self.energy[self.post])
            magnitude=np.maximum(self.floor*self.original_magnitude,
                abs(self.weights.data[self.mask])+self.learning_rate*self.sign*update)
            total=np.bincount(self.post,weights=magnitude,minlength=len(self.state))
            magnitude*=self.budget[self.post]/total[self.post]
            self.weights.data[self.mask]=self.sign*magnitude
        self.baseline+=self.baseline_rate*advantage; self.pending=False
        return advantage
