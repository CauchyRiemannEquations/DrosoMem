"""Controlled computational scheduling, not a physiological timing claim."""
import numpy as np
from flying.brain.mushroom_body import CircuitReservoir


class TimedReservoir(CircuitReservoir):
    def __init__(self, weights, encoder, roles, leak=.6, schedule='sync_one'):
        if schedule not in ('sync_one','mbon_after_kc','sync_two'):
            raise ValueError('Unknown schedule')
        super().__init__(weights,encoder,leak,1)
        self.schedule=schedule
        roles=np.asarray(roles)
        if roles.shape != (weights.shape[0],):raise ValueError('Role count mismatch')
        self.kc=np.flatnonzero(roles=='KC');self.mbon=np.flatnonzero(roles=='MBON')
        if not len(self.kc) or not len(self.mbon):raise ValueError('KC and MBON required')
        self.mbon_weights=self.weights[self.mbon].copy()
        self.mbon_weights.data.flags.writeable=False

    def step(self,digit):
        if self.schedule=='sync_one':return super().step(digit)
        if self.schedule=='sync_two':
            super().step(digit)
            return super().step(digit)
        # Every neuron integrates ONCE. Only MBON reads newly updated KC;
        # KC, DAN, APL and all other MBON inputs retain prior-state semantics.
        previous=self.state.copy();stimulation=self.encoder(int(digit))
        following=(1-self.leak)*previous+self.leak*np.tanh(self.weights@previous+stimulation)
        presynaptic=previous.copy();presynaptic[self.kc]=following[self.kc]
        following[self.mbon]=(1-self.leak)*previous[self.mbon]+self.leak*np.tanh(
            self.mbon_weights@presynaptic+stimulation[self.mbon])
        self.state=following
        return following.copy()
