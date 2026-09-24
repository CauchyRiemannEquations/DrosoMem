"""Reward-modulated one-step policy semi-gradient with optional eligibility trace.

The classifier is supervised-warm-started and FROZEN during reward learning.
This is not an unbiased recurrent policy gradient or a biological dopamine model.
"""
from collections import deque
import hashlib
import numpy as np
from scipy.special import softmax
from flying.brain.plasticity import KCMBONPlasticity

def policy_hash(policy):
    return hashlib.sha256(b''.join(a.tobytes() for a in [policy.indices,policy.model.mean,policy.model.scale,policy.model.weights])).hexdigest()

class RewardPlasticity(KCMBONPlasticity):
    def __init__(self, weights, roles, encoder, policy, leak=.6, microsteps=1, learning_rate=.001,
                 floor=1e-4, trace_decay=.8, reward_delay=0, temperature=2., baseline_rate=.05):
        super().__init__(weights,roles,encoder,leak,microsteps,learning_rate,floor)
        if not 0<=trace_decay<1 or not isinstance(reward_delay,int) or reward_delay<0 or temperature<=0 or not 0<baseline_rate<=1:
            raise ValueError('Invalid reward settings')
        if not np.array_equal(policy.indices,self.mbon):raise ValueError('Policy must observe exactly MBON')
        self.policy=policy;self.trace_decay=trace_decay;self.reward_delay=reward_delay
        self.temperature=temperature;self.baseline_rate=baseline_rate
        self.local_post=np.searchsorted(self.mbon,self.post)
        self.eligibility=np.zeros(len(self.positions));self.baseline=.1
    def forward_policy(self,digit):
        for _ in range(self.microsteps):
            previous=self.state.copy();activation=np.tanh(self.weights@previous+self.encoder(int(digit)))
            self.state=(1-self.leak)*previous+self.leak*activation
        model=self.policy.model
        probabilities=softmax(model.features(self.state[self.mbon])@model.weights/self.temperature,axis=1)[0]
        return probabilities,previous,activation
    def score_gradient(self,previous,activation,probabilities,action):
        model=self.policy.model;matrix=model.weights[:-1]
        feedback=(matrix[:,action]-matrix@probabilities)/(model.scale*self.temperature)
        return self.leak*(1-activation[self.post]**2)*previous[self.pre]*feedback[self.local_post]
    def apply_reward(self,reward):
        advantage=float(reward)-self.baseline
        if self.learning_rate:
            # Per-postsynaptic norm cap; projection makes this a constrained heuristic.
            norm=np.sqrt(np.bincount(self.post,weights=self.eligibility**2,minlength=len(self.state)))
            direction=self.eligibility/np.maximum(1.,norm[self.post])
            magnitude=np.maximum(self.floor*self.original_magnitude,
                abs(self.weights.data[self.mask])+self.learning_rate*advantage*self.sign*direction)
            total=np.bincount(self.post,weights=magnitude,minlength=len(self.state))
            magnitude*=self.budget[self.post]/total[self.post]
            self.weights.data[self.mask]=self.sign*magnitude
        self.baseline+=self.baseline_rate*(float(reward)-self.baseline)
    def fit_reward(self,digits,labels,epochs=20,seed=0,yoked_rewards=None):
        if len(digits)!=len(labels) or not len(digits) or epochs<1:raise ValueError('Invalid training sequence')
        if yoked_rewards is not None:
            if np.shape(yoked_rewards)!=(epochs,len(digits)) or not np.isin(yoked_rewards,[0,1]).all():
                raise ValueError('Yoked rewards must be an aligned binary array')
        initial_policy=policy_hash(self.policy);rng=np.random.default_rng(seed)
        actions=[];true_rewards=[];applied_rewards=[];history=[]
        for epoch in range(epochs):
            self.reset();self.eligibility.fill(0);self.baseline=.1;pending=deque()
            chosen=[];actual=[];used=[];deliveries=0
            for t,(digit,label) in enumerate(zip(digits,labels)):
                prob,previous,activation=self.forward_policy(digit)
                action=min(9,int(np.searchsorted(np.cumsum(prob),rng.random())))
                reward=int(action==int(label))
                applied=reward if yoked_rewards is None else int(yoked_rewards[epoch,t])
                self.eligibility=self.trace_decay*self.eligibility+self.score_gradient(previous,activation,prob,action)
                pending.append(applied)
                if len(pending)>self.reward_delay:
                    self.apply_reward(pending.popleft());deliveries+=1
                chosen.append(action);actual.append(reward);used.append(applied)
            # Deliver the final delayed rewards without advancing the neural state or sampling actions.
            for _ in range(max(0,self.reward_delay-len(digits))):
                self.eligibility*=self.trace_decay
            while pending:
                self.eligibility*=self.trace_decay
                self.apply_reward(pending.popleft());deliveries+=1
            assert deliveries==len(digits)
            actions.append(chosen);true_rewards.append(actual);applied_rewards.append(used)
            history.append(dict(epoch=epoch+1,actual_reward_rate=float(np.mean(actual)),applied_reward_rate=float(np.mean(used)),
                delivered_rewards=deliveries,baseline_final=float(self.baseline)))
        assert initial_policy==policy_hash(self.policy)
        return history,dict(actions=np.asarray(actions,dtype=np.uint8),true_rewards=np.asarray(true_rewards,dtype=np.uint8),
                            applied_rewards=np.asarray(applied_rewards,dtype=np.uint8))
