"""Small feed-forward readout: no recurrent state, position or digit lookup."""
import copy
import hashlib
import numpy as np
from scipy.special import softmax,logsumexp


class NonlinearReadout:
    def __init__(self,indices,hidden=8,seed=0):
        self.indices=np.asarray(indices,dtype=int);self.hidden=hidden;self.seed=seed
        if hidden<1 or not len(self.indices):raise ValueError('Invalid dimensions')

    def initialize(self,states):
        x=np.atleast_2d(states)[:,self.indices]
        self.mean=x.mean(axis=0);self.scale=np.maximum(x.std(axis=0),1e-5)
        rng=np.random.default_rng(self.seed)
        self.parameters={'w1':rng.normal(size=(len(self.indices),self.hidden))/np.sqrt(len(self.indices)),
                         'b1':np.zeros(self.hidden),'w2':rng.normal(size=(self.hidden,10))/np.sqrt(self.hidden),'b2':np.zeros(10)}

    @property
    def parameter_count(self):return sum(p.size for p in self.parameters.values())

    def features(self,states):return (np.atleast_2d(states)[:,self.indices]-self.mean)/self.scale

    def logits(self,states):
        p=self.parameters;h=np.tanh(self.features(states)@p['w1']+p['b1'])
        return h@p['w2']+p['b2']

    def predict(self,states):return self.logits(states).argmax(axis=1)

    def objective(self,states,labels,l2=1e-5):
        z=self.features(states);p=self.parameters;h=np.tanh(z@p['w1']+p['b1']);logits=h@p['w2']+p['b2']
        labels=np.asarray(labels)
        ce=float(np.mean(logsumexp(logits,axis=1)-logits[np.arange(len(labels)),labels]))
        loss=ce+.5*l2*(np.sum(p['w1']**2)+np.sum(p['w2']**2))
        error=softmax(logits,axis=1);error[np.arange(len(labels)),labels]-=1;error/=len(labels)
        hidden_error=(error@p['w2'].T)*(1-h**2)
        gradient={'w1':z.T@hidden_error+l2*p['w1'],'b1':hidden_error.sum(axis=0),
                  'w2':h.T@error+l2*p['w2'],'b2':error.sum(axis=0)}
        return float(loss),gradient,dict(cross_entropy=ce,accuracy=float(np.mean(logits.argmax(axis=1)==labels)))

    def fit(self,states,labels,epochs=2000,learning_rate=.03,l2=1e-5,checkpoints=(400,2000)):
        if len(states)!=len(labels) or not len(labels) or epochs<1 or learning_rate<=0 or l2<0:
            raise ValueError('Invalid training configuration')
        if any(not isinstance(e,int) or e<1 or e>epochs for e in checkpoints):raise ValueError('Invalid checkpoint')
        self.initialize(states);moments={k:np.zeros_like(p) for k,p in self.parameters.items()};variances=copy.deepcopy(moments)
        saved={};history=[]
        for epoch in range(1,epochs+1):
            _,grad,_=self.objective(states,labels,l2)
            for k,p in self.parameters.items():
                moments[k]=.9*moments[k]+.1*grad[k];variances[k]=.999*variances[k]+.001*grad[k]**2
                p-=learning_rate*(moments[k]/(1-.9**epoch))/(np.sqrt(variances[k]/(1-.999**epoch))+1e-8)
            if epoch==1 or epoch%50==0 or epoch in checkpoints:
                loss,_,metrics=self.objective(states,labels,l2);history.append(dict(epoch=epoch,objective=loss,**metrics))
            if epoch in checkpoints:saved[epoch]=copy.deepcopy(self)
        return saved,history

    def digest(self):
        arrays=[self.indices,self.mean,self.scale]+[self.parameters[k] for k in ['w1','b1','w2','b2']]
        return hashlib.sha256(b''.join(a.tobytes() for a in arrays)).hexdigest()
