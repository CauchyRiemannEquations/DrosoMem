import inspect,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import feedback_noise as study


class ToyModel:
    def reset(self):self.state=np.zeros(2);self.inputs=[];self.before=[]
    def step(self,symbol):
        self.before.append(self.state.copy());self.inputs.append(symbol)
        self.state=np.array([.5*symbol,.01*len(self.inputs)])
        return self.state.copy()


class ToyHead:
    def logits(self,x):
        x=np.atleast_2d(x);return np.column_stack([x[:,0],np.full(len(x),.25)])


def test_autonomous_signature_and_generated_feedback_only():
    assert list(inspect.signature(study.autonomous).parameters)==['model','observed','head','prompt','horizon','noise','amplitudes']
    model=ToyModel();a=study.autonomous(model,np.array([0,1]),ToyHead(),[0],4,np.zeros((4,2)),np.zeros(2))
    assert model.inputs==[0,*a['prediction'].tolist()]


def test_observation_corruption_never_writes_back():
    model=ToyModel();noise=np.array([[1.,0.],[0.,0.]])
    a=study.autonomous(model,np.array([0,1]),ToyHead(),[0],2,noise,np.array([.8,0.]))
    assert a['prediction'][0]==0
    np.testing.assert_array_equal(model.before[1],a['state_features'][0])
    assert a['features'][0,0]==.8 and a['state_features'][0,0]==0


def test_prefix_equality_does_not_imply_equal_post_error_outputs():
    from scipy.special import softmax
    target=np.array([1,1,0,0]);obs=np.array([0,1]);noise=np.zeros((4,2));head=ToyHead()
    model=ToyModel();auto=study.autonomous(model,obs,head,[0],4,noise,np.zeros(2))
    model.reset();model.step(0);x=[];p=[]
    for truth in target:
        x.append(model.state.copy());p.append(softmax(head.logits(model.state)[0]));model.step(int(truth))
    x=np.array(x);p=np.array(p);pred=p.argmax(1)
    assert study.verify_pre_error(auto,x,p,pred,target)==2
    assert auto['prediction'][2]!=pred[2]
    assert study.certificate(pred,target,4)['pi_memory_score']==1


def test_zero_and_clipping():
    model=ToyModel();a=study.autonomous(model,np.array([0,1]),ToyHead(),[0],2,np.ones((2,2))*5,np.zeros(2))
    np.testing.assert_array_equal(a['features'],a['state_features']);assert not a['clipped'].any()
    b=study.autonomous(model,np.array([0,1]),ToyHead(),[0],2,np.ones((2,2))*5,np.ones(2))
    assert np.all(b['features']==1) and np.all(b['clipped']==2)


def test_censoring_and_undefined_clean_retention():
    z=np.array([1,2,3]);row=study.certificate(z,z,0)
    assert row['censored'] and row['first_error_position'] is None and row['retention'] is None
    c=study.config();assert study.gate([3,3,3],[np.nan,.2,.2],'confirmation',c) is None


def test_joint_gate_cannot_be_replaced_by_teacher_accuracy():
    c=study.config()
    assert study.gate([3,3,3],[.2,.2,.2],'confirmation',c)
    assert not study.gate([1,1,1],[.2,.2,.2],'confirmation',c)
    assert not study.gate([3,3,3],[.09,.09,.09],'confirmation',c)
    assert not study.gate([10,10,0],[.3,.3,0],'confirmation',c)
