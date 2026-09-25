import unittest
import numpy as np
from flying.models.nonlinear_readout import NonlinearReadout


class NonlinearReadoutTests(unittest.TestCase):
    def test_gradients_match_finite_difference_including_regularization(self):
        rng=np.random.default_rng(10);x=rng.normal(size=(12,7));y=np.arange(12)%10
        model=NonlinearReadout([1,3,5],hidden=3,seed=12);model.initialize(x)
        _,gradient,_=model.objective(x,y,l2=.02)
        for name,locations in {'w1':[(0,0),(2,1)],'b1':[(0,),(2,)],'w2':[(0,0),(2,9)],'b2':[(0,),(9,)]}.items():
            for ix in locations:
                original=model.parameters[name][ix];eps=1e-5
                model.parameters[name][ix]=original+eps;plus=model.objective(x,y,.02)[0]
                model.parameters[name][ix]=original-eps;minus=model.objective(x,y,.02)[0]
                model.parameters[name][ix]=original
                self.assertTrue(np.isclose(gradient[name][ix],(plus-minus)/(2*eps),rtol=1e-5,atol=1e-8))

    def test_parameter_budget_and_observation_mask(self):
        rng=np.random.default_rng(13);x=rng.normal(size=(20,60));m=NonlinearReadout(np.arange(48),seed=9);m.initialize(x)
        self.assertEqual(m.parameter_count,482)
        changed=x.copy();changed[:,48:]+=1e9
        self.assertTrue(np.array_equal(m.logits(x),m.logits(changed)))
        mean=m.mean.copy();m.predict(x*100);self.assertTrue(np.array_equal(mean,m.mean))

    def test_nonlinear_positive_control_xor(self):
        x=np.tile(np.array([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]]),(10,1));y=np.tile([0,1,1,0],10)
        m=NonlinearReadout([0,1],hidden=8,seed=17)
        saved,_=m.fit(x,y,epochs=300,checkpoints=(300,))
        self.assertEqual(float(np.mean(saved[300].predict(x)==y)),1.)

    def test_checkpoints_reproducible_and_predictions_stateless(self):
        rng=np.random.default_rng(5);x=rng.normal(size=(30,4));y=np.arange(30)%10
        a=NonlinearReadout([0,2,3],seed=19);b=NonlinearReadout([0,2,3],seed=19)
        sa,ha=a.fit(x,y,epochs=30,checkpoints=(10,30));sb,hb=b.fit(x,y,epochs=30,checkpoints=(10,30))
        self.assertEqual(ha,hb)
        for epoch in [10,30]:self.assertEqual(sa[epoch].digest(),sb[epoch].digest())
        self.assertNotEqual(sa[10].digest(),sa[30].digest())
        first=a.predict(x[:3]);a.predict(x[10:20]);self.assertTrue(np.array_equal(first,a.predict(x[:3])))
