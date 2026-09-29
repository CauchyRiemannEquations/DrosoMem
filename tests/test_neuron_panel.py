from pathlib import Path
import sys,copy
import numpy as np
import pytest
from scipy import sparse
sys.path.insert(0,str(Path('scripts').resolve()))
from neuron_panel import intervene,MonitoredReservoir,material,validate,panel_masks
from alphabet_memory import SymbolEncoder,read


def test_input_silenced_neuron_can_still_receive_recurrent_activity():
    w=sparse.csr_matrix([[0,.5,0],[.5,0,0],[0,.5,0]])
    bank=np.array([[.5,.5,0.],[.5,.5,0.]])
    mask=np.array([True,False,False]);roles=np.array(['KC','KC','MBON'])
    states=[]
    for input_only in [True,False]:
        weights,effective,dead=intervene(w,bank,mask,input_only)
        model=MonitoredReservoir(weights,SymbolEncoder(effective),roles,.6,'mbon_after_kc')
        model.dead=dead;model.mask=mask;model.epsilon=1e-8;model.mask_active=[]
        model.step(0);states.append(model.step(0))
    assert states[0][0]>0 and states[1][0]==0
    np.testing.assert_array_equal(w.toarray(),[[0,.5,0],[.5,0,0],[0,.5,0]])


def test_confirmation_requires_every_block_and_both_impairments():
    assert material([.1,.1,.1],[.06,.06,.06],True)
    assert not material([.8,.8,-.001],[.6,.6,.6],True)
    assert not material([.2,.2,.2],[.01,.01,.01],True)
    assert not material([.1,.1,.1],[.1,.1,.1],False)


def test_registered_config_rejects_changed_seeds_or_metric():
    c=read('configs/neuron_panel.json');validate(c)
    for key,value in [('gain',.8),('material_effect',.01),('hub_fraction',.1)]:
        changed=copy.deepcopy(c);changed[key]=value
        with pytest.raises(ValueError):validate(changed)
    changed=copy.deepcopy(c);changed['cohorts']['confirmation']['blocks'][0]['seed']=81142
    with pytest.raises(ValueError):validate(changed)


def test_all_observed_slots_kept_after_output_lesion():
    w=sparse.csr_matrix([[0.,0],[.9,0]]);bank=np.array([[.5,0],[.5,0]])
    weights,effective,dead=intervene(w,bank,np.array([False,True]))
    model=MonitoredReservoir(weights,SymbolEncoder(effective),np.array(['KC','MBON']),.6,'mbon_after_kc')
    model.dead=dead;model.mask=dead;model.epsilon=1e-8;model.mask_active=[]
    state=model.step(0)
    assert len(state)==2 and state[0]>0 and state[1]==0
