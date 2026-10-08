"""Synthetic acyclicity, finite-history and exact count safeguards."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from flying.brain.timed_reservoir import TimedReservoir
from flying.training.whole_brain_memory import MappedEncoder,weight_hash
from temporal_mechanism import FactorReservoir
from cycle_structure import rank_dag,graph_structure
from temporal_cycles import finite_certificate,summarize
from cycle_support import config

def fixture(carry=False):
    roles=np.array(['KC','DAN','DAN','MBON']);ids=np.array([40,10,20,1])
    original=sparse.csr_matrix(np.array([[.1,0,0,.2],[.3,0,0,0],[0,.4,0,.1],[.2,0,.5,0]]))
    dag,rank=rank_dag(original,roles,ids)
    patterns=np.zeros((10,4));patterns[:,0]=np.linspace(.05,.5,10)
    base=TimedReservoir(dag,MappedEncoder(patterns),roles,.6,'mbon_after_kc')
    return original,dag,roles,ids,rank,FactorReservoir(base,carry,True)

class Budget:
    def check(self):pass

def test_mask_is_acyclic_preserves_current_path_and_original_coefficients():
    original,dag,roles,ids,rank,_=fixture();before=weight_hash(original)
    summary=graph_structure(original,dag,roles,ids,np.array([3]))
    assert summary['dag_cycle_edges']==0 and summary['global_dependency_depth']==3
    assert summary['protected_kc_to_mbon_all_kept_exactly'] is True
    assert dag[3,0]==original[3,0] and dag[1,0]==.3 and dag[2,1]==.4 and dag[3,2]==.5
    assert dag[0,0]==0 and dag[0,3]==0 and dag[2,3]==0
    assert weight_hash(original)==before
    post,pre=dag.nonzero();assert np.all(rank[pre]<rank[post])

def test_finite_certificate_flushes_arbitrary_initial_state():
    *_,model=fixture();c=config();block=c['blocks'][0]
    arrays,meta=finite_certificate(model,block,3,c,Budget())
    assert meta['distinct_prefix_states'] is True and meta['common_suffix_steps']==4
    np.testing.assert_array_equal(arrays['final_a_state'],arrays['final_b_state'])

def test_certificate_rejects_unremoved_direct_carry():
    *_,model=fixture(True)
    with pytest.raises(AssertionError):finite_certificate(model,config()['blocks'][0],3,config(),Budget())

def table(c):
    rows=[]
    for level,seeds in c['blocks_by_level'].items():
        for seed in seeds:
            for arm in c['arms']:
                for lag in c['lags']:
                    correct=2000 if lag==0 else 400 if arm=='dag_synaptic' and lag==2 else 200
                    accuracy=correct/2000
                    rows.append(dict(level=level,arm=arm,seed=seed,lag=lag,samples=2000,correct=correct,frequency_correct=200,current_correct=200,
                        accuracy=accuracy,baseline_excess=accuracy-.1,chance_adjusted=(accuracy-.1)/.9,frequency_accuracy=.1,current_accuracy=.1))
    return pd.DataFrame(rows)

def test_exact_equality_and_strongest_baseline_are_used():
    c=config();f=table(c);s=summarize(f,c,False)[0]
    assert s['primary_pass'] is True and s['levels']['legacy5']['primary']['minimum_excess_exact']=='1/10'
    selected=(f.level=='legacy5')&(f.arm=='dag_synaptic')&(f.seed==1300006)&(f.lag==2)
    f.loc[selected,'current_correct']=201
    assert summarize(f,c,False)[0]['primary_pass'] is False
    f=table(c);f.loc[selected,'frequency_correct']=201
    assert summarize(f,c,False)[0]['primary_pass'] is False

def test_every_block_is_required_despite_high_mean():
    c=config();f=table(c);f.loc[(f.arm=='dag_synaptic')&(f.lag==2),'correct']=1200
    f.loc[(f.level=='legacy5')&(f.arm=='dag_synaptic')&(f.seed==1300006)&(f.lag==2),'correct']=399
    s=summarize(f,c,False)[0]
    assert s['levels']['legacy5']['primary']['excess']['mean']>.1
    assert s['outcome']=='FAIL'

def test_reference_invalid_and_smoke_scientific_null():
    c=config();f=table(c)
    assert summarize(f,c,True)[0]['primary_pass'] is None
    f.loc[(f.level=='legacy5')&(f.arm=='instantaneous')&(f.seed==1300006)&(f.lag==0),'correct']=1979
    s=summarize(f,c,False)[0]
    assert s['outcome']=='assay-invalid' and s['primary_pass'] is None
