import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path('scripts').resolve()))
from context_memory import fit, predict, rollout, ambiguity, evaluate
from context_reference import infer, verify


def test_weighted_tie_break_and_backoff():
    s=np.array([0,1,0,2]); table=fit(s,4,2,np.array([1,1,1]))
    assert predict(table,[0],2)[0]==1  # equal counts for 1 and 2
    assert predict(table,[3,0],2)[2]==1
    assert predict(table,[3,3],2)[2]==0
    heavier=fit(s,4,2,np.array([1,1,2]))
    assert predict(heavier,[0],2)[0]==2


def test_start_has_no_padding_wrap_or_position_key():
    table=fit(np.array([0,1,2,3]),4,8,np.ones(3))
    assert (0,) in table and (0,1) in table and (0,1,2) in table
    assert (3,0) not in table and max(map(len,table))==3
    assert predict(table,[0,1,2],8)[0]==3


def test_autonomous_feedback_is_not_teacher_forcing():
    table=fit(np.array([0,1,0,2,0,2]),3,1,np.ones(5))
    generated,_,_=rollout(table,[0],5,1)
    assert generated==[2,0,2,0,2]
    assert generated!=[1,0,2,0,2]


def test_ambiguity_floor_known_collision():
    # Evaluation contexts: 0->1, 1->0, 0->2, 2->0, 0->1.
    a=ambiguity(np.array([0,1,0,1,0,2,0,1]),1)
    assert a['conflicting_contexts']==1
    assert a['conflicting_occurrence_fraction']==3/5
    assert a['ambiguity_error_floor']==1/5


@pytest.mark.parametrize('order',[1,2,3,4,5,8])
def test_reference_oracle_and_metrics_on_synthetic_sequences(order):
    s=np.random.default_rng(901).integers(0,4,128); s[:3]=[0,1,0]
    w=np.ones(127); w[2:34]=4
    r=evaluate(s,4,order,w); verify(r,4,order)
    assert len(r['generated'])==125
    assert np.allclose(np.sum(r['probabilities'],axis=1),1)
    assert 0<=r['metrics']['teacher_accuracy']<=1


def test_invalid_training_symbols_and_weights():
    with pytest.raises(ValueError):fit(np.array([0,4]),4,2,np.ones(1))
    with pytest.raises(ValueError):fit(np.array([0,1]),4,2,np.array([np.nan]))
    with pytest.raises(ValueError):fit(np.array([0.,1.]),4,2,np.ones(1))
