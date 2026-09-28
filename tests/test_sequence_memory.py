import json
from pathlib import Path
import numpy as np
import pytest
from flying.data.sequences import SequenceDataset
from flying.training.sequence_memory import prediction_controls, score_metrics, validate
from flying.training.whole_brain_memory import SequenceDataset as LegacyPi


def test_pi_matches_archived_dataset_and_shuffled_multiset():
    original=SequenceDataset('pi',9242).symbols()
    np.testing.assert_array_equal(original,LegacyPi().symbols())
    for seed in range(9242,9247):
        ds=SequenceDataset('shuffled_pi',seed); shuffled=ds.symbols()
        np.testing.assert_array_equal(shuffled[:3],original[:3])
        np.testing.assert_array_equal(np.bincount(shuffled,minlength=10),np.bincount(original,minlength=10))
        assert not np.array_equal(shuffled,original)
        assert ds.identity(shuffled)['sha256']==ds.identity(ds.symbols())['sha256']


@pytest.mark.parametrize('k',[2,4,10,16])
def test_integer_alphabet_random_reproducible_and_bounded(k):
    ds=SequenceDataset('random',9242,alphabet_size=k,prompt=(0,1,0))
    x=ds.symbols();np.testing.assert_array_equal(x,ds.symbols())
    assert x.dtype==np.int64 and len(x)==200 and x.min()>=0 and x.max()<k
    assert not np.array_equal(x,SequenceDataset('random',9243,alphabet_size=k,prompt=(0,1,0)).symbols())
    # Fixed-prompt conditioning is explicit; no change to independent suffix draws.
    rng=np.random.default_rng(np.random.SeedSequence([9242,2201]))
    raw=rng.integers(0,k,size=200,dtype=np.int64)
    np.testing.assert_array_equal(x[3:],raw[3:])


def test_periodic_has_first_order_ceiling():
    x=SequenceDataset('periodic',9242).symbols()
    controls=prediction_controls(x,10,3,np.ones(199))['predictions']
    assert controls['markov1']['exact_prefix_symbols']==197
    assert controls['markov1']['teacher_forced_accuracy']==1.
    assert controls['majority']['exact_prefix_symbols']==1
    np.testing.assert_array_equal(x[:10],[3,1,4,0,2,5,6,7,8,9])


def test_controls_ties_and_unseen_rows_are_explicit():
    x=np.array([0,1,2,1,2])
    controls=prediction_controls(x,4,3,np.ones(4))
    table=np.asarray(controls['transition_counts'])
    np.testing.assert_array_equal(table[3],controls['global_counts'])
    assert controls['predictions']['majority']['generated']==[1,1]
    assert controls['predictions']['markov1']['generated']==[1,2]


def test_scoring_separates_teacher_and_autonomous_and_bits():
    c=json.loads(Path('configs/sequence_memory.json').read_text())
    x=np.array([3,1,4,1,5,9]); pred=np.array([1,0,9]);teacher=x[1:].copy()
    got=score_metrics(x,pred,teacher,np.ones((3,10))/10,c)
    assert got['exact_prefix_symbols']==1 and got['first_error_position']==2
    assert got['teacher_forced_accuracy']==1 and got['autonomous_accuracy']==pytest.approx(2/3)
    assert got['autonomous_prefix_bits']==pytest.approx(np.log2(10))
    assert 'pi_memory_score' not in got


def test_invalid_dataset_and_unregistered_runner_alphabet():
    with pytest.raises(ValueError):SequenceDataset('pi',1,alphabet_size=4,prompt=(0,1,2)).symbols()
    with pytest.raises(ValueError):SequenceDataset('periodic',1,prompt=(9,9,9)).symbols()
    with pytest.raises(ValueError):SequenceDataset('unknown',1).symbols()
    c=json.loads(Path('configs/sequence_memory.json').read_text());validate(c)
    c['dataset']['alphabet_size']=16
    with pytest.raises(ValueError):validate(c)
