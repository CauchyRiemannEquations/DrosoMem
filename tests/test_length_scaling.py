import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pytest
from flying.data.sequences import SequenceDataset
from flying.training.whole_brain_memory import sample_weights

sys.path.insert(0,str(Path('scripts').resolve()))
from run_length_scaling import length_config, validate_spec


def test_random_prefixes_match_longest_stream_for_registered_seeds():
    spec=json.loads(Path('configs/length_scaling.json').read_text())
    validate_spec(spec)
    for block in spec['base']['blocks']:
        longest=SequenceDataset('random',block['dataset_seed'],length=512).symbols()
        for n in spec['lengths']:
            np.testing.assert_array_equal(SequenceDataset('random',block['dataset_seed'],length=n).symbols(),longest[:n])


def test_short_weight_window_covers_only_scored_targets():
    spec=json.loads(Path('configs/length_scaling.json').read_text())
    c=length_config(spec,32)
    weights=sample_weights(31,3,c['prefix_window'],4)
    assert c['eval_length']==29 and c['prefix_window']==29
    np.testing.assert_array_equal(weights[:2],[1,1])
    np.testing.assert_array_equal(weights[2:],np.full(29,4))
    for n in spec['lengths'][1:]:
        c=length_config(spec,n);weights=sample_weights(n-1,3,c['prefix_window'],4)
        assert c['prefix_window']==32 and weights.sum()==n-1+96
        np.testing.assert_array_equal(weights[2:34],np.full(32,4))
        np.testing.assert_array_equal(weights[34:],np.ones(n-35))


def test_invalid_or_duplicate_length_grid_rejected():
    spec=json.loads(Path('configs/length_scaling.json').read_text())
    for lengths in [[],[32,32],[64,32],[3]]:
        with pytest.raises(ValueError):validate_spec(dict(spec,lengths=lengths))
