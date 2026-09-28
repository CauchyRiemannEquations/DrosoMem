import json
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path('scripts').resolve()))
from run_prefix_mass import validate_pair
from flying.training.whole_brain_memory import sample_weights


def test_weight_mass_restores_pilot_and_only_expected_target_indices_change():
    baseline=sample_weights(511,3,32,4);restored=sample_weights(511,3,32,4*479/167)
    np.testing.assert_array_equal(np.flatnonzero(baseline!=restored),np.arange(2,34))
    assert restored[2:34].sum()/restored.sum()==pytest.approx(128/295,abs=1e-15)
    assert baseline[2:34].sum()/baseline.sum()==pytest.approx(128/607)
    np.testing.assert_array_equal(restored[:2],[1,1]);assert len(restored[34:])==477


def test_cache_refit_rejects_confounding_learning_rate_or_data_change():
    b=json.loads(Path('results/scaling_main/n512/config.json').read_text())
    c=json.loads(Path('configs/prefix_mass.json').read_text());validate_pair(b,c)
    with pytest.raises(AssertionError):validate_pair(b,dict(c,learning_rate=.02))
    with pytest.raises(AssertionError):validate_pair(b,dict(c,epochs=3000))
    changed=dict(c,dataset=dict(c['dataset'],prompt=[1,2,3]))
    with pytest.raises(AssertionError):validate_pair(b,changed)
