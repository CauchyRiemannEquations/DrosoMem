from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path('scripts').resolve()))
from normalization_transfer import predict_frozen, transfer, FROZEN, decision
from frozen_state_probe import fit_fold
import pandas as pd


def fixture():
    rng = np.random.default_rng(512)
    x = rng.normal(size=(150, 6)); y = rng.integers(0, 4, (150, 2))
    return fit_fold(x[:100], x[100:], y[:100], y[100:])


def test_identity_and_target_labels_cannot_change_prediction():
    source = fixture(); old = {key: source[key].copy() for key in FROZEN}
    identity = transfer(source, source)
    np.testing.assert_array_equal(identity['scores'], source['scores'])
    target = {key: value.copy() for key, value in source.items()}
    target['ytest'] = (target['ytest']+1) % 4
    target['weights'] += 900; target['bias'] += 400
    moved = transfer(source, target)
    np.testing.assert_array_equal(moved['scores'], identity['scores'])
    for key in FROZEN:
        np.testing.assert_array_equal(source[key], old[key])
        np.testing.assert_array_equal(moved[key], old[key])


def test_prediction_does_not_restandardize_shifted_target_features():
    source = fixture(); x = source['xtest']+2
    score, _ = predict_frozen(source, x)
    expected = (((x-source['mean'])/source['scale'])@source['weights']+source['bias']).reshape(score.shape)
    np.testing.assert_array_equal(score, expected)
    assert not np.array_equal(score, source['scores'])


def test_retention_requires_both_accuracy_tolerance_and_access():
    c = dict(access_margin=.05, retention_tolerance=.05)
    b = pd.DataFrame(dict(frequency_excess=[.2]*5, null_excess=[.2]*5,
        r2_vs_frequency=[.1]*5, transfer_minus_target_refit=[-.04]*5))
    assert decision(b, 'main', c)['portable_decoding']
    b['r2_vs_frequency'] = -.1
    assert not decision(b, 'main', c)['portable_decoding']
    b['r2_vs_frequency'] = .1; b['transfer_minus_target_refit'] = [-.06]*5
    assert not decision(b, 'main', c)['portable_decoding']
