from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0, str(Path('scripts').resolve()))
from moment_alignment import fit_adapter, predict_aligned, transfer, FROZEN, decision
from frozen_state_probe import fit_fold


def fixture():
    rng = np.random.default_rng(913)
    x = rng.normal(size=(150, 6)); y = rng.integers(0, 4, (150, 2))
    return fit_fold(x[:100], x[100:], y[:100], y[100:])


def test_identity_and_frozen_parameters():
    source = fixture(); before = {key: source[key].copy() for key in FROZEN}
    a = transfer(source, source)
    np.testing.assert_array_equal(a['scores'], source['scores'])
    np.testing.assert_array_equal(a['null_scores'], source['null_scores'])
    for key in FROZEN:
        np.testing.assert_array_equal(source[key], before[key])
        np.testing.assert_array_equal(a[key], before[key])


def test_positive_affine_distortion_recovery_on_heldout_rows():
    source = fixture()
    factor = np.array([.2, .5, 1., 2., 5., 10.])
    offset = np.array([2., -5., 3., 1., -9., 6.])
    adapter = fit_adapter(source['xtrain']*factor+offset)
    scores, null = predict_aligned(source, adapter, source['xtest']*factor+offset)
    np.testing.assert_allclose(scores, source['scores'], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(null, source['null_scores'], atol=1e-12, rtol=1e-12)


def test_target_labels_heads_and_test_distribution_do_not_fit_adapter():
    source = fixture(); target = {k: v.copy() for k, v in source.items()}
    a = transfer(source, target)
    target['ytrain'] = (target['ytrain']+2) % 4
    target['ytest'] = (target['ytest']+1) % 4
    target['weights'] += 99; target['bias'] -= 400
    target['mean'] += 5; target['scale'] *= 10
    b = transfer(source, target)
    for key in ['target_mean', 'target_scale', 'scores', 'null_scores']:
        np.testing.assert_array_equal(a[key], b[key])
    target['xtest'][1:] += 1e4
    c = transfer(source, target)
    for key in ['target_mean', 'target_scale']:
        np.testing.assert_array_equal(a[key], c[key])
    np.testing.assert_array_equal(a['scores'][0], c['scores'][0])
    perm = np.arange(len(source['xtest']))[::-1]
    reordered, _ = predict_aligned(source, a, source['xtest'][perm])
    np.testing.assert_allclose(reordered, a['scores'][perm], atol=1e-12, rtol=1e-12)


def test_constant_feature_floor_and_invalid_input():
    adapter = fit_adapter(np.full((10, 3), 4.))
    np.testing.assert_array_equal(adapter['target_scale'], np.full(3, 1e-5))
    np.testing.assert_array_equal(adapter['target_mean'], np.full(3, 4.))
    with pytest.raises(ValueError):
        fit_adapter(np.empty((0, 3)))
    with pytest.raises(ValueError):
        fit_adapter(np.array([[np.nan]]))


def test_improvement_is_distinct_from_access_and_retention():
    config = dict(access_margin=.05, retention_tolerance=.05, alignment_margin=.05)
    b = pd.DataFrame(dict(frequency_excess=[.2]*5, null_excess=[.2]*5,
        r2_vs_frequency=[.1]*5, transfer_minus_target_refit=[-.06]*5,
        aligned_minus_unaligned=[.2]*5))
    d = decision(b, 'main', config)
    assert d['alignment_improvement'] and d['past_access'] and not d['portable_decoding']
    b['transfer_minus_target_refit'] = -.04
    assert decision(b, 'main', config)['portable_decoding']
    b['r2_vs_frequency'] = -.1
    assert not decision(b, 'main', config)['past_access']
    b['aligned_minus_unaligned'] = [.1, .1, .1, -.01, -.01]
    assert not decision(b, 'main', config)['alignment_improvement']
