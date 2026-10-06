"""Prospective TDC safeguards: endpoint conjunction, leakage and tampering."""
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import temporal_memory_curve as experiment
import verify_temporal_memory_curve as verifier
from tdc_support import read, write, array_sha, attempt, seal
from verify_tdc_decision_counts import cell_excess
from fractions import Fraction


def test_primary_is_every_seed_every_lag_and_secondary_cannot_rescue():
    c = read(experiment.CONFIG)
    rows = [dict(level=level, seed=b['seed'], lag=lag, baseline_excess=.2)
            for level in c['levels'] for b in c['blocks'] for lag in c['lags']]
    f = pd.DataFrame(rows)
    assert experiment.summarize(f, c, False)['primary_pass']
    f.loc[(f.level == 'legacy5') & (f.seed == 910006) & (f.lag == 5), 'baseline_excess'] = .099
    s = experiment.summarize(f, c, False)
    assert s['primary_pass'] is False and len(s['failed_primary_cells']) == 1
    f.loc[f.level == 'brain5', 'baseline_excess'] = .9
    assert experiment.summarize(f, c, False)['primary_pass'] is False


def test_controls_are_fitted_without_test_labels():
    train = np.array([0, 1, 0, 1, 0, 1])
    test = np.array([1, 0, 1, 0])
    ytrain = np.column_stack([train[2:], train[1:-1]])
    ytest = np.zeros((2, 2), dtype=int)
    first = experiment.controls(train, test, ytrain, ytest, 2, 2)
    second = experiment.controls(train, test, ytrain, 1-ytest, 2, 2)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(first[1][:, 0], test[2:])
    np.testing.assert_array_equal(first[1][:, 1], 1-test[2:])


def test_exact_threshold_equality_passes_despite_float_subtraction():
    row = dict(samples=2000, correct=600, frequency_correct=400, current_correct=200)
    assert .3-.2 < .1  # why the extra integer-count validation is necessary
    assert cell_excess(row, 10) == Fraction(1, 10)
    row['correct'] = 599
    assert cell_excess(row, 10) < Fraction(1, 10)


def test_interrupted_attempt_is_preserved_and_cannot_be_overwritten(tmp_path):
    c = {'max_seconds': 30, 'max_rss_bytes': 2**32}
    out = tmp_path/'failed'
    with pytest.raises(ValueError, match='deliberate test failure'):
        with attempt(out, c, 'test'):
            write(out/'partial.json', {'case': 1})
            raise ValueError('deliberate test failure')
    assert (out/'partial.json').exists() and read(out/'failure.json')['complete'] is False
    assert read(out/'manifest.json')['complete'] is False
    with pytest.raises(FileExistsError):
        with attempt(out, c, 'test'):
            pass


def test_independent_verifier_rejects_rehashed_wrong_lag_labels(tmp_path):
    source = Path('results/tdc_v2/p1_smoke/legacy5_s910001')
    if not source.exists():
        pytest.skip('Requires archived smoke, available in a full checkout')
    target = tmp_path/'case'
    shutil.copytree(source, target)
    with np.load(target/'case.npz', allow_pickle=False) as a:
        arrays = {n: a[n].copy() for n in a.files}
    arrays['ytest'][:, 2] = (arrays['ytest'][:, 2]+1) % 10
    np.savez_compressed(target/'case.npz', **arrays)
    graph = read(target/'graph.json')
    graph['array_hashes']['ytest'] = array_sha(arrays['ytest'])
    write(target/'graph.json', graph)
    seal(target, {'complete': True})
    c = read(Path('results/tdc_v2/p1_smoke/config.json'))
    class NoResourceLimit:
        def check(self):
            pass
    with pytest.raises(AssertionError):
        verifier.verify_case(target, c, c['blocks'][0], 'legacy5', NoResourceLimit())
