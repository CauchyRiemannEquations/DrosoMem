from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
sys.path.insert(0, str(Path('scripts').resolve()))
from structural_k4 import audit_shuffle, gates
from frozen_state_probe import labels, fit_fold, measures


def test_known_lag_on_independent_streams():
    train = np.random.default_rng(192).integers(0, 4, 1100)
    test = np.random.default_rng(193).integers(0, 4, 600)
    tx, vx = np.arange(100, len(train)), np.arange(100, len(test))
    lags = [0, 1, 2, 8, 32]
    # An exact delay8 register must decode lag8, not current or wrong lag.
    a = fit_fold(np.eye(4)[train[tx-8]], np.eye(4)[test[vx-8]],
                 labels(train, tx, lags), labels(test, vx, lags))
    rows = measures([a], lags)
    assert rows[3]['test_accuracy'] == 1
    assert all(row['test_accuracy'] < .4 for i, row in enumerate(rows) if i != 3)


def test_audit_rejects_incomplete_swaps_and_wrong_weight_multisets():
    a = sparse.csr_matrix(([1., 2., 3., 4.], ([1, 2, 3, 0], [0, 1, 2, 3])), shape=(4, 4))
    with pytest.raises(AssertionError):
        audit_shuffle(a, a, ['KC']*4, dict(requested_swaps=20, accepted_swaps=19))
    b = a.copy(); b.data[0] *= 2
    with pytest.raises(AssertionError):
        audit_shuffle(a, b, ['KC']*4, dict(requested_swaps=20, accepted_swaps=20))


def test_positive_gates_require_access_and_seed_consistency():
    blocks = pd.DataFrame([dict(arm=arm, frequency_excess=.1, null_excess=.1,
                               r2_vs_frequency=.1) for arm in ['real', 'role_shuffled'] for _ in range(5)])
    assert gates(blocks, pd.Series([.04]*5), 'main')['real_wiring_advantage']
    assert not gates(blocks, pd.Series([.08, .08, .08, -.01, -.01]), 'main')['real_wiring_advantage']
    assert not gates(blocks, pd.Series([.029]*5), 'main')['real_wiring_advantage']
    assert not gates(blocks.iloc[:0], pd.Series([.1]*5), 'main')['real_wiring_advantage']
