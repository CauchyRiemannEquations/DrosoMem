from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy import sparse
sys.path.insert(0, str(Path('scripts').resolve()))
from normalization_control import normalized, lipschitz_bound, decision, MonitoredReservoir, perturbation_probe
from alphabet_memory import SymbolEncoder


def test_only_diagonal_factors_change_on_fixed_raw_graph():
    a = sparse.csr_matrix([[0., 2., 1.], [3., 0., 4.], [5., 6., 0.]])
    b = sparse.csr_matrix([[0., 6., 4.], [5., 0., 1.], [3., 2., 0.]])
    wr, dr = normalized(a, b, 'renormalized'); wf, df = normalized(a, b, 'fixed_original')
    np.testing.assert_array_equal(dr, .9/np.array([10, 6, 5]))
    np.testing.assert_array_equal(df, .9/np.array([3, 7, 11]))
    np.testing.assert_allclose(wf.toarray(), (df/dr)[:, None]*wr.toarray())
    assert wr.nnz == wf.nnz == b.nnz
    assert np.max(np.asarray(abs(wf).sum(axis=1))) > 1


def test_scheduled_bound_accounts_for_new_kc_inputs():
    w = sparse.csr_matrix([[.2, .1], [.7, .2]])
    row, bounds = lipschitz_bound(w, ['KC', 'MBON'], .6)
    np.testing.assert_allclose(bounds, [.58, .4+.6*(.7*.58+.2)])
    encoder = SymbolEncoder(np.array([[.5, 0.], [0., 0.]]))
    model = MonitoredReservoir(w, encoder, ['KC', 'MBON'], .6, 'mbon_after_kc')
    result = perturbation_probe(model, [1], np.zeros(64, dtype=int), 889, 1e-6)
    distances = result['perturbation_full_inf']
    assert distances[-1] < distances[0]*1e-3
    assert np.all(distances[1:] <= bounds.max()*distances[:-1]+1e-15)


def test_gate_is_symmetric_and_confirmation_cannot_switch_direction():
    b = pd.DataFrame([dict(arm=arm, frequency_excess=.2, null_excess=.2, r2_vs_frequency=.1)
                      for arm in ['real', 'renormalized', 'fixed_original'] for _ in range(5)])
    for sign in [-1, 1]:
        g = decision(b, pd.Series([sign*.02]*5), 'main')
        assert g['material_normalization_effect'] and g['direction'] == sign
        assert not decision(b, pd.Series([-sign*.02]*3), 'confirmation', main_direction=sign)['material_normalization_effect']
    assert not decision(b, pd.Series([.009]*5), 'main')['material_normalization_effect']
