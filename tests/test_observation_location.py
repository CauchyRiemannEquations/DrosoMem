import sys
from pathlib import Path
import numpy as np
from scipy import sparse
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from observation_location import select_observations
from verify_observation_location import trajectory
from flying.training import whole_brain_memory as core
from alphabet_memory import SymbolEncoder


def test_alternative_excludes_any_direct_input_and_uses_48_unique_kcs():
    roles = np.array(['KC'] * 90 + ['MBON'] * 48)
    bank = np.zeros((4, len(roles)))
    for j in range(4):
        bank[j, j*8:(j+1)*8] = .5
    ids = list(map(str, range(5000, 5000 + len(roles))))
    sites = select_observations(ids, roles, bank, 19)
    assert len(sites['KC_unstimulated']) == len(set(sites['KC_unstimulated'])) == 48
    assert not bank[:, sites['KC_unstimulated']].any()
    assert (roles[sites['MBON']] == 'MBON').all()


def test_observation_does_not_change_mbon_after_kc_schedule():
    roles = np.array(['KC', 'MBON', 'DAN', 'KC'])
    w = sparse.csr_matrix(np.array([[0, .2, .1, 0], [.7, 0, .2, .1], [0, .2, 0, .1], [.1, 0, .3, 0]]))
    bank = np.array([[.5, 0, 0, 0], [0, 0, 0, .5]])
    symbols = np.array([0, 1, 0, 1, 1, 0]); obs = np.array([3, 0])
    model = core.TimedReservoir(w, SymbolEncoder(bank), roles, .6, 'mbon_after_kc')
    x, _ = core.collect(model, obs, symbols, 1e-8)
    y, _, _, _ = trajectory(w, bank, roles, symbols, obs, .6)
    np.testing.assert_array_equal(x, y)
    mbon, _ = core.collect(model, np.array([1]), symbols, 1e-8)
    ref, _, _, _ = trajectory(w, bank, roles, symbols, np.array([1]), .6)
    np.testing.assert_array_equal(mbon, ref)
