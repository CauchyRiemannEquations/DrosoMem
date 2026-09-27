import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

pytest.importorskip('brian2')
from flying.brain.dopamine import CausalDepression, DopamineLIF, replay_depression
from flying.brain.lif import LIFParameters, LIFReservoir
from flying.data.compartments import resolve_compartments
from flying.data.connectome import load_connectome


ROLES = np.array(['KC', 'KC', 'MBON', 'DAN', 'DAN', 'MBON'])
MAPPING = {'gamma1_pedc': {'MBON': [2], 'DAN': [3]}, 'alpha3': {'MBON': [5], 'DAN': [4]}}


def fixture():
    w = sparse.csr_matrix(([160., 20., 80., -3.], ([2, 2, 5, 0], [0, 1, 0, 5])), shape=(6, 6))
    encoder = SimpleNamespace(patterns=np.tile([1., 0., 0., 0., 0., 0.], (10, 1)))
    return w, encoder


def test_analytic_causal_decay_and_compartment_support():
    w, _ = fixture()
    rule = CausalDepression(w, ROLES, [2], [3], dt_ms=1, eligibility_ms=10, learning_rate=.2)
    rule.observe(0, [0]); rule.observe(10, [3])
    assert rule.weights[2, 0] == pytest.approx(160*np.exp(-.2*np.exp(-1)))
    assert rule.weights[2, 1] == 20  # inactive KC unchanged
    assert rule.weights[5, 0] == 80  # other compartment unchanged
    assert rule.weights[0, 5] == -3  # inhibitory support unchanged


@pytest.mark.parametrize('spikes', [[(0, [3]), (10, [0])], [(0, [0]), (10, [4])],
                                  [(0, [0]), (10, [2])], [(0, [3]), (10, [3])]])
def test_absent_causal_or_matching_dopamine_has_no_update(spikes):
    w, _ = fixture(); rule = CausalDepression(w, ROLES, [2], [3])
    for tick, cells in spikes:
        rule.observe(tick, cells)
    np.testing.assert_array_equal(rule.weights.data, w.data)


def test_floor_and_simultaneous_events():
    w, _ = fixture()
    rule = CausalDepression(w, ROLES, [2], [3], learning_rate=100, floor_fraction=.1)
    rule.observe(0, [0, 3])
    assert rule.weights[2, 0] == 16
    rule.observe(1, [0, 3])
    assert rule.weights[2, 0] == 16
    np.testing.assert_array_equal(rule.weights.indices, w.indices)
    np.testing.assert_array_equal(rule.weights.indptr, w.indptr)


def test_learning_uses_no_postsynaptic_spike_or_label():
    w, _ = fixture()
    a = CausalDepression(w, ROLES, [2], [3])
    z = CausalDepression(w, ROLES, [2], [3])
    a.observe(0, [0]); z.observe(0, [0, 2])
    a.observe(1, [3]); z.observe(1, [2, 3])
    np.testing.assert_array_equal(a.weights.data, z.weights.data)


def test_online_brian_updates_equal_independent_closed_form_and_reset():
    w, encoder = fixture()
    r = DopamineLIF(w, encoder, ROLES, MAPPING, eligibility_ms=10.)
    adapted, spikes, events = r.condition(3, 0., 10., [5.], duration_ms=30.)
    reference = replay_depression(w, ROLES, [2], [3], spikes,
        dt_ms=.1, eligibility_ms=10., learning_rate=.1, floor_fraction=.1)
    np.testing.assert_allclose(adapted.data, reference.data, atol=1e-12, rtol=0)
    assert events[0]['tick'] == 51 and events[0]['changed_edges'] == 1
    assert adapted[2, 0] < w[2, 0]
    repeated, repeated_spikes, _ = r.condition(3, 0., 10., [5.], duration_ms=30.)
    np.testing.assert_array_equal(adapted.data, repeated.data)
    for k in spikes:
        np.testing.assert_array_equal(spikes[k], repeated_spikes[k])


def test_wrong_compartment_and_zero_learning_online():
    w, encoder = fixture()
    for kwargs, target in [({}, 'alpha3'), ({'learning_rate': 0.}, 'gamma1_pedc')]:
        r = DopamineLIF(w, encoder, ROLES, MAPPING, **kwargs)
        adapted, _, _ = r.condition(3, 0., 10., [5.], dan_compartment=target, duration_ms=30.)
        np.testing.assert_array_equal(adapted.data, w.data)


def test_no_dopamine_matches_unmodified_lif_spikes():
    w, encoder = fixture()
    r = DopamineLIF(w, encoder, ROLES, MAPPING)
    adapted, spikes, _ = r.condition(3, 0., 50., [], duration_ms=50.)
    base = LIFReservoir(w, encoder, ROLES)
    base.states([3])
    np.testing.assert_array_equal(adapted.data, w.data)
    for k in spikes:
        np.testing.assert_array_equal(spikes[k], base.spike_arrays()[k])


def test_contact_to_si_validation_avoids_round_trip_error():
    w, encoder = fixture()
    w.data[:] = [29., 37., 113., 7.]
    r = DopamineLIF(w, encoder, ROLES, MAPPING)
    adapted, _, _ = r.condition(3, 0., 10., [], duration_ms=10.)
    np.testing.assert_array_equal(adapted.data, w.data)


@pytest.mark.parametrize('circuit', [701, 702])
def test_exact_verified_type_mapping(circuit):
    directory = f'data/flywire_783_mb_left_kc512_s{circuit}'
    _, ids, _ = load_connectome(directory)
    roles, mapping, audit = resolve_compartments(directory, ids)
    assert len(mapping['gamma1_pedc']['DAN']) == 1
    assert len(mapping['gamma1_pedc']['MBON']) == 1
    assert len(mapping['alpha3']['DAN']) == 1
    assert len(mapping['alpha3']['MBON']) == 2
    assert all(roles[i] == 'DAN' for i in mapping['gamma1_pedc']['DAN'])
    assert audit['unmapped_modulatory_or_output_cells'] == 168


def test_registry_rejects_conflicting_role(tmp_path):
    from pathlib import Path
    spec = json.loads(Path('configs/compartments.json').read_text())
    spec['compartments']['gamma1_pedc']['dan_types'] = ['MBON11']
    path = tmp_path/'bad.json'; path.write_text(json.dumps(spec))
    directory = 'data/flywire_783_mb_left_kc512_s701'
    _, ids, _ = load_connectome(directory)
    with pytest.raises(ValueError, match='role'):
        resolve_compartments(directory, ids, path)


def test_rule_rejects_negative_constants_and_duplicate_ticks():
    w, _ = fixture()
    with pytest.raises(ValueError):
        CausalDepression(w, ROLES, [2], [3], eligibility_ms=-1)
    rule = CausalDepression(w, ROLES, [2], [3]); rule.observe(0, [0])
    with pytest.raises(ValueError):
        rule.observe(0, [3])
