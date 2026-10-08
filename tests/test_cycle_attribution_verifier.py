"""Independent M4 certificate safeguards, with no source-data outcomes."""
from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import numpy as np
import pytest
from scipy import sparse

from verify_cycle_attribution_feasibility import (ConeReplay, degree_mass,
    path_delay, validate_witness, walk_support, analytic_current_response,
    independent_degrees, independent_level_summary, independent_summary)


def matched_fixture():
    roles = np.array(['KC', 'KC', 'KC', 'OTHER', 'OTHER', 'OTHER', 'MBON'])
    ids = np.arange(101, 108, dtype=np.int64)
    common = [(0, 6), (0, 3), (1, 3), (2, 3), (1, 4), (2, 4), (1, 5)]
    dag_edges = common + [(3, 4), (3, 5), (4, 5), (4, 6), (5, 6)]
    cycle_edges = common + [(3, 5), (3, 6), (4, 5), (4, 6), (5, 4)]
    def matrix(edges):
        return sparse.csr_matrix((np.full(len(edges), .3),
            ([post for pre, post in edges], [pre for pre, post in edges])), shape=(7, 7))
    return matrix(dag_edges), matrix(cycle_edges), roles, ids, np.array([6])


def witness_for(matrix, roles, ids, observed, selected, depth=4):
    paths = ConeReplay(matrix, roles, ids, observed).canonical_paths(selected)
    if paths is None:
        return dict(symbol=0, certified=False, input_path=[], cycle=[], observed_path=[],
                    anchor_root_id=None, scc_size=None,
                    A=None, B=None, P=None, H=depth, k=None, delay=None, node_count=len(ids),
                    k_same_nodes=None, universal_delay=None, reason='no relevant cycle')
    a, p, b = [path_delay(matrix, roles, path) for path in paths]
    k = max(1, (depth - a - b) // p + 1)
    k_same = max(1, (len(ids) - 1 - a - b) // p + 1)
    return dict(symbol=0, certified=True, input_path=ids[paths[0]].astype(str).tolist(),
                cycle=ids[paths[1]].astype(str).tolist(), observed_path=ids[paths[2]].astype(str).tolist(),
                anchor_root_id=str(ids[paths[1][0]]),
                scc_size=int(ConeReplay(matrix, roles, ids, observed).sizes[
                    ConeReplay(matrix, roles, ids, observed).labels[paths[1][0]]]),
                A=a, B=b, P=p, H=depth, k=k, delay=a + k*p + b,
                node_count=len(ids), k_same_nodes=k_same, universal_delay=a + k_same*p + b,
                reason=None)


def test_local_weight_degree_role_controls_do_not_match_delayed_paths():
    dag, cyclic, roles, ids, observed = matched_fixture()
    for key in degree_mass(dag):
        np.testing.assert_array_equal(degree_mass(dag)[key], degree_mass(cyclic)[key])
    incoming = degree_mass(dag)['incoming_abs_weight']
    np.testing.assert_allclose(incoming[incoming > 0], .9, atol=1e-15, rtol=0)
    for role in set(roles):
        for other in set(roles):
            left = dag[np.flatnonzero(roles == other)][:, np.flatnonzero(roles == role)]
            right = cyclic[np.flatnonzero(roles == other)][:, np.flatnonzero(roles == role)]
            assert left.nnz == right.nnz
    selected = np.array([[0]])
    assert np.any(walk_support(dag, roles, selected, observed) !=
                  walk_support(cyclic, roles, selected, observed))
    c = {'leak': .6, 'input_amplitude': .5}
    np.testing.assert_array_equal(analytic_current_response(dag, roles, selected, observed, c),
                                  analytic_current_response(cyclic, roles, selected, observed, c))


def test_certificate_exceeds_every_same_node_dag_bound():
    _, cyclic, roles, ids, observed = matched_fixture()
    selected = np.array([0])
    cell = witness_for(cyclic, roles, ids, observed, selected)
    checked = validate_witness(cell, 0, selected, observed, cyclic, roles, ids, 4)
    assert checked['certified'] and checked['P'] == 2
    assert checked['delay'] == 6 and checked['universal_delay'] == 8
    assert checked['universal_delay'] > len(ids) - 1


@pytest.mark.parametrize('change', ['edge', 'endpoint', 'cycle', 'universal_bound', 'smaller_repeat'])
def test_rejects_corrupted_saved_witness(change):
    _, cyclic, roles, ids, observed = matched_fixture()
    selected = np.array([0])
    cell = deepcopy(witness_for(cyclic, roles, ids, observed, selected))
    if change == 'edge':
        cell['input_path'] = [str(ids[0]), str(ids[4])]
    elif change == 'endpoint':
        cell['observed_path'][-1] = str(ids[5])
    elif change == 'cycle':
        cell['cycle'][-1] = str(ids[5])
    elif change == 'universal_bound':
        cell['universal_delay'] = len(ids) - 1
    else:
        cell['k_same_nodes'] -= 1
        cell['universal_delay'] -= cell['P']
    with pytest.raises(AssertionError):
        validate_witness(cell, 0, selected, observed, cyclic, roles, ids, 4)


def test_independent_absence_check_and_rejects_false_missing_claim():
    dag, cyclic, roles, ids, observed = matched_fixture()
    selected = np.array([0])
    missing = witness_for(dag, roles, ids, observed, selected)
    assert validate_witness(missing, 0, selected, observed, dag, roles, ids, 4)['certified'] is False
    with pytest.raises(AssertionError):
        validate_witness(missing, 0, selected, observed, cyclic, roles, ids, 4)


def test_cycle_unobservable_from_output_does_not_certify():
    _, cyclic, roles, ids, _ = matched_fixture()
    observed = np.array([2])
    assert ConeReplay(cyclic, roles, ids, observed).canonical_paths(np.array([0])) is None


def test_boolean_support_handles_many_inputs_without_integer_overflow():
    count = 300
    matrix = sparse.csr_matrix((np.ones(count),
                               (np.full(count, count), np.arange(count))), shape=(count + 1, count + 1))
    roles = np.array(['KC'] * count + ['MBON'])
    support = walk_support(matrix, roles, np.array([np.arange(count)]), np.array([count]), 2)
    assert support.dtype == bool and support[0, 0, 0]
    assert not support[0, 1:, 0].any()


def test_degree_obstruction_absence_does_not_prove_feasibility():
    dag, cyclic, roles, _, _ = matched_fixture()
    assert independent_degrees(dag, roles)['status'] == 'UNRESOLVED'
    assert independent_degrees(cyclic, roles)['status'] == 'UNRESOLVED'
    directed_cycle = sparse.csr_matrix((np.ones(3), ([1, 2, 0], [0, 1, 2])), shape=(3, 3))
    obstruction = independent_degrees(directed_cycle, np.array(['OTHER'] * 3))
    assert obstruction['status'] == 'INFEASIBLE'
    assert obstruction['obstructions'] == [dict(rule='missing_source_or_sink',
                                               source_missing=True, sink_missing=True)]


def test_unresolved_summary_and_smoke_null_are_preserved():
    dag, _, roles, ids, _ = matched_fixture()
    degrees = independent_degrees(dag, roles)
    cells = [dict(seed=1, symbol=0, certified=False)]
    record = independent_level_summary(degrees, cells, len(ids), 4)
    assert record['outcome'] == 'UNRESOLVED'
    assert independent_summary({'legacy5': record}, False, 'legacy5')['outcome'] == 'UNRESOLVED'
    assert independent_summary({'legacy5': record}, True, 'legacy5')['outcome'] is None
