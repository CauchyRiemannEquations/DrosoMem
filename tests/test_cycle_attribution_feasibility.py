"""Prospective M4 safeguards on artificial matrices only."""
import numpy as np
import pytest
from scipy import sparse

from cycle_attribution_feasibility import (degree_certificates, walk_support,
                                          cycle_witness, summarize, _zero_observed)


def graph(n, edges, weight=.3):
    pre, post = np.asarray(edges, dtype=np.int64).T
    return sparse.csr_matrix((np.full(len(pre), weight), (post, pre)), shape=(n, n))


def matched_pair():
    # k,p,q are KC; i,a,b OTHER; o MBON. Every active incoming sum is .9.
    roles = np.array(['KC', 'KC', 'KC', 'OTHER', 'OTHER', 'OTHER', 'MBON'])
    common = [(0, 6), (0, 3), (1, 3), (2, 3), (1, 4), (2, 4), (1, 5)]
    dag = graph(7, common + [(3, 4), (3, 5), (4, 5), (4, 6), (5, 6)])
    cyclic = graph(7, common + [(3, 5), (3, 6), (4, 5), (4, 6), (5, 4)])
    return dag, cyclic, roles, np.arange(101, 108, dtype=np.int64)


def test_normalized_same_degree_strength_roles_current_block_counterexample():
    dag, cyclic, roles, ids = matched_pair()
    np.testing.assert_array_equal(np.diff(dag.indptr), np.diff(cyclic.indptr))
    np.testing.assert_array_equal(np.bincount(dag.indices, minlength=7),
                                  np.bincount(cyclic.indices, minlength=7))
    np.testing.assert_array_equal(abs(dag).sum(axis=1), abs(cyclic).sum(axis=1))
    np.testing.assert_array_equal(abs(dag).sum(axis=0), abs(cyclic).sum(axis=0))
    incoming = np.asarray(abs(dag).sum(axis=1)).ravel()
    np.testing.assert_allclose(incoming[incoming > 0], .9, rtol=0, atol=1e-15)
    for pre in range(7):
        np.testing.assert_array_equal(np.sort(dag[:, pre].data), np.sort(cyclic[:, pre].data))
    assert (dag[[6]][:, [0, 1, 2]] != cyclic[[6]][:, [0, 1, 2]]).nnz == 0
    d, c = degree_certificates(dag, roles), degree_certificates(cyclic, roles)
    assert d['role_block_counts'] == c['role_block_counts']
    assert d['status'] == c['status'] == 'UNRESOLVED'
    assert not d['obstructions'] and not c['obstructions']
    ds = walk_support(dag, roles, [0], [6], range(10))
    cs = walk_support(cyclic, roles, [0], [6], range(10))
    assert ds[0, 0, 0] and cs[0, 0, 0]
    assert not ds[0, 5:, :].any() and cs[0, 5:, :].any()
    assert cycle_witness(cyclic, roles, ids, [0], [6], 4)['certified']
    assert not cycle_witness(dag, roles, ids, [0], [6], 4)['certified']
    inputs = np.zeros((1, 7)); inputs[0, 0] = .5
    np.testing.assert_array_equal(_zero_observed(dag, roles, inputs, [6], {'leak': .6}),
                                  _zero_observed(cyclic, roles, inputs, [6], {'leak': .6}))


def test_relevant_cycle_compressed_bounds_and_zero_delay():
    _, cyclic, roles, ids = matched_pair()
    w = cycle_witness(cyclic, roles, ids, [0], [6], 4)
    assert w['certified'] and w['cycle'][0] == w['cycle'][-1]
    assert w['P'] >= 1 and w['delay'] == w['A'] + w['k'] * w['P'] + w['B'] > 4
    assert w['universal_delay'] == w['A'] + w['k_same_nodes'] * w['P'] + w['B'] > 6
    assert w['k'] == max(1, (4 - w['A'] - w['B']) // w['P'] + 1)
    assert w['k_same_nodes'] == max(1, (6 - w['A'] - w['B']) // w['P'] + 1)
    direct = graph(2, [(0, 1)])
    support = walk_support(direct, ['KC', 'MBON'], [0], [1], range(4))
    np.testing.assert_array_equal(support[0, :, 0], [True, False, False, False])
    loop = graph(2, [(0, 1), (1, 0)])
    w = cycle_witness(loop, ['KC', 'MBON'], [11, 12], [0], [1], 0)
    assert w['P'] == 1 and w['A'] == w['B'] == 0


def test_disconnected_or_unobservable_cycles_are_not_certificates():
    roles = ['KC', 'OTHER', 'OTHER', 'MBON', 'OTHER']
    ids = [10, 11, 12, 13, 14]
    isolated = graph(5, [(0, 3), (1, 2), (2, 1)])
    unobservable = graph(5, [(0, 3), (0, 1), (1, 2), (2, 1)])
    for matrix in [isolated, unobservable]:
        assert not cycle_witness(matrix, roles, ids, [0], [3], 4)['certified']
    uninput = graph(5, [(0, 3), (1, 2), (2, 1), (2, 3)])
    assert not cycle_witness(uninput, roles, ids, [0], [3], 4)['certified']


def test_degree_sufficient_obstructions_and_isolated_vertices():
    cycle = graph(4, [(0, 1), (1, 0)])
    c = degree_certificates(cycle, ['KC', 'MBON', 'OTHER', 'OTHER'])
    assert c['active_node_count'] == 2 and c['node_incident_capacity'] == 1
    assert c['global_capacity'] == 1 and c['edge_count'] == 2
    assert c['status'] == 'INFEASIBLE'
    rules = {x['rule'] for x in c['obstructions']}
    assert {'missing_source_or_sink', 'node_incident_capacity', 'global_dag_density',
            'opposite_role_pair_capacity'} <= rules
    within = degree_certificates(graph(3, [(0, 1), (1, 2), (2, 0), (0, 2)]), ['OTHER'] * 3)
    assert 'within_role_density' in {x['rule'] for x in within['obstructions']}
    dag = degree_certificates(graph(4, [(0, 1), (1, 2)]), ['OTHER'] * 4)
    assert dag['status'] == 'UNRESOLVED' and not dag['obstructions']
    empty = degree_certificates(sparse.csr_matrix((3, 3)), ['OTHER'] * 3)
    assert empty['status'] == 'UNRESOLVED' and empty['active_node_count'] == 0


def test_boolean_support_does_not_overflow_many_paths_and_ignores_signs():
    # 256 equal-length paths would cancel in int8 counting; support is boolean.
    edges = [(0, j) for j in range(1, 257)] + [(j, 257) for j in range(1, 257)]
    m = graph(258, edges)
    roles = ['KC'] + ['OTHER'] * 256 + ['MBON']
    result = walk_support(m, roles, [0], [257], [0, 1, 2, 3])
    np.testing.assert_array_equal(result[0, :, 0], [False, False, True, False])
    m.data[::2] *= -1
    np.testing.assert_array_equal(walk_support(m, roles, [0], [257], [0, 1, 2, 3]), result)


def test_smoke_null_unresolved_and_strict_summary_counts():
    level = dict(degree_role_status='UNRESOLVED', all_lag_path_status='UNRESOLVED',
                 outcome='UNRESOLVED', witness_count=0, cell_count=1,
                 cells=[dict(seed=1, symbol=0, certified=False)], degree_obstruction_rules=[])
    assert summarize({'legacy5': level})['outcome'] == 'UNRESOLVED'
    assert summarize({'legacy5': level}, smoke=True)['outcome'] is None
    invalid = dict(level, all_lag_path_status='INFEASIBLE', outcome='INFEASIBLE')
    with pytest.raises(ValueError):
        summarize({'legacy5': invalid})
    invalid = dict(level, witness_count=1)
    with pytest.raises(ValueError):
        summarize({'legacy5': invalid})
    invalid = dict(level, degree_role_status='INFEASIBLE', outcome='INFEASIBLE')
    with pytest.raises(ValueError):
        summarize({'legacy5': invalid})
    valid = dict(level, degree_role_status='INFEASIBLE', outcome='INFEASIBLE',
                 degree_obstruction_rules=['global_dag_density'])
    assert summarize({'legacy5': valid})['outcome'] == 'INFEASIBLE'
    with pytest.raises(ValueError):
        summarize({'legacy5': dict(level, outcome='FEASIBLE')})
