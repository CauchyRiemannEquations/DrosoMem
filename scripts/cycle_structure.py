"""Structure-only deterministic DAG masks and finite-dependency bounds.

Matrices use W[target, source]. No rate updates, input streams, readout fits or
scientific outcomes are computed here. Retained weights keep their original
normalization; no post-mask rescaling is performed.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


def _inputs(weights, roles, ids):
    matrix = weights.copy().tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    roles = np.asarray(roles)
    ids = np.asarray(ids, dtype=np.int64)
    assert matrix.shape == (len(ids), len(ids))
    assert roles.shape == ids.shape and len(np.unique(ids)) == len(ids)
    assert np.isfinite(matrix.data).all()
    return matrix, roles, ids


def _rank(roles, ids):
    group = np.where(roles == 'KC', 0, np.where(roles == 'MBON', 2, 1))
    order = np.lexsort((ids, group))
    rank = np.empty(len(ids), dtype=np.int64)
    rank[order] = np.arange(len(ids), dtype=np.int64)
    return rank


def rank_dag(weights, roles, ids):
    """Copy normalized W and retain only edges that ascend the fixed rank."""
    matrix, roles, ids = _inputs(weights, roles, ids)
    rank = _rank(roles, ids)
    target = np.repeat(np.arange(len(ids)), np.diff(matrix.indptr))
    source = matrix.indices
    keep = rank[source] < rank[target]
    protected = (roles[source] == 'KC') & (roles[target] == 'MBON')
    assert keep[protected].all()
    matrix.data[~keep] = 0
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return matrix, rank


def _degrees_masses(matrix):
    target = np.repeat(np.arange(matrix.shape[0]), np.diff(matrix.indptr))
    return {
        'indegree': np.diff(matrix.indptr).astype(np.int64),
        'outdegree': np.bincount(matrix.indices, minlength=matrix.shape[0]),
        'incoming_abs_weight': np.bincount(target, weights=np.abs(matrix.data),
                                         minlength=matrix.shape[0]),
        'outgoing_abs_weight': np.bincount(matrix.indices, weights=np.abs(matrix.data),
                                         minlength=matrix.shape[0]),
    }


def _depths(dag, roles, rank):
    """Longest weighted-delay DAG paths; global starts every coordinate at 0.

    The mbon_after_kc schedule gives KC->MBON edges delay zero. Every other
    synaptic edge reads the previous state and has delay one. Input-pool paths
    start ALL KC coordinates at zero; unreachable coordinates remain -1.
    """
    outgoing = dag.T.tocsr()
    global_depth = np.zeros(dag.shape[0], dtype=np.int64)
    input_depth = np.full(dag.shape[0], -1, dtype=np.int64)
    input_depth[roles == 'KC'] = 0
    for source in np.argsort(rank):
        targets = outgoing.indices[outgoing.indptr[source]:outgoing.indptr[source + 1]]
        delay = np.where((roles[source] == 'KC') & (roles[targets] == 'MBON'), 0, 1)
        global_depth[targets] = np.maximum(global_depth[targets],
                                          global_depth[source] + delay)
        if input_depth[source] >= 0:
            input_depth[targets] = np.maximum(input_depth[targets],
                                             input_depth[source] + delay)
    return global_depth, input_depth


def audit_arrays(original, dag, roles, ids, observed_indices):
    """Per-coordinate structural evidence suitable for a compressed NPZ."""
    original, roles, ids = _inputs(original, roles, ids)
    dag, _, _ = _inputs(dag, roles, ids)
    rank = _rank(roles, ids)
    expected, expected_rank = rank_dag(original, roles, ids)
    assert (expected != dag).nnz == 0 and np.array_equal(rank, expected_rank)
    observed = np.asarray(observed_indices, dtype=np.int64)
    assert observed.ndim == 1 and np.all((observed >= 0) & (observed < len(ids)))
    assert len(np.unique(observed)) == len(observed)
    assert np.all(roles[observed] == 'MBON')
    global_depth, input_depth = _depths(dag, roles, rank)
    arrays = {'root_ids': ids, 'roles': roles.astype('U5'), 'rank': rank,
              'observed_indices': observed, 'global_dependency_depth': global_depth,
              'input_pool_dependency_depth': input_depth}
    for name, matrix in [('original', original), ('dag', dag)]:
        for statistic, values in _degrees_masses(matrix).items():
            arrays[name + '_' + statistic] = values
    incoming = arrays['original_incoming_abs_weight']
    arrays['retained_incoming_abs_weight_fraction'] = np.divide(
        arrays['dag_incoming_abs_weight'], incoming,
        out=np.zeros_like(incoming), where=incoming > 0)
    return arrays


def _stats(values):
    values = np.asarray(values)
    if not len(values):
        return {'count': 0, 'min': None, 'max': None, 'mean': None,
                'median': None, 'sd': None, 'sum': 0.}
    return {'count': int(len(values)), 'min': float(values.min()),
            'max': float(values.max()), 'mean': float(values.mean()),
            'median': float(np.median(values)), 'sd': float(values.std()),
            'sum': float(values.sum())}


def _scc(matrix):
    count, labels = connected_components(matrix.T, directed=True, connection='strong')
    sizes = np.bincount(labels)
    target = np.repeat(np.arange(matrix.shape[0]), np.diff(matrix.indptr))
    source = matrix.indices
    cycle = (labels[source] == labels[target]) & ((sizes[labels[source]] > 1) | (source == target))
    return {'components': int(count), 'largest_sizes': sorted(map(int, sizes), reverse=True)[:10],
            'nontrivial_components': int((sizes > 1).sum()),
            'cycle_participating_edges': int(cycle.sum()),
            'self_edges': int((source == target).sum())}


def graph_structure(original, dag, roles, ids, observed_indices):
    """Audit the fixed mask, graph cycles, weights, access and finite bounds."""
    original, roles, ids = _inputs(original, roles, ids)
    dag, _, _ = _inputs(dag, roles, ids)
    arrays = audit_arrays(original, dag, roles, ids, observed_indices)
    observed = arrays['observed_indices']
    rank = arrays['rank']
    target = np.repeat(np.arange(len(ids)), np.diff(dag.indptr))
    assert np.all(rank[dag.indices] < rank[target])
    original_scc, dag_scc = _scc(original), _scc(dag)
    assert dag_scc['components'] == len(ids) and dag_scc['cycle_participating_edges'] == 0
    kc = np.flatnonzero(roles == 'KC')
    mbon = np.flatnonzero(roles == 'MBON')
    protected_original = original[mbon][:, kc]
    protected_dag = dag[mbon][:, kc]
    assert (protected_original != protected_dag).nnz == 0
    role_counts = {str(role): int(count) for role, count in zip(*np.unique(roles, return_counts=True))}
    role_blocks = {}
    for matrix_name, matrix in [('original', original), ('dag', dag)]:
        post = np.repeat(np.arange(len(ids)), np.diff(matrix.indptr))
        pre = matrix.indices
        role_blocks[matrix_name] = {}
        for source_role in sorted(role_counts):
            for target_role in sorted(role_counts):
                selector = (roles[pre] == source_role) & (roles[post] == target_role)
                role_blocks[matrix_name][source_role + '->' + target_role] = {
                    'edges': int(selector.sum()),
                    'absolute_weight_sum': float(np.abs(matrix.data[selector]).sum())}
    statistics = {}
    for matrix_name in ['original', 'dag']:
        statistics[matrix_name] = {
            name: _stats(arrays[matrix_name + '_' + name])
            for name in ['indegree', 'outdegree', 'incoming_abs_weight', 'outgoing_abs_weight']}
    active = arrays['original_incoming_abs_weight'] > 0
    fraction = arrays['retained_incoming_abs_weight_fraction']
    input_observed = arrays['input_pool_dependency_depth'][observed]
    return {
        'nodes': len(ids), 'original_edges': int(original.nnz), 'dag_edges': int(dag.nnz),
        'removed_edges': int(original.nnz - dag.nnz), 'roles': role_counts,
        'rank_definition': 'KC first; other roles second; MBON last; ascending numeric root ID within group',
        'normalization': 'original incoming-L1 normalized weights masked without rescaling',
        'original_scc': original_scc, 'dag_scc': dag_scc,
        'strict_rank_increase_on_every_retained_edge': True,
        'dag_cycle_edges': 0, 'protected_kc_to_mbon_edges': int(protected_original.nnz),
        'protected_kc_to_mbon_all_kept_exactly': True,
        'role_blocks': role_blocks, 'per_neuron_statistics': statistics,
        'retained_incoming_mass_fraction_active_rows': _stats(fraction[active]),
        'retained_incoming_mass_fraction_observed_rows': _stats(fraction[observed]),
        'delay_definition': '0 for KC->MBON; 1 for every other edge under mbon_after_kc; no direct carry',
        'global_dependency_depth': int(arrays['global_dependency_depth'].max(initial=0)),
        'observed_dependency_depth': int(arrays['global_dependency_depth'][observed].max(initial=0)),
        'input_pool_definition': 'all neurons with KC role, independent of per-symbol selected inputs',
        'input_pool_count': len(kc), 'input_pool_observed_depth': int(input_observed.max(initial=-1)),
        'input_pool_observed_depths': input_observed.tolist(),
        'input_pool_observed_reachable': int((input_observed >= 0).sum()),
        'observed_count': len(observed),
    }
