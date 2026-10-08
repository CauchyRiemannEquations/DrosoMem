"""Independent M4 structural replay; imports no runner or M3 graph helper.

No trajectory, symbol stream, performance prediction or decoder fit is executed.
Raw source authentication uses the frozen independent TDC source verifier only.
Rank masking, incoming dependency recurrence, boolean walk support, canonical
cycle witnesses, degree obstructions and strict endpoint are recomputed here.
"""
import argparse
import hashlib
from pathlib import Path
import subprocess

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import breadth_first_order, connected_components, shortest_path
from threadpoolctl import threadpool_limits

from verify_temporal_memory_curve import independent_graph, validate_whole_source
from m4_support import (config, history_preserved, read, write, sha, array_sha,
                        attempt, seal, check_manifest)
from tdc_support import environment, git


PROTOCOL_COMMIT = '2c7a5f593c22909652a1374c9338571656411855'
STATISTICS = ('indegree', 'outdegree', 'incoming_abs_weight', 'outgoing_abs_weight')
FLOAT_ATOL = 1e-12
FLOAT_RTOL = 1e-12


def compare_tree(actual, expected, path='root'):
    """Pre-outcome descriptive-reduction tolerance; counts/booleans stay exact."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and set(actual) == set(expected), path
        for key in expected:
            compare_tree(actual[key], expected[key], path + '/' + key)
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), path
        for index, (left, right) in enumerate(zip(actual, expected)):
            compare_tree(left, right, path + '/' + str(index))
    elif isinstance(expected, float):
        assert isinstance(actual, (int, float)) and not isinstance(actual, bool), path
        assert np.isfinite(actual) and abs(actual - expected) <= FLOAT_ATOL + FLOAT_RTOL*abs(expected), (path, actual, expected)
    else:
        assert type(actual) is type(expected) and actual == expected, (path, actual, expected)


def matrix_digest(matrix):
    result = hashlib.sha256()
    for vector in (matrix.data, matrix.indices, matrix.indptr):
        result.update(vector.tobytes())
    return result.hexdigest()


def reconstruct_dag(matrix, roles, ids):
    """Independently sort Python coordinate keys and filter each incoming row."""
    matrix = matrix.tocsr()
    ids = np.asarray(ids, dtype=np.int64)
    roles = np.asarray(roles)
    assert matrix.shape == (len(ids), len(ids)) and roles.shape == ids.shape
    assert len(np.unique(ids)) == len(ids) and np.isfinite(matrix.data).all()
    assert not np.any(matrix.diagonal()) and not np.any(matrix.data == 0)
    order = sorted(range(len(ids)), key=lambda v: (0 if roles[v] == 'KC' else
                   2 if roles[v] == 'MBON' else 1, int(ids[v])))
    rank = np.empty(len(ids), dtype=np.int64)
    rank[order] = np.arange(len(ids))
    values, indices, pointers = [], [], [0]
    for target in range(len(ids)):
        start, end = matrix.indptr[target:target + 2]
        sources = matrix.indices[start:end]
        keep = np.fromiter((rank[int(source)] < rank[target] for source in sources),
                           dtype=bool, count=len(sources))
        values.append(matrix.data[start:end][keep])
        indices.append(sources[keep])
        pointers.append(pointers[-1] + int(keep.sum()))
    dag = sparse.csr_matrix((np.concatenate(values), np.concatenate(indices),
                            np.asarray(pointers, dtype=matrix.indptr.dtype)), shape=matrix.shape)
    dag.sort_indices()
    return dag, rank


def degree_mass(matrix):
    """CSR/CSC degree counts and ordered scatter rather than main bincounts."""
    outgoing = matrix.tocsc()
    incoming_mass, outgoing_mass = np.zeros(matrix.shape[0]), np.zeros(matrix.shape[0])
    target, source = matrix.nonzero()
    np.add.at(incoming_mass, target, np.abs(matrix.data))
    np.add.at(outgoing_mass, source, np.abs(matrix.data))
    return dict(indegree=np.diff(matrix.indptr).astype(np.int64),
                outdegree=np.diff(outgoing.indptr).astype(np.int64),
                incoming_abs_weight=incoming_mass, outgoing_abs_weight=outgoing_mass)


def independent_arrays(original, dag, roles, ids, observed, rank):
    global_depth = np.zeros(len(ids), dtype=np.int64)
    input_depth = np.full(len(ids), -1, dtype=np.int64)
    input_depth[roles == 'KC'] = 0
    # Calculate a target only after all earlier-ranked predecessors, rather than
    # propagating source depths over outgoing edges as in the main helper.
    for target in sorted(range(len(ids)), key=lambda v: int(rank[v])):
        sources = dag.indices[dag.indptr[target]:dag.indptr[target + 1]]
        if not len(sources):
            continue
        assert np.all(rank[sources] < rank[target])
        delay = ((roles[sources] != 'KC') | (roles[target] != 'MBON')).astype(np.int64)
        global_depth[target] = max(int(global_depth[target]), int((global_depth[sources] + delay).max()))
        reachable = input_depth[sources] >= 0
        if reachable.any():
            input_depth[target] = max(int(input_depth[target]),
                                      int((input_depth[sources[reachable]] + delay[reachable]).max()))
    arrays = dict(root_ids=np.asarray(ids, dtype=np.int64), roles=np.asarray(roles, dtype='U5'),
                  rank=rank, observed_indices=np.asarray(observed, dtype=np.int64),
                  global_dependency_depth=global_depth, input_pool_dependency_depth=input_depth)
    for name, matrix in [('original', original), ('dag', dag)]:
        arrays.update({name + '_' + key: value for key, value in degree_mass(matrix).items()})
    incoming = arrays['original_incoming_abs_weight']
    arrays['retained_incoming_abs_weight_fraction'] = np.divide(
        arrays['dag_incoming_abs_weight'], incoming, out=np.zeros(len(ids)), where=incoming > 0)
    return arrays


def describe(values):
    values = np.asarray(values)
    if not len(values):
        return dict(count=0, min=None, max=None, mean=None, median=None, sd=None, sum=0.)
    return dict(count=int(len(values)), min=float(values.min()), max=float(values.max()),
                mean=float(values.mean()), median=float(np.median(values)),
                sd=float(values.std(ddof=0)), sum=float(values.sum()))


def component_description(matrix):
    count, labels = connected_components(matrix, directed=True, connection='strong')
    sizes = np.bincount(labels)
    target, source = matrix.nonzero()
    cycle = (labels[target] == labels[source]) & ((sizes[labels[source]] > 1) | (target == source))
    return dict(components=int(count), largest_sizes=sorted(map(int, sizes), reverse=True)[:10],
                nontrivial_components=int((sizes > 1).sum()),
                cycle_participating_edges=int(cycle.sum()), self_edges=int((target == source).sum()))


def role_blocks(matrix, roles):
    target, source = matrix.nonzero()
    result = {}
    for pre_role in sorted(set(map(str, roles))):
        for post_role in sorted(set(map(str, roles))):
            choose = (roles[source] == pre_role) & (roles[target] == post_role)
            result[pre_role + '->' + post_role] = dict(
                edges=int(choose.sum()), absolute_weight_sum=float(np.abs(matrix.data[choose]).sum()))
    return result


def independent_topology(original, dag, roles, ids, observed, arrays):
    role_count = {role: int((roles == role).sum()) for role in sorted(set(map(str, roles)))}
    kc, mbon = np.flatnonzero(roles == 'KC'), np.flatnonzero(roles == 'MBON')
    protected = original[mbon][:, kc]
    assert (protected != dag[mbon][:, kc]).nnz == 0
    original_scc, dag_scc = component_description(original), component_description(dag)
    assert dag_scc['cycle_participating_edges'] == 0 and dag_scc['components'] == len(ids)
    observed_depth = arrays['input_pool_dependency_depth'][observed]
    fraction = arrays['retained_incoming_abs_weight_fraction']
    return dict(nodes=len(ids), original_edges=int(original.nnz), dag_edges=int(dag.nnz),
                removed_edges=int(original.nnz - dag.nnz), roles=role_count,
                rank_definition='KC first; other roles second; MBON last; ascending numeric root ID within group',
                normalization='original incoming-L1 normalized weights masked without rescaling',
                original_scc=original_scc, dag_scc=dag_scc,
                strict_rank_increase_on_every_retained_edge=True, dag_cycle_edges=0,
                protected_kc_to_mbon_edges=int(protected.nnz), protected_kc_to_mbon_all_kept_exactly=True,
                role_blocks=dict(original=role_blocks(original, roles), dag=role_blocks(dag, roles)),
                per_neuron_statistics={name: {key: describe(arrays[name + '_' + key]) for key in STATISTICS}
                                       for name in ['original', 'dag']},
                retained_incoming_mass_fraction_active_rows=describe(fraction[arrays['original_incoming_abs_weight'] > 0]),
                retained_incoming_mass_fraction_observed_rows=describe(fraction[observed]),
                delay_definition='0 for KC->MBON; 1 for every other edge under mbon_after_kc; no direct carry',
                global_dependency_depth=int(arrays['global_dependency_depth'].max(initial=0)),
                observed_dependency_depth=int(arrays['global_dependency_depth'][observed].max(initial=0)),
                input_pool_definition='all neurons with KC role, independent of per-symbol selected inputs',
                input_pool_count=int(len(kc)), input_pool_observed_depth=int(observed_depth.max(initial=-1)),
                input_pool_observed_depths=observed_depth.tolist(),
                input_pool_observed_reachable=int((observed_depth >= 0).sum()), observed_count=int(len(observed)))


def independent_degrees(matrix, roles):
    """Recalculate the five sufficient certificates from incoming/outgoing rows."""
    incoming = np.diff(matrix.indptr).astype(np.int64)
    outgoing = np.diff(matrix.tocsc().indptr).astype(np.int64)
    active = (incoming != 0) | (outgoing != 0)
    count, edges = int(active.sum()), int(matrix.nnz)
    sources = [int(v) for v in range(len(roles)) if active[v] and incoming[v] == 0]
    sinks = [int(v) for v in range(len(roles)) if active[v] and outgoing[v] == 0]
    capacity, total_capacity = max(0, count - 1), count * (count - 1) // 2
    offenders = [dict(index=v, indegree=int(incoming[v]), outdegree=int(outgoing[v]),
                      incident=int(incoming[v] + outgoing[v]), capacity=capacity)
                 for v in range(len(roles)) if active[v] and incoming[v] + outgoing[v] > capacity]
    sizes = {role: int((roles == role).sum()) for role in sorted(set(map(str, roles)))}
    # Count role totals through sparse submatrices instead of endpoint selectors.
    blocks = {a + '->' + b: int(matrix[np.flatnonzero(roles == b)][:, np.flatnonzero(roles == a)].nnz)
              for a in sizes for b in sizes}
    within = [dict(role=role, nodes=size, edges=blocks[role + '->' + role],
                   capacity=size * (size - 1) // 2,
                   obstructed=blocks[role + '->' + role] > size * (size - 1) // 2)
              for role, size in sizes.items()]
    pairs = []
    names = list(sizes)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            left, right = blocks[first + '->' + second], blocks[second + '->' + first]
            limit = sizes[first] * sizes[second]
            pairs.append(dict(roles=[first, second], r_to_s=left, s_to_r=right,
                              edges=left + right, capacity=limit, obstructed=left + right > limit))
    obstructions = []
    if edges > 0 and (not sources or not sinks):
        obstructions.append(dict(rule='missing_source_or_sink', source_missing=not bool(sources),
                                 sink_missing=not bool(sinks)))
    if offenders:
        obstructions.append(dict(rule='node_incident_capacity',
                                 offender_indices=[item['index'] for item in offenders]))
    if edges > total_capacity:
        obstructions.append(dict(rule='global_dag_density', edges=edges, capacity=total_capacity))
    if any(item['obstructed'] for item in within):
        obstructions.append(dict(rule='within_role_density',
                                 roles=[item['role'] for item in within if item['obstructed']]))
    if any(item['obstructed'] for item in pairs):
        obstructions.append(dict(rule='opposite_role_pair_capacity',
                                 role_pairs=[item['roles'] for item in pairs if item['obstructed']]))
    return dict(status='INFEASIBLE' if obstructions else 'UNRESOLVED',
                certified_obstruction=bool(obstructions), node_count=len(roles), active_node_count=count,
                edge_count=edges, source_indices=sources, sink_indices=sinks,
                nonisolated_source_count=len(sources), nonisolated_sink_count=len(sinks),
                node_incident_capacity=capacity, global_capacity=total_capacity,
                node_incident_offenders=offenders, role_sizes=sizes, role_block_counts=blocks,
                within_roles=within, role_pairs=pairs, obstructions=obstructions)


def independent_mismatch(original, dag, roles, arrays):
    result = {}
    for statistic, field in [('indegree', 'indegree_mismatch_count'), ('outdegree', 'outdegree_mismatch_count'),
                             ('incoming_abs_weight', 'incoming_mass_mismatch_count'),
                             ('outgoing_abs_weight', 'outgoing_mass_mismatch_count')]:
        result[field] = int(np.count_nonzero(arrays['original_' + statistic] != arrays['dag_' + statistic]))
    kc, mbon = np.flatnonzero(roles == 'KC'), np.flatnonzero(roles == 'MBON')
    result.update(current_block_exact=(original[mbon][:, kc] != dag[mbon][:, kc]).nnz == 0,
                  degree_equal=not (result['indegree_mismatch_count'] or result['outdegree_mismatch_count']),
                  weight_mass_equal=not (result['incoming_mass_mismatch_count'] or result['outgoing_mass_mismatch_count']),
                  role_blocks_equal={key: value['edges'] for key, value in role_blocks(original, roles).items()} ==
                                    {key: value['edges'] for key, value in role_blocks(dag, roles).items()},
                  exact_comparison=True)
    return result


def walk_support(matrix, roles, selected, observed, max_lag=20):
    """Per-symbol boolean sparse-vector recurrence on a different representation."""
    delayed = matrix.copy().astype(bool)
    target, source = matrix.nonzero()
    instantaneous = (roles[source] == 'KC') & (roles[target] == 'MBON')
    zero = sparse.csr_matrix((np.ones(int(instantaneous.sum()), dtype=bool),
                              (target[instantaneous], source[instantaneous])), shape=matrix.shape)
    delayed.data[instantaneous] = False
    delayed.eliminate_zeros()
    # The delay0 graph has just KC->MBON, so its square is zero and closure
    # needs exactly the identity plus one zero-delay propagation.
    assert (zero @ zero).nnz == 0
    answer = np.zeros((len(selected), max_lag + 1, len(observed)), dtype=bool)
    for symbol, inputs in enumerate(selected):
        active = np.zeros(matrix.shape[0], dtype=bool)
        active[inputs] = True
        active |= np.asarray(zero @ active).ravel()
        answer[symbol, 0] = active[observed]
        for lag in range(1, max_lag + 1):
            active = np.asarray(delayed @ active).ravel()
            active |= np.asarray(zero @ active).ravel()
            answer[symbol, lag] = active[observed]
    return answer


def analytic_current_response(matrix, roles, selected, observed, c):
    """Closed-form one-symbol, zero-initial-state response, not a trajectory."""
    kc = np.flatnonzero(roles == 'KC')
    local = {int(index): position for position, index in enumerate(kc)}
    stimulus = np.zeros((len(selected), len(kc)))
    for symbol, inputs in enumerate(selected):
        stimulus[symbol, [local[int(index)] for index in inputs]] = c['input_amplitude']
    current_kc = c['leak'] * np.tanh(stimulus)
    return c['leak'] * np.tanh((matrix[observed][:, kc] @ current_kc.T).T)


class ConeReplay:
    """Independent SCC condensation and shortest-distance witness construction."""
    def __init__(self, matrix, roles, ids, observed):
        self.matrix, self.roles, self.ids, self.observed = matrix, roles, ids, observed
        self.outgoing = matrix.T.tocsr().astype(bool)
        self.incoming = matrix.astype(bool)
        self.count, self.labels = connected_components(matrix, directed=True, connection='strong')
        self.sizes = np.bincount(self.labels)
        target, source = matrix.nonzero()
        between = self.labels[source] != self.labels[target]
        self.condensation = sparse.csr_matrix((np.ones(int(between.sum()), dtype=bool),
                                              (self.labels[source[between]], self.labels[target[between]])),
                                             shape=(self.count, self.count))
        self.observable = np.zeros(self.count, dtype=bool)
        reverse = self.condensation.T.tocsr()
        for component in set(map(int, self.labels[observed])):
            self.observable[breadth_first_order(reverse, component, directed=True,
                                                 return_predecessors=False)] = True
        self.minimum = np.full(self.count, np.iinfo(np.int64).max, dtype=np.int64)
        np.minimum.at(self.minimum, self.labels, ids)
        self.component_cache, self.distance_cache = {}, {}

    def reachable_component(self, component):
        component = int(component)
        if component not in self.component_cache:
            answer = np.zeros(self.count, dtype=bool)
            answer[breadth_first_order(self.condensation, component, directed=True,
                                        return_predecessors=False)] = True
            self.component_cache[component] = answer
        return self.component_cache[component]

    def chosen_component(self, selected):
        accessible = np.zeros(self.count, dtype=bool)
        for component in set(map(int, self.labels[selected])):
            accessible |= self.reachable_component(component)
        eligible = np.flatnonzero(accessible & self.observable & (self.sizes > 1))
        if not len(eligible):
            return None
        return min(map(int, eligible), key=lambda component: (-int(self.sizes[component]),
                                                              int(self.minimum[component])))

    def distance_to(self, target):
        target = int(target)
        if target not in self.distance_cache:
            self.distance_cache[target] = shortest_path(self.incoming, directed=True,
                                                         unweighted=True, indices=target)
        return self.distance_cache[target]

    def path(self, start, target):
        start, target = int(start), int(target)
        distances = self.distance_to(target)
        assert np.isfinite(distances[start])
        answer = [start]
        while answer[-1] != target:
            current = answer[-1]
            following = self.outgoing.indices[self.outgoing.indptr[current]:self.outgoing.indptr[current + 1]]
            following = following[distances[following] == distances[current] - 1]
            assert len(following)
            answer.append(int(following[np.argmin(self.ids[following])]))
        return answer

    def canonical_paths(self, selected):
        component = self.chosen_component(selected)
        if component is None:
            return None
        anchor = int(np.flatnonzero(self.ids == self.minimum[component])[0])
        internal = self.outgoing.indices[self.outgoing.indptr[anchor]:self.outgoing.indptr[anchor + 1]]
        internal = internal[self.labels[internal] == component]
        assert len(internal)
        neighbor = int(internal[np.argmin(self.ids[internal])])
        cycle = [anchor] + self.path(neighbor, anchor)
        possible_inputs = selected[np.isfinite(self.distance_to(anchor)[selected])]
        assert len(possible_inputs)
        input_node = int(possible_inputs[np.argmin(self.ids[possible_inputs])])
        reachable = np.zeros(len(self.ids), dtype=bool)
        reachable[breadth_first_order(self.outgoing, anchor, directed=True,
                                     return_predecessors=False)] = True
        possible_observed = self.observed[reachable[self.observed]]
        assert len(possible_observed)
        observer = int(possible_observed[np.argmin(self.ids[possible_observed])])
        return self.path(input_node, anchor), cycle, self.path(anchor, observer)


def path_delay(matrix, roles, coordinates):
    assert coordinates
    result = 0
    for source, target in zip(coordinates[:-1], coordinates[1:]):
        assert matrix[int(target), int(source)] != 0, (source, target)
        result += 0 if roles[source] == 'KC' and roles[target] == 'MBON' else 1
    return result


def validate_witness(cell, symbol, selected, observed, matrix, roles, ids, depth, cone=None):
    """Validate saved edges/endpoints, exact arithmetic and independent existence."""
    assert cell['symbol'] == symbol and type(cell['certified']) is bool
    cone = cone or ConeReplay(matrix, roles, ids, observed)
    expected = cone.canonical_paths(selected)
    if expected is None:
        assert cell['certified'] is False and cell['reason']
        assert cell['anchor_root_id'] is None and cell['scc_size'] is None
        assert cell['H'] == depth and cell['node_count'] == len(ids)
        for key in ['input_path', 'cycle', 'observed_path']:
            assert cell[key] == []
        for key in ['A', 'B', 'P', 'k', 'delay', 'k_same_nodes', 'universal_delay']:
            assert cell[key] is None
        return dict(symbol=symbol, certified=False, independently_absent_relevant_cycle=True)
    assert cell['certified'] is True and cell['reason'] is None
    lookup = {str(root): index for index, root in enumerate(ids)}
    paths = []
    for key, expected_path in zip(['input_path', 'cycle', 'observed_path'], expected):
        assert isinstance(cell[key], list) and all(isinstance(root, str) and root in lookup for root in cell[key])
        actual_path = [lookup[root] for root in cell[key]]
        assert actual_path == expected_path, key
        paths.append(actual_path)
    input_path, cycle, observed_path = paths
    assert input_path[0] in selected and roles[input_path[0]] == 'KC'
    assert observed_path[-1] in observed and roles[observed_path[-1]] == 'MBON'
    assert input_path[-1] == cycle[0] == cycle[-1] == observed_path[0]
    assert len(cycle) >= 3 and len(set(cycle[:-1])) == len(cycle) - 1
    assert cell['anchor_root_id'] == str(ids[cycle[0]])
    assert cell['scc_size'] == int(cone.sizes[cone.labels[cycle[0]]])
    a, p, b = [path_delay(matrix, roles, path) for path in paths]
    assert p >= 1
    k = max(1, (depth - a - b) // p + 1)
    k_same = max(1, (len(ids) - 1 - a - b) // p + 1)
    expected_numbers = dict(A=a, B=b, P=p, H=int(depth), k=k, delay=a + k*p + b,
                            node_count=len(ids), k_same_nodes=k_same,
                            universal_delay=a + k_same*p + b)
    for key, value in expected_numbers.items():
        assert type(cell[key]) is int and cell[key] == value, key
    assert expected_numbers['delay'] > depth
    assert expected_numbers['universal_delay'] > len(ids) - 1
    return dict(symbol=symbol, certified=True, every_edge_and_endpoint_valid=True,
                deterministic_paths_independently_recalculated=True, **expected_numbers)


def independent_level_summary(degrees, cells, nodes, depth):
    witnesses = sum(1 for cell in cells if cell['certified'])
    rules = [item['rule'] for item in degrees['obstructions']]
    return dict(degree_role_status=degrees['status'],
                all_lag_path_status='INFEASIBLE' if witnesses else 'UNRESOLVED',
                outcome='INFEASIBLE' if rules or witnesses else 'UNRESOLVED',
                witness_count=witnesses, cell_count=len(cells), cells=cells,
                degree_obstruction_rules=rules, node_count=nodes,
                reference_global_dependency_bound=depth)


def independent_summary(levels, smoke, primary):
    assert type(smoke) is bool and primary in levels
    return dict(smoke=smoke, outcome=None if smoke else levels[primary]['outcome'],
                primary_level=primary, levels=levels, neural_runs=0, decoder_fits=0,
                graph_searches=0, functional_cycle_effect_claim=False,
                biological_plasticity_performed=False, exact_all_lag_ideal_only=True,
                absence_of_obstruction_is_not_feasible=True)


def source_matrix_record(c, level, original, dag, roles, ids, observed, first_block):
    small_weights, small_patterns, small_roles, small_observed, small_ids = independent_graph(
        c, 'legacy5', first_block['input_seed'])
    del small_weights
    provenance = read(Path(c['cache']) / 'provenance.json')
    required = ['nodes.npz'] if level == 'legacy5' else ['nodes.npz', 'brain5.npz']
    initial_mapping = [small_ids[row != 0].astype(str).tolist() for row in small_patterns]
    graph = dict(level=level, neurons=len(ids), edges=int(original.nnz), threshold=5,
                 weight_sha256=matrix_digest(original), source=provenance['sources'],
                 graph_cache_hashes={name: provenance['files'][name] for name in required},
                 observation_root_ids=ids[observed].astype(str).tolist(),
                 input_root_ids=initial_mapping, input_amplitude=c['input_amplitude'],
                 observed_features=48,
                 input_mapping_sha256=hashlib.sha256(small_ids.tobytes() + small_patterns.tobytes()).hexdigest())
    assert len(small_observed) == 48 and int((small_roles == 'KC').sum()) == 512
    sealed_file = Path('results/tdc_cycles_v1/structural_audit') / (level + '-dag.npz')
    with np.load(sealed_file, allow_pickle=False) as saved:
        assert set(saved.files) == {'data', 'indices', 'indptr', 'shape'}
        np.testing.assert_array_equal(saved['shape'], dag.shape)
        for name in ['data', 'indices', 'indptr']:
            np.testing.assert_array_equal(saved[name], getattr(dag, name))
    return dict(original_weight_sha256=matrix_digest(original), dag_weight_sha256=matrix_digest(dag),
                root_ids_sha256=array_sha(ids), roles_sha256=array_sha(roles.astype('U5')),
                observation_indices_sha256=array_sha(observed), sealed_dag_file_sha256=sha(sealed_file),
                source_graph=graph, sealed_m3_mask_exact=True)


def compare_npz(path, expected):
    with np.load(path, allow_pickle=False) as saved:
        assert set(saved.files) == set(expected)
        for name, values in expected.items():
            assert saved[name].dtype == values.dtype and saved[name].shape == values.shape, name
            if values.dtype.kind == 'f':
                assert np.isfinite(saved[name]).all()
                np.testing.assert_allclose(saved[name], values, atol=FLOAT_ATOL, rtol=FLOAT_RTOL, err_msg=name)
            else:
                np.testing.assert_array_equal(saved[name], values, err_msg=name)


def verify_level(result, destination, c, level, resources):
    destination.mkdir(parents=True, exist_ok=False)
    blocks = [block for block in c['blocks'] if block['seed'] in c['blocks_by_level'][level]]
    assert len(blocks) == len(c['blocks_by_level'][level]) and blocks
    original, first_patterns, roles, observed, ids = independent_graph(c, level, blocks[0]['input_seed'])
    roles = roles.astype('U5')
    del first_patterns
    assert len(observed) == 48 and np.all(roles[observed] == 'MBON')
    dag, rank = reconstruct_dag(original, roles, ids)
    arrays = independent_arrays(original, dag, roles, ids, observed, rank)
    compare_npz(result / 'structural-arrays.npz', arrays)
    np.savez_compressed(destination / 'structural-arrays.npz', **arrays)
    topology = independent_topology(original, dag, roles, ids, observed, arrays)
    degrees = independent_degrees(original, roles)
    mismatch = independent_mismatch(original, dag, roles, arrays)
    matrix_record = source_matrix_record(c, level, original, dag, roles, ids, observed, blocks[0])
    for name, record in [('topology', topology), ('degree-certificates', degrees),
                         ('mismatch', mismatch), ('source-matrix', matrix_record)]:
        compare_tree(read(result / (name + '.json')), record, level + '/' + name)
        write(destination / (name + '.json'), record)
    resources.check()
    cone = ConeReplay(original, roles, ids, observed)
    cells, cases, pairing = [], [], {}
    for block in blocks:
        # Rebuild the encoder from its fixed seed, never the saved index table.
        mapping_weights, patterns, mapping_roles, mapping_observed, mapping_ids = independent_graph(
            c, level, block['input_seed'])
        assert matrix_digest(mapping_weights) == matrix_digest(original)
        np.testing.assert_array_equal(mapping_roles.astype('U5'), roles)
        np.testing.assert_array_equal(mapping_observed, observed)
        np.testing.assert_array_equal(mapping_ids, ids)
        del mapping_weights, mapping_roles, mapping_observed, mapping_ids
        selected = np.array([np.flatnonzero(pattern) for pattern in patterns], dtype=np.int64)
        assert selected.shape == (10, 51) and np.all(roles[selected] == 'KC')
        for row in selected:
            assert len(np.unique(row)) == 51
        del patterns
        replay = dict(input_indices=selected,
                      lag_support_original=walk_support(original, roles, selected, observed, max(c['lags'])),
                      lag_support_dag=walk_support(dag, roles, selected, observed, max(c['lags'])),
                      zero_state_original=analytic_current_response(original, roles, selected, observed, c),
                      zero_state_dag=analytic_current_response(dag, roles, selected, observed, c))
        np.testing.assert_array_equal(replay['zero_state_original'], replay['zero_state_dag'])
        name = 'mapping-s' + str(block['seed'])
        metadata = read(result / (name + '.json'))
        assert set(metadata['array_hashes']) == set(replay)
        with np.load(result / (name + '.npz'), allow_pickle=False) as saved:
            assert set(saved.files) == set(replay)
            for key in saved.files:
                assert array_sha(saved[key]) == metadata['array_hashes'][key], key
            np.testing.assert_array_equal(saved['zero_state_original'], saved['zero_state_dag'])
        compare_npz(result / (name + '.npz'), replay)
        roots = [ids[row].astype(str).tolist() for row in selected]
        expected_metadata = dict(seed=block['seed'], input_seed=block['input_seed'], input_root_ids=roots,
                                 observation_root_ids=ids[observed].astype(str).tolist(),
                                 array_hashes=metadata['array_hashes'], zero_state_current_exact=True,
                                 structural_walk_support_only=True, current_decoding_measured=False)
        compare_tree(metadata, expected_metadata, level + '/' + name)
        saved_witnesses = read(result / ('witnesses-s' + str(block['seed']) + '.json'))
        assert len(saved_witnesses) == 10
        checks = [validate_witness(cell, symbol, selected[symbol], observed, original,
                                   roles, ids, topology['global_dependency_depth'], cone)
                  for symbol, cell in enumerate(saved_witnesses)]
        cells.extend(dict(seed=block['seed'], symbol=cell['symbol'], certified=cell['certified'])
                     for cell in checks)
        np.savez_compressed(destination / (name + '.npz'), **replay)
        write(destination / ('witness-checks-s' + str(block['seed']) + '.json'), checks)
        pairing[block['seed']] = roots
        cases.append(dict(level=level, seed=block['seed'], input_seed=block['input_seed'],
                          mapping_exact=True, boolean_support_exact=True,
                          zero_state_response_analytically_recalculated=True,
                          zero_state_current_exact=True, current_decoding_measured=False,
                          certified_witnesses=sum(cell['certified'] for cell in checks),
                          witness_cells=len(checks)))
        resources.check()
        print(f'Independently verified M4 {level} mapping {block["seed"]}', flush=True)
    summary = independent_level_summary(degrees, cells, len(ids), topology['global_dependency_depth'])
    return summary, cases, pairing


def verify(result, out):
    manifest = check_manifest(result)
    assert type(manifest['smoke']) is bool
    recorded_files = set(manifest['artifacts']) | {'manifest.json'}
    actual_files = {path.relative_to(result).as_posix() for path in result.rglob('*') if path.is_file()}
    assert actual_files == recorded_files, 'Unmanifested stage files'
    c = read(result / 'config.json')
    assert c == config(manifest['smoke'])
    source = read(result / 'source.json')
    assert source['source_commit'] == manifest['source_commit'] and source['tracked_changes'] is False
    assert source['source_tree'] == git('rev-parse', source['source_commit'] + '^{tree}')
    assert source['protocol_commit'] == PROTOCOL_COMMIT
    assert (source['neural_runs'], source['decoder_fits'], source['graph_searches']) == (0, 0, 0)
    subprocess.run(['git', 'merge-base', '--is-ancestor', PROTOCOL_COMMIT, source['source_commit']], check=True)
    required = {'scripts/verify_cycle_attribution_feasibility.py', 'scripts/m4_support.py',
                'scripts/verify_temporal_memory_curve.py', 'scripts/cycle_attribution_feasibility.py',
                'configs/cycle_attribution_feasibility.json', c['protocol'],
                'tests/test_cycle_attribution_verifier.py'}
    assert required.issubset(source['hashes'])
    for name, digest in source['hashes'].items():
        assert sha(name) == digest, name
    for name in ['configs/cycle_attribution_feasibility.json', c['protocol']]:
        frozen = subprocess.check_output(['git', 'show', PROTOCOL_COMMIT + ':' + name])
        assert hashlib.sha256(frozen).hexdigest() == source['hashes'][name], name
    assert subprocess.run(['git', 'diff', '--quiet', 'HEAD']).returncode == 0
    with attempt(out, c, 'm4-independent-structural-validation') as resources:
        preservation = history_preserved(c)
        compare_tree(read(result / 'historical-preservation.json'), preservation)
        write(out / 'environment.json', environment())
        write(out / 'source-validation.json', dict(result_source_commit=source['source_commit'],
              protocol_commit=PROTOCOL_COMMIT, verifier_commit=git('rev-parse', 'HEAD'),
              source_hashes_checked=len(source['hashes']), neural_runs=0, decoder_fits=0))
        write(out / 'source-graph-audit.json', validate_whole_source(c, resources))
        check_manifest(Path('results/tdc_cycles_v1/structural_audit'))
        levels, cases, mapping_roots = {}, [], {}
        for level in c['levels']:
            record, checks, roots = verify_level(result / level, out / level, c, level, resources)
            levels[level] = record
            cases.extend(checks)
            mapping_roots[level] = roots
        for seed, whole_roots in mapping_roots['brain5'].items():
            partial_roots = mapping_roots['legacy5'][seed]
            assert [sorted(row) for row in whole_roots] == [sorted(row) for row in partial_roots]
        summary = independent_summary(levels, manifest['smoke'], c['primary_level'])
        compare_tree(read(result / 'summary.json'), summary)
        write(out / 'recomputed-summary.json', summary)
        write(out / 'case-checks.json', cases)
        assert history_preserved(c) == preservation
        for name, digest in source['hashes'].items():
            assert sha(name) == digest, name
        resources.check()
        write(out / 'checks.json', dict(all_checks_pass=True, outcome=summary['outcome'], levels=levels,
              mapping_cases=len(cases), witness_cells=sum(case['witness_cells'] for case in cases),
              certified_witnesses=sum(case['certified_witnesses'] for case in cases),
              validated_degree_obstructions=sum(len(record['degree_obstruction_rules']) for record in levels.values()),
              raw_source_graphs_authenticated=True, rank_mask_independently_reconstructed=True,
              graph_support_and_analytic_response_independently_recalculated=True,
              deterministic_witness_selection_and_universal_bound_verified=True,
              cross_level_input_root_ids_paired=True, current_decoding_measured=False,
              descriptive_float_atol=FLOAT_ATOL, descriptive_float_rtol=FLOAT_RTOL,
              csr_coefficients_hashes_degrees_support_and_witness_integers_exact=True,
              neural_runs=0, decoder_fits=0, graph_searches=0,
              result_manifest_sha256=sha(result / 'manifest.json'), source_commit=source['source_commit'],
              protocol_commit=PROTOCOL_COMMIT, verifier_commit=git('rev-parse', 'HEAD'),
              verifier_sha256=sha(__file__), historical_preservation=preservation))
    seal(out, dict(complete=True, result_manifest_sha256=sha(result / 'manifest.json')))
    print(dict(all_checks_pass=True, outcome=summary['outcome'], mapping_cases=len(cases),
               certified_witnesses=sum(case['certified_witnesses'] for case in cases),
               neural_runs=0, decoder_fits=0), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(1):
        verify(args.result, args.out)


if __name__ == '__main__':
    main()
