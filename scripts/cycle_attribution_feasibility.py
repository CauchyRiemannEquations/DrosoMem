"""M4 structural impossibility certificates; no neural trajectories or decoders.

Matrices use W[target, source]. Support ignores signs and magnitudes: a walk
certificate establishes a structural dependency only, never a functional gain.
"""
import argparse
from collections import deque
import hashlib
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from threadpoolctl import threadpool_limits


def _matrix(matrix, roles):
    matrix = matrix.copy().tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    roles = np.asarray(roles)
    if matrix.shape != (len(roles), len(roles)):
        raise ValueError('Role count and square matrix must agree')
    if not np.isfinite(matrix.data).all() or np.any(matrix.diagonal() != 0):
        raise ValueError('Finite simple directed matrix without self edges required')
    return matrix, roles


def _role_counts(matrix, roles):
    post = np.repeat(np.arange(len(roles)), np.diff(matrix.indptr))
    pre = matrix.indices
    names = sorted(map(str, np.unique(roles)))
    return {a + '->' + b: int(((roles[pre] == a) & (roles[post] == b)).sum())
            for a in names for b in names}


def degree_certificates(matrix, roles):
    """Five prospective sufficient obstructions; absence stays UNRESOLVED."""
    matrix, roles = _matrix(matrix, roles)
    indegree = np.diff(matrix.indptr)
    outdegree = np.bincount(matrix.indices, minlength=len(roles))
    active = (indegree + outdegree) > 0
    count = int(active.sum())
    edges = int(matrix.nnz)
    sources = np.flatnonzero(active & (indegree == 0))
    sinks = np.flatnonzero(active & (outdegree == 0))
    incident_capacity = max(0, count - 1)
    global_capacity = count * (count - 1) // 2
    offenders = [dict(index=int(i), indegree=int(indegree[i]),
                      outdegree=int(outdegree[i]), incident=int(indegree[i] + outdegree[i]),
                      capacity=incident_capacity)
                 for i in np.flatnonzero(active & (indegree + outdegree > incident_capacity))]
    role_blocks = _role_counts(matrix, roles)
    sizes = {str(r): int((roles == r).sum()) for r in np.unique(roles)}
    within = [dict(role=r, nodes=n, edges=role_blocks[r + '->' + r],
                   capacity=n * (n - 1) // 2,
                   obstructed=role_blocks[r + '->' + r] > n * (n - 1) // 2)
              for r, n in sorted(sizes.items())]
    pairs = []
    names = sorted(sizes)
    for j, a in enumerate(names):
        for b in names[j + 1:]:
            ab, ba = role_blocks[a + '->' + b], role_blocks[b + '->' + a]
            cap = sizes[a] * sizes[b]
            pairs.append(dict(roles=[a, b], r_to_s=ab, s_to_r=ba,
                              edges=ab + ba, capacity=cap, obstructed=ab + ba > cap))
    obstructions = []
    if edges and (not len(sources) or not len(sinks)):
        obstructions.append(dict(rule='missing_source_or_sink',
                                 source_missing=not bool(len(sources)),
                                 sink_missing=not bool(len(sinks))))
    if offenders:
        obstructions.append(dict(rule='node_incident_capacity',
                                 offender_indices=[x['index'] for x in offenders]))
    if edges > global_capacity:
        obstructions.append(dict(rule='global_dag_density', edges=edges,
                                 capacity=global_capacity))
    if any(x['obstructed'] for x in within):
        obstructions.append(dict(rule='within_role_density',
                                 roles=[x['role'] for x in within if x['obstructed']]))
    if any(x['obstructed'] for x in pairs):
        obstructions.append(dict(rule='opposite_role_pair_capacity',
                                 role_pairs=[x['roles'] for x in pairs if x['obstructed']]))
    return dict(status='INFEASIBLE' if obstructions else 'UNRESOLVED',
                certified_obstruction=bool(obstructions), node_count=len(roles),
                active_node_count=count, edge_count=edges,
                source_indices=sources.tolist(), sink_indices=sinks.tolist(),
                nonisolated_source_count=len(sources), nonisolated_sink_count=len(sinks),
                node_incident_capacity=incident_capacity, global_capacity=global_capacity,
                node_incident_offenders=offenders, role_sizes=sizes,
                role_block_counts=role_blocks, within_roles=within, role_pairs=pairs,
                obstructions=obstructions)


def _delay_matrices(matrix, roles):
    matrix, roles = _matrix(matrix, roles)
    post = np.repeat(np.arange(len(roles)), np.diff(matrix.indptr))
    pre = matrix.indices
    zero = (roles[pre] == 'KC') & (roles[post] == 'MBON')
    def support(mask):
        return sparse.csr_matrix((np.ones(int(mask.sum()), dtype=bool),
                                  (post[mask], pre[mask])), shape=matrix.shape)
    return support(~zero), support(zero)


def walk_support(matrix, roles, input_indices, observed, lags):
    """Boolean existence of exact scheduled-delay walks [symbol, lag, obs]."""
    matrix, roles = _matrix(matrix, roles)
    inputs = np.asarray(input_indices, dtype=np.int64)
    if inputs.ndim == 1:
        inputs = inputs[None, :]
    observed = np.asarray(observed, dtype=np.int64)
    lags = list(map(int, lags))
    if not lags or min(lags) < 0 or len(set(lags)) != len(lags):
        raise ValueError('Unique nonnegative requested lags required')
    if inputs.ndim != 2 or np.any(inputs < 0) or np.any(inputs >= len(roles)):
        raise ValueError('Invalid symbol input indices')
    if observed.ndim != 1 or np.any(observed < 0) or np.any(observed >= len(roles)):
        raise ValueError('Invalid observation indices')
    delayed, zero = _delay_matrices(matrix, roles)
    state = np.zeros((len(roles), len(inputs)), dtype=bool)
    for symbol, indices in enumerate(inputs):
        state[indices, symbol] = True
    # A second zero edge is impossible: every delay0 edge ends in MBON.
    state |= zero @ state
    wanted = {lag: i for i, lag in enumerate(lags)}
    result = np.zeros((len(inputs), len(lags), len(observed)), dtype=bool)
    for lag in range(max(lags) + 1):
        if lag in wanted:
            result[:, wanted[lag], :] = state[observed].T
        if lag < max(lags):
            state = delayed @ state
            state |= zero @ state
    return result


def _bfs(adjacency, starts, ids, allowed=None, target=None):
    """Root-sorted shortest edge paths; predecessor of each reached node."""
    parents = np.full(adjacency.shape[0], -1, dtype=np.int64)
    starts = sorted(set(map(int, starts)), key=lambda i: int(ids[i]))
    queue = deque(starts)
    for i in starts:
        parents[i] = i
    while queue:
        node = queue.popleft()
        if target is not None and node == target:
            break
        neighbors = adjacency.indices[adjacency.indptr[node]:adjacency.indptr[node + 1]]
        if allowed is not None:
            neighbors = neighbors[allowed[neighbors]]
        for child in neighbors[np.argsort(ids[neighbors], kind='stable')]:
            if parents[child] < 0:
                parents[child] = node
                queue.append(int(child))
    return parents


def _path(parents, end):
    result = [int(end)]
    while parents[result[-1]] != result[-1]:
        if parents[result[-1]] < 0:
            raise ValueError('Unreachable path endpoint')
        result.append(int(parents[result[-1]]))
    return result[::-1]


def _path_delay(path, roles):
    return sum(0 if roles[a] == 'KC' and roles[b] == 'MBON' else 1
               for a, b in zip(path, path[1:]))


class _CycleIndex:
    """Shared source SCC/paths across fixed mappings; no graph search."""
    def __init__(self, matrix, roles, ids, observed):
        self.matrix, self.roles = _matrix(matrix, roles)
        self.ids = np.asarray(ids, dtype=np.int64)
        if self.ids.shape != (len(roles),) or len(np.unique(self.ids)) != len(roles):
            raise ValueError('Unique source root IDs required')
        self.observed = np.asarray(observed, dtype=np.int64)
        self.outgoing = self.matrix.T.tocsr()
        self.incoming = self.matrix
        _, self.labels = connected_components(self.outgoing, directed=True, connection='strong')
        sizes = np.bincount(self.labels)
        can_observe = _bfs(self.incoming, self.observed, self.ids) >= 0
        candidates = []
        for component in np.flatnonzero(sizes > 1):
            nodes = np.flatnonzero(self.labels == component)
            anchor = int(nodes[np.argmin(self.ids[nodes])])
            if can_observe[anchor]:
                candidates.append((int(component), int(sizes[component]), anchor))
        self.candidates = sorted(candidates, key=lambda x: (-x[1], int(self.ids[x[2]])))
        self.cache = {}

    def component_paths(self, component, size, anchor):
        if component in self.cache:
            return self.cache[component]
        reverse_parents = _bfs(self.incoming, [anchor], self.ids)
        forward_parents = _bfs(self.outgoing, [anchor], self.ids)
        obs = self.observed[forward_parents[self.observed] >= 0]
        endpoint = int(obs[np.argmin(self.ids[obs])])
        observed_path = _path(forward_parents, endpoint)
        neighbors = self.outgoing.indices[self.outgoing.indptr[anchor]:self.outgoing.indptr[anchor + 1]]
        internal = neighbors[self.labels[neighbors] == component]
        neighbor = int(internal[np.argmin(self.ids[internal])])
        return_parents = _bfs(self.outgoing, [neighbor], self.ids,
                              allowed=self.labels == component, target=anchor)
        cycle = [anchor] + _path(return_parents, anchor)
        result = dict(component_size=size, anchor=anchor,
                      reverse_parents=reverse_parents, cycle=cycle,
                      observed_path=observed_path)
        self.cache[component] = result
        return result

    def witness(self, inputs, dag_bound):
        inputs = np.asarray(inputs, dtype=np.int64)
        if inputs.ndim != 1 or np.any(inputs < 0) or np.any(inputs >= len(self.ids)):
            raise ValueError('Valid one-symbol input indices required')
        reachable = _bfs(self.outgoing, inputs, self.ids) >= 0
        for component, size, anchor in self.candidates:
            if not reachable[anchor]:
                continue
            paths = self.component_paths(component, size, anchor)
            parents = paths['reverse_parents']
            available = inputs[parents[inputs] >= 0]
            first = int(available[np.argmin(self.ids[available])])
            input_path = _path(_bfs(self.outgoing, [first], self.ids, target=anchor), anchor)
            cycle = paths['cycle']
            observed_path = paths['observed_path']
            a = _path_delay(input_path, self.roles)
            b = _path_delay(observed_path, self.roles)
            p = _path_delay(cycle, self.roles)
            if p < 1:
                raise ValueError('A directed cycle must include a delayed edge')
            h = int(dag_bound)
            k = max(1, (h - a - b) // p + 1)
            n = len(self.ids)
            universal_k = max(1, (n - 1 - a - b) // p + 1)
            delay, universal_delay = a + k * p + b, a + universal_k * p + b
            assert delay > h and universal_delay > n - 1
            ids = self.ids
            return dict(certified=True,
                        input_path=ids[input_path].astype(str).tolist(),
                        cycle=ids[cycle].astype(str).tolist(),
                        observed_path=ids[observed_path].astype(str).tolist(),
                        anchor_root_id=str(ids[anchor]), scc_size=size,
                        A=a, B=b, P=p, H=h, k=k, delay=delay, node_count=n,
                        k_same_nodes=universal_k, universal_delay=universal_delay,
                        reason=None)
        return dict(certified=False, input_path=[], cycle=[], observed_path=[],
                    anchor_root_id=None, scc_size=None, A=None, B=None, P=None,
                    H=int(dag_bound), k=None, delay=None, node_count=len(self.ids),
                    k_same_nodes=None, universal_delay=None,
                    reason='No positive-delay cycle lies in this input-to-observed cone')


def cycle_witness(matrix, roles, ids, input_indices, observed, dag_bound):
    """One-symbol compressed walk certificate, or explicit absent obstruction."""
    return _CycleIndex(matrix, roles, ids, observed).witness(input_indices, dag_bound)


def summarize(levels, smoke=False, primary_level='legacy5'):
    """Pure stopping rule; no witness is never automatically feasible."""
    for level, record in levels.items():
        if record['degree_role_status'] not in ('INFEASIBLE', 'UNRESOLVED'):
            raise ValueError('Invalid degree certificate status')
        if record['all_lag_path_status'] not in ('INFEASIBLE', 'UNRESOLVED'):
            raise ValueError('Invalid path certificate status')
        cells = record['cells']
        witnesses = sum(bool(cell['certified']) for cell in cells)
        if record['witness_count'] != witnesses or record['cell_count'] != len(cells):
            raise ValueError('Certificate counts disagree with raw cells')
        if (record['all_lag_path_status'] == 'INFEASIBLE') != bool(witnesses):
            raise ValueError('Path status must follow actual witnesses')
        rules = record['degree_obstruction_rules']
        if (record['degree_role_status'] == 'INFEASIBLE') != bool(rules):
            raise ValueError('Degree status must follow actual sufficient obstructions')
        expected = ('INFEASIBLE' if rules or witnesses else 'UNRESOLVED')
        if record['outcome'] != expected:
            raise ValueError('Level outcome must follow exact obstruction rule')
    if primary_level not in levels:
        raise ValueError('Missing primary level')
    return dict(smoke=bool(smoke), outcome=None if smoke else levels[primary_level]['outcome'],
                primary_level=primary_level, levels=levels, neural_runs=0,
                decoder_fits=0, graph_searches=0, functional_cycle_effect_claim=False,
                biological_plasticity_performed=False,
                exact_all_lag_ideal_only=True, absence_of_obstruction_is_not_feasible=True)


def _csr_hash(matrix):
    return hashlib.sha256(matrix.data.tobytes() + matrix.indices.tobytes() +
                          matrix.indptr.tobytes()).hexdigest()


def _zero_observed(matrix, roles, inputs, observed, c):
    """Analytical zero-prior-state current response, without model.step."""
    first = c['leak'] * np.tanh(inputs)
    mixed = np.zeros_like(first)
    mixed[:, np.asarray(roles) == 'KC'] = first[:, np.asarray(roles) == 'KC']
    return c['leak'] * np.tanh((matrix[observed] @ mixed.T).T + inputs[:, observed])


def analyze_level(c, level, out, resources):
    """Source construction and the registered finite structural panel only."""
    import pandas as pd
    from flying.brain.mushroom_body import KCEncoder
    from flying.data.mushroom_body import load_roles
    from flying.training import whole_brain_memory as core
    from cycle_structure import rank_dag, audit_arrays, graph_structure
    from m4_support import read, write, sha, array_sha, check_manifest

    out.mkdir(parents=True, exist_ok=False)
    small, small_ids, _ = core.load_connectome(Path('data/flywire_783_mb_left_kc512_s701'))
    small_roles, _ = load_roles(Path('data/flywire_783_mb_left_kc512_s701'), small_ids)
    small_ids = np.asarray(small_ids, dtype=np.int64)
    blocks = [b for b in c['blocks'] if b['seed'] in c['blocks_by_level'][level]]
    condition = core.NetworkCondition(level, 701, blocks[0]['input_seed'])
    model, observed, graph_info = core.build_model(c['cache'], condition, c)
    original = model.weights.copy().tocsr()
    del model
    if level == 'legacy5':
        ids, roles = small_ids, np.asarray(small_roles, dtype='U5')
    else:
        with np.load(Path(c['cache']) / 'nodes.npz', allow_pickle=False) as nodes:
            ids, roles = nodes['ids'].copy(), nodes['roles'].astype('U5')
    dag, rank = rank_dag(original, roles, ids)
    sealed_folder = Path('results/tdc_cycles_v1/structural_audit')
    check_manifest(sealed_folder)
    sealed_file = sealed_folder / (level + '-dag.npz')
    with np.load(sealed_file, allow_pickle=False) as a:
        sealed_dag = sparse.csr_matrix((a['data'], a['indices'], a['indptr']),
                                      shape=tuple(a['shape']))
    assert np.array_equal(dag.data, sealed_dag.data)
    assert np.array_equal(dag.indices, sealed_dag.indices)
    assert np.array_equal(dag.indptr, sealed_dag.indptr)
    del sealed_dag
    topology = graph_structure(original, dag, roles, ids, observed)
    arrays = audit_arrays(original, dag, roles, ids, observed)
    np.testing.assert_array_equal(rank, arrays['rank'])
    degrees = degree_certificates(original, roles)
    mismatches = {}
    for suffix, name in [('indegree', 'indegree_mismatch_count'),
                         ('outdegree', 'outdegree_mismatch_count'),
                         ('incoming_abs_weight', 'incoming_mass_mismatch_count'),
                         ('outgoing_abs_weight', 'outgoing_mass_mismatch_count')]:
        mismatches[name] = int(np.count_nonzero(arrays['original_' + suffix] != arrays['dag_' + suffix]))
    mismatches.update(current_block_exact=topology['protected_kc_to_mbon_all_kept_exactly'],
                      degree_equal=not (mismatches['indegree_mismatch_count'] or mismatches['outdegree_mismatch_count']),
                      weight_mass_equal=not (mismatches['incoming_mass_mismatch_count'] or mismatches['outgoing_mass_mismatch_count']),
                      role_blocks_equal=_role_counts(original, roles) == _role_counts(dag, roles),
                      exact_comparison=True)
    write(out / 'topology.json', topology)
    write(out / 'degree-certificates.json', degrees)
    write(out / 'mismatch.json', mismatches)
    np.savez_compressed(out / 'structural-arrays.npz', **arrays)
    write(out / 'source-matrix.json', dict(original_weight_sha256=_csr_hash(original),
                                          dag_weight_sha256=_csr_hash(dag),
                                          root_ids_sha256=array_sha(ids), roles_sha256=array_sha(roles),
                                          observation_indices_sha256=array_sha(observed),
                                          sealed_dag_file_sha256=sha(sealed_file),
                                          source_graph=graph_info,
                                          sealed_m3_mask_exact=True))
    resources.check()
    cycle_index = _CycleIndex(original, roles, ids, observed)
    local = pd.Index(ids).get_indexer(small_ids)
    assert (local >= 0).all()
    cells = []
    for block in blocks:
        encoder = KCEncoder(small_roles, block['input_seed'], c['input_fraction'], c['input_amplitude'])
        inputs = np.zeros((10, len(ids)))
        inputs[:, local] = encoder.patterns
        indices = np.asarray([np.flatnonzero(row) for row in inputs], dtype=np.int64)
        assert indices.shape == (10, 51)
        assert np.all(roles[indices] == 'KC')
        mapping = dict(input_indices=indices,
                       lag_support_original=walk_support(original, roles, indices, observed, c['lags']),
                       lag_support_dag=walk_support(dag, roles, indices, observed, c['lags']),
                       zero_state_original=_zero_observed(original, roles, inputs, observed, c),
                       zero_state_dag=_zero_observed(dag, roles, inputs, observed, c))
        np.testing.assert_array_equal(mapping['zero_state_original'], mapping['zero_state_dag'])
        witnesses = [dict(symbol=symbol, **cycle_index.witness(indices[symbol], topology['global_dependency_depth']))
                     for symbol in range(10)]
        cells.extend(dict(seed=block['seed'], symbol=cell['symbol'], certified=cell['certified'])
                     for cell in witnesses)
        name = 'mapping-s' + str(block['seed'])
        np.savez_compressed(out / (name + '.npz'), **mapping)
        write(out / (name + '.json'), dict(seed=block['seed'], input_seed=block['input_seed'],
                                          input_root_ids=[ids[row].astype(str).tolist() for row in indices],
                                          observation_root_ids=ids[observed].astype(str).tolist(),
                                          array_hashes={key: array_sha(value) for key, value in mapping.items()},
                                          zero_state_current_exact=True,
                                          structural_walk_support_only=True,
                                          current_decoding_measured=False))
        write(out / ('witnesses-s' + str(block['seed']) + '.json'), witnesses)
        resources.check()
    count = sum(cell['certified'] for cell in cells)
    rules = [item['rule'] for item in degrees['obstructions']]
    return dict(degree_role_status=degrees['status'],
                all_lag_path_status='INFEASIBLE' if count else 'UNRESOLVED',
                outcome='INFEASIBLE' if rules or count else 'UNRESOLVED',
                witness_count=count, cell_count=len(cells), cells=cells,
                degree_obstruction_rules=rules,
                node_count=len(ids), reference_global_dependency_bound=topology['global_dependency_depth'])


def main():
    from m4_support import config, source_record, history_preserved, attempt, seal, write
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    c = config(args.smoke)
    with threadpool_limits(limits=1), attempt(args.out, c, 'm4-smoke' if args.smoke else 'm4-main') as resources:
        write(args.out / 'config.json', c)
        source = source_record(c)
        write(args.out / 'source.json', source)
        write(args.out / 'historical-preservation.json', history_preserved(c))
        levels = {level: analyze_level(c, level, args.out / level, resources) for level in c['levels']}
        summary = summarize(levels, args.smoke, c['primary_level'])
        write(args.out / 'summary.json', summary)
        resources.check()
    seal(args.out, dict(complete=True, kind='m4-structural-feasibility', smoke=args.smoke,
                       source_commit=source['source_commit']))
    print(dict(out=args.out.as_posix(), smoke=args.smoke, outcome=summary['outcome'],
               witness_counts={level: result['witness_count'] for level, result in levels.items()},
               neural_runs=0, decoder_fits=0))


if __name__ == '__main__':
    main()
