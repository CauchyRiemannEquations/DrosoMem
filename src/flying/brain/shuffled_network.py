"""Directed double edge swaps: preserve exact in/out degrees, N, E.

Weights follow their presynaptic source; outgoing weight multisets are retained.
Incoming strengths and mixing are not guaranteed to be preserved.
"""
import numpy as np
from scipy import sparse

def shuffled_network(a, seed, swaps_per_edge=10):
    coo = a.tocoo(); pre = coo.col.copy(); post = coo.row.copy()
    rng = np.random.default_rng(seed)
    edges = set(zip(pre.tolist(), post.tolist()))
    requested = swaps_per_edge * len(pre); accepted = 0; attempts = 0
    if len(pre) < 2:
        raise ValueError("At least two edges required")
    while accepted < requested and attempts < requested * 30:
        attempts += 1
        i, j = rng.choice(len(pre), 2, replace=False)
        u, v, x, y = int(pre[i]), int(post[i]), int(pre[j]), int(post[j])
        if u == x or v == y or u == y or x == v or (u, y) in edges or (x, v) in edges:
            continue
        edges.remove((u, v)); edges.remove((x, y))
        edges.add((u, y)); edges.add((x, v))
        post[i], post[j] = y, v
        accepted += 1
    out = sparse.csr_matrix((coo.data.copy(), (post, pre)), shape=a.shape)
    overlap = len(edges & set(zip(coo.col.tolist(), coo.row.tolist()))) / len(pre)
    return out, dict(requested_swaps=requested, accepted_swaps=accepted, attempts=attempts, edge_overlap=overlap)
