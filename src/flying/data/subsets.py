"""Seeded connected subsets of the verified v783 graph; no sequence access."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from flying.data.connectome import SOURCE_SHA256, SOURCE_URL, SOURCE_COMMIT, sha256


def connected_indices(a, neurons, seed):
    """Randomized breadth-first growth in the largest weak component.

    Connectivity refers to the undirected support, not strong connectivity.
    Sorted support and an explicit RNG make row-storage order irrelevant.
    """
    if not 2 <= neurons <= a.shape[0]:
        raise ValueError('Invalid subset size')
    support = ((a != 0) + (a.T != 0)).astype(bool).tocsr()
    support.sort_indices()
    _, labels = connected_components(support, directed=False)
    largest = np.flatnonzero(labels == np.argmax(np.bincount(labels)))
    if len(largest) < neurons:
        raise ValueError('Largest weak component too small')
    rng = np.random.default_rng(seed)
    start = int(rng.choice(largest))
    queue = [start]; seen = {start}; cursor = 0
    while len(queue) < neurons:
        current = queue[cursor]; cursor += 1
        neighbors = support.indices[support.indptr[current]:support.indptr[current+1]]
        for nxt in rng.permutation(neighbors):
            nxt = int(nxt)
            if nxt not in seen:
                seen.add(nxt); queue.append(nxt)
                if len(queue) == neurons:
                    break
    return np.array(sorted(queue)), start


def build_subsets(raw, destination, neurons=300, seeds=(101,202,303), min_synapses=5):
    import pyarrow.parquet as pq
    if not 10 <= neurons <= 3000 or min_synapses < 1:
        raise ValueError('Require 10..3000 neurons and a positive threshold')
    if sha256(raw) != SOURCE_SHA256:
        raise ValueError('Unexpected source hash')
    pf = pq.ParquetFile(raw)
    cols = ['Presynaptic_ID','Postsynaptic_ID','Connectivity','Excitatory']
    chunks = []
    for batch in pf.iter_batches(batch_size=250000, columns=cols):
        d = batch.to_pandas()
        keep = (d.Connectivity >= min_synapses) & (d.Presynaptic_ID != d.Postsynaptic_ID)
        chunks.append(d.loc[keep,cols].to_numpy(dtype=np.int64))
    data = np.concatenate(chunks); del chunks
    ids = np.unique(data[:,:2].ravel())
    pre = np.searchsorted(ids,data[:,0]); post = np.searchsorted(ids,data[:,1])
    a = sparse.csr_matrix((data[:,2]*data[:,3],(post,pre)),shape=(len(ids),len(ids)))
    if a.nnz != len(data):
        raise ValueError('Unexpected duplicate source pair')
    del data, pre, post
    results = []
    for seed in seeds:
        selected, start = connected_indices(a, neurons, seed)
        sub = a[selected][:,selected].tocoo(); subids = ids[selected]
        dest = Path(destination)/f'flywire_783_connected_{neurons}_s{seed}'
        dest.mkdir(parents=True,exist_ok=False)
        edges = pd.DataFrame(dict(pre=subids[sub.col],post=subids[sub.row],
                                  count=np.abs(sub.data),sign=np.sign(sub.data)))
        if (edges.groupby('pre')['sign'].nunique()>1).any():
            raise ValueError('Inconsistent source sign')
        edges.sort_values(['pre','post']).to_csv(dest/'edges.csv',index=False)
        (dest/'neurons.json').write_text(json.dumps([str(i) for i in subids],indent=2))
        meta = dict(dataset='FlyWire FAFB',version=783,source='Shiu et al. processed connectivity',
                    source_url=SOURCE_URL,source_commit=SOURCE_COMMIT,source_sha256=SOURCE_SHA256,
                    source_rows=pf.metadata.num_rows,neurons=len(subids),edges=sub.nnz,
                    selection='randomized breadth-first growth in largest weak component; induced subgraph',
                    selection_seed=seed,start_root_id=str(ids[start]),
                    min_synapses=min_synapses,autapses='removed',sign='upstream Excitatory column; model assumption',
                    edges_sha256=sha256(dest/'edges.csv'),neurons_sha256=sha256(dest/'neurons.json'))
        (dest/'provenance.json').write_text(json.dumps(meta,indent=2))
        results.append(meta)
    return results
