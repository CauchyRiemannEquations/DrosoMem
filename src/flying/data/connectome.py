"""Verified Shiu/FlyWire v783 adapter; matrices are W[post, pre]."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse

SOURCE_COMMIT = "91bdd1e7dcf193f3e7ca5a8933497fcef63b7960"
SOURCE_URL = f"https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/{SOURCE_COMMIT}/Connectivity_783.parquet"
SOURCE_SHA256 = "efeb23fb99098e9c390f6869969b2a121a2ee92c833cfc45ecb2c1d8e1af0347"

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()

def extract_subset(path, destination, neurons=300, min_synapses=5):
    """Two batch passes: rank total incident retained synapses; induce subgraph.

    IDs stay int64 in memory and decimal strings on disk. Selection does not
    inspect π or outcomes. This is a hub-biased subset, NOT an anatomical circuit.
    """
    import pyarrow.parquet as pq
    if neurons < 10 or neurons > 3000 or min_synapses < 1:
        raise ValueError("Use 10..3000 neurons and positive synapse threshold")
    if sha256(path) != SOURCE_SHA256:
        raise ValueError("Source checksum mismatch; do not silently change dataset")
    pf = pq.ParquetFile(path)
    cols = ["Presynaptic_ID", "Postsynaptic_ID", "Connectivity", "Excitatory"]
    strength = pd.Series(dtype=float)
    for batch in pf.iter_batches(batch_size=250000, columns=cols):
        d = batch.to_pandas()
        d = d[(d.Connectivity >= min_synapses) & (d.Presynaptic_ID != d.Postsynaptic_ID)]
        for c in cols[:2]:
            strength = strength.add(d.groupby(c).Connectivity.sum(), fill_value=0)
    ranked = sorted(strength.index, key=lambda i: (-strength.loc[i], int(i)))
    ids = np.array(sorted(ranked[:neurons]), dtype=np.int64)
    frames = []
    for batch in pf.iter_batches(batch_size=250000, columns=cols):
        d = batch.to_pandas()
        d = d[d.Presynaptic_ID.isin(ids) & d.Postsynaptic_ID.isin(ids)
              & (d.Connectivity >= min_synapses) & (d.Presynaptic_ID != d.Postsynaptic_ID)]
        frames.append(d)
    edges = pd.concat(frames).rename(columns={cols[0]: "pre", cols[1]: "post", cols[2]: "count", cols[3]: "sign"})
    if edges.duplicated(["pre", "post"]).any():
        raise ValueError("Unexpected duplicate source pairs")
    if (edges.groupby('pre')['sign'].nunique() > 1).any():
        raise ValueError("Inconsistent outgoing signs")
    edges = edges.sort_values(["pre", "post"])
    dest = Path(destination); dest.mkdir(parents=True, exist_ok=True)
    edges.to_csv(dest / "edges.csv", index=False)
    (dest / "neurons.json").write_text(json.dumps([str(i) for i in ids], indent=2))
    meta = dict(dataset="FlyWire FAFB", version=783, source="Shiu et al. processed connectivity",
                source_url=SOURCE_URL, source_commit=SOURCE_COMMIT, source_sha256=SOURCE_SHA256,
                source_rows=pf.metadata.num_rows, neurons=len(ids), edges=len(edges),
                selection="top total incident synapse strength after threshold; induced subgraph",
                min_synapses=min_synapses, autapses="removed", sign="upstream Excitatory column; model assumption",
                edges_sha256=sha256(dest / "edges.csv"), neurons_sha256=sha256(dest / "neurons.json"))
    (dest / "provenance.json").write_text(json.dumps(meta, indent=2))
    return meta

def load_connectome(directory):
    p = Path(directory)
    meta = json.loads((p / "provenance.json").read_text())
    for name in ["edges", "neurons"]:
        file = p / (name + (".csv" if name == "edges" else ".json"))
        if sha256(file) != meta[name + "_sha256"]:
            raise ValueError(f"{name} checksum mismatch")
    ids = json.loads((p / "neurons.json").read_text())
    index = {v: i for i, v in enumerate(ids)}
    d = pd.read_csv(p / "edges.csv", dtype={"pre": str, "post": str})
    if d.duplicated(["pre", "post"]).any() or not d['sign'].isin([-1, 1]).all() or (d['count'] <= 0).any():
        raise ValueError("Invalid edge table")
    pre = d.pre.map(index); post = d.post.map(index)
    if pre.isna().any() or post.isna().any():
        raise ValueError("Unknown neuron ID")
    a = sparse.csr_matrix((d['count'].to_numpy(float) * d['sign'], (post, pre)), shape=(len(ids), len(ids)))
    return a, ids, meta
