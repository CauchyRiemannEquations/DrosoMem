"""Stream the pinned author graph; retain its complete indexed node universe."""
import json
from pathlib import Path
import shutil
import urllib.request

import numpy as np
import pandas as pd
from scipy import sparse

from flying.data.connectome import SOURCE_URL, SOURCE_SHA256, sha256, load_connectome
from flying.data.mushroom_body import (ANNOTATION_URL, ANNOTATION_SHA256,
                                      roles_from_annotations)

COMPLETENESS_URL = SOURCE_URL.replace('Connectivity_783.parquet', 'Completeness_783.csv')
COMPLETENESS_SHA256 = 'bbb847a4cc2caaa7a16349722d220c087317b946d148d4d592d94d250617a311'
SOURCES = {'Connectivity_783.parquet': (SOURCE_URL, SOURCE_SHA256),
           'Completeness_783.csv': (COMPLETENESS_URL, COMPLETENESS_SHA256),
           'annotations.tsv': (ANNOTATION_URL, ANNOTATION_SHA256)}


def download_sources(raw):
    raw = Path(raw)
    raw.mkdir(parents=True, exist_ok=True)
    for name, (url, checksum) in SOURCES.items():
        path = raw/name
        if not path.exists():
            part = path.with_suffix('.part')
            try:
                with urllib.request.urlopen(url, timeout=120) as source, part.open('wb') as dest:
                    shutil.copyfileobj(source, dest)
                if sha256(part) != checksum:
                    raise ValueError(f'Source checksum mismatch: {name}')
                part.replace(path)
            finally:
                part.unlink(missing_ok=True)
        if sha256(path) != checksum:
            raise ValueError(f'Source checksum mismatch: {name}')


def validate_batch(d, ids):
    for label in ['Presynaptic', 'Postsynaptic']:
        indices = d[label+'_Index'].to_numpy()
        roots = d[label+'_ID'].to_numpy()
        if indices.dtype.kind not in 'iu' or roots.dtype.kind not in 'iu':
            raise ValueError('Neuron IDs and indices must stay integer')
        if np.any(indices < 0) or np.any(indices >= len(ids)) or not np.array_equal(ids[indices], roots):
            raise ValueError('Source index/root crosswalk mismatch')
    if (not d.Excitatory.isin([-1, 1]).all() or (d.Connectivity <= 0).any()
            or not np.array_equal(d['Excitatory x Connectivity'], d.Excitatory*d.Connectivity)):
        raise ValueError('Invalid source count/sign')


def input_partition(row, selected, threshold=5):
    """Disjoint absolute-contact accounting for one incoming CSR row.

    Caller supplies the row's post index separately in selected['post'].
    Autapses are allocated first, then weak edges, then outside-node edges.
    """
    pre, values = row.indices, row.data
    contacts = np.abs(values).astype(np.int64)
    auto = pre == selected['post']
    weak = (~auto) & (contacts < threshold)
    outside = (~auto) & (~weak) & (~selected['mask'][pre])
    retained = (~auto) & (~weak) & selected['mask'][pre]
    parts = {name: int(contacts[mask].sum()) for name, mask in
             [('autapse', auto), ('weak', weak), ('outside', outside), ('retained', retained)]}
    parts.update(total=int(contacts.sum()), excitatory=int(contacts[values > 0].sum()),
                 inhibitory=int(contacts[values < 0].sum()))
    assert sum(parts[k] for k in ['autapse', 'weak', 'outside', 'retained']) == parts['total']
    return parts


def prepare(raw, cache, repository=Path('.')):
    import pyarrow.parquet as pq
    raw, cache, repository = Path(raw), Path(cache), Path(repository)
    download_sources(raw)
    cache.mkdir(parents=True, exist_ok=True)
    nodes = pd.read_csv(raw/'Completeness_783.csv', index_col=0)
    ids = nodes.index.to_numpy(dtype=np.int64)
    if not nodes.index.is_unique or not nodes.Completed.eq(True).all():
        raise ValueError('Invalid complete node universe')
    ann = pd.read_csv(raw/'annotations.tsv', sep='\t', dtype=str).fillna('')
    if ann.root_id.duplicated().any():
        raise ValueError('Duplicate annotation ID')
    ann['role'] = roles_from_annotations(ann).fillna('OTHER')
    ann = ann.set_index('root_id').reindex(ids.astype(str))
    missing = ann[ann.role.isna()].index.tolist()
    ann = ann.fillna('')
    roles = ann.role.replace('', 'OTHER').to_numpy(dtype='U5')
    left = ((ann.side == 'left') & ann.role.isin(['KC', 'MBON', 'DAN', 'APL'])).to_numpy()
    left_indices = np.flatnonzero(left)
    left_kc = left & (roles == 'KC')
    left_mbon = np.flatnonzero(left & (roles == 'MBON'))
    pf = pq.ParquetFile(raw/'Connectivity_783.parquet')
    size = pf.metadata.num_rows
    pre, post, data = np.empty(size, np.int32), np.empty(size, np.int32), np.empty(size)
    sign_min = np.ones(len(ids), dtype=np.int8)
    sign_max = -sign_min.copy()
    cursor = 0
    columns = ['Presynaptic_ID', 'Postsynaptic_ID', 'Presynaptic_Index', 'Postsynaptic_Index',
               'Connectivity', 'Excitatory', 'Excitatory x Connectivity']
    for batch in pf.iter_batches(batch_size=250000, columns=columns):
        d = batch.to_pandas()
        validate_batch(d, ids)
        end = cursor+len(d)
        pre[cursor:end], post[cursor:end] = d.Presynaptic_Index, d.Postsynaptic_Index
        data[cursor:end] = d['Excitatory x Connectivity']
        np.minimum.at(sign_min, pre[cursor:end], d.Excitatory)
        np.maximum.at(sign_max, pre[cursor:end], d.Excitatory)
        cursor = end
    if cursor != size or np.any(sign_min < sign_max):
        raise ValueError('Inconsistent source rows/signs')
    full = sparse.csc_matrix((data, (post, pre)), shape=(len(ids), len(ids)))
    if full.nnz != size:
        raise ValueError('Duplicate source pairs')
    full.sort_indices()
    del pre, post, data
    incoming = full.tocsr()
    coverage = []
    for seed in [701, 702]:
        old, old_ids, _ = load_connectome(repository/f'data/flywire_783_mb_left_kc512_s{seed}')
        index = pd.Index(ids).get_indexer(np.asarray(old_ids, dtype=np.int64))
        if np.any(index < 0):
            raise ValueError('Bundled node absent from source')
        mask = np.zeros(len(ids), bool)
        mask[index] = True
        induced = full[index, :][:, index].tocsr()
        induced.setdiag(0)
        induced.data[np.abs(induced.data) < 5] = 0
        induced.eliminate_zeros()
        if (old != induced).nnz:
            raise ValueError('Bundled graph differs from source-induced graph')
        for post_index in left_mbon:
            row = incoming[post_index:post_index+1]
            part = input_partition(row, dict(post=post_index, mask=mask))
            part.update(circuit_seed=seed, root_id=str(ids[post_index]),
                        cell_type=ann.iloc[post_index].cell_type,
                        full_left_kc_contacts=int(np.abs(row.data[left_kc[row.indices]]).sum()),
                        retained_kc_contacts=int(np.abs(row.data[(roles[row.indices] == 'KC')
                            & mask[row.indices] & (np.abs(row.data) >= 5)
                            & (row.indices != post_index)]).sum()))
            coverage.append(part)
    autapses = full.diagonal()
    metadata = dict(source_rows=size, neurons=len(ids),
                    autapse_edges=int(np.count_nonzero(autapses)),
                    autapse_contacts=int(np.abs(autapses).sum()),
                    missing_annotation_ids=missing, left_neurons=len(left_indices),
                    role_counts=pd.Series(roles).value_counts().to_dict(),
                    left_role_counts=pd.Series(roles[left]).value_counts().to_dict(),
                    bundled_induced_graphs_exact=True,
                    sources={k: {'url': v[0], 'sha256': v[1]} for k, v in SOURCES.items()})
    del incoming
    full.setdiag(0)
    full.eliminate_zeros()
    full.sort_indices()
    sparse.save_npz(cache/'brain1.npz', full)
    metadata['brain1_edges'] = full.nnz
    full.data[np.abs(full.data) < 5] = 0
    full.eliminate_zeros()
    sparse.save_npz(cache/'brain5.npz', full)
    metadata['brain5_edges'] = full.nnz
    left_w = full[left_indices, :][:, left_indices].tocsc()
    sparse.save_npz(cache/'left5.npz', left_w)
    metadata['left5_edges'] = left_w.nnz
    np.savez_compressed(cache/'nodes.npz', ids=ids, roles=roles, left_indices=left_indices)
    pd.DataFrame(coverage).to_csv(cache/'coverage.csv', index=False)
    metadata['files'] = {p.name: sha256(p) for p in sorted(cache.iterdir()) if p.suffix in ['.npz', '.csv']}
    (cache/'provenance.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    return metadata


def load_graph(cache, level, seed, repository=Path('.'), encoder_seed=6142):
    from flying.brain.mushroom_body import KCEncoder
    from flying.data.mushroom_body import load_roles
    from types import SimpleNamespace
    cache, repository = Path(cache), Path(repository)
    meta = json.loads((cache/'provenance.json').read_text())
    for name in ['nodes.npz']+([] if level == 'legacy5' else [level+'.npz']):
        if sha256(cache/name) != meta['files'][name]:
            raise ValueError('Graph cache checksum mismatch')
    old_path = repository/f'data/flywire_783_mb_left_kc512_s{seed}'
    old, old_ids, _ = load_connectome(old_path)
    old_roles, _ = load_roles(old_path, old_ids)
    old_ids = np.asarray(old_ids, dtype=np.int64)
    original_patterns = KCEncoder(old_roles, seed=encoder_seed).patterns != 0
    with np.load(cache/'nodes.npz', allow_pickle=False) as nodes:
        ids, roles = nodes['ids'], nodes['roles']
        if level == 'legacy5':
            ids, roles, weights = old_ids, old_roles.astype('U5'), old.tocsc()
        elif level in ['left5', 'brain5', 'brain1']:
            if level == 'left5':
                ids, roles = ids[nodes['left_indices']], roles[nodes['left_indices']]
            weights = sparse.load_npz(cache/(level+'.npz'))
        else:
            raise ValueError('Unknown graph level')
    indices = pd.Index(ids).get_indexer(old_ids)
    if np.any(indices < 0):
        raise ValueError('Input nodes absent from larger graph')
    patterns = np.zeros((10, len(ids)), bool)
    patterns[:, indices] = original_patterns
    return weights, ids, roles, SimpleNamespace(patterns=patterns)
