"""Pinned v783 annotations + real connectivity; explicitly sampled left MB circuit."""
import json
from pathlib import Path
import urllib.request
import shutil
import numpy as np
import pandas as pd
from flying.data.connectome import SOURCE_SHA256,SOURCE_URL,SOURCE_COMMIT,sha256

ANNOTATION_COMMIT='8587524c1748ce5ef2080822a2fc890fc03bf597'
ANNOTATION_SHA256='9a4f8b2f843196074431ebd7cd883536afa1be86c8a4ce90970441e8be81d1be'
ANNOTATION_URL=f'https://raw.githubusercontent.com/flyconnectome/flywire_annotations/{ANNOTATION_COMMIT}/supplemental_files/Supplemental_file1_neuron_annotations.tsv'


def download_annotations(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        temp=path.with_suffix('.part')
        try:
            with urllib.request.urlopen(ANNOTATION_URL,timeout=120) as r,temp.open('wb') as f:
                shutil.copyfileobj(r,f)
            if sha256(temp)!=ANNOTATION_SHA256:raise ValueError('Annotation checksum mismatch')
            temp.replace(path)
        finally:temp.unlink(missing_ok=True)
    if sha256(path)!=ANNOTATION_SHA256:raise ValueError('Annotation checksum mismatch')


def roles_from_annotations(d):
    """Exact source labels, not substring guesses for cell identity."""
    role=d.cell_class.map({'Kenyon_Cell':'KC','MBON':'MBON','DAN':'DAN'})
    role=role.mask(d.cell_type=='APL','APL')
    return role


def sample_kcs(d,count,seed):
    """Uniform sampling without replacement; preserve type labels for audit."""
    if count<10 or count>len(d):raise ValueError('Invalid KC sample size')
    d=d.sort_values('root_id')
    ix=np.random.default_rng(seed).choice(len(d),count,replace=False)
    return d.iloc[np.sort(ix)]


def build_circuits(raw,annotations,destination,kcs=512,seeds=(701,702),threshold=5):
    import pyarrow.parquet as pq
    if sha256(raw)!=SOURCE_SHA256:raise ValueError('Connectivity checksum mismatch')
    if sha256(annotations)!=ANNOTATION_SHA256:raise ValueError('Annotation checksum mismatch')
    if not 10<=kcs<=2000 or threshold<1:raise ValueError('Invalid CPU subset settings')
    ann=pd.read_csv(annotations,sep='\t',dtype=str).fillna('')
    if ann.root_id.duplicated().any():raise ValueError('Duplicate annotation IDs')
    ann['role']=roles_from_annotations(ann)
    left=ann[(ann.side=='left')&ann.role.notna()].copy()
    pool=set(left.root_id)
    pf=pq.ParquetFile(raw)
    cols=['Presynaptic_ID','Postsynaptic_ID','Connectivity','Excitatory']
    frames=[];seen=set()
    for batch in pf.iter_batches(batch_size=250000,columns=cols):
        d=batch.to_pandas()
        # Never pass root IDs through float.
        for c in cols[:2]:
            d[c]=d[c].astype(str);seen.update(set(d[c])&pool)
        d=d[(d[cols[0]].isin(pool))&(d[cols[1]].isin(pool))&(d.Connectivity>=threshold)&(d[cols[0]]!=d[cols[1]])]
        frames.append(d)
    if pool-seen:raise ValueError(f'Annotation IDs absent from connectivity: {sorted(pool-seen)}')
    edges=pd.concat(frames).rename(columns=dict(zip(cols,['pre','post','count','sign'])))
    core=set(left[left.role.isin(['KC','MBON'])].root_id)
    dan=set(left[left.role=='DAN'].root_id)
    touching=edges[((edges.pre.isin(dan))&(edges.post.isin(core)))|((edges.post.isin(dan))&(edges.pre.isin(core)))]
    retained_dan=(set(touching.pre)|set(touching.post))&dan
    # DAN selection uses the full left KC/MBON pool before sampling, independent of pi.
    support=left[(left.role.isin(['MBON','APL']))|left.root_id.isin(retained_dan)]
    results=[]
    for seed in seeds:
        chosen=pd.concat([sample_kcs(left[left.role=='KC'],kcs,seed),support]).sort_values('root_id')
        ids=chosen.root_id.tolist();selected=set(ids)
        e=edges[edges.pre.isin(selected)&edges.post.isin(selected)].sort_values(['pre','post'])
        if e.duplicated(['pre','post']).any():raise ValueError('Duplicate source pairs')
        if (e.groupby('pre')['sign'].nunique()>1).any():raise ValueError('Inconsistent source sign')
        dest=Path(destination)/f'flywire_783_mb_left_kc{kcs}_s{seed}'
        dest.mkdir(parents=True,exist_ok=False)
        chosen[['root_id','role','cell_class','cell_type','hemibrain_type','side','top_nt','status']].to_csv(dest/'annotations.csv',index=False)
        e.to_csv(dest/'edges.csv',index=False)
        (dest/'neurons.json').write_text(json.dumps(ids,indent=2))
        meta=dict(dataset='FlyWire FAFB',version=783,source='Shiu et al. processed connectivity',
                  source_url=SOURCE_URL,source_sha256=SOURCE_SHA256,source_commit=SOURCE_COMMIT,
                  annotation_url=ANNOTATION_URL,annotation_sha256=ANNOTATION_SHA256,annotation_commit=ANNOTATION_COMMIT,
                  selection='left hemisphere; seeded uniform KC sample + all left MBON/APL + DAN with >=threshold direct edge to full left KC/MBON pool',
                  selection_seed=seed,min_synapses=threshold,neurons=len(ids),edges=len(e),
                  role_counts=chosen.role.value_counts().to_dict(),left_pool_role_counts=left.role.value_counts().to_dict(),
                  dan_excluded_no_core_connection=sorted(dan-retained_dan),unmatched_annotation_ids=[],
                  annotation_status_policy='retain source status flags without assuming they mean failed proofreading',
                  sign='upstream Excitatory column retained; DAN modeled as ordinary signed edges, not modulation',
                  edges_sha256=sha256(dest/'edges.csv'),neurons_sha256=sha256(dest/'neurons.json'),roles_sha256=sha256(dest/'annotations.csv'))
        (dest/'provenance.json').write_text(json.dumps(meta,indent=2));results.append(meta)
    return results


def load_roles(directory,ids):
    p=Path(directory);meta=json.loads((p/'provenance.json').read_text())
    if sha256(p/'annotations.csv')!=meta['roles_sha256']:raise ValueError('Roles checksum mismatch')
    d=pd.read_csv(p/'annotations.csv',dtype=str).fillna('')
    if d.root_id.duplicated().any() or set(d.root_id)!=set(ids):raise ValueError('Role/graph ID mismatch')
    d=d.set_index('root_id').loc[ids]
    if not d.role.isin(['KC','MBON','DAN','APL']).all():raise ValueError('Unknown role')
    if not (roles_from_annotations(d)==d.role).all():raise ValueError('Role label mismatch')
    return d.role.to_numpy(),d
