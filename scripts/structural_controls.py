"""Bounded directed structural controls; rows are postsynaptic neurons."""
from collections import defaultdict
import numpy as np
from scipy import sparse


def generate(raw, roles, family, seed, swaps_per_edge=10):
    raw=raw.tocsr(copy=True);raw.sort_indices();co=raw.tocoo();n=raw.shape[0]
    row=co.row.copy();col=co.col.copy();data=co.data.copy();rng=np.random.default_rng(seed)
    log=dict(family=family,seed=seed,proposals=0,accepted=0,requested=0)
    if family=='intact':return raw,log
    if family=='random':
        # Uniform sampling of distinct directed non-self positions, followed by
        # a random global permutation of the complete signed raw-weight list.
        flat=rng.choice(n*(n-1),len(data),replace=False)
        row=flat//(n-1);col=flat%(n-1);col+=col>=row;data=rng.permutation(data)
    elif family=='weight':
        for post in range(n):
            for sign in [-1,1]:
                ix=np.flatnonzero((row==post)&(np.sign(data)==sign))
                data[ix]=rng.permutation(data[ix])
    elif family in ['degree','role']:
        groups=defaultdict(list)
        for e in range(len(data)):
            key=(int(np.sign(data[e])),)
            if family=='role':key+=(str(roles[col[e]]),str(roles[row[e]]))
            groups[key].append(e)
        pools=[None]*len(data)
        for group in groups.values():
            a=np.asarray(group)
            for e in group:pools[e]=a
        occupied=set(zip(row.tolist(),col.tolist()));need=swaps_per_edge*len(data)
        log['requested']=need
        for _ in range(200*len(data)):
            log['proposals']+=1;i=int(rng.integers(len(data)));pool=pools[i];j=int(pool[rng.integers(len(pool))])
            r,s=int(row[i]),int(row[j]);u,v=int(col[i]),int(col[j])
            if r==s or u==v or r==v or s==u or (r,v) in occupied or (s,u) in occupied:continue
            occupied.remove((r,u));occupied.remove((s,v));occupied.add((r,v));occupied.add((s,u))
            col[i],col[j]=v,u;log['accepted']+=1
            if log['accepted']==need:break
        if log['accepted']!=need:raise RuntimeError(('Incomplete swaps',log))
    else:raise ValueError(family)
    result=sparse.csr_matrix((data,(row,col)),shape=raw.shape);result.sort_indices()
    assert result.nnz==raw.nnz and not result.diagonal().any()
    return result,log


def audit(original,control,roles,family):
    a=original.toarray();b=control.toarray();ap=a!=0;bp=b!=0
    assert a.shape==b.shape and ap.sum()==bp.sum() and not np.diag(b).any()
    np.testing.assert_array_equal(np.sort(a[ap]),np.sort(b[bp]))
    if family in ['intact','weight','degree','role']:
        for r in range(len(a)):
            np.testing.assert_array_equal(np.sort(a[r,ap[r]]),np.sort(b[r,bp[r]]))
    if family in ['intact','weight','degree','role']:
        for sign in [-1,1]:
            np.testing.assert_array_equal((np.sign(a)==sign).sum(0),(np.sign(b)==sign).sum(0))
            np.testing.assert_array_equal((np.sign(a)==sign).sum(1),(np.sign(b)==sign).sum(1))
    if family in ['intact','weight']:np.testing.assert_array_equal(np.sign(a),np.sign(b))
    if family=='role':
        for role in sorted(set(roles)):
            for sign in [-1,1]:
                np.testing.assert_array_equal((np.sign(a[:,roles==role])==sign).sum(1),(np.sign(b[:,roles==role])==sign).sum(1))
                np.testing.assert_array_equal((np.sign(a[roles==role,:])==sign).sum(0),(np.sign(b[roles==role,:])==sign).sum(0))
    return dict(nodes=len(a),edges=int(ap.sum()),common_edges=int((ap&bp).sum()),edge_overlap=float((ap&bp).sum()/ap.sum()),
        changed_weight_positions=int(np.count_nonzero(a!=b)),incoming_l1_changed_nodes=int(np.count_nonzero(abs(a).sum(1)!=abs(b).sum(1))),
        outgoing_l1_changed_nodes=int(np.count_nonzero(abs(a).sum(0)!=abs(b).sum(0))),
        in_degree_changed_nodes=int(np.count_nonzero(ap.sum(1)!=bp.sum(1))),out_degree_changed_nodes=int(np.count_nonzero(ap.sum(0)!=bp.sum(0))))
