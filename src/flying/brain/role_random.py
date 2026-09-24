"""Randomize within role blocks, matching the number of plastic parameters."""
import numpy as np
from scipy import sparse

def role_random(a,roles,seed):
    roles=np.asarray(roles);c=a.tocoo();rng=np.random.default_rng(seed);signs=np.ones(len(roles))
    for pre,value in zip(c.col,c.data):signs[pre]=np.sign(value)
    rows=[];cols=[];values=[]
    for source in np.unique(roles):
        pre_ids=np.flatnonzero(roles==source)
        for target in np.unique(roles):
            post_ids=np.flatnonzero(roles==target);mask=(roles[c.col]==source)&(roles[c.row]==target);count=int(mask.sum())
            if not count:continue
            if source==target:
                n=len(pre_ids);slots=rng.choice(n*(n-1),count,replace=False)
                u=slots//(n-1);v=slots%(n-1);v+=v>=u
            else:
                slots=rng.choice(len(pre_ids)*len(post_ids),count,replace=False)
                u=slots//len(post_ids);v=slots%len(post_ids)
            pre=pre_ids[u];post=post_ids[v]
            cols.extend(pre);rows.extend(post);values.extend(rng.permutation(abs(c.data[mask]))*signs[pre])
    return sparse.csr_matrix((values,(rows,cols)),shape=a.shape)
