"""Graph-only definitions for the registered edge intervention panel."""
from collections import Counter, deque
import numpy as np


def edge_betweenness(raw):
    """Unnormalized directed Brandes counts; row=post, column=pre."""
    co=raw.tocoo();n=raw.shape[0];adj=[[] for _ in range(n)]
    for e,(post,pre) in enumerate(zip(co.row,co.col)):adj[pre].append((post,e))
    score=np.zeros(raw.nnz)
    for source in range(n):
        pred=[[] for _ in range(n)];sigma=np.zeros(n);sigma[source]=1
        distance=np.full(n,-1,int);distance[source]=0;queue=deque([source]);stack=[]
        while queue:
            u=queue.popleft();stack.append(u)
            for v,e in adj[u]:
                if distance[v]<0:distance[v]=distance[u]+1;queue.append(v)
                if distance[v]==distance[u]+1:sigma[v]+=sigma[u];pred[v].append((u,e))
        delta=np.zeros(n)
        for v in reversed(stack):
            for u,e in pred[v]:
                contribution=sigma[u]/sigma[v]*(1+delta[v])
                score[e]+=contribution;delta[u]+=contribution
    return score


def select_masks(raw,ids,roles,centrality,block,c):
    co=raw.tocoo();count=raw.nnz;q=int(np.floor(c['edge_fraction']*count))
    dan=(roles[co.row]=='DAN')|(roles[co.col]=='DAN');within=roles[co.row]==roles[co.col]
    masks=dict(intact=np.zeros(count,bool),DAN=dan)
    keys=list(zip(np.sign(co.data).astype(int).tolist(),np.digitize(abs(co.data),c['weight_boundaries']).tolist()))
    histogram=Counter(keys[i] for i in np.where(dan)[0])
    for j,seed in enumerate(block['edge_seeds']['dan_control']):
        rng=np.random.default_rng(seed);mask=np.zeros(count,bool)
        for key in sorted(histogram):
            pool=np.array([i for i,k in enumerate(keys) if k==key])
            mask[rng.choice(pool,histogram[key],replace=False)]=True
        assert Counter(keys[i] for i in np.where(mask)[0])==histogram
        masks[f'dan_control{j}']=mask
    tie=lambda i:(int(ids[co.col[i]]),int(ids[co.row[i]]))
    for name,values in [('weak',abs(co.data)),('strong',-abs(co.data)),('betweenness',-np.round(centrality,6))]:
        order=sorted(range(count),key=lambda i:(float(values[i]),*tie(i)))
        mask=np.zeros(count,bool);mask[order[:q]]=True;masks[name]=mask
    for family,pool in [('uniform',np.arange(count)),('within',np.where(within)[0]),('between',np.where(~within)[0])]:
        if len(pool)<q:raise ValueError('Insufficient registered edge pool')
        for j,seed in enumerate(block['edge_seeds'][family]):
            mask=np.zeros(count,bool);mask[np.random.default_rng(seed).choice(pool,q,replace=False)]=True;masks[f'{family}{j}']=mask
    return masks


def cut_edges(weights,mask):
    """Same CSR enumeration as raw; no normalization or input edits."""
    if mask.shape!=(weights.nnz,) or mask.dtype!=bool:raise ValueError('Invalid edge mask')
    out=weights.copy();out.data[mask]=0;out.eliminate_zeros();out.sort_indices();return out
