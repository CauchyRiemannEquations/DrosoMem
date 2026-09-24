import numpy as np
from scipy import sparse
from flying.brain.mushroom_body import KCEncoder, CircuitReservoir, SelectedReadout, ablate, role_shuffled
from flying.data.connectome import load_connectome
from flying.data.mushroom_body import load_roles
from flying.brain.reservoir import Reservoir
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def test_kc_input_and_microstep_propagation():
    roles=np.array(['KC']*10+['MBON','DAN'])
    encoder=KCEncoder(roles,42,fraction=1)
    assert not encoder.patterns[:,10:].any()
    assert np.array_equal(encoder.patterns,KCEncoder(roles,42,fraction=1).patterns)
    w=sparse.csr_matrix(([.7],([10],[0])),shape=(12,12))
    r=CircuitReservoir(w,encoder,microsteps=4)
    one=Reservoir(w,encoder)
    assert one.step(3)[10]==0
    for _ in range(3): expected=one.step(3)
    assert np.allclose(r.step(3),expected)
    assert expected[10]>0

def test_role_shuffle_and_ablation_invariants():
    a,ids,_=load_connectome(ROOT/'data/flywire_783_mb_left_kc512_s701')
    roles,_=load_roles(ROOT/'data/flywire_783_mb_left_kc512_s701',ids)
    b,log=role_shuffled(a,roles,99,swaps_per_edge=1)
    assert b.nnz==a.nnz and not b.diagonal().any()
    for axis in (0,1):
        assert np.array_equal((a!=0).sum(axis=axis),(b!=0).sum(axis=axis))
    for pre in set(roles):
        for post in set(roles):
            ix=np.flatnonzero(roles==post); iy=np.flatnonzero(roles==pre)
            assert a[ix][:,iy].nnz==b[ix][:,iy].nnz
    for j in range(len(roles)):
        assert np.array_equal(np.sort(a[:,j].data),np.sort(b[:,j].data))
    assert log['accepted_swaps']>0 and log['edge_overlap']<1
    for condition in ('no_dan','no_feedback'):
        c=ablate(a,roles,condition).tocoo()
        assert np.array_equal(np.asarray(a[c.row,c.col]).ravel(),c.data)
        if condition=='no_dan': assert not np.any(roles[c.row]=='DAN') and not np.any(roles[c.col]=='DAN')
        else: assert not np.any((roles[c.row]=='KC') & (roles[c.col]!='KC'))
    assert ablate(a,roles,'leaky_only').nnz==0

def test_readout_cannot_observe_unselected_neurons():
    rng=np.random.default_rng(2); x=rng.normal(size=(40,12)); labels=np.arange(40)%10
    readout=SelectedReadout([10,11]); readout.fit(x,labels,epochs=5)
    changed=x.copy(); changed[:,:10]+=1e6
    assert np.array_equal(readout.predict(x),readout.predict(changed))
    assert readout.model.weights.shape==(3,10)
