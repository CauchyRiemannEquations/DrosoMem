import json
from pathlib import Path
import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from threadpoolctl import threadpool_limits
from flying.brain.diagnostics import normalize_condition,graph_diagnostics,state_diagnostics
from flying.brain.reservoir import Reservoir
from flying.encoding.digit_encoder import DigitEncoder
from flying.data.subsets import connected_indices
from flying.data.connectome import load_connectome
from flying.training.phase2 import validate,run_phase2

ROOT=Path(__file__).resolve().parents[1]


def test_incoming_normalization_preserves_edges_signs_and_contraction():
    a=sparse.csr_matrix([[0,2,-4],[1,0,3],[0,0,0]],dtype=float)
    w=normalize_condition(a,'incoming_l1',.9)
    assert np.array_equal(np.sign(a.toarray()),np.sign(w.toarray()))
    np.testing.assert_allclose(abs(w).sum(axis=1).A.ravel(),[.9,.9,0])
    assert np.max(abs(np.linalg.eigvals(w.toarray())))<=.9+1e-12
    np.testing.assert_allclose(w.toarray()[0,1]/w.toarray()[0,2],-.5)
    with pytest.raises(ValueError):normalize_condition(a,'incoming_l1',1.2)


def test_connected_sampling_is_seeded_induced_and_without_fake_edges():
    rng=np.random.default_rng(1)
    a=sparse.csr_matrix(rng.random((60,60))<.05)
    a=a+sparse.diags(np.ones(59),1,shape=(60,60))
    x,start=connected_indices(a,20,101)
    y,_=connected_indices(a,20,101)
    np.testing.assert_array_equal(x,y)
    assert len(x)==len(np.unique(x))==20 and start in x
    assert connected_components(a[x][:,x],directed=False)[0]==1
    z,_=connected_indices(a,20,202)
    assert not np.array_equal(x,z)


def test_diagnostics_known_graph_and_rank():
    a=sparse.csr_matrix([[0,1,0],[1,0,0],[0,0,0]])
    d=graph_diagnostics(a)
    assert d['isolates']==1 and d['largest_strong']==2 and d['weak_components']==2
    assert state_diagnostics(np.ones((10,3)))['effective_rank']==0
    rank=state_diagnostics(np.arange(10)[:,None]*np.array([[1,2,3]]))['effective_rank']
    assert rank==pytest.approx(1,abs=1e-10)


def test_memoryless_and_leaky_controls_are_distinct():
    enc=DigitEncoder(20)
    zero=sparse.csr_matrix((20,20))
    mem=Reservoir(zero,enc,1).states([1,4,1])
    np.testing.assert_array_equal(mem[0],mem[2])
    leaky=Reservoir(zero,enc,.6).states([1,4,1])
    assert not np.array_equal(leaky[0],leaky[2])


def test_real_connected_subsets():
    for seed in [101,202,303]:
        a,ids,meta=load_connectome(ROOT/f'data/flywire_783_connected_300_s{seed}')
        assert len(ids)==300 and meta['selection_seed']==seed
        assert connected_components(a,directed=False)[0]==1
        assert graph_diagnostics(a)['largest_strong']>1


def test_phase2_rejects_overlapping_test_block():
    c=json.loads((ROOT/'configs/phase2.json').read_text())
    c['heldout_start']=200
    with pytest.raises(ValueError):validate(c)


def test_phase2_reproducibility_and_holdout_leakage(tmp_path,monkeypatch):
    import flying.training.phase2 as mod
    c=json.loads((ROOT/'configs/phase2.json').read_text())
    c.update(graphs={'tiny':'data/flywire_783_subset'},models=['fly','leaky_only','memoryless'],
             normalizations=['incoming_l1'],seeds=[142],train_lengths=[20],heldout_start=30,heldout_digits=5)
    c['training']['epochs']=10
    monkeypatch.setattr(mod,'plot_phase2',lambda *args:None)
    with threadpool_limits(limits=1):
        first=run_phase2(c,tmp_path/'one',ROOT)
        second=run_phase2(c,tmp_path/'two',ROOT)
        original=mod.pi_digits(35);original[30:]=(original[30:]+1)%10
        monkeypatch.setattr(mod,'pi_digits',lambda n:original[:n])
        altered=run_phase2(c,tmp_path/'altered',ROOT)
    import pandas as pd
    pd.testing.assert_frame_equal(first.drop(columns='seconds'),second.drop(columns='seconds'),check_exact=True)
    for col in ['train_accuracy','pi_memory_score','effective_rank']:
        np.testing.assert_array_equal(first[col],altered[col])
    assert (tmp_path/'one/recall.jsonl').read_bytes()==(tmp_path/'altered/recall.jsonl').read_bytes()
    assert (tmp_path/'one/training.csv').read_bytes()==(tmp_path/'altered/training.csv').read_bytes()

    for folder in ['one','two','altered']:
        for filename in ['metrics.jsonl','recall.jsonl']:
            records=[json.loads(line) for line in (tmp_path/folder/filename).read_text().splitlines()]
            assert len(records)==3
            assert {r['model'] for r in records}=={'fly','leaky_only','memoryless'}
