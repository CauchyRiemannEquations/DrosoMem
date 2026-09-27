import json
from pathlib import Path
import numpy as np
import pytest
from scipy import sparse
from flying.training.whole_brain_memory import (
    SequenceDataset, NetworkCondition, ExperimentConfig, build_model, collect,
    rollout, decay_probe, prefix_score, write_json, sha256, load_connectome, load_roles,
    KCEncoder, TimedReservoir, normalize_condition,
)


@pytest.fixture
def graph_cache(tmp_path):
    old, ids, _ = load_connectome('data/flywire_783_mb_left_kc512_s701')
    roles,_=load_roles('data/flywire_783_mb_left_kc512_s701',ids)
    # Extra unobserved, unstimulated neurons plus a different root-ID ordering.
    ids=np.asarray(ids,dtype=np.int64); n=len(ids)
    perm=np.arange(n+2)[::-1]
    full=sparse.block_diag((old,sparse.csr_matrix((2,2)))).tocsr()[perm,:][:,perm]
    all_ids=np.r_[ids,[1,2]][perm]; all_roles=np.r_[roles,['OTHER','OTHER']][perm].astype('U5')
    np.savez(tmp_path/'nodes.npz',ids=all_ids,roles=all_roles,left_indices=np.flatnonzero(all_roles!='OTHER'))
    sparse.save_npz(tmp_path/'brain5.npz',full)
    sparse.save_npz(tmp_path/'brain1.npz',full)
    write_json(tmp_path/'provenance.json',dict(sources={'test':'reordered bundled graph'},
               files={p.name:sha256(p) for p in tmp_path.glob('*.npz')}))
    return tmp_path


def config():
    return json.loads(Path('configs/whole_brain_memory.json').read_text())


def test_legacy_matches_existing_dynamics(graph_cache):
    c=config(); condition=NetworkCondition('legacy5',701,1142)
    model,obs,info=build_model(graph_cache,condition,c)
    old,ids,_=load_connectome('data/flywire_783_mb_left_kc512_s701')
    roles,_=load_roles('data/flywire_783_mb_left_kc512_s701',ids)
    weights=normalize_condition(old,'incoming_l1',.9); weights.sort_indices()
    reference=TimedReservoir(weights,KCEncoder(roles,1142,.1,.5),roles,.6,'mbon_after_kc')
    digits=SequenceDataset(length=40).symbols()
    features,_=collect(model,obs,digits,1e-8)
    np.testing.assert_array_equal(features,reference.states(digits)[:,roles=='MBON'])
    assert len(info['observation_root_ids'])==48
    assert all(len(x)==51 for x in info['input_root_ids'])


def test_mapping_survives_reorder_and_extra_nodes(graph_cache):
    c=config(); a,ao,ai=build_model(graph_cache,NetworkCondition('legacy5',701,7142),c)
    b,bo,bi=build_model(graph_cache,NetworkCondition('brain5',701,7142),c)
    assert ai['input_root_ids']==bi['input_root_ids']
    assert ai['observation_root_ids']==bi['observation_root_ids']
    digits=SequenceDataset(length=40).symbols()
    af,_=collect(a,ao,digits,1e-8); bf,_=collect(b,bo,digits,1e-8)
    np.testing.assert_allclose(af,bf,atol=1e-14,rtol=0)
    assert np.count_nonzero(b.encoder.patterns[:, :2])==0


def test_corrupt_graph_rejected(graph_cache):
    with (graph_cache/'brain5.npz').open('ab') as f: f.write(b'corrupt')
    with pytest.raises(ValueError,match='cache changed'):
        build_model(graph_cache,NetworkCondition('brain5',701,7142),config())


def test_autonomous_rollout_has_no_targets():
    class Model:
        def reset(self): self.seen=[]
        def step(self,d): self.seen.append(d); return np.array([d])
    class Head:
        def logits(self,state):
            result=np.zeros((1,10)); result[0,(int(state[0])+1)%10]=5; return result
    model=Model(); got,p,_=rollout(model,np.array([0]),Head(),[3,1,4],5)
    np.testing.assert_array_equal(got,[5,6,7,8,9])
    assert model.seen==[3,1,4,5,6,7,8,9]
    np.testing.assert_allclose(p.sum(axis=1),1)
    assert prefix_score([5,0,7,8,9],got)==1
    assert prefix_score(got,got)==5


def test_decay_restores_state_and_encoder(graph_cache):
    model,obs,_=build_model(graph_cache,NetworkCondition('legacy5',701,7142),config())
    model.step(3); before=model.state.copy()
    full,observed=decay_probe(model,obs,32)
    np.testing.assert_array_equal(before,model.state)
    assert not model.encoder.silent and len(full)==len(observed)==33
    assert full[-1]<full[0]


def test_config_rejects_invalid_budget_and_dataset():
    c=config(); c['eval_length']=200
    with pytest.raises(ValueError): ExperimentConfig(c).validate()
    with pytest.raises(ValueError): SequenceDataset(alphabet_size=2).symbols()
    np.testing.assert_array_equal(SequenceDataset(offset=2,length=4).symbols(),[4,1,5,9])
