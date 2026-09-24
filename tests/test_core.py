from pathlib import Path
import json
import numpy as np
import pytest
from scipy import sparse
from threadpoolctl import threadpool_limits
from flying.data.pi_digits import pi_digits
from flying.data.connectome import load_connectome
from flying.encoding.digit_encoder import DigitEncoder
from flying.brain.reservoir import Reservoir, normalize
from flying.brain.shuffled_network import shuffled_network
from flying.brain.random_network import random_network
from flying.models.readout import Readout
from flying.evaluation.free_recall import generate, evaluate_recall
from flying.evaluation.metrics import pi_memory_score

ROOT = Path(__file__).resolve().parents[1]


def test_pi_and_precision():
    expected = '314159265358979323846264338327950288419716939937510'
    assert ''.join(map(str, pi_digits(len(expected)))) == expected
    assert np.array_equal(pi_digits(1000)[:len(expected)], pi_digits(len(expected)))


@pytest.mark.parametrize('target,pred,score', [([1,2,3],[0,2,3],0),([1,2,3],[1,2,0],2),([1,2],[1,2],2),([],[],0)])
def test_score(target, pred, score):
    assert pi_memory_score(target,pred) == score


def test_bad_length_not_censored_success():
    with pytest.raises(ValueError):
        pi_memory_score([1,2],[1])


def test_encoding_reproducible_equal_population():
    a = DigitEncoder(100,42); b = DigitEncoder(100,42)
    np.testing.assert_array_equal(a.patterns, b.patterns)
    assert len(np.unique(a.patterns,axis=0)) == 10
    assert np.all(np.count_nonzero(a.patterns,axis=1) == 10)
    with pytest.raises(ValueError): a(-1)


def test_direction_and_previous_state():
    a = sparse.csr_matrix(([.8],([1],[0])),shape=(10,10))
    enc = DigitEncoder(10)
    r = Reservoir(a, enc, leak=1)
    r.state[0] = 1
    expected = np.tanh(a @ r.state + enc(4))
    np.testing.assert_allclose(r.step(4), expected)
    assert r.state[1] > 0
    r.reset(); fresh = r.step(4)
    assert fresh[1] < expected[1]


def test_temporal_causality_and_reset():
    enc = DigitEncoder(20)
    r = Reservoir(sparse.eye(20)*.8, enc)
    a = r.states([3,1,4,1,5])
    b = r.states([3,1,4,9,9])
    np.testing.assert_array_equal(a[:3],b[:3])
    assert not np.allclose(a[1], a[3])  # same input 1, distinct histories
    np.testing.assert_array_equal(a,r.states([3,1,4,1,5]))


def test_real_ids_weights_and_controls():
    a, ids, meta = load_connectome(ROOT/'data/flywire_783_subset')
    assert len(ids) == 300 and all(len(i) == 18 for i in ids)
    assert a.nnz == 3303 and meta['version'] == 783
    d = __import__('pandas').read_csv(ROOT/'data/flywire_783_subset/edges.csv',dtype={'pre':str,'post':str})
    row = d.iloc[0]
    assert a[ids.index(row.post),ids.index(row.pre)] == row['count'] * row['sign']
    b, log = shuffled_network(a,42,2)
    assert b.nnz == a.nnz and log['accepted_swaps'] == 2*a.nnz
    assert log['edge_overlap'] < .9
    for axis in [0,1]:
        np.testing.assert_array_equal((a != 0).sum(axis=axis),(b != 0).sum(axis=axis))
    np.testing.assert_array_equal(abs(a).sum(axis=0),abs(b).sum(axis=0))
    c = random_network(a,42)
    assert c.nnz == a.nnz and c.shape == a.shape and not c.diagonal().any()
    np.testing.assert_array_equal(np.sort(abs(c.data)),np.sort(abs(a.data)))
    with threadpool_limits(limits=1):
        scaled, rho = normalize(a)
        assert np.max(abs(np.linalg.eigvals(scaled.toarray()))) == pytest.approx(.9,abs=1e-8)


class Recorder:
    def __init__(self): self.inputs=[]
    def reset(self): self.inputs=[]
    def step(self,d): self.inputs.append(d); return np.array([d])

class PlusOne:
    def predict(self,x): return np.array([(int(x[0])+1)%10])


def test_free_recall_uses_own_output_and_excludes_prompt():
    r=Recorder()
    pred=generate(r,PlusOne(),[3,1,4],4)
    assert pred.tolist() == [5,6,7,8]
    assert r.inputs == [3,1,4,5,6,7,8]
    score=evaluate_recall(r,PlusOne(),np.array([3,1,4,5,6,0,8]),3,4)
    assert score['pi_memory_score'] == 2
    assert score['first_error_digit_index'] == 5
    assert not score['censored']


def test_readout_training_never_changes_reservoir():
    r=Reservoir(sparse.eye(20)*.8,DigitEncoder(20))
    digits=pi_digits(20); states=r.states(digits[:-1]); before=r.weights.copy()
    ro=Readout(); ro.fit(states,digits[1:],epochs=20)
    assert (r.weights != before).nnz == 0
    assert ro.predict(states).shape == (19,)
    np.testing.assert_allclose(ro.mean,states.mean(axis=0))


def test_heldout_suffix_cannot_change_fit(tmp_path, monkeypatch):
    import flying.training.train as training
    config = json.loads((ROOT/'configs/mvp.json').read_text())
    config.update(train_digits=20,heldout_digits=5,seeds=[42],models=['fly'])
    config['training']['epochs']=10
    for name in ['plot_activity','plot_memory','plot_training','plot_comparison']:
        monkeypatch.setattr(training,name,lambda *args: None)
    original = pi_digits(25)
    with threadpool_limits(limits=1):
        training.run(config,tmp_path/'first',ROOT)
        changed=original.copy();changed[20:]=(changed[20:]+1)%10
        monkeypatch.setattr(training,'pi_digits',lambda length:changed[:length])
        training.run(config,tmp_path/'second',ROOT)
    a=np.load(tmp_path/'first/fly_seed42/readout.npz')
    b=np.load(tmp_path/'second/fly_seed42/readout.npz')
    for key in a.files:
        np.testing.assert_array_equal(a[key],b[key])
    assert json.loads((tmp_path/'first/fly_seed42/recall.json').read_text()) == json.loads((tmp_path/'second/fly_seed42/recall.json').read_text())
