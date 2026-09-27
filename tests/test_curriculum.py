import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flying.models.nonlinear_readout import NonlinearReadout
from flying.training import phase5_curriculum as curriculum


def data():
    rng = np.random.default_rng(7)
    return rng.normal(size=(12,4)), np.arange(12)%10


def test_constant_schedule_is_exactly_static_training():
    states, labels = data(); weights = np.arange(1,13)
    a, b = [NonlinearReadout([0,1,2], seed=31) for _ in range(2)]
    sa, ha = a.fit(states, labels, epochs=6, checkpoints=(2,4,6), sample_weight=weights)
    sb, hb = b.fit(states, labels, epochs=6, checkpoints=(2,4,6),
                   sample_weight_schedule=[(2,weights),(4,weights),(6,weights)])
    assert ha == hb
    assert all(sa[e].digest() == sb[e].digest() for e in sa)


def test_schedule_boundaries_and_continuous_adam_match_manual_updates():
    states, labels = data(); weights = [np.ones(12), np.arange(1,13), np.arange(12,0,-1)]
    model = NonlinearReadout([0,1,2], seed=31)
    saved, _ = model.fit(states, labels, epochs=6, learning_rate=.02, checkpoints=(2,4,6),
                         sample_weight_schedule=list(zip([2,4,6],weights)))
    reference = NonlinearReadout([0,1,2], seed=31); reference.initialize(states)
    moments = {k:np.zeros_like(v) for k,v in reference.parameters.items()}
    variances = {k:v.copy() for k,v in moments.items()}
    for step in range(1,7):
        _, gradients, _ = reference.objective(states, labels, sample_weight=weights[(step-1)//2])
        for key, parameter in reference.parameters.items():
            moments[key] = .9*moments[key] + .1*gradients[key]
            variances[key] = .999*variances[key] + .001*gradients[key]**2
            parameter -= .02*(moments[key]/(1-.9**step))/(np.sqrt(variances[key]/(1-.999**step))+1e-8)
        if step in saved:
            assert saved[step].digest() == reference.digest()
    np.testing.assert_array_equal(model.mean, states[:,:3].mean(axis=0))


@pytest.mark.parametrize('schedule', [[], [(5,np.ones(12))], [(0,np.ones(12)),(6,np.ones(12))],
    [(4,np.ones(12)),(3,np.ones(12)),(6,np.ones(12))], [(6,np.zeros(12))],
    [(6,np.ones(11))], [(6,np.full(12,np.nan))]])
def test_invalid_schedules_fail_before_initializing(schedule):
    states, labels = data(); model = NonlinearReadout([0,1])
    with pytest.raises(ValueError):
        model.fit(states, labels, epochs=6, checkpoints=(6,), sample_weight_schedule=schedule)
    assert not hasattr(model, 'parameters')


def test_static_weights_and_schedule_are_mutually_exclusive():
    states, labels = data()
    with pytest.raises(ValueError):
        NonlinearReadout([0,1]).fit(states, labels, epochs=6, checkpoints=(6,),
            sample_weight=np.ones(12), sample_weight_schedule=[(6,np.ones(12))])


def config_file(tmp_path):
    cfg = json.loads(Path('configs/phase5_curriculum.json').read_text())
    cfg.update(circuits=cfg['circuits'][:1], seeds=[2142], initializations=[0], offsets=[0], stage_epochs=1)
    path = tmp_path/'config.json'; path.write_text(json.dumps(cfg)); return path


def test_checkpoint_resume_replay_and_corruption(tmp_path):
    config = config_file(tmp_path); direct = tmp_path/'direct'; resumed = tmp_path/'resumed'
    curriculum.run(config, direct)
    assert curriculum.run(config, resumed, max_conditions=1)['complete'] is False
    assert curriculum.run(config, resumed, resume=True) == dict(complete=True, trained=1, reused=1)
    pd.testing.assert_frame_equal(pd.read_csv(direct/'evaluations.csv'), pd.read_csv(resumed/'evaluations.csv'))
    assert (direct/'recalls.jsonl').read_bytes() == (resumed/'recalls.jsonl').read_bytes()
    assert curriculum.verify(resumed)['heads_replayed'] == 18
    frame = pd.read_csv(resumed/'evaluations.csv')
    first = frame[frame.epoch == 1].pivot(index=curriculum.KEYS, columns='treatment', values='readout_sha256')
    assert (first.fixed == first.curriculum).all()
    summary = json.loads((resumed/'summary.json').read_text())
    assert summary['all_offsets_recall_passed'] is False
    manifest = json.loads((resumed/'manifest.json').read_text())
    assert all('\\' not in name for name in manifest['file_sha256'])
    checkpoint = resumed/'checkpoints/condition_000.npz'; checkpoint.write_bytes(b'broken')
    with pytest.raises(ValueError, match='Checkpoint'):
        curriculum.run(config, resumed, resume=True)


def test_resume_rejects_config_and_code_changes(tmp_path, monkeypatch):
    config = config_file(tmp_path); out = tmp_path/'paused'
    curriculum.run(config, out, max_conditions=1)
    original = config.read_text(); cfg = json.loads(original); cfg['stage_epochs'] = 2
    config.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match='changed'):
        curriculum.run(config, out, resume=True)
    config.write_text(original)
    actual = curriculum.checkpoint.context
    def changed(cfg):
        result = actual(cfg); result['code']['changed.py'] = 'changed'; return result
    monkeypatch.setattr(curriculum.checkpoint, 'context', changed)
    with pytest.raises(ValueError, match='changed'):
        curriculum.run(config, out, resume=True)


def test_metrics_use_fixed_position_bands():
    class Head:
        def predict(self, states):
            result = np.zeros(199, dtype=int); result[34:66] = 1; return result
        def logits(self, states):
            return np.zeros((199,10))
    got = curriculum.measure(Head(), np.zeros((199,1)), np.zeros(199,dtype=int),
                              dict(prompt_length=3,prefix_window=32))
    assert got['early_accuracy'] == 1
    assert got['accuracy_33_64'] == 0
    assert got['accuracy_65_128'] == got['accuracy_129_197'] == 1
    assert got['later_accuracy'] == 133/165
