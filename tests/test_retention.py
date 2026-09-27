import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flying.training import phase5_curriculum as engine
from flying.training import phase5_retention as retention
from flying.training.phase5_prefix import sample_weights


def config_file(tmp_path):
    cfg = json.loads(Path('configs/phase5_retention.json').read_text())
    cfg.update(circuits=cfg['circuits'][:1], seeds=[3142], initializations=[0], stage_epochs=1)
    path = tmp_path/'config.json'; path.write_text(json.dumps(cfg)); return path


@pytest.mark.parametrize('window', [32,64,128])
def test_normalized_anchor_share_and_untouched_boundaries(window):
    ordinary = sample_weights(199,3,window,4)
    weights = retention.anchored_weights(199,3,window)
    assert np.isclose(weights[2:34].sum()/weights.sum(),128/295,atol=1e-15,rtol=0)
    np.testing.assert_array_equal(weights[:2],np.ones(2))
    np.testing.assert_array_equal(weights[34:],ordinary[34:])
    if window == 32:
        np.testing.assert_array_equal(weights,ordinary)
    else:
        assert weights[2] > 4


@pytest.mark.parametrize('window,anchor', [(16,32),(32,0),(128,199)])
def test_invalid_anchor(window,anchor):
    with pytest.raises(ValueError):
        retention.anchored_weights(199,3,window,anchor)


def test_all_arms_share_first_stage_and_budget():
    rng = np.random.default_rng(71); states = rng.normal(size=(199,48)); labels = np.arange(199)%10
    cfg = json.loads(Path('configs/phase5_retention.json').read_text()); cfg['stage_epochs'] = 2
    fits = [retention.fit_head(states,labels,np.arange(48),{'seed':3142},0,arm,cfg) for arm in retention.ARMS]
    assert len({saved[2].digest() for saved,_ in fits}) == 1
    for saved,history in fits:
        assert list(saved) == [2,4,6] and history[-1]['epoch'] == 6
        np.testing.assert_allclose(saved[6].mean,states.mean(axis=0),atol=1e-15,rtol=0)
        np.testing.assert_array_equal(saved[2].mean,saved[6].mean)
        np.testing.assert_array_equal(saved[2].scale,saved[6].scale)
    assert fits[1][0][6].digest() != fits[2][0][6].digest()


def test_retention_resume_replay_and_wrong_runner(tmp_path):
    config = config_file(tmp_path); direct = tmp_path/'direct'; resumed = tmp_path/'resumed'
    with pytest.raises(ValueError,match='runner'):
        engine.run(config,tmp_path/'wrong')
    retention.run(config,direct)
    assert retention.run(config,resumed,max_conditions=1)['complete'] is False
    assert retention.run(config,resumed,resume=True) == dict(complete=True,trained=1,reused=1)
    pd.testing.assert_frame_equal(pd.read_csv(direct/'evaluations.csv'),pd.read_csv(resumed/'evaluations.csv'))
    assert (direct/'recalls.jsonl').read_bytes() == (resumed/'recalls.jsonl').read_bytes()
    got = retention.verify(resumed)
    assert got['heads_replayed'] == 18 and got['independently_refitted'] == 3
    assert json.loads((resumed/'summary.json').read_text())['primary_success'] is False


@pytest.mark.parametrize('missing',[False,True])
def test_retention_rejects_corrupt_or_missing_checkpoint(tmp_path,missing):
    config = config_file(tmp_path); out = tmp_path/'paused'
    retention.run(config,out,max_conditions=1)
    path = out/'checkpoints/condition_000.npz'
    if missing: path.unlink()
    else: path.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='Checkpoint'):
        retention.run(config,out,resume=True)


def test_retention_rejects_changed_config(tmp_path):
    config = config_file(tmp_path); out = tmp_path/'paused'
    retention.run(config,out,max_conditions=1)
    cfg = json.loads(config.read_text()); cfg['stage_epochs'] = 2; config.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='changed'):
        retention.run(config,out,resume=True)


def test_criteria_do_not_hide_early_failure_behind_mean_gain():
    cfg = {'stage_epochs':2}
    rows = []
    for circuit in ['a','b']:
        for seed in range(3):
            for arm in retention.ARMS:
                for epoch in [2,4,6]:
                    row = dict(circuit=circuit,seed=seed,normalization='incoming_l1',model='fly',
                               schedule='mbon_after_kc',offset=0,initialization=0,treatment=arm,epoch=epoch)
                    row.update({k:1. for k in engine.MEASURES})
                    row['pi_memory_score'] = 32 if epoch == 2 else (100 if arm == 'anchored' else 33)
                    rows.append(row)
    frame = pd.DataFrame(rows)
    assert retention.summary(frame,cfg)['primary_success'] is True
    mask = (frame.circuit=='a') & (frame.seed==0) & (frame.treatment=='anchored') & (frame.epoch==6)
    frame.loc[mask,'pi_memory_score'] = 0
    result = retention.summary(frame,cfg)
    assert result['criteria']['recall'] is True
    assert result['criteria']['retention'] is False and result['primary_success'] is False
