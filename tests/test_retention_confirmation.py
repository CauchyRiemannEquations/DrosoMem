import json
from pathlib import Path

import pandas as pd
import pytest

from flying.training import phase5_curriculum as engine
from flying.training import phase5_retention as discovery
from flying.training import phase5_retention_confirmation as confirmation


def full_config():
    return json.loads(Path('configs/phase5_retention_confirmation.json').read_text())


def passing_frame():
    rows = []
    for offset in confirmation.OFFSETS:
        for circuit in ['a','b']:
            for seed in range(3):
                for initialization in range(3):
                    for arm in confirmation.ARMS:
                        for epoch in [2,4,6]:
                            row = dict(circuit=circuit,seed=seed,normalization='incoming_l1',model='fly',
                                       schedule='mbon_after_kc',offset=offset,initialization=initialization,
                                       treatment=arm,epoch=epoch)
                            row.update({k:1. for k in engine.MEASURES})
                            row['pi_memory_score'] = 32 if epoch == 2 else (40 if arm == 'anchored' else 33)
                            rows.append(row)
    return pd.DataFrame(rows)


def test_requires_success_at_every_segment_without_pooling():
    frame = passing_frame()
    result = confirmation.summary(frame, {'stage_epochs':2})
    assert result['all_offsets_primary_passed'] and result['all_offsets_joint_passed']
    mask = (frame.offset == 1000) & (frame.treatment == 'anchored') & (frame.epoch == 6)
    frame.loc[mask,'pi_memory_score'] = 33
    result = confirmation.summary(frame, {'stage_epochs':2})
    assert result['criteria_by_offset']['0']['primary_success']
    assert not result['criteria_by_offset']['1000']['recall']
    assert not result['all_offsets_primary_passed']


def test_retention_and_later_cost_are_separate_gates():
    frame = passing_frame()
    mask = (frame.offset == 2000) & (frame.treatment == 'anchored') & (frame.epoch == 6)
    frame.loc[mask,'later_accuracy'] = .9
    result = confirmation.summary(frame, {'stage_epochs':2})
    assert result['all_offsets_primary_passed']
    assert not result['all_offsets_joint_passed']
    frame.loc[mask & (frame.circuit == 'a') & (frame.seed == 0) & (frame.initialization == 0), 'pi_memory_score'] = 0
    result = confirmation.summary(frame, {'stage_epochs':2})
    assert result['criteria_by_offset']['2000']['recall']
    assert not result['criteria_by_offset']['2000']['retention']
    assert not result['all_offsets_primary_passed']


def test_missing_segment_cannot_pass():
    frame = passing_frame()
    result = confirmation.summary(frame[frame.offset != 1000], {'stage_epochs':2})
    assert not result['all_offsets_primary_passed']
    assert not result['all_offsets_joint_passed']


@pytest.mark.parametrize('change', [dict(seeds=[3142]), dict(offsets=[0,1000]),
                                    dict(experiment='prefix_share_retention')])
def test_rejects_discovery_seeds_missing_offsets_and_wrong_runner(change):
    cfg = full_config()
    cfg.update(change)
    with pytest.raises(ValueError):
        confirmation.validate(cfg)


def test_cross_segment_stop_resume_replay(tmp_path):
    cfg = full_config()
    cfg.update(circuits=cfg['circuits'][:1],seeds=[4142],initializations=[0],stage_epochs=1)
    path = tmp_path/'config.json'
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):
        discovery.run(path,tmp_path/'wrong')
    direct, resumed = tmp_path/'direct',tmp_path/'resumed'
    confirmation.run(path,direct)
    assert confirmation.run(path,resumed,max_conditions=1)['complete'] is False
    result = confirmation.run(path,resumed,resume=True)
    assert result == dict(complete=True,trained=5,reused=1)
    pd.testing.assert_frame_equal(pd.read_csv(direct/'evaluations.csv'),pd.read_csv(resumed/'evaluations.csv'))
    assert (direct/'recalls.jsonl').read_bytes() == (resumed/'recalls.jsonl').read_bytes()
    result = confirmation.verify(resumed)
    assert result['states_rebuilt'] == 6
    assert result['heads_replayed'] == 54
    assert result['independently_refitted'] == 9
    assert result['refitted_stage_heads'] == 27
    assert result['pi_generators_equal_digits'] == 2200
    assert not json.loads((resumed/'summary.json').read_text())['all_offsets_primary_passed']
    changed = full_config()
    changed.update(cfg, learning_rate=.04)
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='changed'):
        confirmation.run(path,resumed,resume=True)
    path.write_text(json.dumps(cfg))
    (resumed/'checkpoints/condition_001.npz').write_bytes(b'broken')
    with pytest.raises(ValueError,match='Checkpoint'):
        confirmation.run(path,resumed,resume=True)
