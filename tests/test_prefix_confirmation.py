import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from flying.training import phase5_prefix_confirmation as confirmation
from flying.brain.timed_reservoir import TimedReservoir
from flying.brain.mushroom_body import KCEncoder


def test_segment_indexing_and_exact_length():
    all_digits = confirmation.decimal_pi_digits(2200)
    for offset in [0,1000,2000]:
        digits = confirmation.segment_digits(offset,200)
        np.testing.assert_array_equal(digits,all_digits[offset:offset+200])
        assert len(digits)==200
    with pytest.raises(ValueError): confirmation.segment_digits(-1,200)


def test_each_segment_state_resets_to_zero_history():
    roles=np.array(['KC']*10+['MBON'])
    w=sparse.csr_matrix(([.7,.3],([10,0],[0,10])),shape=(11,11))
    encoder=KCEncoder(roles,42,fraction=.5)
    a=TimedReservoir(w,encoder,roles,schedule='mbon_after_kc')
    b=TimedReservoir(w,encoder,roles,schedule='mbon_after_kc')
    a.states([3,1,4,1,5,9])
    segment=confirmation.segment_digits(1000,20)
    np.testing.assert_array_equal(a.states(segment),b.states(segment))


def config_file(tmp_path):
    cfg=json.loads(Path('configs/phase5_prefix_confirmation.json').read_text())
    cfg.update(circuits=cfg['circuits'][:1],seeds=[1142],offsets=[0,40],pi_length=40,epochs=1)
    path=tmp_path/'config.json';path.write_text(json.dumps(cfg))
    return path


def test_interrupted_resume_matches_uninterrupted_and_replay(tmp_path):
    config=config_file(tmp_path)
    direct=tmp_path/'direct';resumed=tmp_path/'resumed'
    confirmation.run(config,direct)
    got=confirmation.run(config,resumed,max_conditions=1)
    assert not got['complete'] and got['trained']==1
    ledger=json.loads((resumed/'progress.json').read_text())
    first_name,first_hash=next(iter(ledger['completed'].items()))
    got=confirmation.run(config,resumed,resume=True)
    assert got==dict(complete=True,trained=3,reused=1)
    assert confirmation.sha256(resumed/first_name)==first_hash
    pd.testing.assert_frame_equal(pd.read_csv(direct/'evaluations.csv'),pd.read_csv(resumed/'evaluations.csv'))
    assert (direct/'recalls.jsonl').read_bytes()==(resumed/'recalls.jsonl').read_bytes()
    for path in (direct/'checkpoints').glob('*.npz'):
        with np.load(path,allow_pickle=False) as a,np.load(resumed/'checkpoints'/path.name,allow_pickle=False) as b:
            assert a.files==b.files
            for name in a.files: np.testing.assert_array_equal(a[name],b[name])
    result=confirmation.verify(resumed)
    assert result['heads_replayed']==24 and result['independently_refitted']==12
    manifest_path=resumed/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    assert all('\\' not in name for name in manifest['file_sha256'])
    manifest['file_sha256']={name.replace('/', '\\'):digest for name,digest in manifest['file_sha256'].items()}
    manifest_path.write_text(json.dumps(manifest))
    assert confirmation.verify(resumed)==result


def test_resume_rejects_config_and_code_changes(tmp_path,monkeypatch):
    config=config_file(tmp_path);out=tmp_path/'paused'
    confirmation.run(config,out,max_conditions=1)
    original=config.read_text();cfg=json.loads(original);cfg['prefix_weight']=5
    config.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='changed'):confirmation.run(config,out,resume=True)
    config.write_text(original)
    real_context=confirmation.context
    def changed(cfg):
        ctx=real_context(cfg);ctx['code']['changed.py']='changed';return ctx
    monkeypatch.setattr(confirmation,'context',changed)
    with pytest.raises(ValueError,match='changed'):confirmation.run(config,out,resume=True)


@pytest.mark.parametrize('missing',[False,True])
def test_resume_rejects_corrupt_or_missing_checkpoint(tmp_path,missing):
    config=config_file(tmp_path);out=tmp_path/'paused'
    confirmation.run(config,out,max_conditions=1)
    ledger=json.loads((out/'progress.json').read_text())
    path=out/next(iter(ledger['completed']))
    if missing:path.unlink()
    else:path.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='Checkpoint'):confirmation.run(config,out,resume=True)
