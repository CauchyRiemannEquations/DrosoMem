import json
from pathlib import Path

import pytest

pytest.importorskip('brian2')
from flying.training import stage_bc


def test_resume_rejects_changed_context_before_reusing_heads(tmp_path, monkeypatch):
    config = json.loads(Path('configs/stage_bc.json').read_text())
    cfg_file = tmp_path/'config.json'
    cfg_file.write_text(json.dumps(config))
    (tmp_path/'progress.json').write_text(json.dumps(dict(context={'code': 'old'}, fingerprint='old')))
    monkeypatch.setattr(stage_bc, 'context', lambda cfg: {'code': 'changed'})
    monkeypatch.setattr(stage_bc, 'grid', lambda cfg: pytest.fail('Must reject before graph loading'))
    with pytest.raises(ValueError, match='Resume'):
        stage_bc.run(cfg_file, tmp_path, resume=True)


def test_verification_rejects_corrupted_artifact_before_simulation(tmp_path, monkeypatch):
    ctx = {'config': {}}
    (tmp_path/'artifact.npz').write_bytes(b'changed')
    (tmp_path/'manifest.json').write_text(json.dumps(dict(context=ctx, file_sha256={'artifact.npz': 'wrong'})))
    monkeypatch.setattr(stage_bc, 'context', lambda cfg: ctx)
    monkeypatch.setattr(stage_bc, 'grid', lambda cfg: pytest.fail('Must reject before simulation'))
    with pytest.raises(ValueError, match='checksum'):
        stage_bc.verify(tmp_path)
