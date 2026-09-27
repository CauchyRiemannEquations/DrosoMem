import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from flying.game.match import Match, PrefixJudge
from flying.game.opponent import ARRAYS, OpponentCatalog

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT/'assets/opponents/fixed32/catalog.json'


class ScriptedOpponent:
    prompt = '314'
    identity = 'test-opponent'

    def __init__(self, digits):
        self.digits = digits
        self.horizon = len(digits)
        self.reset()

    def reset(self):
        self.generated = 0

    def next_digit(self):
        value = int(self.digits[self.generated])
        self.generated += 1
        return value


def test_all_exported_models_replay_archived_full_rollouts_without_targets(monkeypatch):
    catalog = OpponentCatalog(CATALOG)
    assert len(catalog.entries) == 18
    assert len({entry['id'] for entry in catalog.entries}) == 18
    def forbidden(*args, **kwargs):
        raise AssertionError('Opponent requested reference digits')
    monkeypatch.setattr('flying.data.pi_digits.pi_digits', forbidden)
    monkeypatch.setattr('flying.game.match.pi_digits', forbidden)
    for entry in catalog.entries:
        with np.load(CATALOG.parent/entry['artifact'], allow_pickle=False) as bundle:
            assert set(bundle.files) == ARRAYS
            assert all(bundle[name].dtype.kind != 'O' for name in bundle.files)
        with np.load(ROOT/'results/phase5_retention'/entry['source_checkpoint'], allow_pickle=False) as source:
            payload = json.loads(str(source['payload'].item()))
        expected = next(r['prediction'] for r in payload['recalls'] if r['head_index'] == entry['source_head_index'])
        model = catalog.load(entry['id'])
        assert ''.join(str(model.next_digit()) for _ in range(197)) == expected
        with pytest.raises(StopIteration):
            model.next_digit()
        model.reset()
        assert ''.join(str(model.next_digit()) for _ in range(12)) == expected[:12]


def test_selection_is_seeded_and_independent_instances_do_not_share_state():
    catalog = OpponentCatalog(CATALOG)
    a, b = catalog.select(42), catalog.select(42)
    assert a.identity == b.identity
    sequence = [a.next_digit() for _ in range(20)]
    assert [b.next_digit() for _ in range(20)] == sequence
    assert not np.shares_memory(a._reservoir.state, b._reservoir.state)
    assert not a._head.parameters['w1'].flags.writeable
    assert not a._reservoir.weights.data.flags.writeable
    a.reset()
    assert a.next_digit() == sequence[0]
    # Observe the population selected by many seeds without loading each bundle.
    catalog.load = lambda identity: identity
    assert {catalog.select(seed) for seed in range(300)} == {e['id'] for e in catalog.entries}
    with pytest.raises(ValueError):
        catalog.select(-1)


def test_artifact_corruption_is_rejected(tmp_path):
    value = json.loads(CATALOG.read_text())
    entry = value['opponents'][0]
    value['opponents'] = [entry]
    path = tmp_path/'catalog.json'
    path.write_text(json.dumps(value))
    shutil.copyfile(CATALOG.parent/entry['artifact'], tmp_path/entry['artifact'])
    with (tmp_path/entry['artifact']).open('ab') as stream:
        stream.write(b'corrupted')
    with pytest.raises(ValueError, match='checksum'):
        OpponentCatalog(path).select(0)
    entry['artifact'] = '../outside.npz'
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='local NPZ'):
        OpponentCatalog(path)


def test_ten_minute_gate_and_study_material_visibility():
    now = [100.]
    game = Match(ScriptedOpponent('159'), clock=lambda: now[0])
    assert game.study_seconds == 600
    assert game.study_material == '314159'
    with pytest.raises(RuntimeError):
        game.begin_recall()
    with pytest.raises(RuntimeError):
        game.submit('1')
    now[0] = 699.9
    assert game.phase == 'studying'
    now[0] = 700.
    assert game.phase == 'ready'
    assert game.study_material is None
    game.begin_recall()
    assert game.phase == 'recalling'
    assert game.remaining_seconds == 0
    with pytest.raises(RuntimeError):
        game.begin_recall()


@pytest.mark.parametrize('bad', ['', '12', '１', 'a', True, 1.0, -1, 10, None])
def test_invalid_input_advances_neither_player(bad):
    game = Match(ScriptedOpponent('159'), study_seconds=0)
    game.begin_recall()
    with pytest.raises(ValueError):
        game.submit(bad)
    assert game.opponent.generated == 0
    assert game.human.attempts == game.model.attempts == 0
    assert game.rounds == []


def test_same_rule_first_error_and_prompt_excluded():
    game = Match(ScriptedOpponent('159'), study_seconds=0)
    game.begin_recall()
    game.submit('1')
    game.submit('5')
    game.submit('8')
    assert game.winner == 'model'
    assert game.human.score.correct == 2
    assert game.human.score.first_error_position == 3
    assert game.model.score.correct == 3
    assert game.model.score.censored
    with pytest.raises(RuntimeError):
        game.submit()
    record = game.record()
    record['rounds'].clear()
    assert len(game.record()['rounds']) == 3


def test_human_failure_does_not_feed_or_stop_model():
    game = Match(ScriptedOpponent('159'), study_seconds=0)
    game.begin_recall()
    game.submit('0')
    assert game.phase == 'recalling'
    assert game.winner is None
    with pytest.raises(RuntimeError):
        game.record()
    with pytest.raises(RuntimeError):
        game.submit('5')
    game.submit()
    game.submit()
    assert [r['model_digit'] for r in game.rounds] == [1, 5, 9]
    assert game.human.score.correct == 0
    assert game.human.score.attempts == 1
    assert game.winner == 'model'


def test_model_failure_does_not_stop_human():
    game = Match(ScriptedOpponent('009'), study_seconds=0)
    game.begin_recall()
    for digit in '159':
        game.submit(digit)
    assert game.winner == 'human'
    assert game.model.score.correct == 0
    assert game.opponent.generated == 1
    assert [r['model_digit'] for r in game.rounds] == [0, None, None]


@pytest.mark.parametrize('digits,human,score,censored', [('159','159',3,True),('009','0',0,False)])
def test_draws_and_horizon_are_explicit(digits, human, score, censored):
    game = Match(ScriptedOpponent(digits), study_seconds=0)
    game.begin_recall()
    for digit in human:
        game.submit(digit)
    assert game.winner == 'draw'
    assert game.human.score.correct == game.model.score.correct == score
    assert game.human.score.censored == game.model.score.censored == censored


def test_judge_stops_at_first_error():
    judge = PrefixJudge('159')
    judge.submit(1)
    score = judge.submit(0)
    assert (score.correct, score.attempts, score.first_error_position) == (1, 2, 2)
    with pytest.raises(RuntimeError):
        judge.submit(9)


@pytest.mark.parametrize('seconds', [-1, float('nan'), float('inf'), True])
def test_invalid_duration(seconds):
    with pytest.raises(ValueError):
        Match(ScriptedOpponent('159'), study_seconds=seconds)


@pytest.mark.parametrize('horizon', [0, 4, True, 1.5])
def test_invalid_horizon(horizon):
    with pytest.raises(ValueError):
        Match(ScriptedOpponent('159'), horizon=horizon)


@pytest.mark.parametrize('entered,winner,score', [('1\n5\n9\n','draw',3), ('x\n0\n','model',0)])
def test_console_flow_with_korean_windows_encoding(tmp_path, entered, winner, score):
    record = tmp_path/'match.json'
    result = subprocess.run([sys.executable, '-m', 'flying.game', '--seed', '0',
                             '--study-seconds', '0', '--horizon', '3', '--record', str(record)],
                            cwd=ROOT, input=entered, text=True, encoding='cp949', capture_output=True,
                            env={**os.environ, 'PYTHONPATH':str(ROOT/'src'), 'PYTHONIOENCODING':'cp949'},
                            timeout=30)
    assert result.returncode == 0, result.stderr
    saved = json.loads(record.read_text())
    assert saved['winner'] == winner
    assert saved['human']['correct'] == score
    assert saved['model']['correct'] == 3
    assert saved['match_seed'] == 0
    assert saved['catalog_sha256'] == OpponentCatalog(CATALOG).digest
