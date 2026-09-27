"""Clock-gated study phase and identical first-error scoring for both players."""
from dataclasses import asdict, dataclass
from copy import deepcopy
import math
import time

from flying.data.pi_digits import pi_digits


def digit_value(value):
    if type(value) is int and 0 <= value <= 9:
        return value
    if type(value) is str and len(value) == 1 and value in '0123456789':
        return int(value)
    raise ValueError('Enter exactly one digit 0-9')


@dataclass(frozen=True)
class Score:
    correct: int
    attempts: int
    finished: bool
    censored: bool
    first_error_position: int | None


class PrefixJudge:
    def __init__(self, target):
        if not target:
            raise ValueError('Need a nonempty target')
        self._target = tuple(digit_value(digit) for digit in target)
        self.correct = self.attempts = 0
        self.error = None

    @property
    def score(self):
        complete = self.correct == len(self._target)
        return Score(self.correct,self.attempts,self.error is not None or complete,complete,self.error)

    def submit(self, digit):
        digit = digit_value(digit)
        if self.score.finished:
            raise RuntimeError('Player already finished')
        self.attempts += 1
        if digit == self._target[self.correct]:
            self.correct += 1
        else:
            self.error = self.correct+1
        return self.score


class Match:
    def __init__(self, opponent, study_seconds=600, horizon=None, clock=time.monotonic):
        if type(study_seconds) not in (int,float) or not math.isfinite(study_seconds) or study_seconds < 0:
            raise ValueError('Invalid study duration')
        horizon = opponent.horizon if horizon is None else horizon
        if type(horizon) is not int or not 1 <= horizon <= opponent.horizon:
            raise ValueError('Invalid match horizon')
        self.opponent = opponent
        self.prompt = opponent.prompt
        self.horizon = horizon
        self.study_seconds = study_seconds
        self._clock = clock
        self._deadline = clock()+study_seconds
        self._started = False
        digits = ''.join(map(str,pi_digits(len(self.prompt)+horizon)))
        if not digits.startswith(self.prompt):
            raise ValueError('Opponent prompt does not match the game opening')
        self._study_digits = digits
        self.human = PrefixJudge(digits[len(self.prompt):])
        self.model = PrefixJudge(digits[len(self.prompt):])
        self.rounds = []

    @property
    def remaining_seconds(self):
        return max(0., self._deadline-self._clock()) if not self._started else 0.

    @property
    def phase(self):
        if not self._started:
            return 'studying' if self.remaining_seconds > 0 else 'ready'
        return 'finished' if self.human.score.finished and self.model.score.finished else 'recalling'

    @property
    def study_material(self):
        return self._study_digits if self.phase == 'studying' else None

    def begin_recall(self):
        if self.phase != 'ready':
            raise RuntimeError('Study time must finish before recall begins')
        self.opponent.reset()
        self._study_digits = None
        self._started = True

    def submit(self, human_digit=None):
        if self.phase != 'recalling':
            raise RuntimeError('Match is not accepting recall digits')
        if self.human.score.finished:
            if human_digit is not None:
                raise RuntimeError('Human already finished; advance with no input')
        else:
            human_digit = digit_value(human_digit)
        # Validate human input before advancing either player. The opponent is
        # never given human input, correctness feedback or a reference digit.
        model_digit = None if self.model.score.finished else self.opponent.next_digit()
        if human_digit is not None:
            self.human.submit(human_digit)
        if model_digit is not None:
            self.model.submit(model_digit)
        result = dict(round=len(self.rounds)+1,human_digit=human_digit,model_digit=model_digit,
                      human=asdict(self.human.score),model=asdict(self.model.score))
        self.rounds.append(result)
        return deepcopy(result)

    @property
    def winner(self):
        if self.phase != 'finished':
            return None
        a,b = self.human.correct,self.model.correct
        return 'draw' if a == b else ('human' if a > b else 'model')

    def record(self):
        if self.phase != 'finished':
            raise RuntimeError('Cannot record an unfinished match as a result')
        return dict(format_version=1,opponent_id=self.opponent.identity,mode='pretrained',
                    study_seconds=self.study_seconds,prompt=self.prompt,horizon=self.horizon,
                    winner=self.winner,human=asdict(self.human.score),model=asdict(self.model.score),rounds=deepcopy(self.rounds))
