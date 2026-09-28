"""Integer-symbol datasets; ACT II deliberately holds the experiment at K=10."""
from dataclasses import dataclass
import hashlib
import numpy as np
from flying.data.pi_digits import pi_digits, decimal_pi_digits


@dataclass(frozen=True)
class SequenceDataset:
    family: str
    seed: int
    alphabet_size: int = 10
    length: int = 200
    offset: int = 0
    prompt: tuple = (3, 1, 4)
    motif: tuple = (3, 1, 4, 0, 2, 5, 6, 7, 8, 9)

    def symbols(self):
        k = self.alphabet_size
        if type(k) is not int or k < 2 or self.length <= len(self.prompt) or self.offset < 0:
            raise ValueError('Invalid alphabet, length or offset')
        prompt = np.asarray(self.prompt, dtype=np.int64)
        if np.any(prompt < 0) or np.any(prompt >= k):
            raise ValueError('Prompt outside alphabet')
        if self.family in ('pi', 'shuffled_pi'):
            if k != 10:
                raise ValueError('Pi is a decimal dataset; no modulo conversion')
            end = self.offset + self.length
            a, b = pi_digits(end), decimal_pi_digits(end)
            np.testing.assert_array_equal(a, b)
            symbols = a[self.offset:].astype(np.int64)
            if not np.array_equal(symbols[:len(prompt)], prompt):
                raise ValueError('Prompt does not match pi segment')
            if self.family == 'shuffled_pi':
                rng = np.random.default_rng(np.random.SeedSequence([self.seed, 2202]))
                symbols[len(prompt):] = rng.permutation(symbols[len(prompt):])
        elif self.family == 'random':
            if self.offset != 0:
                raise ValueError('Random datasets use a seed, not a pi offset')
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, 2201]))
            symbols = rng.integers(0, k, size=self.length, dtype=np.int64)
            symbols[:len(prompt)] = prompt
        elif self.family == 'periodic':
            motif = np.asarray(self.motif, dtype=np.int64)
            if not len(motif) or np.any(motif < 0) or np.any(motif >= k):
                raise ValueError('Invalid motif')
            symbols = np.resize(motif, self.offset + self.length)[self.offset:].copy()
            if not np.array_equal(symbols[:len(prompt)], prompt):
                raise ValueError('Prompt does not match periodic segment')
        else:
            raise ValueError('Unknown sequence family')
        return symbols

    def identity(self, symbols):
        return dict(**self.__dict__, sha256=hashlib.sha256(symbols.tobytes()).hexdigest(),
                    generator='numpy PCG64; SeedSequence namespace 2201 random / 2202 shuffle',
                    symbols=symbols.tolist(), digit_counts=np.bincount(symbols, minlength=self.alphabet_size).tolist())
