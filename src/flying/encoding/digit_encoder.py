import numpy as np

class DigitEncoder:
    """Fixed positive sparse populations, equal cardinality for all digits."""
    def __init__(self, neurons, seed=42, fraction=0.1, amplitude=0.5):
        if neurons < 10 or not 0 < fraction <= 1 or amplitude <= 0:
            raise ValueError("Invalid encoding parameters")
        rng = np.random.default_rng(seed)
        k = max(1, int(neurons * fraction))
        self.patterns = np.zeros((10, neurons))
        for digit in range(10):
            self.patterns[digit, rng.choice(neurons, k, replace=False)] = amplitude
        self.patterns.flags.writeable = False

    def __call__(self, digit):
        if not isinstance(digit, (int, np.integer)) or not 0 <= digit <= 9:
            raise ValueError("Input must be a digit 0..9")
        return self.patterns[digit]
