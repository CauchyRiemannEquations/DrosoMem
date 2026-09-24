import numpy as np

def next_digit_accuracy(readout, states, targets):
    if len(states) != len(targets) or not len(targets):
        raise ValueError("Nonempty aligned states and targets required")
    return float(np.mean(readout.predict(states) == targets))
