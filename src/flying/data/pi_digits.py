"""π includes leading 3, never the decimal point. No external data request."""
import numpy as np
from mpmath import mp

def pi_digits(length):
    if not isinstance(length, int) or length < 2:
        raise ValueError("length must be an integer >= 2")
    with mp.workdps(length + 30):
        s = str(mp.pi).replace(".", "")[:length]
    return np.array([int(c) for c in s], dtype=np.int64)
