import numpy as np

def pi_memory_score(target, prediction):
    """Consecutive generated digits before first error; prompt excluded.

    Both arrays must describe the same complete evaluation horizon.
    """
    target = np.asarray(target); prediction = np.asarray(prediction)
    if target.ndim != 1 or target.shape != prediction.shape:
        raise ValueError("Equal-length 1D sequences required")
    mismatch = np.flatnonzero(target != prediction)
    return int(mismatch[0]) if len(mismatch) else len(target)
