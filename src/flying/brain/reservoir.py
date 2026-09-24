import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs

def normalize(a, radius=0.9):
    """Global scaling preserves relative weights/signs; no weight learning."""
    a = sparse.csr_matrix(a, dtype=float).copy()
    if a.shape[0] != a.shape[1] or a.nnz == 0 or not 0 < radius <= 2:
        raise ValueError("Need a nonempty square graph and radius in (0, 2]")
    if a.shape[0] <= 512:
        rho = float(np.max(np.abs(np.linalg.eigvals(a.toarray()))))
    else:
        rho = float(np.max(np.abs(eigs(a, k=1, which="LM", v0=np.ones(a.shape[0]), return_eigenvectors=False))))
    if rho < 1e-12:
        raise ValueError("No recurrent spectral mass; choose a less sparse subset")
    return a * (radius / rho), rho

class Reservoir:
    def __init__(self, weights, encoder, leak=0.6):
        if not 0 < leak <= 1:
            raise ValueError("leak must be in (0, 1]")
        self.weights = sparse.csr_matrix(weights).copy()
        self.weights.data.flags.writeable = False
        self.encoder = encoder
        self.leak = leak
        self.reset()

    def reset(self):
        self.state = np.zeros(self.weights.shape[0])

    def step(self, digit):
        self.state = ((1 - self.leak) * self.state + self.leak *
                      np.tanh(self.weights @ self.state + self.encoder(digit)))
        return self.state.copy()

    def states(self, digits, reset=True):
        if reset:
            self.reset()
        return np.array([self.step(int(d)) for d in digits])
