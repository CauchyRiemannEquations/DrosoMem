import numpy as np
from scipy import sparse

def random_network(reference, seed):
    """Uniform directed support, exact N/E, same absolute weight multiset.

Per-neuron outgoing sign retained when observed (otherwise +1).
    """
    rng = np.random.default_rng(seed); n = reference.shape[0]
    coo = reference.tocoo()
    slots = rng.choice(n * (n - 1), len(coo.data), replace=False)
    pre = slots // (n - 1); post = slots % (n - 1)
    post = post + (post >= pre)
    signs = np.ones(n)
    for j, val in zip(coo.col, coo.data):
        signs[j] = np.sign(val)
    data = rng.permutation(np.abs(coo.data)) * signs[pre]
    return sparse.csr_matrix((data, (post, pre)), shape=(n, n))
