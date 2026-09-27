"""Target-free, frozen-readout recall under declared computational perturbations."""
import numpy as np

from flying.brain.timed_reservoir import TimedReservoir
from flying.brain.plasticity import weight_hash

KINDS = ('pulse', 'ongoing', 'edge_dropout')


def drop_edges(weights, fraction, rng):
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError('Invalid removal fraction')
    result = weights.copy()
    count = int(np.floor(fraction * weights.nnz))
    if count:
        result.data[rng.permutation(weights.nnz)[:count]] = 0
        result.eliminate_zeros()
    return result, count


def recall(reservoir, head, roles, prompt, horizon, kind, strength, seed_parts):
    """No targets or correctness feedback enter this function.

    State noise is absolute, dimensionless Gaussian noise, clipped to [-1,1].
    Warmup is clean for state noise. Dropout is active during warmup and recall.
    """
    if kind not in ('clean', *KINDS) or not np.isfinite(strength) or strength < 0:
        raise ValueError('Invalid perturbation')
    if kind == 'clean' and strength != 0:
        raise ValueError('Clean recall cannot have nonzero strength')
    if not prompt or any(d not in '0123456789' for d in prompt) or type(horizon) is not int or horizon < 1:
        raise ValueError('Invalid prompt or horizon')
    # Identical first Gaussian vector for pulse/ongoing; independent dropout stream.
    rng = np.random.default_rng(np.random.SeedSequence([*seed_parts, 2 if kind == 'edge_dropout' else 1]))
    weights, removed = (drop_edges(reservoir.weights, strength, rng) if kind == 'edge_dropout'
                        else (reservoir.weights, 0))
    altered_hash = weight_hash(weights)
    model = TimedReservoir(weights, reservoir.encoder, roles, reservoir.leak, reservoir.schedule)
    for digit in prompt:
        model.step(int(digit))
    digits, clipped = [], 0
    for position in range(horizon):
        if strength > 0 and (kind == 'ongoing' or (kind == 'pulse' and position == 0)):
            state = model.state + strength * rng.standard_normal(len(model.state))
            clipped += int(np.count_nonzero(np.abs(state) > 1))
            model.state = np.clip(state, -1, 1)
        digit = int(head.predict(model.state)[0])
        digits.append(str(digit))
        model.step(digit)
    return dict(prediction=''.join(digits), clipped_coordinates=clipped, removed_edges=removed,
                perturbed_weight_sha256=altered_hash)
