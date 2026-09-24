import numpy as np
from flying.evaluation.metrics import pi_memory_score

def generate(reservoir, readout, prompt, steps):
    """Targets are deliberately absent from the autoregressive generator."""
    if len(prompt) == 0 or steps < 0:
        raise ValueError("Need a nonempty prompt and nonnegative steps")
    reservoir.reset()
    for digit in prompt:
        state = reservoir.step(int(digit))
    prediction = []
    for _ in range(steps):
        digit = int(readout.predict(state)[0])
        prediction.append(digit)
        state = reservoir.step(digit)
    return np.array(prediction, dtype=np.int64)

def evaluate_recall(reservoir, readout, digits, prompt_length, horizon):
    if not 1 <= prompt_length < len(digits) or not 1 <= horizon <= len(digits) - prompt_length:
        raise ValueError("Invalid recall horizon/prompt")
    pred = generate(reservoir, readout, digits[:prompt_length], horizon)
    target = digits[prompt_length:prompt_length + horizon]
    score = pi_memory_score(target, pred)
    return dict(pi_memory_score=score, prompt_length=prompt_length, horizon=horizon,
                correct_total_including_prompt=prompt_length + score,
                censored=score == horizon,
                first_error_digit_index=None if score == horizon else prompt_length + score,
                prompt=''.join(map(str, digits[:prompt_length])),
                target=''.join(map(str, target)), prediction=''.join(map(str, pred)))
