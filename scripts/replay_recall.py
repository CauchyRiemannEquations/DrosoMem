"""Replay an output checkpoint with no fitting and no target input to generation."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from flying.brain.reservoir import Reservoir
from flying.encoding.digit_encoder import DigitEncoder
from flying.models.readout import Readout
from flying.evaluation.free_recall import generate


def replay(folder):
    folder = Path(folder)
    saved = np.load(folder / 'readout.npz', allow_pickle=False)
    encoder = DigitEncoder(saved['encoder'].shape[1])
    encoder.patterns = saved['encoder'].copy()
    encoder.patterns.flags.writeable = False
    brain = Reservoir(sparse.load_npz(folder / 'reservoir.npz'), encoder, float(saved['leak']))
    readout = Readout()
    readout.weights = saved['weights']; readout.mean = saved['mean']; readout.scale = saved['scale']
    evidence = json.loads((folder / 'recall.json').read_text())
    prediction = ''.join(map(str, generate(brain, readout, [int(d) for d in evidence['prompt']], evidence['horizon'])))
    if prediction != evidence['prediction']:
        raise RuntimeError('Checkpoint replay differs from logged prediction')
    print(f"Verified {folder.name}: score {evidence['pi_memory_score']}, full prediction identical")
    return prediction

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('folder')
    replay(p.parse_args().folder)
