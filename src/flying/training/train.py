"""Paired seeds, chronological heldout suffix, fresh-state prefix recall."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits
from flying.data.connectome import load_connectome
from flying.data.pi_digits import pi_digits
from flying.encoding.digit_encoder import DigitEncoder
from flying.brain.reservoir import Reservoir, normalize
from flying.brain.shuffled_network import shuffled_network
from flying.brain.random_network import random_network
from flying.models.readout import Readout
from flying.evaluation.next_digit import next_digit_accuracy
from flying.evaluation.free_recall import evaluate_recall
from flying.visualization.activity import plot_activity
from flying.visualization.pi_memory import plot_memory, plot_training


def run(config, out, base=Path('.')):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)  # never overwrite a previous experiment
    started = time.perf_counter()
    n_train = config['train_digits']; n_test = config['heldout_digits']; prompt = config['prompt_length']
    if not 1 <= prompt < n_train or n_test < 1:
        raise ValueError('Require 1 <= prompt < train_digits and heldout_digits >= 1')
    if not config['seeds'] or not set(config['models']) <= {'fly', 'shuffled', 'random'} or not config['models']:
        raise ValueError('Need seeds and known models')
    a, ids, provenance = load_connectome(Path(base) / config['connectome'])
    # Safety cap induces an explicitly logged smaller graph; never allocates whole-brain dense W.
    max_n = config['reservoir']['max_neurons']
    if not 10 <= max_n <= 3000:
        raise ValueError('CPU MVP cap must be 10..3000 neurons')
    original_n = a.shape[0]
    if original_n > max_n:
        strengths = np.asarray(abs(a).sum(axis=0)).ravel() + np.asarray(abs(a).sum(axis=1)).ravel()
        selected = np.sort(np.argsort(-strengths, kind='stable')[:max_n])
        a = a[selected][:, selected]; ids = [ids[i] for i in selected]
    digits = pi_digits(n_train + n_test)
    (out / 'pi_digits.txt').write_text(''.join(map(str, digits)) + '\n')
    (out / 'config.json').write_text(json.dumps(config, indent=2))
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    (out / 'neuron_ids.json').write_text(json.dumps(ids, indent=2))
    try:
        git = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=base, stderr=subprocess.DEVNULL, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git = None
    manifest = dict(time_utc=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
                    platform=platform.platform(), git_commit=git, original_neurons=original_n,
                    effective_neurons=a.shape[0], cap_applied=original_n > max_n,
                    packages={p: importlib.metadata.version(p) for p in ['numpy','scipy','pandas','matplotlib','mpmath','threadpoolctl']},
                    config_sha256=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                    source_code_sha256={str(p.relative_to(Path(__file__).parents[1])): hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in sorted(Path(__file__).parents[1].rglob('*.py'))},
                    split='train targets indices 1..train_digits-1; heldout targets train_digits..end',
                    selection='final epoch, fixed in advance; no heldout tuning', threads=1)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    # Memoryless empirical P(next digit | current digit), trained only on prefix.
    transitions = np.ones((10,10))
    np.add.at(transitions, (digits[:n_train-1], digits[1:n_train]), 1)
    lookup = transitions.argmax(axis=1)
    memless = dict(train_accuracy=float(np.mean(lookup[digits[:n_train-1]] == digits[1:n_train])),
                   heldout_accuracy=float(np.mean(lookup[digits[n_train-1:-1]] == digits[n_train:])))
    (out / 'memoryless.json').write_text(json.dumps(memless, indent=2))
    rows = []
    for seed in config['seeds']:
        encoder = DigitEncoder(a.shape[0], seed, **config['encoding'])
        for model in config['models']:
            t0 = time.perf_counter(); extra = {}
            if model == 'fly':
                raw = a.copy()
            elif model == 'shuffled':
                raw, extra = shuffled_network(a, seed + 10000)
            else:
                raw = random_network(a, seed + 20000)
            weights, rho = normalize(raw, config['reservoir']['spectral_radius'])
            reservoir = Reservoir(weights, encoder, config['reservoir']['leak'])
            states = reservoir.states(digits[:-1])
            # state at t contains inputs 0..t only. Target is digit t+1.
            x = states[:n_train-1]; y = digits[1:n_train]
            readout = Readout()
            history = readout.fit(x, y, **config['training'])
            train_acc = next_digit_accuracy(readout, x, y)
            heldout_acc = next_digit_accuracy(readout, states[n_train-1:], digits[n_train:])
            recall = evaluate_recall(reservoir, readout, digits[:n_train], prompt, n_train-prompt)
            extended = evaluate_recall(reservoir, readout, digits, prompt, len(digits)-prompt)
            row = dict(model=model, seed=seed, neurons=a.shape[0], edges=raw.nnz,
                       train_accuracy=train_acc, heldout_accuracy=heldout_acc,
                       pi_memory_score=recall['pi_memory_score'], horizon=recall['horizon'], censored=recall['censored'],
                       extended_score=extended['pi_memory_score'], raw_spectral_radius=rho,
                       seconds=time.perf_counter()-t0, **extra)
            rows.append(row)
            folder = out / f'{model}_seed{seed}'; folder.mkdir()
            sparse.save_npz(folder / 'reservoir.npz', weights)
            np.savez_compressed(folder / 'readout.npz', weights=readout.weights, mean=readout.mean,
                                scale=readout.scale, encoder=encoder.patterns, leak=reservoir.leak)
            np.save(folder / 'teacher_states.npy', states)
            pd.DataFrame(history).to_csv(folder / 'training.csv', index=False)
            for name, data in [('recall', recall), ('extended_recall', extended), ('metrics', row)]:
                (folder / f'{name}.json').write_text(json.dumps(data, indent=2))
            plot_training(history, folder / 'training.png')
            plot_memory(recall, folder / 'pi_memory.png')
            plot_activity(states, folder / 'activity.png')
            print(json.dumps(row), flush=True)
    frame = pd.DataFrame(rows); frame.to_csv(out / 'results.csv', index=False)
    summary = frame.groupby('model')[['train_accuracy','heldout_accuracy','pi_memory_score']].agg(['mean','std','min','max'])
    summary.to_csv(out / 'summary.csv')
    plot_comparison(frame, out / 'comparison.png')
    (out / 'runtime.json').write_text(json.dumps(dict(total_seconds=time.perf_counter()-started), indent=2))
    return frame


def plot_comparison(frame, path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10,4))
    colors = ['#0f766e','#d97706','#6366f1']
    names = list(frame.model.unique())
    for ax, col, title in zip(axes, ['pi_memory_score','heldout_accuracy'], ['Free recall: generated correct prefix', 'Heldout next-digit accuracy']):
        for i, name in enumerate(names):
            vals = frame.loc[frame.model == name, col].to_numpy()
            ax.bar(i, vals.mean(), color=colors[i], alpha=.7)
            ax.scatter(i + np.linspace(-.09,.09,len(vals)), vals, c='black', s=20, zorder=3)
        ax.set_xticks(range(len(names)), names); ax.set_title(title, fontsize=11)
        ax.grid(axis='y', alpha=.15)
    axes[0].set_ylabel('Pi Memory Score (prompt excluded)')
    axes[1].axhline(.1, linestyle='--', color='gray', label='Uniform chance 0.10')
    axes[1].legend(fontsize=8); axes[1].set_ylim(0, max(.2, frame.heldout_accuracy.max()+.03))
    fig.tight_layout(); fig.savefig(path,dpi=160); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/mvp.json')
    p.add_argument('--output', default=None)
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text())
    out = args.output or ('outputs/' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    with threadpool_limits(limits=1):
        run(config, out)
    print(f'Saved experiment: {out}')

if __name__ == '__main__':
    main()
