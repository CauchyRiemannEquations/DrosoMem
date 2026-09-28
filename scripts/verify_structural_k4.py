"""Independent saved-artifact alignment, metric, pairing and summary checks."""
import argparse
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from alphabet_memory import check, read
from flying.training import whole_brain_memory as core


@threadpool_limits.wrap(limits=1)
def verify(root):
    manifest = check(root); c = manifest['effective_config']; records = []
    for group in sorted(root.glob('*_c*_s*')):
        m = check(group); identity = m['identity']; metrics = read(group/'metrics.json')
        with np.load(group/'checkpoint.npz', allow_pickle=False) as a:
            w = c['warmup']
            for split, target in [('train', 'ytrain'), ('test', 'ytest')]:
                symbols = a[split+'_symbols']; n = c[split+'_samples']
                expected = np.column_stack([symbols[w-lag:w-lag+n] for lag in c['lags']])
                np.testing.assert_array_equal(expected, a[target])
                np.testing.assert_array_equal(symbols, np.random.default_rng(identity[split+'_seed']).integers(0, 4, w+n).astype(np.uint8))
            z = (a['test_features'][w:]-a['mean'])/a['scale']
            for prefix in ['', 'null_']:
                score = (z@a[prefix+'weights']+a[prefix+'bias']).reshape(a[prefix+'scores'].shape)
                np.testing.assert_array_equal(score, a[prefix+'scores'])
                np.testing.assert_array_equal(score.argmax(axis=2), a[prefix+'predictions'])
            for j, row in enumerate(metrics):
                target = a['ytest'][:, j]
                calculated = dict(test_accuracy=float(np.mean(a['predictions'][:, j] == target)),
                                  null_accuracy=float(np.mean(a['null_predictions'][:, j] == target)),
                                  frequency_accuracy=float(np.mean(a['majority'][:, j] == target)),
                                  train_accuracy=float(np.mean(a['train_scores'][:, j].argmax(axis=1) == a['ytrain'][:, j])),
                                  training_mse=float(np.mean((a['train_scores'][:, j]-np.eye(4)[a['ytrain'][:, j]])**2)))
                frequency = a['bias'].reshape(-1, 4)[j]
                calculated['r2_vs_frequency'] = float(1-((a['scores'][:, j]-np.eye(4)[target])**2).sum()/((np.eye(4)[target]-frequency)**2).sum())
                for key, value in calculated.items():
                    assert abs(row[key]-value) < 1e-14, (group, key)
                records.append(dict(**identity, **row))
    frame = pd.DataFrame(records)
    past = frame[frame.lag.isin(c['primary_lags'])]
    block = past.groupby(['arm', 'seed', 'circuit_seed']).test_accuracy.mean().groupby(['arm', 'seed']).mean()
    table = pd.read_csv(root/'seed-blocks.csv').set_index(['arm', 'seed'])
    for key, value in block.items():
        assert abs(table.loc[key, 'test_accuracy']-value) < 1e-14
    summary = read(root/'summary.json')
    if manifest['cohort'] != 'smoke':
        delta = (block.loc['real']-block.loc['role_shuffled']).to_numpy()
        stats = summary['real_minus_shuffled']['test_accuracy']
        assert abs(stats['mean']-delta.mean()) < 1e-14
        assert abs(stats['median']-np.median(delta)) < 1e-14
        assert abs(stats['variance']-delta.var(ddof=1)) < 1e-14
        rng = np.random.default_rng(c['bootstrap_seed'])
        means = [np.mean(delta[rng.integers(0, len(delta), len(delta))]) for _ in range(c['bootstrap_draws'])]
        np.testing.assert_allclose(stats['bootstrap95'], np.quantile(means, [.025, .975]), atol=1e-14)
        required = 4 if manifest['cohort'] == 'main' else 3
        real = table.loc['real']
        access = bool(real.frequency_excess.mean() >= .05 and real.null_excess.mean() >= .05 and
                      ((real.frequency_excess >= .05) & (real.null_excess >= .05)).sum() >= required and
                      real.r2_vs_frequency.mean() > 0)
        assert summary['gates']['real_wiring_advantage'] == bool(access and delta.mean() >= .03 and (delta > 0).sum() >= required)
    historical = read(root/'prior-results-sha256.json')
    for path, digest in {**historical, **manifest['context']['source_sha256']}.items():
        assert core.sha256(path) == digest, path
    return dict(root=str(root), manifest_sha256=core.sha256(root/'manifest.json'),
                verified_graph_runs=len(records)//len(c['lags']), lag_rows=len(records),
                artifact_hashes=len(manifest['artifacts']), old_results_unchanged=len(historical),
                alignment_predictions_metrics_bootstrap_gates_verified=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('roots', type=Path, nargs='+'); args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    checked = [verify(root) for root in args.roots]
    completed = subprocess.run([sys.executable, '-m', 'pytest', '-q'], capture_output=True, text=True)
    (args.out/'pytest.txt').write_text(completed.stdout+completed.stderr, encoding='utf-8')
    completed.check_returncode()
    core.write_json(args.out/'checks.json', dict(studies=checked, pytest_exit_code=completed.returncode,
                    verifier_sha256=core.sha256(__file__), environment=core.environment()))
    print(checked); print(completed.stdout)
