"""Independent artifact-based audit of train-only moment-aligned frozen transfer."""
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
    manifest = check(root); c = manifest['config']; rows = []; counts = {}
    for path in sorted(root.glob('*/*/*/manifest.json')):
        dest = path.parent; record = check(dest); identity = record['identity']
        sp, tp = Path(record['source_path']), Path(record['target_path']); check(sp); check(tp)
        assert core.sha256(sp/'manifest.json') == record['source_manifest_sha256']
        assert core.sha256(tp/'manifest.json') == record['target_manifest_sha256']
        assert record['target_fits'] == 0 and record['target_moment_fits'] == 1
        op = Path(record['unaligned_path']); check(op)
        assert core.sha256(op/'manifest.json') == record['unaligned_manifest_sha256']
        with np.load(dest/'checkpoint.npz', allow_pickle=False) as a, np.load(sp/'checkpoint.npz', allow_pickle=False) as source, np.load(tp/'checkpoint.npz', allow_pickle=False) as target:
            for key in ['mean', 'scale', 'weights', 'bias', 'null_weights', 'null_bias']:
                np.testing.assert_array_equal(a[key], source[key])
            for key in ['train_symbols', 'test_symbols', 'input_patterns', 'observed_indices', 'ytrain', 'ytest']:
                np.testing.assert_array_equal(source[key], target[key])
            np.testing.assert_array_equal(a['xtest'], target['test_features'][100:])
            n = len(a['ytest']); symbols = target['test_symbols']
            truth = np.column_stack([symbols[100-lag:100-lag+n] for lag in c['lags']])
            np.testing.assert_array_equal(a['ytest'], truth)
            # Compute moments independently from complete archived training trajectory.
            train = target['train_features'][100:]
            mu = np.mean(train, axis=0); scale = np.maximum(np.std(train, axis=0), 1e-5)
            np.testing.assert_array_equal(a['target_mean'], mu)
            np.testing.assert_array_equal(a['target_scale'], scale)
            z = (target['test_features'][100:]-mu)/scale
            with np.load(op/'checkpoint.npz', allow_pickle=False) as old:
                np.testing.assert_array_equal(a['unaligned_scores'], old['scores'])
            oldz = (target['xtest']-source['mean'])/source['scale']
            np.testing.assert_array_equal(a['unaligned_scores'],
                (oldz@source['weights']+source['bias']).reshape(a['scores'].shape))
            for prefix in ['', 'null_']:
                score = (z@source[prefix+'weights']+source[prefix+'bias']).reshape(n, len(c['lags']), 4)
                np.testing.assert_array_equal(a[prefix+'scores'], score)
                np.testing.assert_array_equal(a[prefix+'predictions'], score.argmax(axis=2))
            np.testing.assert_array_equal(a['source_within_scores'], source['scores'])
            np.testing.assert_array_equal(a['target_refit_scores'], target['scores'])
            measured = read(dest/'metrics.json')
            for j, row in enumerate(measured):
                y = truth[:, j]; onehot = np.eye(4)[y]
                accuracy = float(np.mean(a['predictions'][:, j] == y))
                frequency = float(np.mean(source['bias'].reshape(-1, 4)[j].argmax() == y))
                null = float(np.mean(a['null_predictions'][:, j] == y))
                within = float(np.mean(source['scores'][:, j].argmax(axis=1) == y))
                refit = float(np.mean(target['scores'][:, j].argmax(axis=1) == y))
                unaligned = float(np.mean(a['unaligned_scores'][:, j].argmax(axis=1) == y))
                error = np.square(a['scores'][:, j]-onehot).sum()
                denominator = np.square(onehot-source['bias'].reshape(-1, 4)[j]).sum()
                checked = dict(test_accuracy=accuracy, frequency_accuracy=frequency, null_accuracy=null,
                    frequency_excess=accuracy-frequency, null_excess=accuracy-null, r2_vs_frequency=float(1-error/denominator),
                    test_mse=float(error/onehot.size), source_within_accuracy=within, target_refit_accuracy=refit,
                    transfer_minus_source=accuracy-within, transfer_minus_target_refit=accuracy-refit,
                    unaligned_accuracy=unaligned, aligned_minus_unaligned=accuracy-unaligned)
                for key, value in checked.items():
                    np.testing.assert_allclose(row[key], value, atol=1e-14, rtol=1e-12)
                rows.append(dict(**identity, lag=c['lags'][j], **checked))
        counts[identity['cohort']] = counts.get(identity['cohort'], 0)+1
    assert counts == dict(confirmation=12, main=20, smoke=2), counts
    frame = pd.DataFrame(rows); metrics = list(checked)
    blocks = frame[frame.lag.isin(c['primary_lags'])].groupby(['cohort', 'direction', 'seed', 'circuit_seed'])[metrics].mean().groupby(['cohort', 'direction', 'seed']).mean()
    saved = pd.read_csv(root/'seed-blocks.csv').set_index(['cohort', 'direction', 'seed'])
    np.testing.assert_allclose(saved[metrics].sort_index(), blocks[metrics].sort_index(), atol=1e-13, rtol=1e-12)
    summary = read(root/'summary.json'); gates = {}
    for (cohort, direction), b in blocks.reset_index().query("cohort != 'smoke'").groupby(['cohort', 'direction']):
        key = f'{cohort}/{direction}'; required = 4 if cohort == 'main' else 3
        access = bool(b.frequency_excess.mean() >= .05 and b.null_excess.mean() >= .05 and
            ((b.frequency_excess >= .05) & (b.null_excess >= .05)).sum() >= required and b.r2_vs_frequency.mean() > 0)
        loss = b.transfer_minus_target_refit
        retention = bool(loss.mean() >= -.05 and (loss >= -.05).sum() >= required)
        improved = bool(b.aligned_minus_unaligned.mean() >= .05 and (b.aligned_minus_unaligned > 0).sum() >= required)
        gates[key] = dict(alignment_improvement=improved, past_access=access, retention_tolerance=retention, portable_decoding=access and retention)
        for metric in metrics:
            values = b[metric].to_numpy(); stats = summary['cells'][key][metric]
            for name, value in [('mean', values.mean()), ('median', np.median(values)), ('variance', values.var(ddof=1))]:
                np.testing.assert_allclose(stats[name], value, atol=1e-13, rtol=1e-12)
            rng = np.random.default_rng(c['bootstrap_seed'])
            draws = [np.mean(values[rng.integers(0, len(values), len(values))]) for _ in range(c['bootstrap_draws'])]
            np.testing.assert_allclose(stats['bootstrap95'], np.quantile(draws, [.025, .975]), atol=1e-12, rtol=1e-12)
    assert gates == summary['gates']
    for outcome, field in [('supported_improvement', 'alignment_improvement'), ('supported_access', 'past_access'), ('supported_portability', 'portable_decoding')]:
        expected = ['_to_'.join(pair) for pair in c['directions'] if all(gates[f'{cohort}/{"_to_".join(pair)}'][field] for cohort in ['main', 'confirmation'])]
        assert summary[outcome] == expected
    assert summary['both_directions_portable'] == (len(summary['supported_portability']) == 2)
    for p, h in {**read(root/'prior-results-sha256.json'), **manifest['context']['source_sha256']}.items():
        assert core.sha256(p) == h, p
    return dict(manifest_sha256=core.sha256(root/'manifest.json'), directions=counts, verified_lag_rows=len(rows),
                new_target_fits=0, target_moment_fits=34, pairing_scores_metrics_bootstrap_gates_verified=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--out', type=Path, required=True)
    p.add_argument('source', type=Path); args = p.parse_args(); args.out.mkdir(parents=True, exist_ok=False)
    verified = verify(args.source)
    tests = subprocess.run([sys.executable, '-m', 'pytest', '-q'], capture_output=True, text=True)
    (args.out/'pytest.txt').write_text(tests.stdout+tests.stderr, encoding='utf-8'); tests.check_returncode()
    core.write_json(args.out/'checks.json', dict(verification=verified, pytest_exit_code=tests.returncode,
        verifier_sha256=core.sha256(__file__), environment=core.environment()))
    print(verified); print(tests.stdout)
