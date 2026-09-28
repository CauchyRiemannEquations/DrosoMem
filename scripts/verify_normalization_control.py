"""Independent saved matrix, target, metric, pairing and decision verification."""
import argparse
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits
from alphabet_memory import read, check
from flying.training import whole_brain_memory as core


@threadpool_limits.wrap(limits=1)
def verify(root, direction=None):
    manifest = check(root); c = manifest['effective_config']; rows = []; pairs = {}
    for group in sorted(root.glob('*_c*_s*')):
        run = check(group); identity = run['identity']; arm = identity['arm']
        graph = read(group/'graph.json'); neural = read(group/'neural.json')
        directory = Path(f'data/flywire_783_mb_left_kc512_s{identity["circuit_seed"]}')
        original, ids, _ = core.load_connectome(directory); roles, _ = core.load_roles(directory, ids)
        raw = sparse.load_npz(group/'raw-graph.npz'); weights = sparse.load_npz(group/'weights.npz')
        assert graph['raw_sha256'] == core.weight_hash(raw) and graph['weight_sha256'] == core.weight_hash(weights)
        denominator = np.asarray(abs(raw if arm == 'renormalized' else original).sum(axis=1)).ravel()
        expected = np.divide(.9, denominator, out=np.zeros(len(ids)), where=denominator > 0)
        with np.load(group/'checkpoint.npz', allow_pickle=False) as a:
            np.testing.assert_array_equal(expected, a['normalization_factors'])
            np.testing.assert_array_equal(raw.toarray()*expected[:, None], weights.toarray())
            dense = abs(weights.toarray()); row_l1 = dense.sum(axis=1)
            np.testing.assert_allclose(row_l1, a['row_l1'], atol=1e-14)
            bounds = .4+.6*row_l1; kc = np.asarray(roles) == 'KC'
            for i in np.flatnonzero(np.asarray(roles) == 'MBON'):
                bounds[i] = .4+.6*(sum(dense[i, ~kc])+sum(dense[i, kc]*(.4+.6*row_l1[kc])))
            np.testing.assert_allclose(bounds, a['row_lipschitz_bounds'], atol=1e-14)
            assert graph['contraction_certified'] == bool(bounds.max() < 1)
            assert a['max_abs_by_stream'].max() <= 1+1e-12
            perturb = np.random.default_rng(identity['perturb_seed']).uniform(-1e-6, 1e-6, len(ids))
            np.testing.assert_array_equal(perturb, a['initial_perturbation'])
            assert neural['initial_difference_ratio256'] == float(a['perturbation_full_inf'][-1]/a['perturbation_full_inf'][0])
            assert neural['empirical_forgetting'] == bool(neural['initial_difference_ratio256'] <= c['forgetting_ratio'])
            for split, label in [('train', 'ytrain'), ('test', 'ytest')]:
                w, n = c['warmup'], c[split+'_samples']
                generated = np.random.default_rng(identity[split+'_seed']).integers(0, 4, w+n).astype(np.uint8)
                np.testing.assert_array_equal(generated, a[split+'_symbols'])
                np.testing.assert_array_equal(np.column_stack([generated[w-lag:w-lag+n] for lag in c['lags']]), a[label])
            z = (a['test_features'][100:]-a['mean'])/a['scale']
            for prefix in ['', 'null_']:
                score = (z@a[prefix+'weights']+a[prefix+'bias']).reshape(a[prefix+'scores'].shape)
                np.testing.assert_array_equal(score, a[prefix+'scores'])
                np.testing.assert_array_equal(score.argmax(axis=2), a[prefix+'predictions'])
            metrics = read(group/'metrics.json')
            for j, row in enumerate(metrics):
                yt = a['ytest'][:, j]; oh = np.eye(4)[yt]
                accuracy = float(np.mean(a['predictions'][:, j] == yt))
                null = float(np.mean(a['null_predictions'][:, j] == yt)); freq = float(np.mean(a['majority'][:, j] == yt))
                freqvec = a['bias'].reshape(-1, 4)[j]
                checked = dict(test_accuracy=accuracy, null_accuracy=null, frequency_accuracy=freq,
                    frequency_excess=accuracy-freq, null_excess=accuracy-null,
                    train_accuracy=float(np.mean(a['train_scores'][:, j].argmax(axis=1) == a['ytrain'][:, j])),
                    training_mse=float(np.mean((a['train_scores'][:, j]-np.eye(4)[a['ytrain'][:, j]])**2)),
                    test_mse=float(np.mean((a['scores'][:, j]-oh)**2)),
                    r2_vs_frequency=float(1-((a['scores'][:, j]-oh)**2).sum()/((oh-freqvec)**2).sum()))
                for key, value in checked.items():
                    assert abs(row[key]-value) < 1e-14, (group, key)
                rows.append(dict(**identity, **row))
            key = (identity['seed'], identity['circuit_seed'])
            pairs.setdefault(key, {})[arm] = dict(raw=graph['raw_sha256'], factors=a['normalization_factors'].copy(),
                paired={name: a[name].copy() for name in ['train_symbols', 'test_symbols', 'input_patterns', 'observed_indices', 'initial_perturbation']})
    for triple in pairs.values():
        assert triple['renormalized']['raw'] == triple['fixed_original']['raw']
        np.testing.assert_array_equal(triple['real']['factors'], triple['fixed_original']['factors'])
        for arm in ['renormalized', 'fixed_original']:
            for key in triple['real']['paired']:
                np.testing.assert_array_equal(triple['real']['paired'][key], triple[arm]['paired'][key])
    frame = pd.DataFrame(rows); metrics = [name for name in checked]
    raw = frame[frame.lag.isin(c['primary_lags'])].groupby(['arm', 'seed', 'circuit_seed'])[metrics].mean()
    blocks = raw.groupby(['arm', 'seed']).mean()
    saved = pd.read_csv(root/'seed-blocks.csv').set_index(['arm', 'seed'])
    np.testing.assert_allclose(saved[metrics].sort_index(), blocks[metrics].sort_index(), atol=1e-14, rtol=0)
    summary = read(root/'summary.json')
    if manifest['cohort'] != 'smoke':
        required = 4 if manifest['cohort'] == 'main' else 3
        access = {}
        for arm in ['real', 'renormalized', 'fixed_original']:
            b = blocks.loc[arm]
            access[arm] = bool(b.frequency_excess.mean() >= .05 and b.null_excess.mean() >= .05 and
                ((b.frequency_excess >= .05) & (b.null_excess >= .05)).sum() >= required and b.r2_vs_frequency.mean() > 0)
        for name, left, right in [('renormalized_minus_fixed', 'renormalized', 'fixed_original'),
                ('real_minus_renormalized', 'real', 'renormalized'), ('real_minus_fixed', 'real', 'fixed_original')]:
            delta = blocks.loc[left]-blocks.loc[right]
            for metric in metrics:
                x = delta[metric].to_numpy(); stats = summary['contrasts'][name][metric]
                assert abs(stats['mean']-x.mean()) < 1e-14
                assert abs(stats['median']-np.median(x)) < 1e-14
                assert abs(stats['variance']-x.var(ddof=1)) < 1e-14
                rng = np.random.default_rng(c['bootstrap_seed'])
                resampled = [np.mean(x[rng.integers(0, len(x), len(x))]) for _ in range(c['bootstrap_draws'])]
                np.testing.assert_allclose(stats['bootstrap95'], np.quantile(resampled, [.025, .975]), atol=1e-14, rtol=0)
        delta = blocks.loc['renormalized'].test_accuracy-blocks.loc['fixed_original'].test_accuracy
        if direction is None:
            assert manifest['cohort'] == 'main'; direction = int(np.sign(delta.mean()))
        passed = bool(direction != 0 and direction*delta.mean() >= c['material_effect'] and
                      (direction*delta > 0).sum() >= required and access['renormalized'] and access['fixed_original'])
        assert summary['gates'] == dict(direction=direction, past_access=access, material_normalization_effect=passed)
    for path, digest in {**read(root/'prior-results-sha256.json'), **manifest['context']['source_sha256']}.items():
        assert core.sha256(path) == digest, path
    return dict(root=root.as_posix(), graph_runs=len(pairs)*3, lag_rows=len(rows),
                manifest_sha256=core.sha256(root/'manifest.json'), verified=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True); p.add_argument('roots', type=Path, nargs='+')
    args = p.parse_args(); args.out.mkdir(parents=True, exist_ok=False)
    checked = []; direction = None
    for root in args.roots:
        m = check(root)
        checked.append(verify(root, direction if m['cohort'] == 'confirmation' else None))
        if m['cohort'] == 'main':
            direction = read(root/'summary.json')['gates']['direction']
    tests = subprocess.run([sys.executable, '-m', 'pytest', '-q'], capture_output=True, text=True)
    (args.out/'pytest.txt').write_text(tests.stdout+tests.stderr, encoding='utf-8'); tests.check_returncode()
    core.write_json(args.out/'checks.json', dict(studies=checked, verifier_sha256=core.sha256(__file__),
        pytest_exit_code=tests.returncode, environment=core.environment()))
    print(checked); print(tests.stdout)
