"""Train-only label-free moment adapter; frozen source heads across normalization."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from alphabet_memory import read, check
from frozen_state_probe import Budget, verify_fold
from context_memory import estimate
from flying.training import whole_brain_memory as core

FROZEN = ['mean', 'scale', 'weights', 'bias', 'null_weights', 'null_bias']
METRICS = ['test_accuracy', 'frequency_accuracy', 'null_accuracy', 'frequency_excess',
           'null_excess', 'r2_vs_frequency', 'test_mse', 'source_within_accuracy',
           'target_refit_accuracy', 'transfer_minus_source', 'transfer_minus_target_refit', 'unaligned_accuracy', 'aligned_minus_unaligned']


def fit_adapter(training_features):
    """No labels, test features or target head parameters enter this API."""
    x = np.asarray(training_features)
    if x.ndim != 2 or len(x) == 0 or not np.isfinite(x).all():
        raise ValueError('Expected nonempty finite training feature matrix')
    return dict(target_mean=x.mean(axis=0), target_scale=np.maximum(x.std(axis=0), 1e-5))


def predict_aligned(source, adapter, features):
    z = (np.asarray(features)-adapter['target_mean'])/adapter['target_scale']
    shape = (len(z), source['bias'].size//4, 4)
    return ((z@source['weights']+source['bias']).reshape(shape),
            (z@source['null_weights']+source['null_bias']).reshape(shape))


def transfer(source, target):
    from normalization_transfer import transfer as unaligned_transfer
    a = unaligned_transfer(source, target)
    a['unaligned_scores'] = a['scores'].copy()
    a.update(fit_adapter(target['xtrain']))
    a['scores'], a['null_scores'] = predict_aligned(source, a, target['xtest'])
    a['predictions'] = a['scores'].argmax(axis=2)
    a['null_predictions'] = a['null_scores'].argmax(axis=2)
    for value in a.values():
        assert np.isfinite(value).all()
    return a


def verify_transfer(source, target, a):
    # Source-only exact refits and independent augmented solver. No target fitting.
    verify_fold(source)
    for key in FROZEN:
        np.testing.assert_array_equal(a[key], source[key])
    np.testing.assert_array_equal(a['xtest'], target['xtest'])
    np.testing.assert_array_equal(a['ytest'], target['ytest'])
    np.testing.assert_array_equal(a['source_within_scores'], source['scores'])
    np.testing.assert_array_equal(a['target_refit_scores'], target['scores'])
    z = (source['xtrain']-source['mean'])/source['scale']
    np.testing.assert_array_equal(a['target_mean'], target['xtrain'].mean(axis=0))
    np.testing.assert_array_equal(a['target_scale'], np.maximum(target['xtrain'].std(axis=0), 1e-5))
    xtest = (target['xtest']-a['target_mean'])/a['target_scale']
    mapped = source['mean']+source['scale']*xtest
    mapped_z = (mapped-source['mean'])/source['scale']
    targets = np.eye(4)[source['ytrain']].reshape(len(z), -1)
    augmented = np.vstack([z, np.eye(z.shape[1])])
    for label, truth in [('', targets), ('null_', np.roll(targets, len(targets)//2, axis=0))]:
        bias = truth.mean(axis=0)
        rhs = np.vstack([truth-bias, np.zeros((z.shape[1], truth.shape[1]))])
        fitted = np.linalg.lstsq(augmented, rhs, rcond=None)[0]
        reference = (xtest@fitted+bias).reshape(a[label+'scores'].shape)
        np.testing.assert_allclose(reference, a[label+'scores'], atol=1e-9, rtol=1e-9)
        np.testing.assert_array_equal(reference.argmax(axis=2), a[label+'predictions'])
        replay = (xtest@a[label+'weights']+a[label+'bias']).reshape(a[label+'scores'].shape)
        np.testing.assert_array_equal(replay, a[label+'scores'])
        mapped_score = (mapped_z@source[label+'weights']+source[label+'bias']).reshape(replay.shape)
        np.testing.assert_allclose(mapped_score, replay, atol=1e-9, rtol=1e-9)


def measures(a, lags):
    rows = []
    for j, lag in enumerate(lags):
        truth = a['ytest'][:, j]; onehot = np.eye(4)[truth]
        accuracy = float(np.mean(a['predictions'][:, j] == truth))
        frequency = float(np.mean(a['frequency_predictions'][:, j] == truth))
        null = float(np.mean(a['null_predictions'][:, j] == truth))
        within = float(np.mean(a['source_within_scores'][:, j].argmax(axis=1) == truth))
        refit = float(np.mean(a['target_refit_scores'][:, j].argmax(axis=1) == truth))
        unaligned = float(np.mean(a['unaligned_scores'][:, j].argmax(axis=1) == truth))
        error = float(np.sum((a['scores'][:, j]-onehot)**2))
        denominator = float(np.sum((onehot-a['bias'].reshape(-1, 4)[j])**2))
        rows.append(dict(lag=lag, test_accuracy=accuracy, frequency_accuracy=frequency, null_accuracy=null,
            frequency_excess=accuracy-frequency, null_excess=accuracy-null, r2_vs_frequency=1-error/denominator,
            test_mse=error/onehot.size, source_within_accuracy=within, target_refit_accuracy=refit,
            transfer_minus_source=accuracy-within, transfer_minus_target_refit=accuracy-refit,
            unaligned_accuracy=unaligned, aligned_minus_unaligned=accuracy-unaligned))
    return rows


def decision(blocks, cohort, c):
    required = 4 if cohort == 'main' else 3
    access = bool(blocks.frequency_excess.mean() >= c['access_margin'] and blocks.null_excess.mean() >= c['access_margin']
        and ((blocks.frequency_excess >= c['access_margin']) & (blocks.null_excess >= c['access_margin'])).sum() >= required
        and blocks.r2_vs_frequency.mean() > 0)
    delta = blocks.transfer_minus_target_refit
    tolerance = bool(delta.mean() >= -c['retention_tolerance'] and (delta >= -c['retention_tolerance']).sum() >= required)
    improvement = bool(blocks.aligned_minus_unaligned.mean() >= c['alignment_margin'] and
        (blocks.aligned_minus_unaligned > 0).sum() >= required)
    return dict(alignment_improvement=improvement, past_access=access,
        retention_tolerance=tolerance, portable_decoding=access and tolerance)


def summarize(rows, out, c):
    frame = pd.DataFrame(rows); frame.to_csv(out/'raw-lag-table.csv', index=False)
    raw = frame[frame.lag.isin(c['primary_lags'])].groupby(['cohort', 'direction', 'seed', 'circuit_seed'])[METRICS].mean().reset_index()
    raw.to_csv(out/'raw-past-table.csv', index=False)
    blocks = raw.groupby(['cohort', 'direction', 'seed'])[METRICS].mean().reset_index(); blocks.to_csv(out/'seed-blocks.csv', index=False)
    cells, gates = {}, {}
    for (cohort, direction), b in blocks[blocks.cohort != 'smoke'].groupby(['cohort', 'direction']):
        key = f'{cohort}/{direction}'; cells[key] = {}
        for metric in METRICS:
            stats = estimate(b[metric], c)
            if not (metric.startswith('transfer_minus_') or metric == 'aligned_minus_unaligned'):
                stats = {k: v for k, v in stats.items() if k not in ['paired_dz', 'wins', 'ties', 'losses']}
            cells[key][metric] = stats
        gates[key] = decision(b, cohort, c)
    directions = ['_to_'.join(pair) for pair in c['directions']]
    supported_access = [d for d in directions if all(gates[f'{cohort}/{d}']['past_access'] for cohort in ['main', 'confirmation'])]
    supported_portability = [d for d in directions if all(gates[f'{cohort}/{d}']['portable_decoding'] for cohort in ['main', 'confirmation'])]
    supported_improvement = [d for d in directions if all(gates[f'{cohort}/{d}']['alignment_improvement'] for cohort in ['main', 'confirmation'])]
    result = dict(supported_improvement=supported_improvement, cells=cells, gates=gates, supported_access=supported_access, supported_portability=supported_portability,
        both_directions_portable=len(supported_portability) == 2)
    core.write_json(out/'summary.json', result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, cohort in zip(axes, ['main', 'confirmation']):
        for d, color in zip(directions, ['#e17430', '#239666']):
            b = frame[(frame.cohort == cohort) & (frame.direction == d)].groupby('lag')[['test_accuracy', 'target_refit_accuracy', 'unaligned_accuracy']].mean().reindex(c['lags'])
            ax.plot(range(len(b)), b.test_accuracy, 'o-', color=color, label=d.replace('renormalized', 'R').replace('fixed_original', 'F'))
            ax.plot(range(len(b)), b.unaligned_accuracy, ':', color=color, label='unaligned')
            ax.plot(range(len(b)), b.target_refit_accuracy, '--', color=color, label='target refit')
        ax.axhline(.25, ls=':', color='gray'); ax.set(xticks=range(len(c['lags'])), xticklabels=c['lags'],
            xlabel='Symbol lag', ylabel='Independent test accuracy', ylim=(0, 1.03), title=cohort); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out/'transfer-curve.png', dpi=180); plt.close(fig)
    return result


def audit_pair(source, target, source_graph, target_graph):
    for key in ['raw_sha256', 'observed_root_ids', 'input_root_ids', 'input_mapping_sha256']:
        assert source_graph[key] == target_graph[key], key
    for key in ['train_symbols', 'test_symbols', 'input_patterns', 'observed_indices', 'ytrain', 'ytest']:
        np.testing.assert_array_equal(source[key], target[key], err_msg=key)
    for archive in [source, target]:
        np.testing.assert_array_equal(archive['xtest'], archive['test_features'][100:])
        np.testing.assert_array_equal(archive['xtrain'], archive['train_features'][100:])


@threadpool_limits.wrap(limits=1)
def run(config, out):
    c = read(config)
    assert c['directions'] == [['renormalized', 'fixed_original'], ['fixed_original', 'renormalized']]
    assert c['lags'] == [0, 1, 2, 3, 4, 5, 8, 12, 16, 24, 32] and c['primary_lags'] == [1, 2, 3, 4, 5, 8]
    assert c['scale_floor'] == 1e-5 and c['alignment_margin'] == .05
    assert c['alpha'] == 1 and c['retention_tolerance'] == .05 and c['access_margin'] == .05
    out.mkdir(parents=True, exist_ok=False); budget = Budget(c)
    try:
        context = core.source_context(c)
        for p in [str(config), 'scripts/moment_alignment.py', 'scripts/normalization_transfer.py', 'scripts/verify_moment_alignment.py', 'scripts/alphabet_memory.py', 'scripts/frozen_state_probe.py', 'scripts/context_memory.py']:
            context['source_sha256'][p] = core.sha256(p)
        prior = {p: core.sha256(p) for p in subprocess.check_output(['git', 'ls-files', 'results/'], text=True).splitlines()}
        core.write_json(out/'prior-results-sha256.json', prior); core.write_json(out/'config.json', c)
        rows, baselines, diagnostics, sources = [], [], [], {}
        oldroot = Path(c['unaligned_source']); check(oldroot)
        sources[oldroot.as_posix()] = core.sha256(oldroot/'manifest.json')
        for cohort in ['smoke', 'main', 'confirmation']:
            root = Path(c['sources'][cohort]); manifest = check(root); settings = manifest['effective_config']
            assert manifest['cohort'] == cohort and settings['lags'] == c['lags']
            expected_seeds = {'smoke': [34001], 'main': list(range(34142, 34147)), 'confirmation': list(range(41142, 41145))}[cohort]
            assert [b['seed'] for b in settings['blocks']] == expected_seeds
            assert settings['circuit_seeds'] == ([701] if cohort == 'smoke' else [701, 702])
            sources[root.as_posix()] = core.sha256(root/'manifest.json')
            for block in settings['blocks']:
                for circuit in settings['circuit_seeds']:
                    pair = {}
                    for arm in ['renormalized', 'fixed_original']:
                        p = root/f'{arm}_c{circuit}_s{block["seed"]}'; check(p)
                        with np.load(p/'checkpoint.npz', allow_pickle=False) as a:
                            pair[arm] = dict(a)
                    for source_arm, target_arm in c['directions']:
                        budget.check(); direction = source_arm+'_to_'+target_arm
                        sp = root/f'{source_arm}_c{circuit}_s{block["seed"]}'; tp = root/f'{target_arm}_c{circuit}_s{block["seed"]}'
                        source, target = pair[source_arm], pair[target_arm]
                        audit_pair(source, target, read(sp/'graph.json'), read(tp/'graph.json'))
                        verify_fold(source)  # Baseline before cross-condition prediction.
                        identity = dict(cohort=cohort, direction=direction, circuit_seed=circuit, **block)
                        # Reproduce archived unaligned transfer before new aligned predictions.
                        from normalization_transfer import transfer as old_transfer
                        olddest = oldroot/cohort/f'c{circuit}_s{block["seed"]}'/direction
                        check(olddest); unaligned = old_transfer(source, target)
                        with np.load(olddest/'checkpoint.npz', allow_pickle=False) as oldsaved:
                            for key in oldsaved.files:
                                np.testing.assert_array_equal(oldsaved[key], unaligned[key], err_msg=key)
                        a = transfer(source, target); verify_transfer(source, target, a)
                        np.testing.assert_array_equal(a['unaligned_scores'], unaligned['scores'])
                        dest = out/cohort/f'c{circuit}_s{block["seed"]}'/direction; dest.mkdir(parents=True)
                        np.savez_compressed(dest/'checkpoint.npz', **a)
                        replay = transfer(source, target)
                        with np.load(dest/'checkpoint.npz', allow_pickle=False) as saved:
                            for key in saved.files:
                                np.testing.assert_array_equal(saved[key], replay[key], err_msg=key)
                        measured = measures(a, c['lags']); old = read(sp/'metrics.json'); targetold = read(tp/'metrics.json')
                        for row, ref, targetref in zip(measured, old, targetold):
                            assert row['source_within_accuracy'] == ref['test_accuracy']
                            assert row['target_refit_accuracy'] == targetref['test_accuracy']
                        rows.extend(dict(**identity, **r) for r in measured)
                        core.write_json(dest/'metrics.json', measured)
                        baselines.append(dict(**identity, source=sp.as_posix(), exact_source_refit=True, exact_unaligned_replay=True,
                            unaligned_accuracy=float(np.mean([r['unaligned_accuracy'] for r in measured if r['lag'] in c['primary_lags']])),
                            primary_accuracy=float(np.mean([r['source_within_accuracy'] for r in measured if r['lag'] in c['primary_lags']]))))
                        diagnostic = dict(**identity, mean_abs_train_mean_shift_z=float(abs(a['train_mean_shift_z']).mean()),
                            median_train_scale_ratio=float(np.median(a['train_scale_ratio'])),
                            mean_train_feature_correlation=float(a['train_feature_correlation'].mean()),
                            mean_paired_train_cosine=float(a['paired_train_cosine'].mean()))
                        diagnostics.append(diagnostic)
                        core.write_json(dest/'manifest.json', dict(identity=identity, timestamp=datetime.now(timezone.utc).isoformat(),
                            source_path=sp.as_posix(), target_path=tp.as_posix(), source_manifest_sha256=core.sha256(sp/'manifest.json'),
                            target_manifest_sha256=core.sha256(tp/'manifest.json'), target_fits=0, target_moment_fits=1, adapter_parameters=96,
                            unaligned_path=olddest.as_posix(), unaligned_manifest_sha256=core.sha256(olddest/'manifest.json'), exact_replay=True,
                            coefficients_per_task=196, diagnostic=diagnostic,
                            artifacts={p.name: core.sha256(p) for p in dest.iterdir() if p.is_file()}))
            print(f'{cohort}: moment-aligned frozen transfer and independent verification complete', flush=True)
        summary = summarize(rows, out, c)
        core.write_json(out/'baselines.json', baselines); pd.DataFrame(diagnostics).to_csv(out/'feature-drift.csv', index=False)
        for p, h in {**prior, **context['source_sha256']}.items():
            assert core.sha256(p) == h, p
        budget.check(); usage = budget.close()
        core.write_json(out/'verification.json', dict(frozen_directions=len(baselines), new_target_fits=0, target_moment_fits=len(baselines), unaligned_exact_replays=len(baselines),
            exact_replays=len(baselines), source_only_verification_refits=len(baselines), independent_lstsq_checks=len(baselines),
            prior_files_unchanged=len(prior), budget=usage))
        core.write_json(out/'manifest.json', dict(config=c, context=context, source_manifests=sources,
            timestamp=datetime.now(timezone.utc).isoformat(), environment=core.environment(),
            artifacts={p.relative_to(out).as_posix(): core.sha256(p) for p in out.rglob('*') if p.is_file()}))
        print(dict(supported_access=summary['supported_access'], supported_portability=summary['supported_portability'], budget=usage), flush=True)
    finally:
        budget.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=Path('configs/moment_alignment.json'))
    p.add_argument('--out', type=Path, required=True); a = p.parse_args(); run(a.config, a.out)
