"""Locked observation-location diagnostic; recurrent dynamics never depend on readout sites."""
import argparse
import hashlib
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits
from flying.training import whole_brain_memory as core
from alphabet_memory import read, check, SymbolEncoder
from structural_k4 import build
from edge_panel import anatomy
from edge_masks import cut_edges
from frozen_state_probe import labels, fit_fold, measures, Budget
from context_memory import estimate
from pathway_memory import seal

CONFIG_HASH = '158f97ab140cf9bd6c4f9424b08df57091a9897101e0dbca46394a943bfba681'


def select_observations(ids, roles, bank, seed):
    pool = np.flatnonzero((roles == 'KC') & ~np.any(bank != 0, axis=0))
    pool = np.array(sorted(pool, key=lambda i: int(ids[i])))
    if len(pool) < 48:
        raise ValueError('Insufficient unstimulated KCs; do not substitute another set')
    return dict(MBON=np.flatnonzero(roles == 'MBON'),
                KC_unstimulated=np.sort(np.random.default_rng(seed).choice(pool, 48, replace=False)))


def execute(c, ci, block, arm):
    base, _, graph, raw = build(c, ci, block['seed'])
    _, ids, roles = anatomy(ci)
    bank = base.encoder.patterns.copy()
    sites = select_observations(ids, roles, bank, block['observation_seed'])
    with np.load(Path(c['mask_bank']) / f'c{ci}_s{block["mask_block_seed"]}.npz') as z:
        mask = z[arm]
    weights = cut_edges(base.weights, mask)
    model = core.TimedReservoir(weights, SymbolEncoder(bank), roles, c['leak'], c['schedule'])
    union = np.concatenate([sites[k] for k in c['sites']])
    assert len(union) == len(set(union)) == 96
    common = dict(input_patterns=bank, edge_mask=mask)
    for split in ['train', 'test']:
        symbols = np.random.default_rng(block[split + '_seed']).integers(0, 4, c['warmup'] + c[split + '_samples']).astype(np.uint8)
        x, diag = core.collect(model, union, symbols, c['activity_epsilon'])
        common[split + '_symbols'] = symbols
        common[split + '_active_counts'] = diag['active_counts']
        common[split + '_full_norm'] = diag['full_norm']
        common[split + '_union_features'] = x
        if split == 'train':
            decays = {k: core.decay_probe(model, obs, c['decay_steps']) for k, obs in sites.items()}
    results = {}
    for j, (site, obs) in enumerate(sites.items()):
        a = {k: v for k, v in common.items() if not k.endswith('_union_features')}
        a['observed_indices'] = obs
        a['observed_ids'] = np.array([str(ids[i]) for i in obs])
        for split in ['train', 'test']:
            a[split + '_features'] = common[split + '_union_features'][:, j * 48:(j + 1) * 48].copy()
        warm = c['warmup']; x = a['train_features'][warm:]
        a.update(fit_fold(x, a['test_features'][warm:],
                         labels(a['train_symbols'], np.arange(warm, len(a['train_symbols'])), c['lags']),
                         labels(a['test_symbols'], np.arange(warm, len(a['test_symbols'])), c['lags']), c['alpha']))
        a['decay_full'], a['decay_observed'] = decays[site]
        metrics = measures([a], c['lags'])
        for k, row in enumerate(metrics):
            row['training_mse'] = float(np.mean((a['train_scores'][:, k] - np.eye(4)[a['ytrain'][:, k]]) ** 2))
        n = dict(**core.state_diagnostics(x), observed_sparsity=float(np.mean(abs(x) <= c['activity_epsilon'])),
                 mean_active_neurons=float(a['train_active_counts'][warm:].mean()),
                 observed_norm=float(np.linalg.norm(x, axis=1).mean()),
                 decay_ratio32=float(a['decay_observed'][-1] / a['decay_observed'][0]) if a['decay_observed'][0] else None)
        for value in a.values():
            if np.asarray(value).dtype.kind not in 'US':
                assert np.isfinite(value).all()
        results[site] = (a, metrics, n)
    graph.update(arm=arm, removed_count=int(mask.sum()), original_weight_sha256=graph['weight_sha256'],
                 weight_sha256=core.weight_hash(weights), observation_changes_dynamics=False,
                 renormalized_after_cut=False, mask_bank=c['mask_bank'], mask_block_seed=block['mask_block_seed'])
    return results, graph, weights


def aggregate(frame, c):
    cols = ['test_accuracy', 'frequency_excess', 'null_excess', 'r2_vs_frequency']
    cases = frame[frame.lag.isin(c['primary_lags'])].groupby(['cohort', 'seed', 'circuit_seed', 'arm', 'site'])[cols].mean()
    return cases.groupby(['cohort', 'seed', 'arm', 'site'])[cols].mean().reset_index()


def infer(blocks, c):
    summary = {}; pairs = []
    for cohort in ['discovery', 'confirmation']:
        q = blocks[blocks.cohort == cohort]
        cells = {}; access = {}; losses = {}; extra = {}
        for site in c['sites']:
            t = q[q.site == site]
            for arm in ['intact', 'DAN_MBON']:
                b = t[t.arm == arm].set_index('seed')
                cells[site + '/' + arm] = {k: estimate(b[k], c) for k in ['test_accuracy', 'frequency_excess', 'null_excess', 'r2_vs_frequency']}
                access[site + '/' + arm] = bool((b.frequency_excess >= .05).all() and (b.null_excess >= .05).all() and b.r2_vs_frequency.mean() > 0)
            intact = t[t.arm == 'intact'].set_index('seed').test_accuracy
            target = t[t.arm == 'DAN_MBON'].set_index('seed').test_accuracy
            ctrl = t[t.arm.str.contains('_control')].groupby('seed').test_accuracy.mean()
            losses[site] = intact - target; extra[site] = ctrl - target
            pairs.extend(dict(cohort=cohort, site=site, seed=int(seed), intact=float(intact[seed]), target=float(target[seed]),
                              control=float(ctrl[seed]), loss=float(losses[site][seed]), extra=float(extra[site][seed])) for seed in target.index)
        interaction = losses['MBON'] - losses['KC_unstimulated']
        gates = dict(access=all(access[k] for k in ['MBON/intact', 'KC_unstimulated/intact', 'KC_unstimulated/DAN_MBON']),
                     mbon_loss=bool(losses['MBON'].mean() >= .05 and (losses['MBON'] > 0).all()),
                     kc_retention=bool(losses['KC_unstimulated'].mean() <= .02),
                     interaction=bool(interaction.mean() >= .05 and (interaction > 0).all()))
        summary[cohort] = dict(cells=cells, access=access, loss={k: estimate(v, c) for k, v in losses.items()},
                               extra={k: estimate(v, c) for k, v in extra.items()}, interaction=estimate(interaction, c),
                               gates=gates, primary_pass=all(gates.values()),
                               secondary_mbon_specificity=bool(access['MBON/intact'] and extra['MBON'].mean() >= .05 and (extra['MBON'] > 0).all()))
    summary['confirmed_primary'] = all(summary[k]['primary_pass'] for k in ['discovery', 'confirmation'])
    summary['confirmed_secondary_mbon'] = all(summary[k]['secondary_mbon_specificity'] for k in ['discovery', 'confirmation'])
    summary['cohorts_pooled'] = False
    return summary, pd.DataFrame(pairs)


def baseline(c):
    old = read(Path('configs/pathway_memory.json')); b = old['cohorts']['discovery']['blocks'][0]
    block = dict(b, observation_seed=204001, mask_block_seed=b['seed'])
    result, _, _ = execute(c, 701, block, 'intact')
    path = Path('results/pathway_memory/discovery/intact_c701_s181142'); check(path)
    with np.load(path / 'checkpoint.npz') as z:
        common = sorted(set(z.files) & set(result['MBON'][0]))
        for key in common:
            np.testing.assert_array_equal(z[key], result['MBON'][0][key], err_msg=key)
    assert result['MBON'][1] == read(path / 'metrics.json')
    return dict(source=path.as_posix(), manifest_sha256=core.sha256(path / 'manifest.json'), exact_common_arrays=common, exact_metrics=True)


@threadpool_limits.wrap(limits=1)
def run(config, out):
    c = read(config); assert core.fingerprint(c) == CONFIG_HASH
    assert not subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip(), 'Commit before execution'
    for path, digest in [(c['source'], c['source_manifest_sha256']), (c['mask_bank'], c['mask_manifest_sha256'])]:
        check(Path(path)); assert core.sha256(Path(path) / 'manifest.json') == digest
    context = core.source_context(c)
    dependencies = ['observation_location', 'verify_observation_location', 'pathway_memory', 'pathway_masks',
                    'edge_panel', 'edge_masks', 'structural_k4', 'kc_ablation', 'kc_frozen',
                    'alphabet_memory', 'frozen_state_probe', 'normalization_transfer', 'context_memory', 'verify_neuron_panel']
    paths = [Path('scripts') / (name + '.py') for name in dependencies] + [config, Path('docs/observation-location-selection.json')]
    context['source_sha256'].update({p.as_posix(): core.sha256(p) for p in paths})
    for p, h in context['source_sha256'].items():
        assert hashlib.sha256(subprocess.check_output(['git', 'show', context['git_commit'] + ':' + p])).hexdigest() == h, p
    out.mkdir(parents=True, exist_ok=False); budget = Budget(c)
    try:
        prior = {p: core.sha256(p) for p in subprocess.check_output(['git', 'ls-files', 'results/'], text=True).splitlines()}
        core.write_json(out / 'prior-results-sha256.json', prior); core.write_json(out / 'config.json', c)
        core.write_json(out / 'baseline.json', baseline(c)); print('Archived III-D MBON baseline exactly reproduced', flush=True)
        rows = []; neural = []; count = 0
        for cohort, settings in c['cohorts'].items():
            cc = dict(c, **settings)
            for b in cc['blocks']:
                for ci in cc['circuit_seeds']:
                    for arm in c['arms']:
                        budget.check(); result, graph, weights = execute(cc, ci, b, arm)
                        replay, g2, w2 = execute(cc, ci, b, arm)
                        assert graph == g2 and core.weight_hash(weights) == core.weight_hash(w2)
                        for site, (a, metrics, n) in result.items():
                            for key in a:
                                np.testing.assert_array_equal(a[key], replay[site][0][key], err_msg=key)
                            assert metrics == replay[site][1] and n == replay[site][2]
                            ident = dict(cohort=cohort, seed=b['seed'], circuit_seed=ci, arm=arm, site=site)
                            dest = out / cohort / f'{arm}_{site}_c{ci}_s{b["seed"]}'; dest.mkdir(parents=True)
                            np.savez_compressed(dest / 'checkpoint.npz', **a); sparse.save_npz(dest / 'weights.npz', weights)
                            core.write_json(dest / 'metrics.json', metrics); core.write_json(dest / 'neural.json', n); core.write_json(dest / 'graph.json', graph)
                            rows.extend(dict(**ident, **r) for r in metrics); neural.append(dict(**ident, **n)); count += 1
                            seal(dest, cc, context, identity={**ident, **b}, full_replay=True, independent_lstsq=True)
                    print(f'{cohort} c{ci} s{b["seed"]}:10 fits/replays', flush=True)
        frame = pd.DataFrame(rows); frame.to_csv(out / 'raw-lag-table.csv', index=False)
        blocks = aggregate(frame, c); blocks.to_csv(out / 'seed-blocks.csv', index=False)
        summary, pairs = infer(blocks, c); core.write_json(out / 'summary.json', summary); pairs.to_csv(out / 'paired-differences.csv', index=False)
        pd.DataFrame(neural).to_csv(out / 'neural-diagnostics.csv', index=False)
        for p, h in {**prior, **context['source_sha256']}.items():
            assert core.sha256(p) == h, p
        assert count == 130; budget.check()
        core.write_json(out / 'verification.json', dict(fits=count, exact_replays=count, prior_files_unchanged=len(prior), budget=budget.close()))
        seal(out, c, context, environment=core.environment())
        print({k: summary[k] for k in ['confirmed_primary', 'confirmed_secondary_mbon']}, flush=True)
    except Exception as exc:
        core.write_json(out / 'failure.json', dict(error=repr(exc))); raise
    finally:
        budget.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--config', type=Path, default=Path('configs/observation_location.json')); p.add_argument('--out', type=Path, required=True)
    args = p.parse_args(); run(args.config, args.out)
