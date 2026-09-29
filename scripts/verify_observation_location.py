"""Independent trajectory, observation, decoder and inference audit."""
import argparse
import hashlib
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits
from flying.training import whole_brain_memory as core
from alphabet_memory import read, check, symbol_bank
from frozen_state_probe import verify_fold, Budget
from verify_neuron_panel import audit_metrics, audit_stats


def trajectory(weights, bank, roles, symbols, obs, leak):
    state = np.zeros(len(roles)); kc = np.flatnonzero(roles == 'KC'); mbon = np.flatnonzero(roles == 'MBON')
    wmb = weights[mbon].copy(); features = []; active = []; norms = []
    for symbol in symbols:
        old = state.copy(); u = bank[int(symbol)]
        new = (1 - leak) * old + leak * np.tanh(weights @ old + u)
        mixed = old.copy(); mixed[kc] = new[kc]
        new[mbon] = (1 - leak) * old[mbon] + leak * np.tanh(wmb @ mixed + u[mbon])
        state = new; features.append(state[obs].copy())
        active.append(np.count_nonzero(abs(state) > 1e-8)); norms.append(np.linalg.norm(state))
    return np.asarray(features), np.asarray(active), np.asarray(norms), state


@threadpool_limits.wrap(limits=1)
def verify(root, out):
    m = check(root); c = m['config']; out.mkdir(parents=True, exist_ok=False); budget = Budget(c)
    try:
        check(Path(c['mask_bank'])); assert core.sha256(Path(c['mask_bank']) / 'manifest.json') == c['mask_manifest_sha256']
        selection = read('docs/observation-location-selection.json')['selection']; rows = []; count = 0
        for path in sorted(root.glob('*/*/manifest.json')):
            dest = path.parent; cm = check(dest); i = cm['identity']; cc = cm['config']
            source = Path(f'data/flywire_783_mb_left_kc512_s{i["circuit_seed"]}')
            raw, ids, _ = core.load_connectome(source); raw.sort_indices(); roles, _ = core.load_roles(source, ids); roles = np.asarray(roles)
            with np.load(dest / 'checkpoint.npz') as z:
                a = dict(z)
            bank = symbol_bank(roles, i['seed'], c['input_fraction'], c['input_amplitude'])[:4]
            np.testing.assert_array_equal(bank, a['input_patterns'])
            selected = next(x for x in selection if x['seed'] == i['seed'] and x['circuit_seed'] == i['circuit_seed'])
            obs = a['observed_indices']; np.testing.assert_array_equal(obs, selected[i['site'] + '_indices'])
            if i['site'] == 'MBON':
                np.testing.assert_array_equal(obs, np.flatnonzero(roles == 'MBON'))
            else:
                pool = np.array(sorted([j for j, role in enumerate(roles) if role == 'KC' and not bank[:, j].any()], key=lambda j: int(ids[j])))
                expected = np.sort(np.random.default_rng(i['observation_seed']).choice(pool, 48, replace=False))
                np.testing.assert_array_equal(obs, expected); assert not bank[:, obs].any()
            assert len(set(obs)) == 48
            np.testing.assert_array_equal(a['observed_ids'], [str(ids[j]) for j in obs])
            with np.load(Path(c['mask_bank']) / f'c{i["circuit_seed"]}_s{i["mask_block_seed"]}.npz') as z:
                np.testing.assert_array_equal(a['edge_mask'], z[i['arm']])
            co = raw.tocoo()
            if i['arm'] == 'DAN_MBON':
                np.testing.assert_array_equal(a['edge_mask'], (roles[co.col] == 'DAN') & (roles[co.row] == 'MBON'))
            w = raw.copy(); strengths = np.asarray(abs(raw).sum(1)).ravel()
            w.data = w.data * c['gain'] / np.repeat(np.maximum(strengths, 1e-300), np.diff(w.indptr))
            # The independent formula is numerically compared to original normalization.
            original = core.normalize_condition(raw, c['normalization'], c['gain']); original.sort_indices()
            np.testing.assert_allclose(w.data, original.data, atol=1e-15, rtol=1e-14)
            original.data[a['edge_mask']] = 0; original.eliminate_zeros()
            weights = sparse.load_npz(dest / 'weights.npz'); assert core.weight_hash(weights) == core.weight_hash(original)
            for split in ['train', 'test']:
                symbols = np.random.default_rng(i[split + '_seed']).integers(0, 4, cc['warmup'] + cc[split + '_samples']).astype(np.uint8)
                np.testing.assert_array_equal(symbols, a[split + '_symbols'])
                x, active, norm, state = trajectory(weights, bank, roles, symbols, obs, c['leak'])
                np.testing.assert_array_equal(x, a[split + '_features']); np.testing.assert_array_equal(active, a[split + '_active_counts']); np.testing.assert_array_equal(norm, a[split + '_full_norm'])
                np.testing.assert_array_equal(x[cc['warmup']:], a['x' + split])
                truth = np.column_stack([symbols[np.arange(cc['warmup'], len(symbols)) - lag] for lag in c['lags']])
                np.testing.assert_array_equal(truth, a['y' + split])
            verify_fold(a, c['alpha']); metrics = read(dest / 'metrics.json'); audit_metrics(a, metrics)
            ident = {k: i[k] for k in ['cohort', 'seed', 'circuit_seed', 'arm', 'site']}
            rows.extend(dict(**ident, **r) for r in metrics); count += 1; budget.check()
            if count % 20 == 0:
                print(f'Independently verified {count} observation heads', flush=True)
        f = pd.DataFrame(rows); saved = pd.read_csv(root / 'raw-lag-table.csv'); keys = ['cohort', 'seed', 'circuit_seed', 'arm', 'site', 'lag']
        pd.testing.assert_frame_equal(f[saved.columns].sort_values(keys).reset_index(drop=True), saved.sort_values(keys).reset_index(drop=True), check_dtype=False, atol=1e-12, rtol=1e-12)
        cols = ['test_accuracy', 'frequency_excess', 'null_excess', 'r2_vs_frequency']
        cases = f[f.lag.isin(c['primary_lags'])].groupby(['cohort', 'seed', 'circuit_seed', 'arm', 'site'])[cols].mean()
        blocks = cases.groupby(['cohort', 'seed', 'arm', 'site'])[cols].mean().reset_index()
        saved = pd.read_csv(root / 'seed-blocks.csv'); pd.testing.assert_frame_equal(blocks[saved.columns], saved, check_dtype=False, atol=1e-12, rtol=1e-12)
        summary = read(root / 'summary.json'); primary = []; secondary = []; pairs = []
        for cohort in ['discovery', 'confirmation']:
            ss = summary[cohort]; loss = {}; access = {}; extra = {}
            for site in c['sites']:
                b = blocks[(blocks.cohort == cohort) & (blocks.site == site)]
                for arm in ['intact', 'DAN_MBON']:
                    t = b[b.arm == arm].set_index('seed'); key = site + '/' + arm
                    for col in cols:
                        audit_stats(t[col], ss['cells'][key][col], c)
                    access[key] = bool((t.frequency_excess >= .05).all() and (t.null_excess >= .05).all() and t.r2_vs_frequency.mean() > 0)
                intact = b[b.arm == 'intact'].set_index('seed').test_accuracy
                target = b[b.arm == 'DAN_MBON'].set_index('seed').test_accuracy
                ctrl = b[b.arm.isin(c['arms'][2:])].groupby('seed').test_accuracy.mean()
                loss[site] = intact - target; extra[site] = ctrl - target
                audit_stats(loss[site], ss['loss'][site], c); audit_stats(extra[site], ss['extra'][site], c)
                pairs.extend(dict(cohort=cohort, site=site, seed=int(seed), intact=float(intact[seed]), target=float(target[seed]), control=float(ctrl[seed]), loss=float(loss[site][seed]), extra=float(extra[site][seed])) for seed in target.index)
            interaction = loss['MBON'] - loss['KC_unstimulated']; audit_stats(interaction, ss['interaction'], c)
            gates = dict(access=all(access[k] for k in ['MBON/intact', 'KC_unstimulated/intact', 'KC_unstimulated/DAN_MBON']),
                         mbon_loss=bool(loss['MBON'].mean() >= .05 and (loss['MBON'] > 0).all()), kc_retention=bool(loss['KC_unstimulated'].mean() <= .02),
                         interaction=bool(interaction.mean() >= .05 and (interaction > 0).all()))
            assert access == ss['access'] and gates == ss['gates'] and all(gates.values()) == ss['primary_pass']; primary.append(all(gates.values()))
            test = bool(access['MBON/intact'] and extra['MBON'].mean() >= .05 and (extra['MBON'] > 0).all()); assert test == ss['secondary_mbon_specificity']; secondary.append(test)
        assert all(primary) == summary['confirmed_primary'] and all(secondary) == summary['confirmed_secondary_mbon'] and not summary['cohorts_pooled']
        saved = pd.read_csv(root / 'paired-differences.csv'); pd.testing.assert_frame_equal(pd.DataFrame(pairs)[saved.columns], saved, check_dtype=False, atol=1e-12, rtol=1e-12)
        for p, h in read(root / 'prior-results-sha256.json').items():
            assert core.sha256(p) == h, p
        for p, h in m['context']['source_sha256'].items():
            assert hashlib.sha256(subprocess.check_output(['git', 'show', m['context']['git_commit'] + ':' + p])).hexdigest() == h, p
        assert count == 130 and len(rows) == 1430
        core.write_json(out / 'checks.json', dict(all_checks_pass=True, heads=count, trajectories=2*count, metric_rows=len(rows),
                        primary_confirmed=all(primary), secondary_confirmed=all(secondary), prior_files_unchanged=len(read(root / 'prior-results-sha256.json')),
                        result_manifest_sha256=core.sha256(root / 'manifest.json'), verifier_sha256=core.sha256(__file__), budget=budget.close()))
        print('All observation-location audits passed', flush=True)
    except Exception as exc:
        core.write_json(out / 'failure.json', dict(error=repr(exc))); raise
    finally:
        budget.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('root', type=Path); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); verify(a.root, a.out)
