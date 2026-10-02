"""Independent graph, teacher, head and full-autonomous replay audit."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits

from flying.data.sequences import SequenceDataset
from flying.training import sequence_memory as seq
from flying.training import whole_brain_memory as core
from alphabet_memory import read, check
from frozen_state_probe import Budget
from pathway_memory import seal
from relative_noise import read_table
from verify_allocation_noise import independent_head, independent_noise
from verify_feedback_noise import reference, first_error, additional_metrics
from verify_research_suite import teacher_reference
import fresh_coordinate_sd as run


def exact_files(root):
    manifest = check(root)
    listed = set(manifest['artifacts'])
    present = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p != root / 'manifest.json'}
    assert listed == present
    return manifest


def compare_rollout(saved, actual, target, row, clean_prefix):
    assert saved.keys() == actual.keys()
    for name in saved:
        if name == 'prediction':
            np.testing.assert_array_equal(saved[name], actual[name])
        else:
            np.testing.assert_allclose(saved[name], actual[name], atol=1e-12, rtol=1e-10, err_msg=name)
    prefix = first_error(actual['prediction'], target)
    assert int(row.exact_prefix_symbols) == prefix
    assert int(row.clean_prefix) == clean_prefix
    expected_retention = min(prefix / clean_prefix, 1.) if clean_prefix else np.nan
    if np.isnan(expected_retention):
        assert pd.isna(row.retention)
    else:
        np.testing.assert_allclose(row.retention, expected_retention, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(row.accuracy, np.mean(actual['prediction'] == target), atol=1e-12)
    additional_metrics(actual, target, row)


def verify_summary(frame, c, root):
    z = frame[frame.calibration != 'clean']
    keys = ['cohort', 'family', 'seed', 'topology', 'calibration']
    blocks = z.groupby(keys)[run.METRICS].agg(lambda x: np.mean(x.to_numpy())).reset_index()
    saved_blocks = read_table(root / 'seed-blocks.csv')
    pd.testing.assert_frame_equal(saved_blocks.sort_values(keys).reset_index(drop=True),
                                  blocks.sort_values(keys).reset_index(drop=True),
                                  check_exact=False, atol=1e-12, rtol=1e-12)
    saved = read_table(root / 'paired-differences.csv')
    summary = read(root / 'summary.json')
    gates = {}
    for (co, fam, control), rows in saved.groupby(['cohort', 'family', 'control']):
        need = 4 if co == 'discovery' else 3
        for row in rows.itertuples():
            cell = blocks[(blocks.cohort == co) & (blocks.family == fam) & (blocks.seed == row.seed)]
            for metric in run.METRICS:
                def value(top, calibration):
                    t = cell[(cell.topology == top) & (cell.calibration == calibration)]
                    assert len(t) == 1
                    return float(t.iloc[0][metric])
                own = value('intact', 'own') - value(control, 'own')
                coordinate = value('intact', 'coordinate') - value(control, 'coordinate')
                for column, expected in [('own_advantage', own), ('coordinate_advantage', coordinate),
                                         ('incremental_attenuation', coordinate-own)]:
                    actual = getattr(row, metric + '_' + column)
                    if np.isnan(expected):
                        assert np.isnan(actual)
                    else:
                        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
        a = rows.exact_prefix_symbols_coordinate_advantage.to_numpy()
        b = rows.retention_coordinate_advantage.to_numpy()
        gates[f'{co}/{fam}/{control}'] = None if not np.isfinite(b).all() else bool(
            a.mean() >= c['minimum_prefix_advantage'] and
            b.mean() >= c['minimum_retention_advantage'] and
            np.count_nonzero(a > 0) >= need and np.count_nonzero(b > 0) >= need)
    assert gates == summary['descriptive_gates']
    primary = [gates[f'{co}/random/degree'] for co in c['cohorts']]
    assert summary['primary_confirmed'] == (None if any(x is None for x in primary) else all(primary))
    assert summary['fresh_model_fits'] == 96 and summary['fresh_model_data_graph_seeds']
    assert not summary['cohorts_pooled']


@threadpool_limits.wrap(limits=1)
def verify(root, out):
    manifest = exact_files(root)
    c = manifest['config']
    assert manifest['context'] == run.context(c) == read(root / 'source-hashes.json')
    out.mkdir(parents=True, exist_ok=False)
    budget = Budget(dict(max_seconds=c['verification_seconds'], max_rss_bytes=c['max_rss_bytes']))
    counts = dict(cases=0, graph_reconstructions=0, teacher_replays=0,
                  full_autonomous_replays=0, teacher_certificates=0, exact_head_refits=0)
    max_error = 0.
    try:
        status = read(root / 'verification.json')
        assert status['complete']
        smoke = status['smoke']
        bc = read(c['base_config'])
        h = 8 if smoke else bc['eval_length']
        all_rows = []
        for case in run.cases(c, smoke):
            budget.check()
            path = root / run.stem(case)
            exact_files(path)
            meta = read(path / 'case.json')
            assert meta['case'] == case and meta['new_fit'] and meta['training_updates'] == 2000
            model, obs, graph, raw, weights = run.build(c, case)
            assert meta['graph'] == graph
            counts['graph_reconstructions'] += 1
            if case['topology'] != 'intact':
                np.testing.assert_array_equal(sparse.load_npz(path / 'raw-graph.npz').toarray(), raw.toarray())
                np.testing.assert_array_equal(sparse.load_npz(path / 'weights.npz').toarray(), weights.toarray())
            with np.load(path / 'checkpoint.npz') as archive:
                cp = dict(archive)
            assert all(np.isfinite(v).all() for v in cp.values())
            dataset = SequenceDataset(case['family'], case['dataset_seed'], **bc['dataset'])
            symbols = dataset.symbols()
            np.testing.assert_array_equal(cp['symbols'], symbols)
            assert meta['dataset'] == dataset.identity(symbols)
            independently_collected = teacher_reference(model, obs, symbols[:-1])
            np.testing.assert_allclose(independently_collected, cp['features'], atol=1e-12, rtol=1e-10)
            counts['teacher_replays'] += 1
            head = run.feedback.load_head(cp, case['seed'])
            assert head.digest() == meta['head_sha256'] and head.parameter_count == 482
            if case['family'] == 'random' and case['topology'] == 'intact' and case['circuit_seed'] == 701 and case['block_index'] == 0:
                condition = core.NetworkCondition('legacy5', 701, case['seed'])
                again, _, _ = seq.fit(cp['features'], symbols, condition, bc)
                assert again.digest() == head.digest()
                counts['exact_head_refits'] += 1
            target = symbols[3:3+h]
            teacher = cp['features'][2:2+h]
            ids = np.asarray(graph['observation_root_ids'], dtype=np.int64)
            seeds = c['smoke_noise_seeds'] if smoke else c['noise_seeds']
            noise = np.stack([independent_noise(ids, case['seed'], case['circuit_seed'], ns, h) for ns in seeds])
            with np.load(path / 'calibration.npz') as archive:
                bank = dict(archive)
            np.testing.assert_array_equal(bank['standard_normals'], noise)
            np.testing.assert_array_equal(bank['target'], target)
            np.testing.assert_array_equal(bank['observed_root_ids'], ids)
            sd = np.sqrt(np.mean((cp['features'] - cp['features'].mean(0))**2, axis=0))
            np.testing.assert_allclose(bank['training_sd'], sd, atol=1e-18, rtol=1e-12)
            assert meta['zero_sd_coordinates'] == int(np.sum(sd == 0))
            rows = read_table(path / 'rollouts.csv')
            certs = read_table(path / 'certificates.csv')
            assert len(rows) == 19 and len(certs) == 18
            all_rows.extend(rows.to_dict('records'))
            clean_row = rows[rows.calibration == 'clean'].iloc[0]
            with np.load(path / 'clean.npz') as archive:
                clean = dict(archive)
            clean_again = reference(model, obs, cp, symbols[:3], h, noise[0], np.zeros(48))
            clean_prefix = first_error(clean_again['prediction'], target)
            assert meta['clean_prefix'] == clean_prefix
            compare_rollout(clean, clean_again, target, clean_row, clean_prefix)
            counts['full_autonomous_replays'] += 1
            for di, dose in enumerate(c['relative_strengths']):
                q = float(np.median(sd)); u = float(np.sqrt(np.mean(sd**2)))
                independent_amplitudes = [dose * q * sd / u, dose * sd]
                for ai, cal in enumerate(c['calibrations']):
                    amp = independent_amplitudes[ai]
                    np.testing.assert_allclose(bank['amplitudes'][di, ai], amp, atol=1e-18, rtol=1e-12)
                    for ni, ns in enumerate(seeds):
                        z = rows[(rows.calibration == cal) & (rows.strength == dose) & (rows.noise_seed == ns)]
                        t = certs[(certs.calibration == cal) & (certs.strength == dose) & (certs.noise_seed == ns)]
                        assert len(z) == len(t) == 1
                        row, cert = z.iloc[0], t.iloc[0]
                        file = path / row.artifact
                        with np.load(file) as archive:
                            saved = dict(archive)
                        again = reference(model, obs, cp, symbols[:3], h, noise[ni], amp)
                        compare_rollout(saved, again, target, row, clean_prefix)
                        max_error = max(max_error, float(np.max(np.abs(saved['probabilities'] - again['probabilities']))))
                        x = np.clip(teacher + noise[ni] * amp, -1, 1)
                        p = independent_head(x, cp)
                        np.testing.assert_allclose(bank['teacher_probabilities'][ai, ni, di], p, atol=1e-12, rtol=1e-10)
                        assert int(cert.exact_prefix_symbols) == first_error(p.argmax(1), target)
                        np.testing.assert_allclose(cert.teacher_accuracy, np.mean(p.argmax(1) == target), atol=1e-12)
                        counts['full_autonomous_replays'] += 1
                        counts['teacher_certificates'] += 1
            counts['cases'] += 1
            print(f"Verified {counts['cases']}/{status['cases']}: {run.stem(case)}", flush=True)
        assert counts['cases'] == status['cases']
        assert counts['full_autonomous_replays'] == status['actual_paths']
        assert counts['teacher_certificates'] == status['certificates']
        if not smoke:
            verify_summary(pd.DataFrame(all_rows), c, root)
        budget.check()
        core.write_json(out / 'checks.json', dict(all_checks_pass=True, smoke=smoke,
                        result_manifest_sha256=core.sha256(root / 'manifest.json'),
                        counts=counts, max_probability_error=max_error,
                        environment=core.environment(), budget=budget.close()))
        seal(out, c, run.context(c), purpose='Independent fresh-model full replay')
    except Exception as exc:
        core.write_json(out / 'failure.json', dict(error=repr(exc), counts=counts, budget=budget.close()))
        seal(out, c, run.context(c), purpose='Preserved incomplete independent audit')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('result', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    verify(args.result, args.out)
