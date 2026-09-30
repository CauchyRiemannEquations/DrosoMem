"""Independent reconstruction of every coordinate-allocation counterfactual."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from flying.training import whole_brain_memory as core
from alphabet_memory import read, check
from pathway_memory import seal
from frozen_state_probe import Budget
from verify_neuron_panel import audit_stats
from relative_noise import read_table
import allocation_noise as run


def independent_head(x, cp):
    hidden = np.tanh(((x-cp['mean'])/cp['scale']) @ cp['w1']+cp['b1'])
    logits = hidden @ cp['w2']+cp['b2']
    exponential = np.exp(logits-np.max(logits, axis=1)[:, None])
    return exponential/np.sum(exponential, axis=1)[:, None]


def independent_noise(ids, seed, circuit, ns, h):
    order = sorted(int(v) for v in ids)
    positions = [order.index(int(v)) for v in ids]
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([ns, seed, circuit, 2])))
    return np.asarray([rng.standard_normal(len(order))[positions] for _ in range(h)])


def verify_statistics(f, c, root):
    metrics = ['flat_accuracy', 'allocated_accuracy', 'clean_accuracy', 'gain', 'remaining_clean_loss']
    pairs, blocks, doses, gates = [], [], [], {}
    for co, seeds in c['cohorts'].items():
        for stage in ['archived', 'fresh']:
            streams = [c['archived_noise_seed']] if stage == 'archived' else c['fresh_noise_seeds']
            for g in c['conditions']:
                group = []
                for seed in seeds:
                    cell = []
                    for ci in c['circuit_seeds']:
                        for ns in streams:
                            for r in c['relative_strengths']:
                                a = f[(f.cohort == co) & (f.level == g) & (f.seed == seed) &
                                      (f.circuit_seed == ci) & (f.noise_seed == ns) & (f.strength == r)]
                                assert len(a) == 2 and set(a.arm) == {'flat', 'training_sd'}
                                flat = a[a.arm == 'flat'].iloc[0]; allocated = a[a.arm == 'training_sd'].iloc[0]
                                row = dict(cohort=co, stage=stage, level=g, seed=seed, circuit_seed=ci, noise_seed=ns,
                                    strength=r, flat_accuracy=float(flat.accuracy), allocated_accuracy=float(allocated.accuracy),
                                    clean_accuracy=float(flat.clean_accuracy),
                                    gain=(int(allocated.correct_count)-int(flat.correct_count))/197,
                                    remaining_clean_loss=float(flat.clean_accuracy-allocated.accuracy))
                                cell.append(row); pairs.append(row)
                    values = {k: float(np.mean([x[k] for x in cell])) for k in metrics}
                    block = dict(cohort=co, stage=stage, level=g, seed=seed, **values)
                    blocks.append(block); group.append(values)
                    for r in c['relative_strengths']:
                        doses.append(dict(cohort=co, stage=stage, level=g, seed=seed, strength=r,
                            **{k: float(np.mean([x[k] for x in cell if x['strength'] == r])) for k in metrics}))
                key = f'{co}/{stage}/{g}'; values = [x['gain'] for x in group]
                gates[key] = bool(np.mean(values) >= .05 and np.count_nonzero(np.asarray(values) > 0) >= (4 if co == 'discovery' else 3))
                saved = read(root/'summary.json')['statistics'][key]
                for k in metrics:
                    audit_stats([x[k] for x in group], saved[k], c)
    for filename, rows, keys in [
        ('paired-differences.csv', pairs, ['cohort', 'stage', 'level', 'seed', 'circuit_seed', 'noise_seed', 'strength']),
        ('seed-blocks.csv', blocks, ['cohort', 'stage', 'level', 'seed']),
        ('dose-seed-table.csv', doses, ['cohort', 'stage', 'level', 'seed', 'strength'])]:
        expected = pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)
        saved = read_table(root/filename).sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(saved[expected.columns], expected, check_exact=False, atol=1e-12, rtol=1e-12)
    s = read(root/'summary.json')
    confirmed = [g for g in c['conditions'] if all(gates[f'{co}/{stage}/{g}'] for co in c['cohorts'] for stage in ['archived', 'fresh'])]
    assert s['gates'] == gates and s['confirmed'] == confirmed
    assert s['primary_confirmed'] == ('brain1' in confirmed)
    assert not s['autonomous_recall_tested'] and not s['new_model_holdout'] and not s['cohorts_pooled']


@threadpool_limits.wrap(limits=1)
def verify(root, out):
    out.mkdir(parents=True, exist_ok=False)
    m = check(root); c = m['config']; current = run.context(c)
    for k in ['source_sha256', 'adapter_sha256', 'config_sha256', 'completion_plan_sha256']:
        assert current[k] == m['context'][k]
    budget = Budget(dict(max_seconds=c['verification_seconds'], max_rss_bytes=c['max_rss_bytes']))
    try:
        source = Path(c['source']); obs_source = Path(c['observation_source'])
        check(source); check(obs_source)
        assert core.sha256(source/'manifest.json') == c['source_manifest_sha256']
        assert core.sha256(obs_source/'manifest.json') == c['observation_manifest_sha256']
        run_meta = read(root/'verification.json'); smoke = run_meta['smoke']; h = 8 if smoke else 197
        cases = run.old.case_list(c)
        if smoke:
            cases = [x for x in cases if x['seed'] == 7142 and x['circuit_seed'] == 701]
        allrows, banks = [], {}; count = zeros = archive_matches = clean_count = 0; error = 0.
        for number, case in enumerate(cases, 1):
            budget.check(); name = run.old.stem(case); path = root/name; check(path)
            src = source/name; obs_src = obs_source/name; meta = read(src/'case.json'); case_meta = read(path/'case.json')
            assert case_meta['source_manifest_sha256'] == core.sha256(src/'manifest.json')
            assert case_meta['observation_manifest_sha256'] == core.sha256(obs_src/'manifest.json')
            assert case_meta['case'] == case and case_meta['head_unchanged'] and case_meta['new_neural_trajectories'] == 0
            assert case_meta['graph'] == meta['graph'] and case_meta['head_sha256'] == meta['head_sha256']
            with np.load(src/'checkpoint.npz') as z:
                cp = dict(z)
            target = cp['digits'][3:3+h]; np.testing.assert_array_equal(cp['digits'], core.SequenceDataset().symbols())
            with np.load(src/'dose0_teacher.npz') as z:
                clean = z['features'][:h]; old_probs = z['probabilities'][:h]; old_pred = z['prediction'][:h]
            clean_prob = independent_head(clean, cp)
            np.testing.assert_allclose(clean_prob, old_probs, atol=1e-12, rtol=1e-10)
            np.testing.assert_array_equal(clean_prob.argmax(1), old_pred); clean_count += 1
            train = cp['features']; sd = np.sqrt(np.sum((train-train.mean(0))**2, axis=0)/len(train))
            q = float(np.median(sd)); weights = np.sqrt(len(sd))*sd/np.linalg.norm(sd)
            ids = np.array(meta['graph']['observation_root_ids'], dtype=np.int64)
            with np.load(obs_src/'noise.npz') as z:
                archived = z['standard_normals'][:h]; np.testing.assert_array_equal(ids, z['observed_root_ids'])
            noise = np.stack([archived, *[independent_noise(ids, case['seed'], case['circuit_seed'], n, h) for n in c['fresh_noise_seeds']]])
            key = (case['seed'], case['circuit_seed'])
            if key in banks:
                np.testing.assert_array_equal(noise, banks[key][1]); np.testing.assert_array_equal(ids, banks[key][0])
            banks[key] = (ids, noise)
            f = read_table(path/'metrics.csv'); assert len(f) == 24 and f.horizon.eq(h).all()
            allrows.extend(f.to_dict('records'))
            with np.load(path/'evaluations.npz') as saved:
                np.testing.assert_array_equal(saved['observed_root_ids'], ids); np.testing.assert_array_equal(saved['targets'], target)
                np.testing.assert_array_equal(saved['standard_normals'], noise)
                np.testing.assert_allclose(saved['training_sd'], sd, atol=1e-18, rtol=1e-12)
                np.testing.assert_allclose(saved['allocation_weights'], weights, atol=1e-12, rtol=1e-12)
                for a in [np.ones(48), weights]:
                    zero = np.maximum(-1, np.minimum(1, clean+noise[0]*(0*a)))
                    np.testing.assert_array_equal(zero, clean)
                    np.testing.assert_array_equal(independent_head(zero, cp), clean_prob); zeros += 1
                for j, r in enumerate(c['relative_strengths']):
                    for k, ns in enumerate([c['archived_noise_seed'], *c['fresh_noise_seeds']]):
                        for a, arm in enumerate(['flat', 'training_sd']):
                            amp = r*q*(np.ones(48) if a == 0 else weights)
                            np.testing.assert_allclose(amp, saved['amplitudes'][j, a], atol=1e-18, rtol=1e-12)
                            np.testing.assert_allclose(np.sum(amp**2), 48*(r*q)**2, atol=1e-18, rtol=1e-12)
                            delta = noise[k]*amp; unclipped = clean+delta; x = np.maximum(-1, np.minimum(1, unclipped))
                            probs = independent_head(x, cp); prediction = probs.argmax(1); correct = prediction == target
                            for actual, expected in [(saved['features'][k, j, a], x), (saved['probabilities'][k, j, a], probs)]:
                                np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-10)
                                error = max(error, float(np.max(np.abs(actual-expected))))
                            np.testing.assert_array_equal(saved['predictions'][k, j, a], prediction)
                            np.testing.assert_array_equal(saved['position_accuracy'][k, j, a], correct)
                            if k == 0 and a == 0:
                                with np.load(obs_src/f'instantaneous{j+1}.npz') as z:
                                    np.testing.assert_allclose(x, z['features'][:h], atol=1e-12, rtol=1e-10)
                                    np.testing.assert_array_equal(prediction, z['prediction'][:h]); archive_matches += 1
                            match = f[(f.noise_seed == ns) & (f.strength == r) & (f.arm == arm)]
                            assert len(match) == 1; row = match.iloc[0]
                            for field, value in case.items():
                                assert row[field] == value
                            assert row.stage == ('archived' if k == 0 else 'fresh')
                            expected = dict(correct_count=int(np.count_nonzero(correct)), accuracy=float(correct.sum()/h),
                                accuracy_first32=float(np.mean(correct[:32])), clean_accuracy=float(np.mean(old_pred == target)),
                                target_probability=float(np.mean([probs[t, y] for t, y in enumerate(target)])),
                                expected_squared_energy=float(np.dot(amp, amp)),
                                realized_injected_energy=float(np.sum(delta*delta)/h),
                                realized_clipped_energy=float(np.sum((x-clean)**2)/h),
                                standardized_feature_mse=float(np.sum(((x-clean)/cp['scale'])**2)/(h*48)),
                                clipping_fraction=float(np.count_nonzero((unclipped < -1) | (unclipped > 1))/(h*48)))
                            if h > 32:
                                expected['accuracy_tail'] = float(np.mean(correct[32:]))
                            else:
                                assert pd.isna(row.accuracy_tail)
                            for field, value in expected.items():
                                np.testing.assert_allclose(row[field], value, atol=1e-12, rtol=1e-10, err_msg=field)
                            count += 1
            print(f'Audit {number}/{len(cases)} allocation cases', flush=True)
        f = pd.DataFrame(allrows); saved = read_table(root/'raw-metrics.csv')
        pd.testing.assert_frame_equal(saved[f.columns], f, check_exact=False, atol=1e-12, rtol=1e-12)
        if not smoke:
            verify_statistics(f, c, root)
        assert (run_meta['cases'], run_meta['evaluations'], run_meta['zero_checks'], run_meta['archived_flat_matches']) == (len(cases), count, zeros, archive_matches)
        prior = read(root/'prior-artifacts.json')
        for p, digest in prior.items():
            assert core.sha256(p) == digest, p
        budget.check()
        core.write_json(out/'checks.json', dict(all_checks_pass=True, smoke=smoke, cases=len(cases),
            independent_evaluations=count, zero_checks=zeros, clean_replays=clean_count, archived_flat_matches=archive_matches,
            maximum_numeric_error=error, new_neural_trajectories=0, prior_files_unchanged=len(prior),
            result_manifest_sha256=core.sha256(root/'manifest.json'), budget=budget.close()))
        seal(out, c, m['context'], purpose='Independent allocation counterfactual and statistics audit')
    except Exception as exc:
        core.write_json(out/'failure.json', dict(error=repr(exc), budget=budget.close()))
        seal(out, c, m['context'], purpose='Preserved allocation verification failure')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('source', type=Path)
    parser.add_argument('--out', type=Path, required=True); args = parser.parse_args()
    verify(args.source, args.out)
