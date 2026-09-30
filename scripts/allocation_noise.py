"""Frozen-head coordinate noise allocation at equal expected pre-clipping energy."""
import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import softmax
from threadpoolctl import threadpool_limits
from flying.training import whole_brain_memory as core
from alphabet_memory import read, check
from pathway_memory import seal
from context_memory import estimate
from frozen_state_probe import Budget
from relative_noise import read_table
import act5_robustness as old


def config():
    return read('configs/allocation_noise.json')


def context(c):
    x = core.source_context(c)
    names = ['allocation_noise', 'verify_allocation_noise', 'act5_robustness',
             'relative_noise', 'alphabet_memory', 'pathway_memory', 'context_memory',
             'frozen_state_probe', 'verify_neuron_panel']
    x['adapter_sha256'] = {f'scripts/{n}.py': core.sha256(f'scripts/{n}.py') for n in names}
    x['completion_plan_sha256'] = core.sha256(c['completion_plan'])
    return x


def allocation(training):
    x = np.asarray(training)
    if x.ndim != 2 or min(x.shape) < 2 or not np.isfinite(x).all():
        raise ValueError('Invalid clean training features')
    sd = x.std(axis=0, ddof=0)
    q = float(np.median(sd))
    rms = float(np.sqrt(np.mean(sd**2)))
    if not np.isfinite([q, rms]).all() or q <= 0 or rms <= 0:
        raise ValueError('Unidentifiable training variation')
    return sd, q, sd / rms


def fresh_bank(observed, seed, circuit, noise_seed, horizon):
    ids = np.asarray(observed, dtype=np.int64)
    if ids.ndim != 1 or len(set(ids)) != len(ids):
        raise ValueError('Observation IDs must be unique')
    canonical = np.sort(ids)
    ix = pd.Index(canonical).get_indexer(ids)
    rng = np.random.default_rng(np.random.SeedSequence([noise_seed, seed, circuit, 2]))
    return rng.standard_normal((horizon, len(ids)))[:, ix]


def perturb(clean, noise, amplitudes):
    if (clean.shape != noise.shape or amplitudes.shape != (clean.shape[1],)
            or not all(np.isfinite(x).all() for x in [clean, noise, amplitudes])
            or np.any(amplitudes < 0)):
        raise ValueError('Invalid coordinate perturbation')
    return np.clip(clean + noise * amplitudes, -1, 1)


def gain(flat_count, allocated_count, horizon):
    return (int(allocated_count) - int(flat_count)) / horizon


def gate(values, cohort, c):
    a = np.asarray(values)
    if not np.isfinite(a).all():
        return False
    return bool(a.mean() >= c['accuracy_gain'] and (a > 0).sum() >= (4 if cohort == 'discovery' else 3))


def confirmed(gates, c):
    return [g for g in c['conditions']
            if all(gates[f'{co}/{stage}/{g}'] for co in c['cohorts'] for stage in ['archived', 'fresh'])]


def measures(clean, noise, amp, features, probs, target, scale):
    correct = probs.argmax(1) == target
    injected = noise * amp
    return dict(correct_count=int(correct.sum()), accuracy=float(correct.mean()),
                accuracy_first32=float(correct[:32].mean()),
                accuracy_tail=float(correct[32:].mean()) if len(target) > 32 else None,
                target_probability=float(probs[np.arange(len(target)), target].mean()),
                expected_squared_energy=float(np.sum(amp**2)),
                realized_injected_energy=float(np.mean(np.sum(injected**2, axis=1))),
                realized_clipped_energy=float(np.mean(np.sum((features-clean)**2, axis=1))),
                standardized_feature_mse=float(np.mean(((features-clean)/scale)**2)),
                clipping_fraction=float(np.mean(np.abs(clean+injected) > 1)))


def summarize(rows, c, out):
    f = pd.DataFrame(rows)
    keys = ['cohort', 'seed', 'circuit_seed', 'level', 'stage', 'noise_seed', 'strength']
    flat = f[f.arm == 'flat']; allocated = f[f.arm == 'training_sd']
    p = flat.merge(allocated, on=keys, suffixes=('_flat', '_allocated'), validate='one_to_one')
    p['gain'] = (p.correct_count_allocated-p.correct_count_flat)/p.horizon_flat
    paired = p[keys].copy()
    paired['flat_accuracy'] = p.accuracy_flat; paired['allocated_accuracy'] = p.accuracy_allocated
    paired['clean_accuracy'] = p.clean_accuracy_flat; paired['gain'] = p.gain
    paired['remaining_clean_loss'] = p.clean_accuracy_flat-p.accuracy_allocated
    paired.to_csv(out/'paired-differences.csv', index=False)
    metrics = ['flat_accuracy', 'allocated_accuracy', 'clean_accuracy', 'gain', 'remaining_clean_loss']
    b = paired.groupby(['cohort', 'stage', 'seed', 'level'])[metrics].mean().reset_index()
    b.to_csv(out/'seed-blocks.csv', index=False)
    paired.groupby(['cohort', 'stage', 'seed', 'level', 'strength'])[metrics].mean().reset_index().to_csv(out/'dose-seed-table.csv', index=False)
    stats, gates = {}, {}
    for co in c['cohorts']:
        for stage in ['archived', 'fresh']:
            for g in c['conditions']:
                z = b[(b.cohort == co) & (b.stage == stage) & (b.level == g)]
                key = f'{co}/{stage}/{g}'
                stats[key] = {k: estimate(z[k], c) for k in metrics}
                gates[key] = gate(z.gain, co, c)
    result = dict(statistics=stats, gates=gates, confirmed=confirmed(gates, c),
                  primary_confirmed='brain1' in confirmed(gates, c),
                  new_model_holdout=False, autonomous_recall_tested=False, cohorts_pooled=False)
    core.write_json(out/'summary.json', result)
    return result


@threadpool_limits.wrap(limits=1)
def run(c, out, smoke=False):
    out.mkdir(parents=True, exist_ok=False)
    ctx = context(c); budget = Budget(c); started = time.monotonic()
    try:
        core.write_json(out/'config.json', c); core.write_json(out/'started.json', ctx)
        core.write_json(out/'environment.json', core.environment())
        src = Path(c['source']); obs_src = Path(c['observation_source'])
        for path, digest in [(src, c['source_manifest_sha256']), (obs_src, c['observation_manifest_sha256'])]:
            assert core.sha256(path/'manifest.json') == digest
            check(path)
        check(Path(c['validation_source']))
        valid = read(Path(c['validation_source'])/'checks.json')
        assert valid['all_checks_pass'] and valid['result_manifest_sha256'] == c['observation_manifest_sha256']
        prior = {p.as_posix(): core.sha256(p) for p in Path('results').rglob('*') if p.is_file() and out not in p.parents}
        core.write_json(out/'prior-artifacts.json', prior)
        cases = old.case_list(c)
        if smoke:
            cases = [x for x in cases if x['seed'] == 7142 and x['circuit_seed'] == 701]
        h = 8 if smoke else 197
        rows, banks, identities = [], {}, {}
        zeros = clean_checks = archive_matches = 0
        for number, case in enumerate(cases, 1):
            budget.check(); name = old.stem(case); parent = src/name; oldobs = obs_src/name
            meta = read(parent/'case.json'); path = out/name; path.mkdir()
            with np.load(parent/'checkpoint.npz') as z:
                cp = dict(z)
            with np.load(parent/'dose0_teacher.npz') as z:
                clean = z['features'][:h]; original_prediction = z['prediction'][:h]; original_probs = z['probabilities'][:h]
            assert cp['features'].shape == (199, 48) and clean.shape == (h, 48)
            head = core.NonlinearReadout(np.arange(48), 8, core.nonlinear_seed(case['seed'], 0))
            head.mean = cp['mean']; head.scale = cp['scale']; head.parameters = {k: cp[k] for k in ['w1', 'b1', 'w2', 'b2']}
            assert head.digest() == meta['head_sha256'] and head.parameter_count == 482
            clean_probs = softmax(head.logits(clean), axis=1)
            np.testing.assert_array_equal(clean_probs.argmax(1), original_prediction)
            np.testing.assert_allclose(clean_probs, original_probs, atol=1e-12, rtol=1e-10); clean_checks += 1
            target = cp['digits'][3:3+h]; clean_accuracy = float(np.mean(original_prediction == target))
            sd, q, weights = allocation(cp['features'])
            observed = np.asarray(meta['graph']['observation_root_ids'], dtype=np.int64)
            key = (case['seed'], case['circuit_seed'])
            identity = {k: meta['graph'][k] for k in ['input_mapping_sha256', 'input_root_ids', 'observation_root_ids']}
            if key in identities:
                assert identities[key] == identity
            identities[key] = identity
            with np.load(oldobs/'noise.npz') as z:
                np.testing.assert_array_equal(z['observed_root_ids'], observed)
                archived = z['standard_normals'][:h]
            if key not in banks:
                banks[key] = np.stack([archived, *[fresh_bank(observed, *key, n, h) for n in c['fresh_noise_seeds']]])
            noise = banks[key]; np.testing.assert_array_equal(noise[0], archived)
            for arm in [np.ones(48), weights]:
                zero = perturb(clean, noise[0], 0*arm)
                np.testing.assert_array_equal(zero, clean)
                np.testing.assert_array_equal(softmax(head.logits(zero), axis=1), clean_probs); zeros += 1
            shape = (4, 3, 2)
            features = np.empty((*shape, h, 48)); probs = np.empty((*shape, h, 10))
            amplitudes = np.empty((3, 2, 48)); local = []
            for j, r in enumerate(c['relative_strengths']):
                amplitudes[j] = np.stack([np.full(48, r*q), r*q*weights])
                np.testing.assert_allclose(np.sum(amplitudes[j]**2, axis=1), 48*(r*q)**2, rtol=1e-12, atol=1e-18)
                for k, ns in enumerate([c['archived_noise_seed'], *c['fresh_noise_seeds']]):
                    for a, arm in enumerate(['flat', 'training_sd']):
                        x = perturb(clean, noise[k], amplitudes[j, a]); p = softmax(head.logits(x), axis=1)
                        features[k, j, a] = x; probs[k, j, a] = p
                        if k == 0 and a == 0:
                            with np.load(oldobs/f'instantaneous{j+1}.npz') as z:
                                np.testing.assert_array_equal(x, z['features'][:h])
                                np.testing.assert_allclose(p, z['probabilities'][:h], atol=1e-12, rtol=1e-10)
                                np.testing.assert_array_equal(p.argmax(1), z['prediction'][:h]); archive_matches += 1
                        row = dict(**case, stage='archived' if k == 0 else 'fresh', noise_seed=ns,
                                   strength=r, arm=arm, horizon=h, clean_accuracy=clean_accuracy,
                                   **measures(clean, noise[k], amplitudes[j, a], x, p, target, head.scale))
                        local.append(row); rows.append(row)
            np.savez_compressed(path/'evaluations.npz', features=features, probabilities=probs,
                                predictions=probs.argmax(-1), position_accuracy=probs.argmax(-1) == target,
                                standard_normals=noise, amplitudes=amplitudes, training_sd=sd,
                                allocation_weights=weights, observed_root_ids=observed, targets=target)
            pd.DataFrame(local).to_csv(path/'metrics.csv', index=False)
            assert head.digest() == meta['head_sha256']
            core.write_json(path/'case.json', dict(case=case, source_case=parent.as_posix(),
                source_manifest_sha256=core.sha256(parent/'manifest.json'),
                observation_case=oldobs.as_posix(), observation_manifest_sha256=core.sha256(oldobs/'manifest.json'),
                head_sha256=head.digest(), head_unchanged=True, graph=meta['graph'],
                horizon=h, training_scale=q, smoke=smoke, new_neural_trajectories=0))
            seal(path, c, ctx, purpose='Matched expected observation-noise energy')
            print(f'{number}/{len(cases)} allocation cases complete', flush=True)
        pd.DataFrame(rows).to_csv(out/'raw-metrics.csv', index=False)
        if not smoke:
            summarize(rows, c, out)
        assert context(c) == ctx
        for p, digest in prior.items():
            assert core.sha256(p) == digest, p
        budget.check()
        core.write_json(out/'verification.json', dict(cases=len(cases), evaluations=len(rows), zero_checks=zeros,
            clean_replays=clean_checks, archived_flat_matches=archive_matches, new_neural_trajectories=0,
            matched_inputs_observations=True, smoke=smoke, prior_files_unchanged=len(prior),
            seconds=time.monotonic()-started, budget=budget.close()))
        seal(out, c, ctx, purpose='Allocation smoke' if smoke else 'Allocation main, including fresh noise')
    except Exception as exc:
        core.write_json(out/'failure.json', dict(error=repr(exc), budget=budget.close(), smoke=smoke))
        seal(out, c, ctx, purpose='Preserved allocation execution failure')
        raise


def plot(root, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    m = check(root); c = m['config']; f = read_table(root/'dose-seed-table.csv')
    out.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True, layout='constrained')
    for i, co in enumerate(c['cohorts']):
        for j, g in enumerate(c['conditions']):
            ax = axes[i, j]; z = f[(f.cohort == co) & (f.level == g)]
            for stage, style in [('archived', '--'), ('fresh', '-')]:
                sub = z[z.stage == stage]
                for metric, color in [('flat_accuracy', '#b95d30'), ('allocated_accuracy', '#246caa')]:
                    avg = sub.groupby('strength')[metric].mean()
                    ax.plot(range(3), avg, style, color=color, marker='o', label=f'{stage}: {metric.split("_")[0]}')
                for _, block in sub.groupby('seed'):
                    ax.plot(range(3), block.sort_values('strength').allocated_accuracy, style, color='#246caa', alpha=.15, lw=.7)
            ax.axhline(z.clean_accuracy.mean(), color='gray', ls=':', label='clean')
            ax.set_xticks(range(3), [f'{r:g}' for r in c['relative_strengths']]); ax.set_ylim(0, 1)
            ax.set_title(f'{co}: {g}'); ax.set_xlabel('Relative noise r (unequal spacing)'); ax.grid(alpha=.15)
            if j == 0:
                ax.set_ylabel('Teacher-forced position accuracy')
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Equal expected noise energy: flat versus training-SD allocation')
    fig.savefig(out/'allocation-noise-curves.png', dpi=160); plt.close(fig)
    seal(out, c, m['context'], result_manifest_sha256=core.sha256(root/'manifest.json'), purpose='Allocation diagnostic plot')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=['run', 'plot'])
    parser.add_argument('--source', type=Path); parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true'); args = parser.parse_args()
    if args.command == 'run':
        run(config(), args.out, args.smoke)
    else:
        plot(args.source, args.out)
