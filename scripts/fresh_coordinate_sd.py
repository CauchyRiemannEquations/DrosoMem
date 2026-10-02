"""Fresh-fit structural confirmation under own and coordinate-SD observation noise."""
import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import softmax
from threadpoolctl import threadpool_limits

from flying.data.sequences import SequenceDataset
from flying.training import sequence_memory as seq
from flying.training import whole_brain_memory as core
from alphabet_memory import read
from context_memory import estimate
from frozen_state_probe import Budget
from pathway_memory import seal
import allocation_noise as allocation
import feedback_noise as feedback
import research_suite as suite
import structural_controls as structural


CONFIG = 'configs/fresh_coordinate_sd.json'
METRICS = ['exact_prefix_symbols', 'retention']


def context(c):
    paths = sorted(Path('src/flying').rglob('*.py'))
    paths += [Path(c['protocol']), Path(c['base_config']), Path(CONFIG)]
    paths += [Path('scripts') / (n + '.py') for n in
              ['fresh_coordinate_sd', 'verify_fresh_coordinate_sd', 'structural_controls',
               'feedback_noise', 'allocation_noise', 'research_suite']]
    for ci in c['circuit_seeds']:
        paths += sorted(Path(f'data/flywire_783_mb_left_kc512_s{ci}').glob('*'))
    return {p.as_posix(): core.sha256(p) for p in paths}


def cases(c, smoke=False):
    cohorts = {'discovery': [c['cohorts']['discovery'][0]]} if smoke else c['cohorts']
    for cohort, blocks in cohorts.items():
        for bi, (seed, ds) in enumerate(blocks):
            for family in (['random'] if smoke else c['families']):
                for ci in ([701] if smoke else c['circuit_seeds']):
                    for topology in c['topologies']:
                        yield dict(cohort=cohort, block_index=bi, seed=seed,
                                   dataset_seed=ds, family=family, circuit_seed=ci,
                                   level='legacy5', topology=topology)


def stem(case):
    return suite.stem(case)


def control_seed(c, case):
    if case['topology'] == 'intact':
        return None
    block = case['block_index'] + (0 if case['cohort'] == 'discovery' else 5)
    return c['graph_seed_base'] + 10 * block + 2 * (case['circuit_seed'] - 701) + (case['topology'] == 'role')


def build(c, case):
    bc = read(c['base_config'])
    folder = Path(f"data/flywire_783_mb_left_kc512_s{case['circuit_seed']}")
    original, ids, provenance = core.load_connectome(folder)
    roles, _ = core.load_roles(folder, ids)
    ids = np.asarray(ids, dtype=np.int64)
    roles = np.asarray(roles)
    patterns = core.KCEncoder(roles, case['seed'], bc['input_fraction'], bc['input_amplitude']).patterns
    raw = original
    log = None
    if case['topology'] != 'intact':
        raw, log = structural.generate(original, roles, case['topology'], control_seed(c, case), c['swaps_per_edge'])
        audit = structural.audit(original, raw, roles, case['topology'])
    else:
        audit = None
    weights = core.normalize_condition(raw, bc['normalization'], bc['gain'])
    weights.sort_indices()
    observed = np.flatnonzero(roles == 'MBON')
    assert len(ids) == 686 and len(observed) == 48 and raw.nnz == provenance['edges']
    model = core.TimedReservoir(weights, core.MappedEncoder(patterns), roles, bc['leak'], bc['schedule'])
    graph = dict(topology=case['topology'], circuit_seed=case['circuit_seed'], neurons=len(ids),
                 edges=int(raw.nnz), raw_sha256=core.weight_hash(raw),
                 weight_sha256=core.weight_hash(weights), graph_seed=control_seed(c, case),
                 generation=log, structural_audit=audit,
                 observation_root_ids=ids[observed].astype(str).tolist(),
                 input_mapping_sha256=hashlib.sha256(ids.tobytes() + patterns.tobytes()).hexdigest(),
                 source_file_sha256={p.name: core.sha256(p) for p in sorted(folder.glob('*'))})
    return model, observed, graph, raw, weights


def amplitudes(features, strength):
    sd, q, weights = allocation.allocation(features)
    if not np.isfinite([q, *sd]).all() or q <= 0:
        raise ValueError('Invalid training observation SD')
    return np.stack([strength * q * weights, strength * sd]), sd, q


def execute(c, case, out, smoke, budget):
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    bc = read(c['base_config'])
    dataset = SequenceDataset(case['family'], case['dataset_seed'], **bc['dataset'])
    symbols = dataset.symbols()
    model, obs, graph, raw, weights = build(c, case)
    features, diagnostics = core.collect(model, obs, symbols[:-1], bc['activity_epsilon'])
    condition = core.NetworkCondition('legacy5', case['circuit_seed'], case['seed'])
    head, history, _ = seq.fit(features, symbols, condition, bc)
    assert head.parameter_count == 482 and np.isfinite(features).all()
    cp = dict(symbols=symbols, features=features, mean=head.mean, scale=head.scale,
              **head.parameters, **diagnostics)
    np.savez_compressed(out / 'checkpoint.npz', **cp)
    core.write_json(out / 'training.json', history)
    if case['topology'] != 'intact':
        sparse.save_npz(out / 'raw-graph.npz', raw)
        sparse.save_npz(out / 'weights.npz', weights)
    h = 8 if smoke else bc['eval_length']
    target = symbols[3:3+h]
    teacher = features[2:2+h]
    ids = np.asarray(graph['observation_root_ids'], dtype=np.int64)
    seeds = c['smoke_noise_seeds'] if smoke else c['noise_seeds']
    noise = np.stack([allocation.fresh_bank(ids, case['seed'], case['circuit_seed'], ns, h) for ns in seeds])
    clean = feedback.autonomous(model, obs, head, symbols[:3], h, noise[0], np.zeros(48))
    np.savez_compressed(out / 'clean.npz', **clean)
    clean_prefix = core.prefix_score(target, clean['prediction'])
    rows = [dict(**case, calibration='clean', strength=0., noise_seed=seeds[0],
                 artifact='clean.npz', **suite.rollout_metrics(clean, target, head, clean, graph['neurons']))]
    certs = []
    prob_bank = np.empty((2, 3, 3, h, 10))
    amp_bank = np.empty((3, 2, 48))
    pclean = softmax(head.logits(teacher), axis=1)
    feedback.verify_pre_error(clean, teacher, pclean, pclean.argmax(1), target)
    for di, dose in enumerate(c['relative_strengths']):
        amps, sd, q = amplitudes(features, dose)
        amp_bank[di] = amps
        for ai, cal in enumerate(c['calibrations']):
            for ni, ns in enumerate(seeds):
                budget.check()
                amp = amps[ai]
                noisy_teacher = np.clip(teacher + noise[ni] * amp, -1, 1)
                p = softmax(head.logits(noisy_teacher), axis=1)
                prob_bank[ai, ni, di] = p
                pred = p.argmax(1)
                identity = dict(**case, calibration=cal, strength=dose, noise_seed=ns)
                certs.append(dict(**identity, **suite.endpoint(pred, target, clean_prefix),
                                  teacher_accuracy=float(np.mean(pred == target)),
                                  expected_energy=float(amp @ amp)))
                actual = feedback.autonomous(model, obs, head, symbols[:3], h, noise[ni], amp)
                stop = feedback.verify_pre_error(actual, noisy_teacher, p, pred, target)
                np.testing.assert_allclose(actual['state_features'][:stop], teacher[:stop], atol=1e-12, rtol=1e-10)
                name = f'{cal}_dose{di}_noise{ni}.npz'
                np.savez_compressed(out / name, **actual)
                rows.append(dict(**identity, artifact=name,
                                 teacher_accuracy=float(np.mean(pred == target)),
                                 expected_energy=float(amp @ amp),
                                 **suite.rollout_metrics(actual, target, head, clean, graph['neurons'])))
    np.savez_compressed(out / 'calibration.npz', standard_normals=noise, amplitudes=amp_bank,
                        training_sd=sd, observed_root_ids=ids, target=target,
                        teacher_probabilities=prob_bank)
    pd.DataFrame(rows).to_csv(out / 'rollouts.csv', index=False)
    pd.DataFrame(certs).to_csv(out / 'certificates.csv', index=False)
    loss, _, _ = head.objective(features, symbols[1:], bc['l2'],
                                core.sample_weights(199, 3, bc['prefix_window'], bc['prefix_weight']))
    core.write_json(out / 'case.json', dict(case=case, graph=graph, dataset=dataset.identity(symbols),
                    head_sha256=head.digest(), head_parameters=482, training_updates=bc['epochs'],
                    new_fit=True, train_loss=loss,
                    training_accuracy=float(np.mean(head.predict(features) == symbols[1:])),
                    clean_prefix=clean_prefix, own_q=q, zero_sd_coordinates=int(np.sum(sd == 0)),
                    horizon=h, smoke=smoke, actual_paths=len(rows), certificates=len(certs),
                    seconds=time.monotonic()-started))
    seal(out, c, context(c), purpose='Fresh-model coordinate-SD case')
    return rows, certs


def summarize(frame, c, out):
    noisy = frame[frame.calibration != 'clean']
    group = ['cohort', 'family', 'seed', 'topology', 'calibration']
    blocks = noisy.groupby(group)[METRICS].agg(lambda x: np.mean(x.to_numpy())).reset_index()
    blocks.to_csv(out / 'seed-blocks.csv', index=False)
    noisy.groupby(group + ['strength'])[METRICS].agg(lambda x: np.mean(x.to_numpy())).reset_index().to_csv(out / 'dose-seed-table.csv', index=False)
    pairs = []
    for (co, family, seed), z in blocks.groupby(['cohort', 'family', 'seed']):
        v = {(r.topology, r.calibration): r for r in z.itertuples()}
        for control in ['degree', 'role']:
            row = dict(cohort=co, family=family, seed=int(seed), control=control)
            for metric in METRICS:
                own = getattr(v[('intact', 'own')], metric) - getattr(v[(control, 'own')], metric)
                coord = getattr(v[('intact', 'coordinate')], metric) - getattr(v[(control, 'coordinate')], metric)
                row.update({metric + '_own_advantage': own,
                            metric + '_coordinate_advantage': coord,
                            metric + '_incremental_attenuation': coord - own})
            pairs.append(row)
    pair_frame = pd.DataFrame(pairs)
    pair_frame.to_csv(out / 'paired-differences.csv', index=False)
    stats = {}
    gates = {}
    for (co, family, control), z in pair_frame.groupby(['cohort', 'family', 'control']):
        key = f'{co}/{family}/{control}'
        stats[key] = {m: estimate(z[m], c) if np.isfinite(z[m]).all() else None
                      for m in z.columns if m not in ['cohort', 'family', 'control', 'seed']}
        a = z.exact_prefix_symbols_coordinate_advantage.to_numpy()
        b = z.retention_coordinate_advantage.to_numpy()
        need = 4 if co == 'discovery' else 3
        gates[key] = None if not np.isfinite(b).all() else bool(
            a.mean() >= c['minimum_prefix_advantage'] and
            b.mean() >= c['minimum_retention_advantage'] and
            np.count_nonzero(a > 0) >= need and np.count_nonzero(b > 0) >= need)
    keys = [f'{co}/random/degree' for co in c['cohorts']]
    vals = [gates[k] for k in keys]
    core.write_json(out / 'summary.json', dict(statistics=stats, descriptive_gates=gates,
                    primary_keys=keys, primary_confirmed=None if any(v is None for v in vals) else all(vals),
                    primary='coordinate_intact_minus_degree', fresh_model_fits=96,
                    fresh_model_data_graph_seeds=True, cohorts_pooled=False))


@threadpool_limits.wrap(limits=1)
def run(c, out, smoke=False):
    out.mkdir(parents=True, exist_ok=False)
    budget = Budget(c)
    source = context(c)
    core.write_json(out / 'config.json', c)
    core.write_json(out / 'source-hashes.json', source)
    rows, certs = [], []
    try:
        panel = list(cases(c, smoke))
        for i, case in enumerate(panel, 1):
            r, t = execute(c, case, out / stem(case), smoke, budget)
            rows.extend(r); certs.extend(t)
            print(f'Fresh fit {i}/{len(panel)}: {stem(case)}', flush=True)
        pd.DataFrame(rows).to_csv(out / 'raw-rollouts.csv', index=False)
        pd.DataFrame(certs).to_csv(out / 'raw-certificates.csv', index=False)
        if not smoke:
            summarize(pd.DataFrame(rows), c, out)
        assert context(c) == source
        budget.check()
        core.write_json(out / 'verification.json', dict(complete=True, smoke=smoke,
                        cases=len(panel), new_fits=len(panel), actual_paths=len(rows),
                        certificates=len(certs), environment=core.environment(), budget=budget.close()))
        seal(out, c, source, purpose='Fresh-model coordinate-SD main' if not smoke else 'Excluded smoke')
    except Exception as exc:
        core.write_json(out / 'failure.json', dict(error=repr(exc), budget=budget.close()))
        seal(out, c, source, purpose='Preserved incomplete fresh-model attempt')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=CONFIG)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run(read(args.config), args.out, args.smoke)
