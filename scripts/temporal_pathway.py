"""Cross current and historical DAN→MBON cuts on the same input streams."""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import softmax
from threadpoolctl import threadpool_limits

from flying.training import whole_brain_memory as core
from alphabet_memory import read, check
from context_memory import estimate
from frozen_state_probe import Budget, fit_fold, labels, measures
from observation_location import select_observations
from pathway_memory import seal
from structural_k4 import build as build_k4
import feedback_noise as feedback
import fresh_coordinate_sd as fresh
import research_suite as suite


CONFIG = 'configs/temporal_pathway.json'


def context(c):
    paths = sorted(Path('src/flying').rglob('*.py'))
    paths += [Path(c['protocol']), Path(CONFIG), Path('configs/sequence_memory.json')]
    paths += [Path('scripts') / (n + '.py') for n in
              ['temporal_pathway', 'verify_temporal_pathway', 'structural_k4',
               'observation_location', 'frozen_state_probe', 'fresh_coordinate_sd',
               'feedback_noise', 'research_suite']]
    for ci in c['circuit_seeds']:
        paths += sorted(Path(f'data/flywire_783_mb_left_kc512_s{ci}').glob('*'))
    return {p.as_posix(): core.sha256(p) for p in paths}


def stem(cohort, block, ci):
    return f'{cohort}_c{ci}_s{block["seed"]}'


def cut_model(base, roles):
    co = base.weights.tocoo()
    target = (roles[co.row] == 'MBON') & (roles[co.col] == 'DAN')
    assert int(target.sum()) == 189
    weight = sparse.csr_matrix((co.data[~target], (co.row[~target], co.col[~target])),
                               shape=base.weights.shape)
    weight.sort_indices()
    assert weight.nnz == base.weights.nnz - 189
    return core.TimedReservoir(weight, base.encoder, roles, base.leak, base.schedule), weight


def build(c, ci, block):
    base, mbons, graph, _ = build_k4(c, ci, block['seed'])
    folder = Path(f'data/flywire_783_mb_left_kc512_s{ci}')
    _, ids, _ = core.load_connectome(folder)
    roles, _ = core.load_roles(folder, ids)
    roles = np.asarray(roles)
    sites = select_observations(ids, roles, base.encoder.patterns, block['observation_seed'])
    assert np.array_equal(sites['MBON'], mbons)
    cut, cut_weights = cut_model(base, roles)
    union = np.concatenate([sites[k] for k in c['sites']])
    assert len(np.unique(union)) == 96
    meta = dict(graph=graph, cut_weight_sha256=core.weight_hash(cut_weights),
                removed_edges=189, observed_root_ids={k: [ids[i] for i in v] for k, v in sites.items()})
    return base, cut, sites, union, meta


def one_step(model, previous, symbol):
    """The registered crossed-state equation; verifier uses TimedReservoir.step."""
    w = model.weights
    stimulation = model.encoder(int(symbol))
    following = (1-model.leak)*previous + model.leak*np.tanh(w.dot(previous) + stimulation)
    staged = previous.copy()
    staged[model.kc] = following[model.kc]
    following[model.mbon] = (1-model.leak)*previous[model.mbon] + model.leak*np.tanh(
        model.mbon_weights.dot(staged) + stimulation[model.mbon])
    return following


def crossed_features(base, cut, symbols, union):
    base.reset(); cut.reset()
    output = np.empty((4, len(symbols), len(union)))
    state_distance = np.empty((len(symbols), 2))
    for t, symbol in enumerate(symbols):
        previous_base = base.state.copy()
        previous_cut = cut.state.copy()
        intact = base.step(int(symbol))
        persistent = cut.step(int(symbol))
        current_only = one_step(cut, previous_base, symbol)
        history_only = one_step(base, previous_cut, symbol)
        for j, state in enumerate([intact, current_only, history_only, persistent]):
            output[j, t] = state[union]
        state_distance[t] = [np.linalg.norm(intact-persistent),
                             np.linalg.norm(intact[union]-persistent[union])]
    assert np.isfinite(output).all()
    return output, state_distance


def features_for(states, arm, site, c, warmup=0):
    ai = c['arms'].index(arm)
    si = c['sites'].index(site)
    return states[ai, warmup:, si*48:(si+1)*48]


def execute_case(c, ci, block, cohort, out, smoke, budget):
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    base, cut, sites, union, graph = build(c, ci, block)
    ntrain = c['smoke_train_samples'] if smoke else c['train_samples']
    ntest = c['smoke_test_samples'] if smoke else c['test_samples']
    streams = {}
    for split, n in [('train', ntrain), ('test', ntest)]:
        symbols = np.random.default_rng(block[split+'_seed']).integers(
            0, c['alphabet_size'], c['warmup']+n).astype(np.uint8)
        states, distance = crossed_features(base, cut, symbols, union)
        streams[split] = dict(symbols=symbols, states=states, distance=distance)
        np.savez_compressed(out / f'{split}-states.npz', **streams[split])
    ytrain = labels(streams['train']['symbols'], np.arange(c['warmup'], len(streams['train']['symbols'])), c['lags'])
    ytest = labels(streams['test']['symbols'], np.arange(c['warmup'], len(streams['test']['symbols'])), c['lags'])
    rows = []
    fits = {}
    for site in c['sites']:
        for arm in c['arms']:
            budget.check()
            xtrain = features_for(streams['train']['states'], arm, site, c, c['warmup'])
            xtest = features_for(streams['test']['states'], arm, site, c, c['warmup'])
            fold = fit_fold(xtrain, xtest, ytrain, ytest, c['alpha'])
            fits[(site, arm)] = fold
            name = f'{site}_{arm}_refit.npz'
            np.savez_compressed(out / name, **fold)
            for result in measures([fold], c['lags']):
                rows.append(dict(cohort=cohort, seed=block['seed'], circuit_seed=ci,
                                 site=site, arm=arm, mode='refit', **result))
        intact = fits[(site, 'intact')]
        for arm in c['arms']:
            x = features_for(streams['test']['states'], arm, site, c, c['warmup'])
            p = ((x-intact['mean'])/intact['scale'])@intact['weights']+intact['bias']
            p = p.reshape(len(x), len(c['lags']), c['alphabet_size'])
            prediction = p.argmax(2)
            np.savez_compressed(out / f'{site}_{arm}_frozen.npz', scores=p, prediction=prediction)
            for j, lag in enumerate(c['lags']):
                rows.append(dict(cohort=cohort, seed=block['seed'], circuit_seed=ci,
                                 site=site, arm=arm, mode='frozen', lag=lag,
                                 test_accuracy=float(np.mean(prediction[:, j] == ytest[:, j]))))
    pd.DataFrame(rows).to_csv(out / 'lag-metrics.csv', index=False)
    core.write_json(out / 'case.json', dict(cohort=cohort, block=block, circuit_seed=ci,
                    graph=graph, warmup=c['warmup'], train_samples=ntrain, test_samples=ntest,
                    site_indices={k: v.tolist() for k, v in sites.items()},
                    independent_refits=8, seconds=time.monotonic()-started, smoke=smoke))
    seal(out, c, context(c), purpose='Crossed temporal pathway case')
    return rows


def infer(frame, c):
    selected = frame[frame.lag.isin(c['primary_lags'])]
    cells = selected.groupby(['cohort','seed','circuit_seed','site','arm','mode']).test_accuracy.mean().reset_index()
    blocks = cells.groupby(['cohort','seed','site','arm','mode']).test_accuracy.mean().reset_index()
    pairs = []
    for (co, seed, site, mode), z in blocks.groupby(['cohort','seed','site','mode']):
        v = z.set_index('arm').test_accuracy
        assert set(v.index) == set(c['arms'])
        pairs.append(dict(cohort=co, seed=int(seed), site=site, mode=mode,
                          intact=float(v['intact']),
                          persistent_loss=float(v['intact']-v['persistent']),
                          current_only_loss=float(v['intact']-v['current_only']),
                          history_only_loss=float(v['intact']-v['history_only'])))
    pair_frame = pd.DataFrame(pairs)
    stats, gates = {}, {}
    for (co, site, mode), z in pair_frame.groupby(['cohort','site','mode']):
        key = f'{co}/{site}/{mode}'
        stats[key] = {col: estimate(z[col], c) for col in
                      ['intact','persistent_loss','current_only_loss','history_only_loss']}
        if site == 'MBON' and mode == 'refit':
            p = z.persistent_loss.to_numpy()
            d = z.current_only_loss.to_numpy()
            h = z.history_only_loss.to_numpy()
            gates[co] = dict(persistent=bool(p.mean() >= c['minimum_persistent_loss'] and (p > 0).all()),
                             current=bool(d.mean() >= c['minimum_current_loss'] and (d > 0).all()),
                             dominance=bool(d.mean() >= c['minimum_current_fraction']*p.mean()),
                             history=bool(h.mean() <= c['maximum_history_loss']))
    return cells, blocks, pair_frame, dict(statistics=stats, gates=gates,
           primary_confirmed=all(all(z.values()) for z in gates.values()),
           primary_site='MBON', primary_mode='refit', cohorts_pooled=False)


def parent_cases():
    parent = read('configs/fresh_coordinate_sd.json')
    for cohort, blocks in parent['cohorts'].items():
        for bi, (seed, ds) in enumerate(blocks):
            for ci in parent['circuit_seeds']:
                yield parent, dict(cohort=cohort, block_index=bi, seed=seed,
                                   dataset_seed=ds, family='random', circuit_seed=ci,
                                   level='legacy5', topology='intact')


def autonomous_arm(base, cut, head, prompt, horizon, arm):
    base.reset(); cut.reset()
    for symbol in prompt[:2]:
        base.step(int(symbol)); cut.step(int(symbol))
    previous_base = base.state.copy(); previous_cut = cut.state.copy()
    intact = base.step(int(prompt[2])); persistent = cut.step(int(prompt[2]))
    current = one_step(cut, previous_base, prompt[2])
    history = one_step(base, previous_cut, prompt[2])
    features, probabilities, prediction = [], [], []
    obs = base.mbon
    for t in range(horizon):
        state = dict(intact=intact, current_only=current,
                     history_only=history, persistent=persistent)[arm]
        x = state[obs].copy()
        p = softmax(head.logits(x[None,:]), axis=1)[0]
        digit = int(p.argmax())
        features.append(x); probabilities.append(p); prediction.append(digit)
        if t+1 < horizon:
            previous_base = base.state.copy(); previous_cut = cut.state.copy()
            intact = base.step(digit); persistent = cut.step(digit)
            current = one_step(cut, previous_base, digit)
            history = one_step(base, previous_cut, digit)
    return dict(features=np.asarray(features), probabilities=np.asarray(probabilities),
                prediction=np.asarray(prediction))


def autonomous_panel(c, out, budget):
    parent_root = Path(c['parent'])
    assert check(parent_root)
    rows = []
    for _, case in parent_cases():
        budget.check()
        source = parent_root / suite.stem(case)
        check(source)
        with np.load(source / 'checkpoint.npz') as archive:
            cp = dict(archive)
        head = feedback.load_head(cp, case['seed'])
        base, _, _, _, _ = fresh.build(read('configs/fresh_coordinate_sd.json'), case)
        folder = Path(f'data/flywire_783_mb_left_kc512_s{case["circuit_seed"]}')
        _, ids, _ = core.load_connectome(folder)
        roles, _ = core.load_roles(folder, ids)
        cut, weight = cut_model(base, np.asarray(roles))
        target = cp['symbols'][3:]
        case_out = out / suite.stem(case)
        case_out.mkdir()
        for arm in c['arms']:
            path = autonomous_arm(base, cut, head, cp['symbols'][:3], len(target), arm)
            if arm == 'intact':
                with np.load(source / 'clean.npz') as clean:
                    np.testing.assert_array_equal(path['prediction'], clean['prediction'])
                    np.testing.assert_allclose(path['probabilities'], clean['probabilities'], atol=1e-12, rtol=1e-10)
            np.savez_compressed(case_out / (arm+'.npz'), **path)
            prefix = core.prefix_score(target, path['prediction'])
            rows.append(dict(**case, arm=arm, exact_prefix_symbols=prefix,
                             accuracy=float(np.mean(path['prediction']==target)),
                             parent_manifest_sha256=core.sha256(source/'manifest.json'),
                             parent_head_sha256=head.digest(), cut_weight_sha256=core.weight_hash(weight)))
        seal(case_out, c, context(c), purpose='Target-free autonomous crossed-state check')
    frame = pd.DataFrame(rows)
    frame.to_csv(out / 'autonomous-rollouts.csv', index=False)
    return frame


@threadpool_limits.wrap(limits=1)
def run(c, out, smoke=False):
    out.mkdir(parents=True, exist_ok=False)
    source = context(c)
    budget = Budget(c)
    core.write_json(out/'config.json', c)
    core.write_json(out/'source-hashes.json', source)
    rows = []
    try:
        panel = [('smoke', c['smoke'], 701)] if smoke else [
            (co, block, ci) for co, blocks in c['cohorts'].items()
            for block in blocks for ci in c['circuit_seeds']]
        for i, (co, block, ci) in enumerate(panel, 1):
            rows.extend(execute_case(c, ci, block, co, out/stem(co, block, ci), smoke, budget))
            print(f'Crossed panel {i}/{len(panel)}: {stem(co, block, ci)}', flush=True)
        frame = pd.DataFrame(rows)
        frame.to_csv(out/'raw-lag-metrics.csv', index=False)
        if not smoke:
            cells, blocks, pairs, summary = infer(frame, c)
            cells.to_csv(out/'circuit-cells.csv', index=False)
            blocks.to_csv(out/'seed-blocks.csv', index=False)
            pairs.to_csv(out/'paired-differences.csv', index=False)
            autonomous = autonomous_panel(c, out, budget)
            summary['autonomous_paths'] = len(autonomous)
            core.write_json(out/'summary.json', summary)
        assert source == context(c)
        budget.check()
        core.write_json(out/'verification.json', dict(complete=True, smoke=smoke,
                        cases=len(panel), refits=8*len(panel),
                        autonomous_paths=0 if smoke else 64,
                        environment=core.environment(), budget=budget.close()))
        seal(out, c, source, purpose='Temporal pathway main' if not smoke else 'Excluded temporal smoke')
    except Exception as exc:
        core.write_json(out/'failure.json', dict(error=repr(exc), budget=budget.close()))
        seal(out, c, source, purpose='Preserved incomplete temporal attempt')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=CONFIG)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run(read(args.config), args.out, args.smoke)
