"""Phase 6 coverage and resource-bounded whole-brain feasibility CLI."""
import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

from flying.brain.lif import LIFReservoir, LIFParameters
from flying.brain.sparse_lif import SparseLIFReservoir
from flying.data.connectome import sha256
from flying.data.pi_digits import decimal_pi_digits
from flying.data.whole_brain import prepare, load_graph

CONFIG = Path('configs/phase6.json')
MBON11 = 720575940623201833


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.part')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def probes():
    count = json.loads(CONFIG.read_text())['probe_windows']
    return {'digit3': [3]*count, 'digit1': [1]*count, 'pi20': decimal_pi_digits(count).tolist()}


def context():
    paths = sorted(Path('src/flying').rglob('*.py')) + [CONFIG, Path('docs/phase6-protocol.md'),
            Path('requirements-phase6-lock.txt'), Path('requirements-lif-lock.txt')]
    for seed in [701, 702]:
        paths += sorted(Path(f'data/flywire_783_mb_left_kc512_s{seed}').glob('*'))
    return {p.as_posix(): sha256(p) for p in paths if p.is_file()}


def environment():
    import psutil
    return dict(python=sys.version, platform=platform.platform(), processor=platform.processor(),
                packages={name: importlib.metadata.version(name) for name in
                          ['numpy', 'scipy', 'pandas', 'pyarrow', 'psutil', 'Brian2', 'threadpoolctl']},
                host_total_ram_bytes=psutil.virtual_memory().total,
                host_available_ram_bytes=psutil.virtual_memory().available,
                numerical_threads=1)


def record(model, digits):
    counts = model.states(digits)
    arrays = dict(counts=counts, v_mv=model.v.copy(), g_mv=model.g.copy())
    arrays.update(model.spike_arrays())
    return arrays


def assert_exact(expected, actual):
    if set(expected) != set(actual):
        raise ValueError('Replay array keys differ')
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=f'Replay mismatch: {key}')


def oracle(cache, out):
    """Brian2 is only invoked on the two legacy networks, never the full graph."""
    cfg = json.loads(CONFIG.read_text())
    rows = []
    for seed in cfg['circuit_seeds']:
        w, ids, roles, enc = load_graph(cache, 'legacy5', seed, encoder_seed=cfg['encoder_seed'])
        parameters = LIFParameters(dt_ms=cfg['dt_ms'])
        optimized = SparseLIFReservoir(w, enc, roles, parameters)
        brian = LIFReservoir(w, enc, roles, parameters)
        for name, digits in probes().items():
            result = record(optimized, digits)
            counts = brian.states(digits)
            np.testing.assert_array_equal(result['counts'], counts)
            for key, value in brian.spike_arrays().items():
                np.testing.assert_array_equal(result[key], value)
            bv = np.asarray(brian.neurons.v[:]/brian.b.mV)
            bg = np.asarray(brian.neurons.g[:]/brian.b.mV)
            for key, value in [('v_mv', bv), ('g_mv', bg)]:
                np.testing.assert_allclose(result[key], value, atol=cfg['oracle_state_atol_mv'], rtol=0)
            rows.append(dict(seed=seed, probe=name, spikes=len(result['spike_ticks']),
                             spikes_counts_exact=True,
                             max_v_error_mv=float(np.max(np.abs(result['v_mv']-bv))),
                             max_g_error_mv=float(np.max(np.abs(result['g_mv']-bg)))))
        del brian, optimized
        gc.collect()
    write_json(out/'oracle.json', dict(cases=rows, passed=True, context=context()))


def worker(cache, out, level, seed):
    start = time.perf_counter()
    cfg = json.loads(CONFIG.read_text())
    w, ids, roles, enc = load_graph(cache, level, seed, encoder_seed=cfg['encoder_seed'])
    graph_load = time.perf_counter()-start
    matrix_bytes = sum(a.nbytes for a in [w.data, w.indices, w.indptr])
    init = time.perf_counter()
    model = SparseLIFReservoir(w, enc, roles, LIFParameters(dt_ms=cfg['dt_ms']))
    init_time = time.perf_counter()-init
    arrays = dict(node_ids=ids, roles=roles, input_patterns=enc.patterns)
    metrics = []
    mbon = roles == 'MBON'
    mbon11 = int(np.flatnonzero(ids == MBON11)[0])
    for name, digits in probes().items():
        started = time.perf_counter()
        first = record(model, digits)
        first_wall = time.perf_counter()-started
        started = time.perf_counter()
        repeated = record(model, digits)
        repeat_wall = time.perf_counter()-started
        assert_exact(first, repeated)
        counts = first['counts']
        metrics.append(dict(probe=name, first_wall_seconds=first_wall, repeat_wall_seconds=repeat_wall,
                            exact_repeat=True, total_spikes=int(counts.sum()),
                            active_neurons=int(np.count_nonzero(counts.sum(axis=0))),
                            mbon_spikes=int(counts[:, mbon].sum()),
                            active_mbons=int(np.count_nonzero(counts[:, mbon].sum(axis=0))),
                            mbon11_spikes=int(counts[:, mbon11].sum())))
        arrays.update({f'{name}_{k}': v for k, v in first.items()})
        del first, repeated
        print(f'{level} s{seed} {name}: exact repeat, {metrics[-1]["total_spikes"]} spikes', flush=True)
    stem = f'{level}_s{seed}'
    temporary = out/(stem+'.part.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(out/(stem+'.npz'))
    with np.load(out/(stem+'.npz'), allow_pickle=False) as saved:
        assert_exact(arrays, {k: saved[k] for k in saved.files})
    result = dict(level=level, circuit_seed=seed, neurons=len(ids), edges=w.nnz,
                  sparse_csc_bytes=matrix_bytes, graph_load_seconds=graph_load,
                  model_init_seconds=init_time, delay_ring_bytes=model.pending.nbytes,
                  parameters=vars(model.parameters), probes=metrics,
                  input_cells_per_digit=enc.patterns.sum(axis=1).tolist(),
                  archive_sha256=sha256(out/(stem+'.npz')), context=context())
    write_json(out/(stem+'.json'), result)


def supervised(arguments, log, cfg):
    import psutil
    peak, started = 0, time.perf_counter()
    reason = None
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with log.open('w', encoding='utf-8') as output:
        child = subprocess.Popen([sys.executable, '-m', 'flying.training.phase6', *arguments],
                                 stdout=output, stderr=subprocess.STDOUT, env=env)
        process = psutil.Process(child.pid)
        while child.poll() is None:
            try:
                # Windows venv python.exe can be a small redirector process.
                # Measure the executing interpreter and any descendants too.
                rss = tree_rss(process)
                peak = max(peak, rss)
            except psutil.NoSuchProcess:
                break
            if peak > cfg['worker_rss_limit_bytes']:
                reason = 'rss_limit'
            if time.perf_counter()-started > cfg['worker_wall_limit_seconds']:
                reason = 'wall_limit'
            if reason:
                for descendant in reversed(process.children(recursive=True)):
                    try:
                        descendant.kill()
                    except psutil.NoSuchProcess:
                        pass
                child.kill()
                break
            time.sleep(cfg['sample_seconds'])
        code = child.wait()
    result = dict(wall_seconds=time.perf_counter()-started, sampled_peak_rss_bytes=peak,
                  rss_scope='sum of worker process tree RSS (includes Windows venv redirector)',
                  sample_seconds=cfg['sample_seconds'], returncode=code, resource_failure=reason)
    write_json(log.with_suffix('.resources.json'), result)
    if code or reason:
        raise RuntimeError(f'Worker failed ({reason or code}); inspect {log}')
    return result


def tree_rss(process):
    import psutil
    total = 0
    for member in [process, *process.children(recursive=True)]:
        try:
            total += member.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    return total


def run(cache, out):
    cfg = json.loads(CONFIG.read_text())
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new output directory; never overwrite research results')
    out.mkdir(parents=True, exist_ok=True)
    start_context = context()
    write_json(out/'environment.json', environment())
    shutil.copyfile(cache/'coverage.csv', out/'coverage.csv')
    shutil.copyfile(cache/'provenance.json', out/'graph-provenance.json')
    for name in ['build.log', 'build.resources.json']:
        shutil.copyfile(cache/name, out/name)
    args = ['--cache', str(cache), '--out', str(out)]
    supervised(['oracle', *args], out/'oracle.log', cfg)
    for level in cfg['levels']:
        for seed in cfg['circuit_seeds']:
            stem = f'{level}_s{seed}'
            supervised(['worker', *args, '--level', level, '--seed', str(seed)], out/(stem+'.log'), cfg)
            print(f'Completed {stem}', flush=True)
    if context() != start_context:
        raise RuntimeError('Source/protocol changed during run')
    conditions = len(cfg['levels'])*len(cfg['circuit_seeds'])
    manifest = dict(context=start_context, conditions=conditions, probes=conditions*len(probes()),
                    exact_reset_repeats=conditions*len(probes()),
                    files={p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file()})
    write_json(out/'manifest.json', manifest)
    verify(out)


def verify(out, comparison=None):
    manifest = json.loads((out/'manifest.json').read_text())
    if manifest['context'] != context():
        raise ValueError('Source/protocol differs from recorded context')
    for name, digest in manifest['files'].items():
        if sha256(out/name) != digest:
            raise ValueError(f'Artifact checksum mismatch: {name}')
    if comparison is not None:
        verify(comparison)
        for name in manifest['files']:
            if name.endswith('.npz'):
                with np.load(out/name, allow_pickle=False) as a, np.load(comparison/name, allow_pickle=False) as b:
                    assert_exact({k: a[k] for k in a.files}, {k: b[k] for k in b.files})
    print('Verified artifact hashes'+(' and all regenerated arrays' if comparison else ''), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'build', 'run', 'oracle', 'worker', 'verify'])
    parser.add_argument('--raw', type=Path, default=Path('data/raw'))
    parser.add_argument('--cache', type=Path, default=Path('outputs/phase6-graphs'))
    parser.add_argument('--out', type=Path, default=Path('outputs/phase6'))
    parser.add_argument('--level', choices=['legacy5', 'left5', 'brain5', 'brain1'])
    parser.add_argument('--seed', type=int, choices=[701, 702])
    parser.add_argument('--compare', type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.command == 'prepare':
            args.cache.mkdir(parents=True, exist_ok=True)
            supervised(['build', '--raw', str(args.raw), '--cache', str(args.cache)],
                       args.cache/'build.log', json.loads(CONFIG.read_text()))
        elif args.command == 'build':
            started = time.perf_counter()
            result = prepare(args.raw, args.cache)
            print(json.dumps(dict(neurons=result['neurons'], edges=result['brain1_edges'],
                                  wall_seconds=time.perf_counter()-started)), flush=True)
        elif args.command == 'run':
            run(args.cache, args.out)
        elif args.command == 'oracle':
            oracle(args.cache, args.out)
        elif args.command == 'worker':
            if args.level is None or args.seed is None:
                parser.error('worker requires --level and --seed')
            worker(args.cache, args.out, args.level, args.seed)
        else:
            verify(args.out, args.compare)


if __name__ == '__main__':
    main()
