"""TDC evidence IO and resource accounting; no scientific scoring logic."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
import traceback

import numpy as np
import psutil
from threadpoolctl import threadpool_info


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def write(p, x):
    p = Path(p)
    tmp = p.with_suffix(p.suffix + '.part')
    tmp.write_text(json.dumps(x, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(p)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def array_sha(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], text=True).strip()


def environment():
    packages = ['numpy', 'scipy', 'pandas', 'psutil', 'matplotlib', 'threadpoolctl', 'pyarrow']
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                processor=platform.processor(), logical_cpu_count=psutil.cpu_count(),
                total_ram_bytes=psutil.virtual_memory().total,
                packages={n: importlib.metadata.version(n) for n in packages},
                blas=threadpool_info(), thread_variables={n: os.environ.get(n) for n in
                ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']})


class Resources:
    def __init__(self, c):
        self.c = c
        self.start = time.perf_counter()
        self.proc = psutil.Process()
        self.peak = 0
        self.os_peak = 0
        self.stop = threading.Event()
        self.sample()
        self.thread = threading.Thread(target=self.monitor, daemon=True)
        self.thread.start()

    def sample(self):
        m = self.proc.memory_info()
        self.peak = max(self.peak, m.rss)
        self.os_peak = max(self.os_peak, getattr(m, 'peak_wset', 0))

    def monitor(self):
        while not self.stop.wait(.05):
            self.sample()

    def check(self):
        self.sample()
        if time.perf_counter()-self.start > self.c['max_seconds'] or max(self.peak, self.os_peak) > self.c['max_rss_bytes']:
            raise RuntimeError('Predeclared resource budget exceeded; preserve attempt')

    def close(self):
        self.stop.set()
        self.thread.join()
        self.sample()
        return dict(seconds=time.perf_counter()-self.start, peak_sampled_rss_bytes=self.peak,
                    os_process_peak_working_set_bytes=self.os_peak or None, sample_seconds=.05)


@contextmanager
def attempt(out, c, kind):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    r = Resources(c)
    write(out/'attempt.json', dict(kind=kind, complete=False, started_utc=datetime.now(timezone.utc).isoformat(), source_commit=git('rev-parse', 'HEAD')))
    try:
        yield r
        r.check()
        write(out/'resources.json', r.close())
        write(out/'attempt.json', dict(kind=kind, complete=True, ended_utc=datetime.now(timezone.utc).isoformat(), source_commit=git('rev-parse', 'HEAD')))
    except BaseException as e:
        write(out/'resources.json', r.close())
        write(out/'failure.json', dict(kind=kind, complete=False, exception=type(e).__name__, message=str(e), traceback=traceback.format_exc(), source_commit=git('rev-parse', 'HEAD')))
        seal(out, dict(kind=kind, complete=False))
        raise


def seal(out, meta):
    files = {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob('*'))
             if p.is_file() and p != out/'manifest.json'}
    write(out/'manifest.json', dict(**meta, artifacts=files))


def check_manifest(root):
    m = read(root/'manifest.json')
    for n, h in m['artifacts'].items():
        assert sha(root/n) == h, (root, n)
    assert m.get('complete', True), root
    return m


def source_record(c, config):
    # Experimental inputs and scripts must be committed before any trajectory.
    assert subprocess.run(['git', 'diff', '--quiet', 'HEAD']).returncode == 0, 'Tracked changes before execution'
    paths = sorted(Path('src/flying').rglob('*.py'))
    paths += [Path('scripts')/n for n in ['temporal_memory_curve.py', 'verify_temporal_memory_curve.py', 'tdc_support.py', 'alphabet_memory.py', 'structural_k4.py']]
    for n in ['temporal_null_ensemble.py', 'verify_temporal_null_ensemble.py']:
        if (Path('scripts')/n).exists():
            paths.append(Path('scripts')/n)
    paths += [Path(config), Path(c['protocol']), Path('requirements-act1-lock.txt')]
    for circuit in c['circuit_seeds']:
        paths += sorted(Path(f'data/flywire_783_mb_left_kc512_s{circuit}').glob('*'))
    cache = Path(c['cache'])
    for n in ['nodes.npz', 'brain5.npz', 'provenance.json']:
        if (cache/n).exists():
            paths.append(cache/n)
    paths += sorted((cache.parent/'raw').glob('*'))
    return dict(source_commit=git('rev-parse', 'HEAD'), source_tree=git('rev-parse', 'HEAD^{tree}'),
                tracked_changes=False, hashes={p.as_posix(): sha(p) for p in paths}, environment=environment())


def old_tree_unchanged(baseline):
    # Comparing Git blobs protects every historical archive, including sparse ones.
    changed = git('diff', '--name-only', baseline, 'HEAD', '--', 'results/').splitlines()
    assert all(n.startswith('results/tdc_v2/') for n in changed), changed
    assert not git('diff', '--name-only', 'HEAD', '--', 'results/'), 'Modified tracked result bytes'
    return dict(historical_git_result_blobs_unchanged=True, baseline_commit=baseline)
