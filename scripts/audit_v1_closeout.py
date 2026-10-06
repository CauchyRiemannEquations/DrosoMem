"""Read-only v1 audit; seal Git inventory and materialized evidence separately."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def write(p, x):
    p.write_text(json.dumps(x, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def audit(repo, out):
    start = time.perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    git = lambda *args: subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()
    commit = git('rev-parse', 'HEAD')
    entries = {}
    for line in git('ls-tree', '-r', commit).splitlines():
        meta, name = line.split('\t', 1)
        entries[name] = dict(zip(['mode', 'type', 'git_blob'], meta.split()))
    write(out/'baseline-git-tree.json', entries)
    files = {name: digest(repo/name) for name in entries if name.startswith('results/') and (repo/name).is_file()}
    write(out/'materialized-result-sha256.json', files)
    byte_mismatch = []
    for name in files:
        data = (repo/name).read_bytes()
        blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        if blob != entries[name]['git_blob']:
            byte_mismatch.append(name)
    artifact_checks, unresolved, mismatches = [], [], []
    for p in sorted((repo/'results').rglob('manifest.json')):
        if p.relative_to(repo).as_posix() not in entries:
            continue
        m = json.loads(p.read_text(encoding='utf-8'))
        for name, expected in m.get('artifacts', {}).items():
            if isinstance(expected, dict):
                expected = expected.get('sha256')
            if not isinstance(expected, str) or len(expected) != 64:
                unresolved.append(dict(manifest=p.relative_to(repo).as_posix(), key=name, reason='unrecognized checksum schema'))
                continue
            candidates = [p.parent/name, repo/name]
            q = next((q for q in candidates if q.is_file()), None)
            if q is None:
                unresolved.append(dict(manifest=p.relative_to(repo).as_posix(), key=name, reason='not materialized in sparse checkout'))
                continue
            actual = digest(q)
            row = dict(manifest=p.relative_to(repo).as_posix(), artifact=q.relative_to(repo).as_posix(), expected=expected, actual=actual, matches=actual == expected)
            artifact_checks.append(row)
            if actual != expected:
                mismatches.append(row)
    write(out/'artifact-checks.json', artifact_checks)
    write(out/'unresolved-historical-checks.json', unresolved)
    latest = {}
    for study in ['followup3', 'followup4', 'followup4b', 'followup5']:
        root = repo/'results'/f'{study}_main'
        v = repo/'results'/f'{study}_main_validation'
        summary = json.loads((root/'summary.json').read_text())
        check = json.loads((v/'checks.json').read_text())
        assert check['all_checks_pass'] and check['result_manifest_sha256'] == digest(root/'manifest.json')
        latest[study] = dict(summary=summary, validation=check, manifest_sha256=digest(root/'manifest.json'))
    b = pd.read_csv(repo/'results/followup3_main/raw-probes.csv')
    b = b[b.k == 10].groupby(['seed', 'family']).excess.mean()
    iid = b.xs('iid', level='family')
    markov = b.xs('markov', level='family')
    assert (iid >= .05).all() and (markov >= .05).all()
    f = pd.read_csv(repo/'results/followup4_main/raw-cases.csv')
    a = f.groupby(['seed', 'level', 'arm']).accuracy.mean()
    scale = a.xs(('brain5', 'intact'), level=('level', 'arm')) - a.xs(('legacy5', 'intact'), level=('level', 'arm'))
    assert not (scale.mean() >= .05 and (scale > 0).all())
    # Whole-brain pathway values and all local functional failures remain separate.
    g = pd.read_csv(repo/'results/followup4b_main/raw-cases.csv').groupby('seed').real_minus_shuffled.mean()
    assert not (g.mean() >= .03 and (g > 0).all())
    assert latest['followup5']['summary']['functional_response'] is False
    assert latest['followup5']['summary']['useful_fixed_readout'] is False
    write(out/'latest-evidence.json', latest)
    write(out/'recomputed-gates.json', dict(iid_lag2_excess_by_seed=iid.to_dict(), markov_lag2_excess_by_seed=markov.to_dict(), whole_minus_partial_by_seed=scale.to_dict(), real_minus_null_by_seed=g.to_dict(), historical_access=True, whole_brain_advantage=False, real_wiring_advantage=False, local_functional_response=False, local_fixed_decoder_benefit=False))
    arrays = 0
    nonfinite = []
    for name in files:
        if name.endswith('.npz'):
            with np.load(repo/name, allow_pickle=False) as a:
                for k in a.files:
                    x = a[k]
                    arrays += 1
                    if x.dtype.kind in 'fci' and not np.isfinite(x).all():
                        nonfinite.append([name, k])
    # Document links are checked against the full Git inventory, not sparse disk.
    import re
    links_missing = []
    for name in ['README.md', 'docs/research-status.md', 'docs/research-roadmap.md', 'docs/next-work.md']:
        for target in re.findall(r'\]\(([^)]+)\)', (repo/name).read_text(encoding='utf-8')):
            if '://' in target or target.startswith('#'):
                continue
            import posixpath
            dest = posixpath.normpath(posixpath.join(str(Path(name).parent).replace('\\', '/'), target.split('#')[0]))
            if dest not in entries:
                links_missing.append([name, target])
    report = dict(baseline_commit=commit, baseline_tree=git('rev-parse', commit + '^{tree}'), tracked_result_files=sum(n.startswith('results/') for n in entries), materialized_result_files=len(files), excluded_result_files=sum(n.startswith('results/') for n in entries)-len(files), materialized_git_blob_mismatches=byte_mismatch, artifact_checks=len(artifact_checks), artifact_mismatches=mismatches, unresolved_checks=len(unresolved), numerical_arrays_inspected=arrays, nonfinite_arrays=nonfinite, missing_document_targets=links_missing, seconds=time.perf_counter()-start, scope='All Git result paths sealed by blob ID; checksum/array audit covers materialized archives only. Historical state trajectories are not all rerun. Recent gates and validation-parent hashes checked independently.')
    write(out/'audit.json', report)
    write(out/'manifest.json', dict(source_commit=commit, artifacts={p.name: digest(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    audit(a.repo, a.out)
