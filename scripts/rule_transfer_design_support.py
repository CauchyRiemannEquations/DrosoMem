"""M10 evidence IO/provenance only; no task or prediction algorithms."""
import hashlib
from pathlib import Path
import subprocess
from counterfactual_support import tree, materialized_checks
from tdc_support import read, write, sha, git, environment, attempt, seal

BASELINE = '81ad81f25c445c8964599d35f52e0bef51ce4755'
PREREGISTRATION = '61b6139028720b708463e442c149cd74f8d2bd2b'
CONFIG = Path('configs/rule_transfer_design.json')
OVERVIEWS = {'README.md', 'docs/research-status.md', 'docs/research-roadmap.md', 'docs/next-work.md'}
SOURCES = [CONFIG, Path('docs/rule-transfer-design-protocol.md'), Path('docs/rule-transfer-prior-audit.md')]
SOURCES += [Path('scripts') / n for n in ['rule_transfer_design.py', 'verify_rule_transfer_design.py', 'audit_rule_transfer_design.py', 'run_rule_transfer_design_guards.py', 'rule_transfer_design_support.py', 'tdc_support.py', 'counterfactual_support.py']]
SOURCES += [Path('tests/test_rule_transfer_design.py'), Path('requirements-act1-lock.txt')]


def config():
    c = read(CONFIG)
    for p in SOURCES[:3]:
        frozen = subprocess.check_output(['git', 'show', PREREGISTRATION + ':' + p.as_posix()])
        assert sha(p) == hashlib.sha256(frozen).hexdigest(), ('Registered bytes changed', p)
    assert c['baseline_commit'] == BASELINE and c['neural_model_fits'] == 0
    return c


def history(check_bytes=False):
    old, now = tree(BASELINE), tree('HEAD')
    protected = {n: h for n, h in old.items() if n not in OVERVIEWS}
    assert all(now.get(n) == h for n, h in protected.items()), 'Historical Git identity changed'
    assert not set(git('diff', '--name-only', 'HEAD').splitlines()).intersection(protected)
    answer = dict(baseline_commit=BASELINE, protected_paths=len(protected),
                  prior_result_git_identities=sum(n.startswith('results/') for n in protected),
                  protected_git_identities_preserved=True, old_results_freshly_replayed=False)
    if check_bytes:
        answer.update(materialized_checks(protected))
    return answer


def source():
    assert not git('diff', '--name-only', 'HEAD'), 'Commit scientific source before outcomes'
    rev = git('rev-parse', 'HEAD')
    hashes = {p.as_posix(): sha(p) for p in SOURCES}
    for name, h in hashes.items():
        assert h == hashlib.sha256(subprocess.check_output(['git', 'show', rev + ':' + name])).hexdigest()
    return dict(source_commit=rev, protocol_commit=PREREGISTRATION,
                source_tree=git('rev-parse', 'HEAD^{tree}'), hashes=hashes, environment=environment())


def unchanged(record):
    assert git('rev-parse', 'HEAD') == record['source_commit']
    assert all(sha(n) == h for n, h in record['hashes'].items())
    assert not git('diff', '--name-only', 'HEAD')


def budget(c):
    return dict(max_seconds=c['max_seconds_per_stage'], max_rss_bytes=c['max_process_rss_bytes'])


def authenticate(root):
    m = read(root / 'manifest.json')
    names = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p.name != 'manifest.json'}
    assert m['complete'] and names == set(m['artifacts'])
    assert all(sha(root / n) == h for n, h in m['artifacts'].items())
    record = read(root / 'source.json')
    assert record['protocol_commit'] == PREREGISTRATION
    for n, h in record['hashes'].items():
        assert h == hashlib.sha256(subprocess.check_output(['git', 'show', record['source_commit'] + ':' + n])).hexdigest()
    return record
