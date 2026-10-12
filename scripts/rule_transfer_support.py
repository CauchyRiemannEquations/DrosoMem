"""M11 frozen provenance, IO and resource support; no scientific algorithms."""
import hashlib
from pathlib import Path
import subprocess
from counterfactual_support import tree, materialized_checks, authenticate_git_sources
from tdc_support import read, write, sha, array_sha, git, environment, attempt, seal

BASELINE = '5467811412cb36a0a56f6137e98f93b9e31f8206'
PREREGISTRATION = '68b49d37e79a9a00c95f00baaf44c73d61147c6f'
CONFIG = Path('configs/rule_transfer.json')
OVERVIEWS = {'README.md', 'docs/research-status.md', 'docs/research-roadmap.md', 'docs/next-work.md'}


def config(smoke=False):
    c = read(CONFIG)
    for p in [CONFIG, Path(c['protocol']), Path('docs/rule-transfer-prior-audit-m11.md')]:
        assert sha(p) == hashlib.sha256(subprocess.check_output(['git', 'show', PREREGISTRATION + ':' + p.as_posix()])).hexdigest()
    assert c['baseline_commit'] == BASELINE and not c['recurrent_learning'] and not c['performance_selection']
    if smoke:
        c.update(models=c['smoke_models'], input_seeds=c['smoke_input_seeds'], families=[c['smoke_family']], training_rows=88)
    return c


def coverage(root, complete=True):
    m = read(root / 'manifest.json')
    names = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p != root / 'manifest.json'}
    assert set(m['artifacts']) == names
    assert all(sha(root / n) == h for n, h in m['artifacts'].items())
    assert m['complete'] is complete
    return m


def parents(c):
    for key, digest in [('task_parent', 'task_parent_manifest_sha256'), ('task_validation', 'task_validation_manifest_sha256')]:
        assert sha(Path(c[key]) / 'manifest.json') == c[digest]
        coverage(Path(c[key]))
    checks = read(Path(c['task_validation']) / 'checks.json')
    assert checks['all_checks_pass'] and checks['outcome'] == 'READY' and checks['neural_model_fits'] == 0
    p = Path(c['graph_parent'])
    assert sha(p / 'manifest.json') == c['graph_parent_manifest_sha256']
    for s in range(1840001, 1840011):
        coverage(p / 'graphs' / f'legacy5_g{s}')
        assert read(p / 'graphs' / f'legacy5_g{s}' / 'audit.json')['construction_rule_pass']
    return dict(task_ready=True, partial_bank_members=10, all_used_parent_checksums_pass=True)


def history(bytes_check=False):
    old, now = tree(BASELINE), tree('HEAD')
    protected = {n: h for n, h in old.items() if n not in OVERVIEWS}
    assert all(now.get(n) == h for n, h in protected.items())
    assert not set(git('diff', '--name-only', 'HEAD').splitlines()).intersection(protected)
    report = dict(baseline_commit=BASELINE, protected_paths=len(protected),
                  prior_result_git_identities=sum(n.startswith('results/') for n in protected),
                  protected_git_identities_preserved=True, old_results_freshly_replayed=False)
    if bytes_check:
        report.update(materialized_checks(protected))
    return report


def source(c):
    assert not git('diff', '--name-only', 'HEAD'), 'Commit source before outcomes'
    revision = git('rev-parse', 'HEAD')
    paths = list(Path('src/flying').rglob('*.py')) + list(Path('scripts').glob('*.py'))
    paths += [CONFIG, Path(c['protocol']), Path('docs/rule-transfer-prior-audit-m11.md'),
              Path('tests/test_rule_transfer.py'), Path('requirements-act1-lock.txt'),
              Path('configs/rule_transfer_design.json'), Path('configs/structural_specificity_design.json')]
    paths += list(Path(c['source_data']).glob('*'))
    for key in ['task_parent', 'task_validation']:
        paths += list(Path(c[key]).glob('*'))
    parent = Path(c['graph_parent'])
    paths += [parent / 'manifest.json', parent / 'definitions/legacy5/definition.json', parent / 'definitions/legacy5/definition.npz']
    for s in range(1840001, 1840011):
        paths += list((parent / 'graphs' / f'legacy5_g{s}').glob('*'))
    names = sorted({p.as_posix() for p in paths if p.is_file()})
    hashes = {n: sha(n) for n in names}
    authenticate_git_sources(names, hashes, revision)
    return dict(source_commit=revision, source_tree=git('rev-parse', 'HEAD^{tree}'),
                protocol_commit=PREREGISTRATION, hashes=hashes, environment=environment())


def unchanged(record):
    assert git('rev-parse', 'HEAD') == record['source_commit'] and not git('diff', '--name-only', 'HEAD')
    assert all(sha(n) == h for n, h in record['hashes'].items())


def authenticate_source(record):
    assert record['protocol_commit'] == PREREGISTRATION
    authenticate_git_sources(sorted(record['hashes']), record['hashes'], record['source_commit'])


def case_key(family, model, seed):
    return f'{family}_{model}_s{seed}'


def artifact_limit(out, c):
    n = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    assert n <= c['max_stage_artifact_bytes'], 'Registered disk artifact cap exceeded'
    return n


def baseline(out):
    c, record = config(), source(config())
    with attempt(out, c, 'm11-baseline'):
        write(out / 'source.json', record)
        write(out / 'parent-checks.json', parents(c))
        write(out / 'history.json', history(True))
        unchanged(record)
    seal(out, dict(complete=True))


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--baseline', type=Path, required=True)
    baseline(p.parse_args().baseline)
