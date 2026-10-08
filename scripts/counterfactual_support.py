"""M6 frozen parent/source provenance and immutable historical identity guards."""
import argparse
import hashlib
from pathlib import Path
import subprocess

from tdc_support import (read, write, sha, array_sha, git, environment, attempt,
                         seal, check_manifest)

BASELINE = '43d6b068af853dd317241fc611f0df78b7110fca'
PREREGISTRATION = 'c2263270503c1d106dce7069a496c30c924e04f6'
CONFIG = Path('configs/counterfactual_tracking.json')
NAMESPACE = Path('results/counterfactual_tracking_v1')
OVERVIEWS = {'README.md', 'docs/research-status.md', 'docs/research-roadmap.md', 'docs/next-work.md'}


def tree(revision):
    return {line.split('\t', 1)[1]: line.split('\t', 1)[0].split()[2]
            for line in git('ls-tree', '-r', revision).splitlines()}


def config(smoke=False):
    c = read(CONFIG)
    assert (c['baseline_commit'], c['namespace'], c['protocol']) == (BASELINE, NAMESPACE.as_posix(), 'docs/counterfactual-tracking-protocol.md')
    assert c['parent_namespace'] == 'results/functional_sensitivity_v1/main'
    assert c['parent_source_commit'] == 'c541993484f6778a031d71ad4eb1798c748f6ce7'
    assert c['parent_protocol_commit'] == '37212abeb6cfc1a4907e08348ff41030f3edfb8b'
    assert c['parent_manifest_sha256'] == '98eb3b4c3c49934365caa30608fd50b677dd883b7e3ed722271db4aceeaf73af'
    assert c['levels'] == ['legacy5', 'brain5'] and c['primary_level'] == 'legacy5' and c['circuit_seeds'] == [701]
    assert c['blocks'] == [dict(seed=1600001+i, parent_seed=1500001+i,
        input_seed=1510001+i, stream_seed=1610001+i) for i in range(6)]
    assert c['blocks_by_level'] == {'legacy5': list(range(1600001, 1600007)), 'brain5': list(range(1600001, 1600004))}
    assert (c['alphabet_size'], c['input_fraction'], c['input_amplitude'], c['gain'], c['leak'], c['alpha']) == (10, .1, .5, .9, .6, 1.)
    assert c['normalization'] == 'incoming_l1' and c['schedule'] == 'mbon_after_kc'
    assert c['carry_multiplier'] == 0. and c['synaptic_history'] is True
    assert (c['warmup'], c['test_samples'], c['probe_count'], c['primary_lag']) == (200, 2000, 64, 2)
    assert c['minimum_joint_accuracy'] == .9 and c['performance_eligibility_gate'] is False and c['new_trained_parameters'] == 0
    assert (c['bootstrap_seed'], c['bootstrap_draws'], c['score_atol'], c['score_rtol']) == (1620001, 10000, 1e-10, 1e-10)
    assert (c['neural_workers'], c['max_seconds'], c['max_rss_bytes']) == (4, 7200, 4294967296)
    assert c['graph_searches'] == 0 and c['biological_plasticity_performed'] is False
    frozen = subprocess.check_output(['git', 'show', PREREGISTRATION+':'+CONFIG.as_posix()])
    assert hashlib.sha256(frozen).hexdigest() == sha(CONFIG), 'Registered config bytes changed'
    if smoke:
        c['test_samples'] = c['smoke']['test_samples']
        c['blocks_by_level'] = {level: blocks[:1] for level, blocks in c['blocks_by_level'].items()}
    return c


def materialized_checks(inventory):
    names = sorted(n for n in inventory if Path(n).is_file())
    result = subprocess.run(['git', 'hash-object', '--stdin-paths'], input='\n'.join(names)+'\n',
        text=True, capture_output=True, check=True)
    actual = result.stdout.splitlines()
    assert len(actual) == len(names)
    assert all(inventory[n] == value for n, value in zip(names, actual)), 'Historical bytes changed'
    results = sum(n.startswith('results/') for n in names)
    return dict(materialized_protected_files_checked=len(names), materialized_prior_result_files_checked=results,
        sparse_excluded_prior_result_identities=sum(n.startswith('results/') for n in inventory)-results,
        all_materialized_blob_checks_pass=True, sparse_paths_freshly_replayed=False)


def history_preserved(c=None):
    old = read(NAMESPACE/'baseline_audit/baseline-git-blobs.json')
    assert old == tree(BASELINE)
    protected = {n: value for n, value in old.items() if n not in OVERVIEWS}
    current = tree('HEAD')
    assert all(current.get(n) == value for n, value in protected.items()), 'Historical Git identity changed'
    assert not set(git('diff', '--name-only', 'HEAD').splitlines()).intersection(protected)
    return dict(baseline_commit=BASELINE, prior_result_identities_preserved=sum(n.startswith('results/') for n in protected),
        protected_baseline_paths=len(protected), historical_scientific_source_data_protocol_config_tests_preserved=True,
        m1_outcome_unchanged='assay-invalid', m2_outcome_unchanged='PASS', m3_outcome_unchanged='PASS',
        m4_outcome_unchanged='INFEASIBLE', m5_outcome_unchanged='PASS', p2_material_gates_unchanged='FAIL',
        **materialized_checks(protected))


def parent_paths(c):
    root = Path(c['parent_namespace'])
    paths = [root/name for name in ['manifest.json', 'source.json', 'config.json']]
    for level in c['levels']:
        # Always authenticate the complete registered parent bank, even smoke.
        for block in c['blocks']:
            if level == 'brain5' and block['seed'] > 1600003:
                continue
            folder = root/f'{level}_s{block["parent_seed"]}'
            paths.extend(folder/name for name in ['manifest.json', 'graph.json', 'case.npz'])
    return paths


def authenticate_git_sources(tracked, hashes, revision):
    """Stream Git blobs: old full parent cases need not be copied into RAM."""
    inventory = tree(revision)
    process = subprocess.Popen(['git', 'cat-file', '--batch'], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        process.stdin.write(('\n'.join(inventory[n] for n in tracked)+'\n').encode())
        process.stdin.close()
        for name in tracked:
            header = process.stdout.readline().decode().split()
            assert len(header) == 3 and header[1] == 'blob', name
            remaining = int(header[2]); digest = hashlib.sha256()
            while remaining:
                data = process.stdout.read(min(1048576, remaining))
                assert data, name
                digest.update(data); remaining -= len(data)
            assert process.stdout.read(1) == b'\n'
            assert digest.hexdigest() == hashes[name], ('Uncommitted captured source', name)
        assert process.wait() == 0
    finally:
        if process.poll() is None:
            process.kill(); process.wait()
        process.stdout.close(); process.stderr.close()


def source_record(c):
    assert subprocess.run(['git', 'diff', '--quiet', 'HEAD']).returncode == 0
    revision = git('rev-parse', 'HEAD')
    paths = sorted(Path('src/flying').rglob('*.py'))
    paths += [CONFIG, Path(c['protocol']), Path('requirements-act1-lock.txt')]
    paths += [Path('scripts')/name for name in ['counterfactual_support.py', 'counterfactual_tracking.py',
        'verify_counterfactual_tracking.py', 'audit_counterfactual_tracking.py', 'run_counterfactual_guards.py',
        'temporal_mechanism.py', 'temporal_memory_curve.py', 'verify_temporal_memory_curve.py', 'tdc_support.py', 'tdc_pool.py']]
    paths += [Path('tests')/name for name in ['test_counterfactual_tracking.py', 'test_counterfactual_tracking_verifier.py']]
    paths += sorted(Path('data/flywire_783_mb_left_kc512_s701').glob('*'))
    cache = Path(c['cache'])
    paths += [cache/name for name in ['nodes.npz', 'brain5.npz', 'provenance.json']]
    paths += sorted((cache.parent/'raw').glob('*'))
    paths += parent_paths(c)
    assert all(p.is_file() for p in paths)
    names = [p.as_posix() for p in paths]
    assert len(names) == len(set(names))
    tracked = [n for n in names if not n.startswith('outputs/')]
    subprocess.run(['git', 'ls-files', '--error-unmatch', '--', *tracked], check=True, stdout=subprocess.DEVNULL)
    hashes = {n: sha(n) for n in names}
    authenticate_git_sources(tracked, hashes, revision)
    for name in [CONFIG.as_posix(), c['protocol']]:
        frozen = subprocess.check_output(['git', 'show', PREREGISTRATION+':'+name])
        assert hashlib.sha256(frozen).hexdigest() == hashes[name]
    assert sha(Path(c['parent_namespace'])/'manifest.json') == c['parent_manifest_sha256']
    subprocess.run(['git', 'merge-base', '--is-ancestor', PREREGISTRATION, revision], check=True)
    assert git('rev-parse', 'HEAD') == revision and subprocess.run(['git', 'diff', '--quiet', 'HEAD']).returncode == 0
    return dict(source_commit=revision, source_tree=git('rev-parse', revision+'^{tree}'), protocol_commit=PREREGISTRATION,
        tracked_changes=False, every_tracked_hash_authenticated_at_capture=True,
        parent_bank_frozen=True, new_trained_parameters=0, hashes=hashes, environment=environment())


def assert_source_unchanged(source):
    assert git('rev-parse', 'HEAD') == source['source_commit']
    assert subprocess.run(['git', 'diff', '--quiet', 'HEAD']).returncode == 0
    assert all(sha(n) == value for n, value in source['hashes'].items()), 'Source/data/parent bytes changed during stage'


def baseline_audit(out):
    c = config()
    with attempt(out, c, 'm6-baseline-audit'):
        out = Path(out)
        inventory = tree(BASELINE)
        write(out/'baseline-git-blobs.json', inventory)
        protected = {n: value for n, value in inventory.items() if n not in OVERVIEWS}
        old = Path('results/functional_sensitivity_v1')
        manifests = entries = 0
        for p in sorted(old.rglob('manifest.json')):
            m = check_manifest(p.parent); manifests += 1; entries += len(m['artifacts'])
        assert sha(Path(c['parent_namespace'])/'manifest.json') == c['parent_manifest_sha256']
        for stage in ['main', 'main_validation']:
            for name, value in read(old/stage/'source.json')['hashes'].items():
                assert sha(name) == value
        for name, value in read(old/'final_audit/document-sha256.json').items():
            assert sha(name) == value
        write(out/'audit.json', dict(all_checks_pass=True, baseline_commit=BASELINE,
            prior_result_identities=sum(n.startswith('results/') for n in inventory), protected_baseline_paths=len(protected),
            m5_manifests_checked=manifests, m5_checksum_entries=entries,
            parent_main_manifest_sha256=sha(Path(c['parent_namespace'])/'manifest.json'),
            no_historical_neural_replay_performed=True, **materialized_checks(protected)))
        write(out/'environment.json', environment())
    seal(out, dict(complete=True, kind='m6-baseline-audit', baseline_commit=BASELINE))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, required=True)
    baseline_audit(parser.parse_args().baseline)
