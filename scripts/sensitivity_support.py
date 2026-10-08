"""M5 registered inputs, byte-exact provenance and immutable history guards."""
import argparse
import hashlib
from pathlib import Path
import subprocess

from tdc_support import (read, write, sha, array_sha, git, environment, attempt,
                         seal, check_manifest)

BASELINE = '4a72db553f80317c282f48561e33b5edbfc3a765'
PREREGISTRATION = '37212abeb6cfc1a4907e08348ff41030f3edfb8b'
CONFIG = Path('configs/temporal_functional_sensitivity.json')
NAMESPACE = Path('results/functional_sensitivity_v1')
OVERVIEWS = {'README.md', 'docs/research-status.md', 'docs/research-roadmap.md', 'docs/next-work.md'}


def tree(revision):
    return {line.split('\t', 1)[1]: line.split('\t', 1)[0].split()[2]
            for line in git('ls-tree', '-r', revision).splitlines()}


def config(smoke=False):
    c = read(CONFIG)
    assert (c['baseline_commit'], c['namespace'], c['protocol']) == (
        BASELINE, NAMESPACE.as_posix(), 'docs/temporal-functional-sensitivity-protocol.md')
    assert c['levels'] == ['legacy5', 'brain5'] and c['primary_level'] == 'legacy5'
    assert c['circuit_seeds'] == [701]
    assert c['blocks'] == [dict(seed=1500001+i, input_seed=1510001+i,
                              train_seed=1520001+i, test_seed=1530001+i) for i in range(6)]
    assert c['blocks_by_level'] == {'legacy5': list(range(1500001, 1500007)),
                                  'brain5': list(range(1500001, 1500004))}
    assert (c['alphabet_size'], c['gain'], c['leak'], c['input_fraction'], c['input_amplitude'], c['alpha']) == (10, .9, .6, .1, .5, 1.)
    assert c['normalization'] == 'incoming_l1' and c['schedule'] == 'mbon_after_kc'
    assert c['carry_multiplier'] == 0. and c['synaptic_history'] is True
    assert (c['warmup'], c['train_samples'], c['test_samples'], c['probe_count']) == (200, 4000, 2000, 32)
    assert c['lags'] == list(range(21))
    assert c['finite_difference_probe_indices'] == [0, 10, 21, 31]
    assert c['finite_difference_epsilons'] == [1e-4, 1e-5]
    assert (c['finite_difference_atol'], c['finite_difference_rtol']) == (2e-9, 2e-5)
    assert (c['complex_step_size'], c['complex_step_atol'], c['complex_step_rtol']) == (1e-20, 1e-12, 1e-9)
    assert (c['reduction_atol'], c['reduction_rtol']) == (1e-12, 1e-12)
    assert (c['minimum_current_gain'], c['primary_lag'], c['maximum_primary_relative_gain']) == (1e-8, 5, .1)
    assert (c['bootstrap_seed'], c['bootstrap_draws'], c['replacement_eta']) == (1540001, 10000, 1.)
    assert (c['neural_workers'], c['max_seconds'], c['max_rss_bytes']) == (4, 7200, 4294967296)
    assert c['graph_searches'] == 0 and c['biological_plasticity_performed'] is False
    frozen = subprocess.check_output(['git', 'show', f'{PREREGISTRATION}:{CONFIG.as_posix()}'])
    assert hashlib.sha256(frozen).hexdigest() == sha(CONFIG), 'Registered config changed'
    if smoke:
        c.update({name: c['smoke'][name] for name in ['train_samples', 'test_samples']})
        c['blocks_by_level'] = {level: blocks[:1] for level, blocks in c['blocks_by_level'].items()}
    return c


def materialized_checks(inventory):
    names = sorted(name for name in inventory if Path(name).is_file())
    p = subprocess.run(['git', 'hash-object', '--stdin-paths'], input='\n'.join(names)+'\n',
                       text=True, capture_output=True, check=True)
    actual = p.stdout.splitlines()
    assert len(actual) == len(names)
    assert all(inventory[name] == value for name, value in zip(names, actual)), 'Historical materialized bytes changed'
    result_count = sum(name.startswith('results/') for name in names)
    return dict(materialized_protected_files_checked=len(names),
                materialized_prior_result_files_checked=result_count,
                sparse_excluded_prior_result_identities=sum(n.startswith('results/') for n in inventory)-result_count,
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
                m4_outcome_unchanged='INFEASIBLE', p2_material_gates_unchanged='FAIL',
                **materialized_checks(protected))


def source_record(c):
    assert subprocess.run(['git', 'diff', '--quiet', 'HEAD']).returncode == 0, 'Tracked edits before source capture'
    revision = git('rev-parse', 'HEAD')
    paths = sorted(Path('src/flying').rglob('*.py'))
    paths += [CONFIG, Path(c['protocol']), Path('requirements-act1-lock.txt')]
    paths += [Path('scripts')/name for name in ['sensitivity_support.py', 'sensitivity_dynamics.py',
        'temporal_functional_sensitivity.py', 'verify_temporal_functional_sensitivity.py',
        'audit_temporal_functional_sensitivity.py', 'run_sensitivity_guards.py',
        'temporal_mechanism.py', 'temporal_memory_curve.py', 'verify_temporal_memory_curve.py',
        'cycle_attribution_feasibility.py', 'tdc_support.py', 'tdc_pool.py']]
    paths += [Path('tests')/name for name in ['test_temporal_functional_sensitivity.py', 'test_functional_sensitivity_verifier.py']]
    paths += sorted(Path('data/flywire_783_mb_left_kc512_s701').glob('*'))
    cache = Path(c['cache'])
    paths += [cache/name for name in ['nodes.npz', 'brain5.npz', 'provenance.json']]
    paths += sorted((cache.parent/'raw').glob('*'))
    assert all(p.is_file() for p in paths)
    tracked = [p.as_posix() for p in paths if not p.as_posix().startswith('outputs/')]
    subprocess.run(['git', 'ls-files', '--error-unmatch', '--', *tracked], check=True, stdout=subprocess.DEVNULL)
    hashes = {p.as_posix(): sha(p) for p in paths}
    inventory = tree(revision)
    # Batched cat-file preserves exact recorded Git bytes without one process
    # per source file, and detects edits during capture rather than trusting HEAD.
    blob_ids = [inventory[name] for name in tracked]
    cat = subprocess.run(['git', 'cat-file', '--batch'], input=('\n'.join(blob_ids)+'\n').encode(),
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    data, cursor = cat.stdout, 0
    for name in tracked:
        end = data.index(b'\n', cursor)
        fields = data[cursor:end].decode().split()
        assert fields[1] == 'blob'
        count = int(fields[2]); start = end+1
        assert hashlib.sha256(data[start:start+count]).hexdigest() == hashes[name], ('Uncommitted captured source', name)
        cursor = start+count+1
    assert cursor == len(data)
    for name in [CONFIG.as_posix(), c['protocol']]:
        frozen = subprocess.check_output(['git', 'show', PREREGISTRATION+':'+name])
        assert hashlib.sha256(frozen).hexdigest() == hashes[name], 'Prospective protocol/config changed'
    subprocess.run(['git', 'merge-base', '--is-ancestor', PREREGISTRATION, revision], check=True)
    assert git('rev-parse', 'HEAD') == revision and subprocess.run(['git', 'diff', '--quiet', 'HEAD']).returncode == 0
    return dict(source_commit=revision, source_tree=git('rev-parse', revision+'^{tree}'),
                protocol_commit=PREREGISTRATION, tracked_changes=False,
                every_tracked_hash_authenticated_at_capture=True, hashes=hashes, environment=environment())


def assert_source_unchanged(source):
    assert git('rev-parse', 'HEAD') == source['source_commit'], 'HEAD changed during stage'
    assert subprocess.run(['git', 'diff', '--quiet', 'HEAD']).returncode == 0, 'Tracked edits during stage'
    assert all(sha(name) == value for name, value in source['hashes'].items()), 'Captured source/data changed during stage'


def baseline_audit(out):
    c = config()
    with attempt(out, c, 'm5-baseline-audit'):
        out = Path(out)
        inventory = tree(BASELINE)
        write(out/'baseline-git-blobs.json', inventory)
        protected = {n: value for n, value in inventory.items() if n not in OVERVIEWS}
        prior = Path('results/cycle_attribution_v1')
        manifest_count = checksum_count = 0
        for p in sorted(prior.rglob('manifest.json')):
            m = check_manifest(p.parent)
            manifest_count += 1; checksum_count += len(m['artifacts'])
        failure = read(prior/'validation-failures.json')
        assert len(failure) == 1 and failure[0]['source_validation_failed'] is True
        assert sha(prior/failure[0]['snapshot_path']) == failure[0]['recorded_sha256']
        for name, value in read(prior/'final_audit/document-sha256.json').items():
            assert sha(name) == value
        for name, value in read(prior/'main/source.json')['hashes'].items():
            assert sha(name) == value
        write(out/'audit.json', dict(all_checks_pass=True, baseline_commit=BASELINE,
            prior_result_identities=sum(n.startswith('results/') for n in inventory),
            protected_baseline_paths=len(protected), m4_manifests_checked=manifest_count,
            m4_checksum_entries=checksum_count, retained_m4_guard_validation_failure=True,
            m4_main_manifest_sha256=sha(prior/'main/manifest.json'),
            no_historical_neural_replay_performed=True, **materialized_checks(protected)))
        write(out/'environment.json', environment())
    seal(out, dict(complete=True, kind='m5-baseline-audit', baseline_commit=BASELINE))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, required=True)
    baseline_audit(parser.parse_args().baseline)
