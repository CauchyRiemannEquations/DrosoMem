"""M4 fixed inputs, authenticated source context and historical identity seal."""
import argparse
import hashlib
from pathlib import Path
import subprocess

from tdc_support import (read, write, sha, array_sha, git, environment, attempt,
                         seal, check_manifest)

BASELINE = '5afe8f28cc1d820973b9156c20501a0d31c88a26'
PREREGISTRATION = '2c7a5f593c22909652a1374c9338571656411855'
INITIAL_PREREGISTRATION = 'a1def337bbe3358bf278a4bb39c19ee1eb9c9a86'
CONFIG = Path('configs/cycle_attribution_feasibility.json')
NAMESPACE = Path('results/cycle_attribution_v1')
OVERVIEWS = {'README.md', 'docs/research-status.md',
             'docs/research-roadmap.md', 'docs/next-work.md'}


def tree(revision):
    return {line.split('\t', 1)[1]: line.split('\t', 1)[0].split()[2]
            for line in git('ls-tree', '-r', revision).splitlines()}


def config(smoke=False):
    c = read(CONFIG)
    assert c['baseline_commit'] == BASELINE and c['namespace'] == NAMESPACE.as_posix()
    assert c['protocol'] == 'docs/cycle-attribution-feasibility-protocol.md'
    assert c['levels'] == ['legacy5', 'brain5'] and c['circuit_seeds'] == [701]
    assert c['primary_level'] == 'legacy5'
    assert c['blocks'] == [{'seed': 1400001+i, 'input_seed': 1410001+i} for i in range(6)]
    assert c['blocks_by_level'] == {'legacy5': list(range(1400001, 1400007)),
                                    'brain5': list(range(1400001, 1400004))}
    assert (c['alphabet_size'], c['input_fraction'], c['input_amplitude'], c['gain']) == (10, .1, .5, .9)
    assert (c['leak'], c['carry_multiplier'], c['normalization'], c['schedule']) == (.6, 0., 'incoming_l1', 'mbon_after_kc')
    assert c['lags'] == list(range(21)) and c['smoke_blocks'] == 1
    assert c['degree_certificates'] == ['missing_source_or_sink', 'node_incident_capacity',
        'global_dag_density', 'within_role_density', 'opposite_role_pair_capacity']
    assert (c['neural_runs'], c['decoder_fits'], c['graph_searches']) == (0, 0, 0)
    assert (c['max_seconds'], c['max_rss_bytes']) == (7200, 4294967296)
    # Authenticate the complete config, including explanatory algorithm strings.
    frozen = subprocess.check_output(['git', 'show', f'{PREREGISTRATION}:{CONFIG.as_posix()}'])
    assert hashlib.sha256(frozen).hexdigest() == sha(CONFIG), 'Preregistered config bytes changed'
    if smoke:
        c['blocks_by_level'] = {level: blocks[:1] for level, blocks in c['blocks_by_level'].items()}
    return c


def materialized_blob_checks(inventory):
    names = sorted(name for name in inventory if Path(name).is_file())
    result = subprocess.run(['git', 'hash-object', '--stdin-paths'],
        input='\n'.join(names)+'\n', text=True, capture_output=True, check=True)
    actual = result.stdout.splitlines()
    assert len(names) == len(actual)
    assert all(inventory[name] == value for name, value in zip(names, actual)), 'Materialized baseline bytes changed'
    count = sum(name.startswith('results/') for name in names)
    return {'materialized_protected_files_checked': len(names),
            'materialized_prior_result_files_checked': count,
            'sparse_excluded_prior_result_identities': sum(n.startswith('results/') for n in inventory)-count,
            'all_materialized_blob_checks_pass': True,
            'sparse_paths_freshly_replayed': False}


def history_preserved(c=None):
    old = read(NAMESPACE/'baseline_audit/baseline-git-blobs.json')
    assert old == tree(BASELINE), 'Baseline identity seal changed'
    protected = {name: value for name, value in old.items() if name not in OVERVIEWS}
    current = tree('HEAD')
    assert all(current.get(name) == value for name, value in protected.items()), 'Historical tracked identity changed'
    changes = set(git('diff', '--name-only', 'HEAD').splitlines())
    assert not changes.intersection(protected), 'Historical working-tree edit'
    return {'baseline_commit': BASELINE, 'prior_result_identities_preserved': sum(n.startswith('results/') for n in protected),
            'protected_baseline_paths': len(protected),
            'all_old_scientific_source_data_config_protocol_tests_preserved': True,
            'm1_outcome_unchanged': 'assay-invalid', 'm2_outcome_unchanged': 'PASS',
            'm3_outcome_unchanged': 'PASS', 'p2_material_gates_unchanged': 'FAIL',
            **materialized_blob_checks(protected)}


def source_record(c):
    assert subprocess.run(['git', 'diff', '--quiet', 'HEAD']).returncode == 0, 'Tracked edits before execution'
    revision = git('rev-parse', 'HEAD')
    revision_tree = git('rev-parse', 'HEAD^{tree}')
    paths = sorted(Path('src/flying').rglob('*.py'))
    paths += [CONFIG, Path(c['protocol']), Path('requirements-act1-lock.txt')]
    paths += [Path('scripts')/name for name in ['m4_support.py', 'cycle_attribution_feasibility.py',
        'verify_cycle_attribution_feasibility.py', 'audit_cycle_attribution_feasibility.py',
        'run_cycle_attribution_guards.py',
        'cycle_structure.py', 'tdc_support.py', 'verify_temporal_memory_curve.py']]
    paths += [Path('tests')/name for name in ['test_cycle_attribution_feasibility.py', 'test_cycle_attribution_verifier.py']]
    paths += sorted(Path('data/flywire_783_mb_left_kc512_s701').glob('*'))
    cache = Path(c['cache'])
    paths += [cache/name for name in ['nodes.npz', 'brain5.npz', 'provenance.json']]
    paths += sorted((cache.parent/'raw').glob('*'))
    for p in paths:
        assert p.is_file(), p
    tracked_paths = [p.as_posix() for p in paths if not p.as_posix().startswith('outputs/')]
    subprocess.run(['git', 'ls-files', '--error-unmatch', '--', *tracked_paths],
                   check=True, stdout=subprocess.DEVNULL)
    protocol_bytes = subprocess.check_output(['git', 'show', f'{PREREGISTRATION}:{c["protocol"]}'])
    assert hashlib.sha256(protocol_bytes).hexdigest() == sha(c['protocol']), 'Prospective protocol changed'
    subprocess.run(['git', 'merge-base', '--is-ancestor', PREREGISTRATION, 'HEAD'], check=True)
    hashes = {p.as_posix(): sha(p) for p in paths}
    assert git('rev-parse', 'HEAD') == revision, 'HEAD changed during source capture'
    assert subprocess.run(['git', 'diff', '--quiet', 'HEAD']).returncode == 0, 'Tracked edits during source capture'
    return {'source_commit': revision, 'source_tree': revision_tree,
            'protocol_commit': PREREGISTRATION, 'initial_protocol_commit': INITIAL_PREREGISTRATION,
            'tracked_changes': False, 'hashes': hashes,
            'environment': environment(), 'neural_runs': 0, 'decoder_fits': 0, 'graph_searches': 0}


def baseline_audit(out):
    c = config()
    with attempt(out, c, 'm4-baseline-audit'):
        out = Path(out)
        inventory = tree(BASELINE)
        write(out/'baseline-git-blobs.json', inventory)
        protected = {name: value for name, value in inventory.items() if name not in OVERVIEWS}
        checks = materialized_blob_checks(protected)
        m3 = Path('results/tdc_cycles_v1')
        manifests = entries = source_entries = 0
        for p in sorted(m3.rglob('manifest.json')):
            m = check_manifest(p.parent)
            manifests += 1
            entries += len(m['artifacts'])
        for stage in ['smoke', 'main']:
            source = read(m3/stage/'source.json')
            for name, value in source['hashes'].items():
                assert sha(name) == value, name
                source_entries += 1
        cache = Path(c['cache'])
        provenance = read(cache/'provenance.json')
        for name in ['nodes.npz', 'brain5.npz']:
            assert sha(cache/name) == provenance['files'][name]
        for name, context in provenance['sources'].items():
            assert sha(cache.parent/'raw'/name) == context['sha256']
        write(out/'audit.json', {'all_checks_pass': True, 'baseline_commit': BASELINE,
            'prior_result_identities': sum(n.startswith('results/') for n in inventory),
            'protected_baseline_paths': len(protected), 'm3_manifests_checked': manifests,
            'm3_checksum_entries': entries, 'm3_recorded_source_entries': source_entries,
            'm3_main_manifest_sha256': sha(m3/'main/manifest.json'),
            'no_historical_neural_replay_performed': True,
            'neural_runs': 0, 'decoder_fits': 0, **checks})
        write(out/'environment.json', environment())
    seal(out, {'complete': True, 'kind': 'm4-baseline-audit', 'baseline_commit': BASELINE})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True)
    baseline_audit(parser.parse_args().baseline)
