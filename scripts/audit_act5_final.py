"""Question-based closeout of the bounded ACT V programme and its three diagnostics."""
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
from alphabet_memory import read, check
from pathway_memory import seal
from flying.training import whole_brain_memory as core


def audit(out):
    out.mkdir(parents=True, exist_ok=False)
    roots = ['act5_closeout', 'act5_main', 'act5_validation', 'act5_analysis',
             'relative_noise_closeout', 'relative_noise_main', 'relative_noise_validation',
             'observation_noise_closeout', 'observation_noise_main', 'observation_noise_validation',
             'allocation_noise_tests', 'allocation_noise_smoke', 'allocation_noise_smoke_validation',
             'allocation_noise_main', 'allocation_noise_validation', 'allocation_noise_analysis']
    evidence = []
    for name in roots:
        path = Path('results')/name; m = check(path)
        evidence.append(dict(root=path.as_posix(), manifest_sha256=core.sha256(path/'manifest.json'), artifacts=len(m['artifacts'])))
    first = read('results/act5_closeout/audit.json')
    assert first['bounded_rate_model_programme_complete']
    assert core.sha256('docs/act5-robustness-results.md') == first['report_sha256']
    assert core.sha256('docs/act5-model-assumptions.md') == first['model_assumptions_sha256']
    assert first['coverage'].keys() == {'pulse', 'ongoing', 'edge_dropout', 'neuron_dropout', 'weight_noise'}
    for item in first['evidence']:
        path = Path('results')/item['study']; check(path)
        assert core.sha256(path/'manifest.json') == item['manifest_sha256']
    relative = read('results/relative_noise_closeout/audit.json')
    assert relative['fixed_scope_complete'] and relative['original_smoke_failure_preserved']
    for item in relative['evidence']:
        path = Path(item['path']); check(path)
        assert core.sha256(path/'manifest.json') == item['manifest_sha256']
    assert core.sha256('docs/relative-noise-results.md') == relative['report_sha256']
    observation = read('results/observation_noise_closeout/audit.json')
    assert observation['two_studies_complete']
    assert core.sha256('docs/observation-noise-results.md') == observation['report_sha256']
    assert Path('results/relative_noise_smoke_validation/failure.json').is_file()
    main = read('results/allocation_noise_main/verification.json')
    validation = read('results/allocation_noise_validation/checks.json')
    smoke = read('results/allocation_noise_smoke_validation/checks.json')
    assert (main['cases'], main['evaluations'], main['zero_checks'], main['archived_flat_matches']) == (48, 1152, 96, 144)
    assert (validation['cases'], validation['independent_evaluations'], validation['zero_checks'], validation['archived_flat_matches']) == (48, 1152, 96, 144)
    assert validation['all_checks_pass'] and not validation['smoke']
    assert validation['result_manifest_sha256'] == core.sha256('results/allocation_noise_main/manifest.json')
    assert smoke['all_checks_pass'] and smoke['smoke'] and smoke['independent_evaluations'] == 72
    assert main['new_neural_trajectories'] == validation['new_neural_trajectories'] == 0
    suites = list(ET.parse('results/allocation_noise_tests/tests.xml').getroot().iter('testsuite'))
    assert sum(int(x.attrib.get('failures', 0))+int(x.attrib.get('errors', 0)) for x in suites) == 0
    assert sum(int(x.attrib['tests']) for x in suites) == 22
    for p, digest in read('results/allocation_noise_tests/tested-source.json').items():
        assert core.sha256(p) == digest
    manifest = read('results/allocation_noise_main/manifest.json'); c = manifest['config']
    assert core.sha256(c['completion_plan']) == manifest['context']['completion_plan_sha256']
    s = read('results/allocation_noise_main/summary.json')
    reports = ['docs/allocation-noise-results.md', 'docs/act5-final-results.md']
    core.write_json(out/'audit.json', dict(bounded_act5_complete=True, physiological_validation_complete=False,
        evidence=evidence, original_five_families=list(first['coverage']),
        original_primary_confirmed=first['primary_confirmed'], original_confirmed_comparisons=first['confirmed_comparisons'],
        original_confirmed_absolute_robustness=first['confirmed_absolute_robustness'],
        relative_noise_confirmed=relative['confirmed_comparisons'],
        observation_history_confirmed=observation['confirmed_history_cost'],
        allocation_confirmed=s['confirmed'], allocation_primary_confirmed=s['primary_confirmed'],
        new_allocation_evaluations=1152, new_neural_trajectories_in_allocation=0, targeted_tests=22,
        prior_artifacts_unchanged=main['prior_files_unchanged'], historical_failures_preserved=True,
        report_sha256={p:core.sha256(p) for p in reports},
        completion_plan_sha256=core.sha256(c['completion_plan']),
        biological_questions_remaining=['Physiological noise and parameter calibration', 'Other individuals and dynamics',
                                       'Correlated noise and adaptive learning'],
        no_further_experiments_in_this_scope=True))
    seal(out, c, manifest['context'], purpose='ACT V final bounded completion audit', audit_script_sha256=core.sha256(__file__))
    print('ACT V bounded programme complete; physiological validation remains unestablished')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, required=True)
    audit(p.parse_args().out)
