"""Audit the two-study delivery and keep failed scientific/engineering results visible."""
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
from alphabet_memory import read,check
from pathway_memory import seal
from flying.training import whole_brain_memory as core


def audit(out):
    roots=['relative_noise_closeout','observation_noise_main','observation_noise_validation','observation_noise_analysis','observation_noise_tests']
    evidence=[]
    for name in roots:
        p=Path('results')/name;m=check(p);evidence.append(dict(study=name,manifest_sha256=core.sha256(p/'manifest.json'),artifacts=len(m['artifacts'])))
    prior=read('results/relative_noise_closeout/audit.json');assert prior['fixed_scope_complete']
    assert prior['report_sha256']==core.sha256('docs/relative-noise-results.md')
    m=read('results/observation_noise_main/manifest.json');v=read('results/observation_noise_validation/checks.json')
    assert v['all_checks_pass'] and (v['cases'],v['independent_instantaneous_evaluations'],v['first_step_matches'])==(48,144,144)
    assert v['new_full_neural_trajectory_replays']==0 and v['result_manifest_sha256']==core.sha256('results/observation_noise_main/manifest.json')
    run=read('results/observation_noise_main/verification.json')
    assert (run['new_instantaneous_evaluations'],run['archived_full_state_evaluations'],run['clean_replays'])==(144,144,48)
    cases=list(Path('results/observation_noise_main').glob('*/case.json'));assert len(cases)==48
    for p in cases:
        z=read(p);assert z['head_unchanged'] and not z['internal_dynamics_recomputed']
        assert (z['observed_neurons'],z['positions'])==(48,197)
        assert core.sha256(Path(z['source_case'])/'manifest.json')==z['source_manifest_sha256']
    suites=list(ET.parse('results/observation_noise_tests/tests.xml').getroot().iter('testsuite'))
    assert sum(int(s.attrib.get('failures',0))+int(s.attrib.get('errors',0)) for s in suites)==0
    assert sum(int(s.attrib['tests']) for s in suites)==24
    source=Path(m['config']['source']);assert core.sha256(source/'manifest.json')==m['config']['source_manifest_sha256']
    s=read('results/observation_noise_main/summary.json');out.mkdir(parents=True,exist_ok=False)
    core.write_json(out/'audit.json',dict(two_studies_complete=True,evidence=evidence,
        relative_noise_primary_confirmed=prior['primary_confirmed'],observation_history_primary_confirmed=s['primary_confirmed'],
        confirmed_history_cost=s['confirmed_history_cost'],within_registered_near_tolerance=s['within_registered_near_tolerance'],
        new_autonomous_runs_in_second_study=0,independent_fresh_cohort_in_second_study=False,formal_equivalence_claimed=False,
        report_sha256=core.sha256('docs/observation-noise-results.md'),relative_report_sha256=core.sha256('docs/relative-noise-results.md'),
        old_parser_failure_preserved=True,physiological_validation=False,
        next_proposal='Redistribute observation noise in proportion to each MBON training SD at matched expected total squared noise; fixed head; not executed'))
    seal(out,m['config'],m['context'],purpose='Two-study completion audit',audit_script_sha256=core.sha256(__file__))
    print('Both requested studies complete; diagnostic and memory conclusions kept separate')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();audit(a.out)
