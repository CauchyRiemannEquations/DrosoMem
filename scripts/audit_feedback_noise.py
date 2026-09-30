"""Close the single autonomous-feedback extension without reopening ACT V."""
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
from alphabet_memory import read,check
from pathway_memory import seal
from flying.training import whole_brain_memory as core


def audit(out):
    roots=['feedback_noise_tests','feedback_noise_smoke','feedback_noise_smoke_validation',
           'feedback_noise_main','feedback_noise_validation','feedback_noise_analysis','act5_final_closeout']
    evidence=[]
    for name in roots:
        root=Path('results')/name;m=check(root)
        evidence.append(dict(root=root.as_posix(),manifest_sha256=core.sha256(root/'manifest.json'),artifacts=len(m['artifacts'])))
    m=read('results/feedback_noise_main/manifest.json');c=m['config']
    actual=read('results/feedback_noise_main/verification.json');v=read('results/feedback_noise_validation/checks.json')
    assert (actual['cases'],actual['certificates'],actual['actual_neural_trajectories'],actual['pre_error_checks'])==(48,2016,336,336)
    assert not actual['smoke'] and actual['matched_inputs_observations']
    assert v['all_checks_pass'] and (v['certificates'],v['actual_rollouts_checked'],v['independent_full_rollouts'])==(2016,336,140)
    assert not v['all_whole_rollouts_independently_replayed']
    assert v['result_manifest_sha256']==core.sha256('results/feedback_noise_main/manifest.json')
    smoke=read('results/feedback_noise_smoke_validation/checks.json')
    assert smoke['all_checks_pass'] and smoke['smoke'] and (smoke['certificates'],smoke['independent_full_rollouts'])==(126,21)
    tests=list(ET.parse('results/feedback_noise_tests/tests.xml').getroot().iter('testsuite'))
    assert sum(int(t.attrib.get('failures',0))+int(t.attrib.get('errors',0)) for t in tests)==0
    assert sum(int(t.attrib['tests']) for t in tests)==16
    for p,digest in read('results/feedback_noise_tests/tested-source.json').items():assert core.sha256(p)==digest
    old=read('results/act5_final_closeout/audit.json');assert old['bounded_act5_complete']
    for p,digest in old['report_sha256'].items():assert core.sha256(p)==digest
    summary=read('results/feedback_noise_main/summary.json')
    assert summary['decision_stage']=='fresh' and not summary['fresh_model_cohort']
    out.mkdir(parents=True,exist_ok=False)
    core.write_json(out/'audit.json',dict(fixed_scope_complete=True,act5_closeout_unchanged=True,
        evidence=evidence,primary_confirmed=summary['primary_confirmed'],confirmed=summary['confirmed'],
        archived_certificates=1152,prospective_fresh_certificates=864,actual_neural_trajectories=336,
        independently_replayed_neural_trajectories=140,all_whole_rollouts_independently_replayed=False,
        new_model_holdout=False,new_internal_learning=False,physiological_validation=False,
        report_sha256=core.sha256('docs/feedback-noise-results.md'),
        research_brief_sha256=core.sha256('docs/additional-research-brief.md'),
        prior_files_unchanged=actual['prior_files_unchanged'],targeted_tests=16,
        next_experiment_executed=False))
    seal(out,c,m['context'],audit_script_sha256=core.sha256(__file__),purpose='Single autonomous-feedback extension completion')
    print('Feedback extension complete; ACT V closeout and prior claims preserved')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);audit(p.parse_args().out)
