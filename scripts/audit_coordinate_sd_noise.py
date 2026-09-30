"""Closeout integrity, coverage and historical-report preservation."""
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
from flying.training import whole_brain_memory as core
from alphabet_memory import read,check
from pathway_memory import seal
from verify_coordinate_sd_noise import exact_files


def audit(out):
    root=Path('results/coordinate_sd_noise');m=exact_files(root)
    v=read('results/coordinate_sd_noise_validation/checks.json')
    r=read(root/'verification.json');smoke=read('results/coordinate_sd_noise_smoke_validation/checks.json')
    assert v['all_checks_pass'] and not v['smoke'] and r['complete'] and not r['smoke']
    assert v['result_manifest_sha256']==core.sha256(root/'manifest.json')
    assert (r['cases'],r['actual_paths'],r['certificates'],r['new_fits'])==(96,2688,2592,0)
    assert (r['unique_fresh_noisy_settings'],r['repeated_noisy_settings'],r['clean_parent_repeats'])==(2304,288,96)
    assert (v['independent_full_paths'],v['independent_teacher_paths'],v['exact_clean_replays'],v['graph_audits'])==(2688,96,96,64)
    assert v['intact_null_path_pairs']==288
    assert smoke['all_checks_pass'] and smoke['smoke'] and smoke['independent_full_paths']==84
    evidence={}
    for suffix in ['', '_validation','_smoke','_smoke_validation','_analysis','_tests']:
        p=Path(str(root)+suffix);exact_files(p);evidence[p.as_posix()]=core.sha256(p/'manifest.json')
    analysis=read('results/coordinate_sd_noise_analysis/checks.json')
    assert analysis['all_checks_pass'] and analysis['result_manifest_sha256']==v['result_manifest_sha256']
    assert read('results/coordinate_sd_noise_analysis/manifest.json')['analysis_script_sha256']==core.sha256('scripts/analyze_coordinate_sd_noise.py')
    tests=list(ET.parse('results/coordinate_sd_noise_tests/pytest.xml').getroot().iter('testsuite'))
    assert sum(int(x.attrib['tests']) for x in tests)==32
    assert sum(int(x.attrib.get('failures',0))+int(x.attrib.get('errors',0))+int(x.attrib.get('skipped',0)) for x in tests)==0
    for p,digest in read('results/coordinate_sd_noise_tests/tested-source.json').items():assert core.sha256(p)==digest,p
    for old in ['results/research_suite_closeout/audit.json','results/act5_final_closeout/audit.json','results/graph_relative_noise_closeout/audit.json']:
        for p,digest in read(old)['report_sha256'].items():assert core.sha256(p)==digest,p
    old=read('results/feedback_noise_closeout/audit.json')
    assert core.sha256('docs/feedback-noise-results.md')==old['report_sha256']
    assert core.sha256('docs/additional-research-brief.md')==old['research_brief_sha256']
    programme=[]
    stages=[('feedback_noise',336,140,'actual_rollouts_checked','independent_full_rollouts'),
            ('research_suite_sequence',420,350,'actual_paths_checked','independent_full_paths'),
            ('research_suite_structure',672,672,'actual_paths_checked','independent_full_paths'),
            ('research_suite_correlation',52,52,'actual_paths_checked','independent_full_paths'),
            ('graph_relative_noise',1824,1824,'independent_full_paths','independent_full_paths'),
            ('coordinate_sd_noise',2688,2688,'independent_full_paths','independent_full_paths')]
    for stage,paths,replays,pathkey,replaykey in stages:
        source=Path('results')/('feedback_noise_main' if stage=='feedback_noise' else stage)
        validation=Path('results')/(stage+'_validation')
        check(source);check(validation);vv=read(validation/'checks.json')
        assert vv['all_checks_pass'] and vv['result_manifest_sha256']==core.sha256(source/'manifest.json')
        assert vv[pathkey]==paths and vv[replaykey]==replays
        programme.append(dict(stage=stage,result_path=source.as_posix(),validation_path=validation.as_posix(),
            actual_paths=paths,independent_full_replays=replays,
            manifest_sha256=core.sha256(source/'manifest.json'),validation_manifest_sha256=core.sha256(validation/'manifest.json')))
    assert sum(x['actual_paths'] for x in programme)==5992
    assert sum(x['independent_full_replays'] for x in programme)==5726
    out.mkdir(parents=True,exist_ok=False)
    core.write_json(out/'audit.json',dict(complete=True,evidence=evidence,
        primary_confirmed=read(root/'summary.json')['primary_confirmed'],new_models=0,new_fits=0,
        actual_main_paths=2688,independent_main_replays=2688,unique_fresh_noisy_settings=2304,
        repeated_intact_settings=288,clean_parent_repeats=96,all_sealed_historical_reports_unchanged=True,
        report_sha256={p:core.sha256(p) for p in ['docs/coordinate-sd-noise-protocol.md','docs/coordinate-sd-noise-results.md','docs/additional-research-closeout.md']},
        bounded_observation_extension_closed=True,programme_evidence=programme,
        programme_actual_paths=5992,programme_independent_full_replays=5726,
        counts_are_executions_not_distinct_models=True,
        physiology=False,whole_brain_rewiring=False,next_experiment_executed=False))
    seal(out,m['config'],m['context'],audit_script_sha256=core.sha256(__file__),purpose='Calibration research closeout')
    print('Coordinate-SD noise and bounded extension closeout verified')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);audit(p.parse_args().out)
