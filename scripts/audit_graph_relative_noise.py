"""Closeout integrity, coverage and historical-report preservation."""
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
from flying.training import whole_brain_memory as core
from alphabet_memory import read,check
from pathway_memory import seal
from verify_graph_relative_noise import exact_files


def audit(out):
    root=Path('results/graph_relative_noise');m=exact_files(root)
    v=read('results/graph_relative_noise_validation/checks.json')
    r=read(root/'verification.json');smoke=read('results/graph_relative_noise_smoke_validation/checks.json')
    assert v['all_checks_pass'] and not v['smoke'] and r['complete'] and not r['smoke']
    assert v['result_manifest_sha256']==core.sha256(root/'manifest.json')
    assert (r['cases'],r['actual_paths'],r['certificates'],r['new_fits'])==(96,1824,1728,0)
    assert (r['unique_fresh_noisy_settings'],r['repeated_noisy_settings'],r['clean_parent_repeats'])==(1440,288,96)
    assert (v['independent_full_paths'],v['independent_teacher_paths'],v['exact_clean_replays'],v['graph_audits'])==(1824,96,96,64)
    assert v['intact_null_path_pairs']==288
    assert smoke['all_checks_pass'] and smoke['smoke'] and smoke['independent_full_paths']==57
    evidence={}
    for suffix in ['', '_validation','_smoke','_smoke_validation','_analysis','_tests']:
        p=Path(str(root)+suffix);exact_files(p);evidence[p.as_posix()]=core.sha256(p/'manifest.json')
    analysis=read('results/graph_relative_noise_analysis/checks.json')
    assert analysis['all_checks_pass'] and analysis['result_manifest_sha256']==v['result_manifest_sha256']
    assert read('results/graph_relative_noise_analysis/manifest.json')['analysis_script_sha256']==core.sha256('scripts/analyze_graph_relative_noise.py')
    tests=list(ET.parse('results/graph_relative_noise_tests/pytest.xml').getroot().iter('testsuite'))
    assert sum(int(x.attrib['tests']) for x in tests)==26
    assert sum(int(x.attrib.get('failures',0))+int(x.attrib.get('errors',0))+int(x.attrib.get('skipped',0)) for x in tests)==0
    for p,digest in read('results/graph_relative_noise_tests/tested-source.json').items():assert core.sha256(p)==digest,p
    for old in ['results/research_suite_closeout/audit.json','results/act5_final_closeout/audit.json']:
        for p,digest in read(old)['report_sha256'].items():assert core.sha256(p)==digest,p
    old=read('results/feedback_noise_closeout/audit.json')
    assert core.sha256('docs/feedback-noise-results.md')==old['report_sha256']
    assert core.sha256('docs/additional-research-brief.md')==old['research_brief_sha256']
    out.mkdir(parents=True,exist_ok=False)
    core.write_json(out/'audit.json',dict(complete=True,evidence=evidence,
        primary_confirmed=read(root/'summary.json')['primary_confirmed'],new_models=0,new_fits=0,
        actual_main_paths=1824,independent_main_replays=1824,unique_fresh_noisy_settings=1440,
        repeated_intact_settings=288,clean_parent_repeats=96,all_sealed_historical_reports_unchanged=True,
        report_sha256={p:core.sha256(p) for p in ['docs/graph-relative-noise-protocol.md','docs/graph-relative-noise-results.md']},
        physiology=False,whole_brain_rewiring=False,next_experiment_executed=False))
    seal(out,m['config'],m['context'],audit_script_sha256=core.sha256(__file__),purpose='Calibration research closeout')
    print('Graph-relative noise closeout verified')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);audit(p.parse_args().out)
