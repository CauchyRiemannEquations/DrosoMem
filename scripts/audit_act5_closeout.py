"""Verify the declared ACT V completion boundary without promoting failed hypotheses."""
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from flying.training import whole_brain_memory as core
from alphabet_memory import read,check
from pathway_memory import seal


def run(out):
    roots=['act5_smoke','act5_smoke_validation','act5_tests','act5_resource_probe','act5_main','act5_validation','act5_analysis']
    evidence=[]
    for name in roots:
        p=Path('results')/name;m=check(p)
        evidence.append(dict(study=name,manifest_sha256=core.sha256(p/'manifest.json'),artifacts=len(m['artifacts'])))
    main=read('results/act5_main/manifest.json');c=main['config'];v=read('results/act5_validation/checks.json')
    cases=list(Path('results/act5_main').glob('*/case.json'));assert len(cases)==48;paired_prefix_checks=0
    for path in cases:
        case=read(path);child=check(path.parent)
        assert child['config']==c and child['context']==main['context']
        assert case['horizon']==197 and case['readout_parameters']==482 and case['graph']['observed_features']==48
        assert case['unchanged_weights_and_head'] and not case['smoke']
        frame=pd.read_csv(path.parent/'metrics.csv')
        for row in frame[frame['mode']=='teacher'].to_dict('records'):
            other=frame[(frame['mode']=='autonomous')&(frame.kind==row['kind'])&(frame.strength==row['strength'])]
            assert len(other)==1 and other.pi_memory_score.iloc[0]==row['pi_memory_score']
            paired_prefix_checks+=1
    assert paired_prefix_checks==288
    assert v['all_checks_pass'] and v['result_manifest_sha256']==core.sha256('results/act5_main/manifest.json')
    assert (v['cases'],v['metric_rows'],v['independent_trajectories'],v['exact_fresh_refits'])==(48,1296,440,6)
    r=read('results/act5_main/verification.json');assert r['actual_trajectories']==1056 and r['zero_aliases']==240
    assert r['matched_inputs_observations'] and not r['smoke']
    probe=read('results/act5_resource_probe/checks.json');assert probe['all_arrays_exact'] and probe['arrays_checked']==207
    assert probe['resource']['scope']=='worker_process_tree'
    for resource in read('results/act5_main/resources.json'):
        assert resource['scope']=='worker_process_tree' and resource['peak_sampled_rss_bytes']<c['worker_rss_bytes']
    cache=read(Path(c['cache'])/'provenance.json')
    for name in ['nodes.npz','brain1.npz','brain5.npz']:assert core.sha256(Path(c['cache'])/name)==cache['files'][name]
    tests=ET.parse('results/act5_tests/tests.xml').getroot();suites=list(tests.iter('testsuite'))
    assert sum(int(s.attrib.get('failures',0))+int(s.attrib.get('errors',0)) for s in suites)==0
    assert sum(int(s.attrib['tests'])-int(s.attrib.get('skipped',0)) for s in suites)==252
    assert core.sha256('results/phase2_robustness/manifest.json')=='f01a3ea5c32274cf5af1a4c8163c5beebee7797f5abad45eef49f0d8470d1b7d'
    old=read('results/phase2_robustness/verification.json');assert old['source_heads_reconstructed']==72 and old['full_recalls_replayed']==16128
    assert old['exact_predictions_and_perturbations'] and old['original_weights_and_readouts_unchanged']
    summary=read('results/act5_main/summary.json');report=Path('docs/act5-robustness-results.md');assert report.is_file()
    out.mkdir(parents=True,exist_ok=False)
    core.write_json(out/'audit.json',dict(bounded_rate_model_programme_complete=True,physiological_validation_complete=False,
        evidence=evidence,coverage={k:'executed, independently checked within declared replay scope' for k in c['strengths']},
        paired_teacher_autonomous_first_error_checks=paired_prefix_checks,
        primary_confirmed=summary['primary_confirmed'],confirmed_comparisons=summary['confirmed_comparisons'],
        confirmed_absolute_robustness=summary['confirmed_absolute_robustness'],report_sha256=core.sha256(report),
        historical_robustness=dict(models=72,full_replays=16128,manifest_sha256=core.sha256('results/phase2_robustness/manifest.json')),
        engineering_discrepancy='Initial smoke RSS measured launcher only; preserved and excluded. Corrected process-tree measurements validated before main.',
        unresolved_extensions=['Physiological parameter/noise calibration','Other dynamics and task distributions','Correlated/population-specific perturbations','Perturbation-trained readouts'],
        source_graph_cache=cache['files'],model_assumptions_sha256=core.sha256('docs/act5-model-assumptions.md')))
    seal(out,c,main['context'],audit_script_sha256=core.sha256(__file__),purpose='Five-family ACT V completion and limitations audit')
    print('ACT V bounded rate-model programme complete; physiological validation not claimed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.out)
