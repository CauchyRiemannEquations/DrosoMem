"""Finite closeout of all three additional computational studies."""
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from flying.training import whole_brain_memory as core
from alphabet_memory import read,check
from pathway_memory import seal


def audit(out):
    expected={
        'sequence':dict(cases=96,certificates=1728,actual_paths=420,fresh_fits=36,independent_full_paths=350,exact_refits=38),
        'structure':dict(cases=96,certificates=1728,actual_paths=672,fresh_fits=64,independent_full_paths=672,exact_refits=64),
        'correlation':dict(cases=96,certificates=5184,actual_paths=52,fresh_fits=0,independent_full_paths=52,exact_refits=0)}
    smoke_expected={'sequence':(4,72,28),'structure':(3,54,21),'correlation':(2,108,26)}
    reports=['docs/sequence-noise-results.md','docs/structure-noise-results.md','docs/correlated-noise-results.md',
             'docs/additional-research-results.md','docs/additional-research-completion-plan.md']
    evidence=[];decisions={}
    for stage,target in expected.items():
        root=Path('results')/('research_suite_'+stage);m=check(root)
        v=read(str(root)+'_validation/checks.json');actual=read(root/'verification.json')
        for k in ['cases','certificates','actual_paths','fresh_fits']:assert actual[k]==target[k],(stage,k)
        assert not actual['smoke'] and actual['actual_paths']==actual['pre_error_checks']
        for k in ['independent_full_paths','exact_refits']:assert v[k]==target[k],(stage,k)
        assert v['all_checks_pass'] and v['actual_paths_checked']==target['actual_paths'] and v['certificates']==target['certificates']
        assert v['result_manifest_sha256']==core.sha256(root/'manifest.json')
        if stage=='structure':assert v['structural_audits']==64
        if stage=='correlation':assert v['reused_rho0_certificates']==1728
        smoke=read(str(root)+'_smoke_validation/checks.json')
        assert smoke['all_checks_pass'] and smoke['smoke']
        assert (smoke['cases'],smoke['certificates'],smoke['independent_full_paths'])==smoke_expected[stage]
        for suffix in ['', '_validation','_smoke','_smoke_validation','_analysis']:
            path=Path(str(root)+suffix);check(path)
            evidence.append(dict(path=path.as_posix(),manifest_sha256=core.sha256(path/'manifest.json')))
        decisions[stage]=read(root/'summary.json')['primary_confirmed']
    rho_root=Path('results/research_suite_correlation_rho_analysis');rho_manifest=check(rho_root)
    rho=read(rho_root/'checks.json')
    assert rho['all_checks_pass'] and rho['seed_block_rows']==192 and rho['groups']==48
    assert rho['result_manifest_sha256']==core.sha256('results/research_suite_correlation/manifest.json')
    assert rho_manifest['analysis_script_sha256']==core.sha256('scripts/analyze_research_suite_correlation.py')
    evidence.append(dict(path=rho_root.as_posix(),manifest_sha256=core.sha256(rho_root/'manifest.json')))
    # Intact structural cells deliberately repeat the exact sequence exposure.
    # The runner's reused_exposure flag marks rho0 only, not all parent repeats.
    repeated_structure=0;scale_rows=[]
    for p in Path('results/research_suite_structure').glob('*/case.json'):
        meta=read(p)
        if meta['case']['topology']!='intact':
            scale_rows.append(dict(**meta['case'],own_q=meta['own_q'],used_q=meta['used_q'],ratio=meta['own_q']/meta['used_q']))
            continue
        parent=Path(meta['parent'])
        with np.load(p.parent/'certificates.npz') as a,np.load(parent/'certificates.npz') as b:
            assert a.files==b.files
            for key in a.files:np.testing.assert_array_equal(a[key],b[key])
        repeated_structure+=meta['certificates']
    assert repeated_structure==576
    scale_summary={}
    for topology in ['degree','role']:
        values=np.array([x['ratio'] for x in scale_rows if x['topology']==topology])
        assert len(values)==32 and np.isfinite(values).all()
        scale_summary[topology]=dict(cells=len(values),minimum=float(values.min()),median=float(np.median(values)),maximum=float(values.max()))
    check(Path('results/research_suite_tests'))
    suites=list(ET.parse('results/research_suite_tests/tests.xml').getroot().iter('testsuite'))
    assert sum(int(x.attrib['tests']) for x in suites)==28
    assert sum(int(x.attrib.get('errors',0))+int(x.attrib.get('failures',0)) for x in suites)==0
    for p,digest in read('results/research_suite_tests/tested-source.json').items():assert core.sha256(p)==digest,p
    old=read('results/feedback_noise_closeout/audit.json')
    assert core.sha256('docs/feedback-noise-results.md')==old['report_sha256']
    assert core.sha256('docs/additional-research-brief.md')==old['research_brief_sha256']
    for p,digest in read('results/act5_final_closeout/audit.json')['report_sha256'].items():assert core.sha256(p)==digest,p
    out.mkdir(parents=True,exist_ok=False)
    core.write_json(out/'audit.json',dict(bounded_additional_programme_complete=True,stages=decisions,
        evidence=evidence,main_certificates=8640,distinct_certificate_exposure_settings=6336,
        correlated_rho0_parent_repeats=1728,structural_intact_parent_repeats=repeated_structure,
        total_parent_repeat_evaluations=2304,
        posthoc_structural_signal_scale=dict(definition='own training SD median / paired intact training SD median',
            rows=scale_rows,summary=scale_summary,changes_primary_endpoint=False,new_experiment=False),
        main_actual_evaluation_paths=1144,independent_main_full_replays=1074,
        new_head_fits=100,independent_head_refits=102,new_internal_learning=False,
        physiological_validation=False,whole_brain_structural_controls=False,old_sealed_reports_unchanged=True,
        report_sha256={p:core.sha256(p) for p in reports},next_experiment_executed=False))
    seal(out,m['config'],m['context'],audit_script_sha256=core.sha256(__file__),purpose='All three bounded extensions complete')
    print('Three-study closeout passed; physiological and whole-brain structural extensions remain unexecuted')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);audit(p.parse_args().out)
