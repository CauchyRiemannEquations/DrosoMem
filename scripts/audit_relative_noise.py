"""Close the fixed relative-noise scope without changing prior outcomes."""
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
from alphabet_memory import check,read
from pathway_memory import seal
from flying.training import whole_brain_memory as core
from relative_noise import read_table


def audit(out):
    roots=['relative_noise_smoke','relative_noise_smoke_validation','relative_noise_tests',
        'relative_noise_tests_v2','relative_noise_smoke_v2','relative_noise_smoke_v2_validation',
        'relative_noise_smoke_equivalence','relative_noise_main','relative_noise_validation','relative_noise_analysis']
    evidence=[]
    for name in roots:
        path=Path('results')/name;m=check(path)
        evidence.append(dict(path=path.as_posix(),manifest_sha256=core.sha256(path/'manifest.json'),artifacts=len(m['artifacts'])))
    root=Path('results/relative_noise_main');m=read(root/'manifest.json');v=read('results/relative_noise_validation/checks.json')
    assert v['all_checks_pass'] and (v['cases'],v['metric_rows'],v['independent_trajectories'],v['exact_fresh_refits'])==(48,432,160,6)
    assert v['result_manifest_sha256']==core.sha256(root/'manifest.json')
    run=read(root/'verification.json');assert not run['smoke'] and run['matched_inputs_observations']
    assert (run['actual_trajectories'],run['zero_aliases'])==(384,48)
    pairs=0
    for path in root.glob('*/case.json'):
        meta=read(path);assert meta['readout_parameters']==482 and meta['graph']['observed_features']==48
        assert meta['horizon']==197 and meta['unchanged_weights_and_head'] and meta['training_scale']>0
        f=read_table(path.parent/'metrics.csv')
        for r in f[f['mode']=='teacher'].to_dict('records'):
            other=f[(f['mode']=='autonomous')&(f.strength==r['strength'])&~f.reused_clean]
            assert len(other)==1 and other.pi_memory_score.iloc[0]==r['pi_memory_score'];pairs+=1
    assert pairs==192
    e=read('results/relative_noise_smoke_equivalence/checks.json');assert e['all_arrays_exact'] and e['arrays']==231
    assert not read('results/relative_noise_smoke_validation/failure.json')['all_checks_pass']
    suites=list(ET.parse('results/relative_noise_tests_v2/tests.xml').getroot().iter('testsuite'))
    assert sum(int(s.attrib.get('failures',0))+int(s.attrib.get('errors',0)) for s in suites)==0
    assert sum(int(s.attrib['tests']) for s in suites)==20
    assert core.sha256('results/act5_main/manifest.json')=='31bf94bd27006a4efe99cbbe202204dfb5d762184d3b2e8229e95db0dad1878d'
    s=read(root/'summary.json');out.mkdir(parents=True,exist_ok=False)
    core.write_json(out/'audit.json',dict(fixed_scope_complete=True,physiological_calibration=False,evidence=evidence,
        primary_confirmed=s['primary_confirmed'],confirmed_comparisons=s['confirmed_comparisons'],
        paired_mode_first_error_checks=pairs,original_smoke_failure_preserved=True,round_trip_array_equivalence=e,
        previous_absolute_noise_result_unchanged=True,report_sha256=core.sha256('docs/relative-noise-results.md')))
    seal(out,m['config'],m['context'],purpose='Relative-noise scope completion',audit_script_sha256=core.sha256(__file__))
    print('Relative-noise fixed scope complete; prior absolute-noise study preserved')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();audit(a.out)
