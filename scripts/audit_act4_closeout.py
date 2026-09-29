"""Evidence integrity and bounded ACT IV completion coverage; no new experiments."""
import argparse
from pathlib import Path
from alphabet_memory import read,check
from flying.training import whole_brain_memory as core
from pathway_memory import seal


def run(out):
    roots=['phase5b','readout_dependency','local_reward','reward_direction','reward_noise','reward_trajectory','reward_margin'];records=[]
    for name in roots:
        root=Path('results')/name;m=read(root/'manifest.json');files=m.get('artifacts',m.get('file_sha256'))
        assert files is not None
        for p,h in files.items():assert core.sha256(root/p)==h,(name,p)
        validation=root/'verification.json' if name=='phase5b' else Path('results')/(name+'_validation')/'checks.json'
        v=read(validation)
        if name=='phase5b':
            assert v['all_saved_arrays_and_payloads_exact'] and v['manifest_sha256']==core.sha256(root/'manifest.json')
        else:assert v['all_checks_pass'] and v['result_manifest_sha256']==core.sha256(root/'manifest.json')
        records.append(dict(study=name,root_manifest_sha256=core.sha256(root/'manifest.json'),artifact_count=len(files),validation_path=validation.as_posix(),validation_sha256=core.sha256(validation),all_checks_pass=True))
    required=['docs/reward-margin-results.md','docs/act4-results.md','results/reward_margin_validation/tests.xml','results/reward_margin_analysis/margin-sensitivity.png']
    for p in required:assert Path(p).is_file(),p
    a=read('results/readout_dependency/summary.json');l=read('results/local_reward/summary.json');d=read('results/reward_direction/summary.json');n=read('results/reward_noise/summary.json');t=read('results/reward_trajectory/summary.json');g=read('results/reward_margin/summary.json')
    coverage=[
        dict(question='A/B frozen real versus matched/random + trained readout',execution='complete',finding='Lag2 access near ceiling; no original-graph advantage',source='readout_dependency'),
        dict(question='C plastic connectivity + simple decoder',execution='complete',finding='Artificial vector teacher supports internal recoding, not increased capacity',source='readout_dependency'),
        dict(question='D zero-fitted-parameter fixed-code proxy',execution='complete',finding='A fixed external interpretation remains; literal decoder-free biology untested',source='readout_dependency'),
        dict(question='One scalar-reward local eligibility rule',execution='complete',finding='Full coding and representation-improvement criteria fail',source='local_reward'),
        dict(question='Earlier dopamine/compartment rule',execution='complete_in_gamma1_pedc_scope',finding='Local causal controls pass; neural response and pi recall benefit not established',source='phase5b'),
        dict(question='Initial local direction and noise mismatch',execution='complete',finding='Full initial alignment and matched-noise rescue fail',source='reward_direction + reward_noise'),
        dict(question='Finite directional utility through learning',execution='complete_inconclusive',finding='Common finite radius below preregistered resolution',source='reward_trajectory'),
        dict(question='Analytic margin sensitivity with fresh seeds',execution='complete',finding=g['confirmed'],source='reward_margin')]
    assert a['confirmed']['internal_coding'] and not a['confirmed']['representation_improvement'] and not a['confirmed']['original_graph_advantage']
    assert not l['confirmed']['reward_coding'] and not l['confirmed']['representation_improvement'];assert not d['confirmed_noisy_alignment'];assert not t['primary_identifiable']
    out.mkdir(parents=True,exist_ok=False);core.write_json(out/'audit.json',dict(bounded_programme_complete=True,all_biological_questions_answered=False,shutdown_is_not_scientific_criterion=True,evidence=records,coverage=coverage,
        unresolved_extensions=['Physiological validation and literal decoder-free biology','Whole-brain local learning','Autonomous recall improvement under local-only learning','Other preregistered plasticity families, not automatically required sweeps'],
        documentation_discrepancies=[dict(path='docs/phase5b-results.md',issue='Historical next-step text says Phase 6 unfinished; later phase6 and ACT I artifacts establish completed scoped execution',resolution='Preserve dated historical text; current closeout/roadmap takes precedence')],
        new_margin_confirmed=g['confirmed'],deliverable_sha256={p:core.sha256(p) for p in required}))
    m=check(Path('results/reward_margin'));seal(out,m['config'],m['context'],audit_script_sha256=core.sha256(__file__),purpose='Bounded ACT IV evidence integrity and question coverage')
    print('Seven study roots checked; bounded ACT IV programme complete; biological questions remain open')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.out)
