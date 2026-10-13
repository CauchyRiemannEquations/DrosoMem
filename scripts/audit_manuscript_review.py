"""Audit revised scientific prose/counts against sealed evidence; no model runs.

Manual scientific review is recorded separately. This script authenticates
inputs and checks selected quantitative/structural claims, not prose entailment
or external scientific peer review.
"""
import argparse
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
from pathlib import Path
import re

from audit_research_synthesis import GitBlobs, local_links
from audit_manuscript_draft import citations, numerical_table, evidence
from counterfactual_support import tree, materialized_checks
from tdc_support import read, write, sha, git, environment, attempt, seal

BASELINE = 'f916fe65a66263230d229a31e22d14fbc41bae6f'
CONFIG = 'configs/manuscript_scientific_review.json'
SCRIPT = 'scripts/audit_manuscript_review.py'


class EvidenceReader:
    def __init__(self, blobs):
        self.blobs, self.hashes = blobs, {}
    def __call__(self, name):
        name = Path(name).as_posix()
        digest = sha(name)
        assert digest == self.blobs.blob(BASELINE, name), ('baseline evidence bytes', name)
        self.hashes[name] = digest
        return read(name)


def protected_history(c):
    old, now = tree(BASELINE), tree('HEAD')
    protected = {n: oid for n, oid in old.items() if n not in c['mutable_baseline_paths']}
    assert all(now.get(n) == oid for n, oid in protected.items())
    assert not set(git('diff', '--name-only', 'HEAD').splitlines()).intersection(protected)
    assert all(n.startswith(c['namespace'] + '/') for n in now.keys() - old.keys() if n.startswith('results/'))
    return dict(baseline_commit=BASELINE, protected_paths=len(protected),
                preserved_prior_result_identities=sum(n.startswith('results/') for n in protected),
                previous_manuscript_audit_unchanged=True, scientific_source_and_outcomes_unchanged=True,
                bibliography_and_literature_ledger_unchanged=True, **materialized_checks(protected))


def expanded_numbers(facts, draft, reader):
    result = {}
    prose = ' '.join(draft.split())
    # P1: use per-block integer counts, not a rounded report or pooled lags.
    rows = [row for p in Path(facts['P1']['main']).glob('*_s*/metrics.json') for row in reader(p)]
    curve = []
    for lag in range(1, 6):
        cells = [r for r in rows if r['level'] == 'legacy5' and r['lag'] == lag]
        assert len(cells) == 6
        value = Fraction(sum(r['correct'] for r in cells), sum(r['samples'] for r in cells))
        percentage = (Decimal(value.numerator)*100/Decimal(value.denominator)).quantize(
            Decimal('0.01'), rounding=ROUND_HALF_UP)
        assert f'{percentage}%' in draft
        assert all(Fraction(r['correct'], r['samples']) - max(Fraction(1, 10),
                   Fraction(r['frequency_correct'], r['samples']),
                   Fraction(r['current_correct'], r['samples'])) >= Fraction(1, 10) for r in cells)
        curve.append(dict(lag=lag, mean_accuracy=str(value)))
    assert all(r['correct'] == r['current_correct'] == r['samples'] for r in rows if r['lag'] == 0)
    result['P1'] = curve
    # P2: adjusted-score contrasts are converted back to raw percentage points.
    p2 = reader(Path(facts['P2']['main']) / 'summary.json')['levels']
    for level in p2.values():
        raw_gap = 90 * Fraction(level['real_minus_null_mean_exact'])
        signed = ('−' if raw_gap < 0 else '+') + f'{abs(float(raw_gap)):.3f}pp'
        assert signed in draft and not level['registered_rule_pass']
    for key in ['real_score', 'null_mean']:
        assert f'{100*(.1+.9*p2["legacy5"][key]):.3f}%' in draft
    result['P2'] = {k: dict(raw_gap_pp=str(90*Fraction(v['real_minus_null_mean_exact']))) for k, v in p2.items()}
    # M3: all saved DAG lag2 cells plus graph-specific certificates.
    m3rows = [row for p in Path(facts['M3']['main']).glob('*_dag_synaptic_s*/metrics.json') for row in reader(p)]
    m3 = {}
    for level, blocks, expected in [('legacy5', 6, (11998, 12000)), ('brain5', 3, (6000, 6000))]:
        cells = [r for r in m3rows if r['level'] == level and r['lag'] == 2]
        counts = (sum(r['correct'] for r in cells), sum(r['samples'] for r in cells))
        assert len(cells) == blocks and counts == expected
        assert f'{counts[0]:,}/{counts[1]:,}' in draft
        root = Path(facts['M3']['main']) / (level + '_dag_synaptic_s1300001')
        top = reader(root / 'topology.json')
        cert = reader(root / 'certificate.json')
        assert f'{top["dag_edges"]:,}' in draft
        assert top['dag_cycle_edges'] == 0 and top['protected_kc_to_mbon_all_kept_exactly']
        assert top['global_dependency_depth'] == cert['global_dependency_depth']
        assert f'bound {top["global_dependency_depth"]}' in prose
        m3[level] = dict(correct=counts[0], samples=counts[1], retained_edges=top['dag_edges'],
                         global_dependency_depth=top['global_dependency_depth'])
    result['M3'] = m3
    # M4: incident obstruction and all registered witness cells.
    m4 = reader(Path(facts['M4']['main']) / 'legacy5/degree-certificates.json')
    offenders = m4['node_incident_offenders']
    assert len(offenders) == 1 and offenders[0]['incident'] == 1140 > m4['node_incident_capacity'] == 682
    assert m4['active_node_count'] == 683
    assert sum(v['witness_count'] for v in facts['M4']['archived_summary']['levels'].values()) == 90
    result['M4'] = dict(incident=1140, dag_incident_capacity=682, active_nodes=683, witness_cells=90)
    # M6: report pairs, each of which comprises two worlds.
    m6 = {}
    for level, expected in [('legacy5', 384), ('brain5', 192)]:
        cells = facts['M6']['archived_summary']['levels'][level]['cells']
        assert sum(x['samples'] for x in cells) == sum(x['joint_correct'] for x in cells) == expected
        m6[level] = expected
    assert '576 pairs, each with two worlds' in draft
    result['M6'] = dict(pairs=m6, worlds_per_pair=2)
    # M8: finite bank change ranges and the shared partial proposal restriction.
    m8 = facts['M8']['archived_summary']['levels']
    for level, row in m8.items():
        interval = f'{100*(1-row["maximum_delayed_overlap"]):.2f}–{100*(1-row["minimum_delayed_overlap"]):.2f}%'
        assert interval in draft
    for index in range(1, 11):
        graph = reader(Path(facts['M8']['main']) / f'graphs/legacy5_g184{index:04d}/audit.json')
        assert graph['proposal_pool_edges'] == 602 and graph['delayed_edges'] == 1835
    result['M8'] = dict(partial_proposal_edges=602, partial_delayed_edges=1835,
                        all_ten_partial_proposal_scopes_checked=True)
    # M9: all 87 cases, two readouts/case, 64 pairs/case.
    m9 = facts['M9']['archived_summary']
    assert m9['cases'] == 87 and m9['lag_heads'] == 174 and m9['probe_pairs'] == 5568
    assert m9['new_fitted_affine_parameters'] == 174*490
    result['M9'] = dict(cases=87, heads=174, joint_pairs=5568)
    # M10: rule identities are not episode counts. No task/model execution.
    metadata = reader(Path(facts['M10']['main']) / 'audit-metadata.json')
    rule_counts = {k: len({(r['a'], r['b']) for r in v}) for k, v in metadata.items()}
    assert rule_counts == dict(train=55, instance_test=55, parameter_test=44)
    primary = [r for r in metadata['parameter_test'] if r['a'] in [7, 8, 9]]
    assert len(primary) == 330 and len({(r['a'], r['b']) for r in primary}) == 33
    assert len(metadata['parameter_test']) == 440
    assert 'spanning 33 coefficient pairs' in draft and 'spanning 44 (a,b) pairs' in draft
    result['M10'] = dict(unique_rules=rule_counts, primary_episodes=330, primary_rules=33, all_heldout_episodes=440)
    # M11: same counts retained; per-rule and zero-fit references also checked.
    rows = reader(Path(facts['M11']['main']) / 'metrics.json')
    completions = {}
    for cohort, seeds, a9_count in [('discovery', range(2710001, 2710004), 5),
                                  ('confirmation', range(2710004, 2710007), 10)]:
        for a in [7, 8, 9, 10]:
            cells = [r for r in rows if r['family'] == 'affine' and r['model'] == 'real'
                     and r['split'] == 'parameter_test' and r['group'] == f'a{a}' and r['input_seed'] in seeds]
            assert len(cells) == 3 and sum(r['episodes'] for r in cells) == 330
            assert sum(r['completed'] for r in cells) == (a9_count if a == 9 else 0)
        completions[cohort] = dict(a7=0, a8=0, a9=a9_count, a10=0, episodes_per_multiplier=330)
    shift = [r for r in rows if r['family'] == 'affine' and r['model'] == 'shift'
             and r['split'] == 'parameter_test' and r['group'] == 'nonperiod2']
    assert len(shift) == 6 and all(r['completed'] == 0 for r in shift)
    controls = reader(Path(facts['M11']['main']) / 'fixed-lag-reference-metrics.json')
    for group, episodes, completed in [('a10', 110, 110), ('nonperiod2', 330, 0)]:
        cells = [r for r in controls if r['family'] == 'affine' and r['model'] == 'fixed_lag2'
                 and r['split'] == 'parameter_test' and r['group'] == group]
        assert len(cells) == 1 and (cells[0]['episodes'], cells[0]['completed']) == (episodes, completed)
    result['M11'] = dict(source_per_multiplier=completions, shift_primary_completion=0,
                         lag2_period2_completed=110, lag2_primary_completed=0)
    return result


def scope_table(facts, draft):
    rows = re.findall(r'^\| (P[12]|M\d+) \| (.*?) \| (.*?) \| (.*?) \|$', draft, re.M)
    assert len(rows) == 13 and {r[0] for r in rows} == set(facts)
    output = {}
    for identity, panel, heads, status in rows:
        s = facts[identity]['archived_summary']
        number = int(heads.split()[0])
        expected = s.get('lag_heads', s.get('fits', 0))
        assert number == expected and status == facts[identity]['resolved_status']
        output[identity] = dict(panel=panel, new_neural_heads=number, outcome=status)
    assert facts['P1']['archived_summary']['cases'] == 12
    assert facts['P2']['archived_summary']['cases'] == 21*6 + 11*3
    for k, arms in [('M1',4),('M2',4),('M3',3)]:
        assert facts[k]['archived_summary']['cases'] == arms*9
    assert facts['M9']['archived_summary']['cases'] == 72+15
    return output


def audit(out):
    c = read(CONFIG)
    assert c['baseline_commit'] == BASELINE and c['namespace'] == 'results/manuscript_review_v1'
    assert c['new_model_runs'] == c['new_model_fits'] == c['new_historical_neural_replays'] == 0
    assert c['own_paper_author_information'] == c['external_peer_review'] == c['external_submission'] == False
    assert out.resolve().is_relative_to(Path(c['namespace']).resolve())
    assert not git('diff', '--name-only', 'HEAD'), 'Commit review/source before audit'
    revision, blobs = git('rev-parse', 'HEAD'), GitBlobs()
    reader = EvidenceReader(blobs)
    names = c['documents'] + [CONFIG,SCRIPT,c['literature'],'docs/manuscript-references.bib',
        'scripts/audit_manuscript_draft.py','scripts/audit_research_synthesis.py',
        'scripts/counterfactual_support.py','scripts/tdc_support.py']
    hashes = {n: sha(n) for n in names}
    try:
        assert all(h == blobs.blob(revision, n) for n,h in hashes.items())
        draft = Path('docs/manuscript-draft.md').read_text(encoding='utf-8')
        note = Path('docs/manuscript-scientific-review.md').read_text(encoding='utf-8')
        assert all(f'| {finding} |' in note for finding in c['review_findings'])
        assert 'All 440 coefficients' not in draft and 'report the null advantage' not in draft
        assert '## 6. Data, code and provenance' in draft
        for placeholder in ['Author draft','Authors, affiliations','Paper authorship/',
                            'correspondence and declarations','Human authorship']:
            assert placeholder not in draft
        assert 'G_0<10⁻⁸' in draft and 'ξ∼N(0,I₄₈)' in draft and 'current KC values' in draft
        assert 'is not plotted here' in draft and 'empirical-over-null advantage' in draft
        with attempt(out, c, 'internal-scientific-manuscript-review-audit') as resources:
            write(out/'source.json', dict(source_commit=revision,baseline_commit=BASELINE,hashes=hashes,environment=environment()))
            write(out/'history.json', protected_history(c))
            facts, parent = evidence(c,draft)
            write(out/'parent-checks.json',parent)
            write(out/'existing-numeric-checks.json',numerical_table(facts,draft))
            write(out/'expanded-numeric-checks.json',expanded_numbers(facts,draft,reader))
            write(out/'study-scope-checks.json',scope_table(facts,draft))
            write(out/'citation-checks.json',citations(c,draft))
            for n in c['figure_paths']:
                assert sha(n) == blobs.blob(BASELINE,n)
            write(out/'figure-checks.json',{n:sha(n) for n in c['figure_paths']})
            write(out/'authenticated-selected-inputs.json',reader.hashes)
            write(out/'document-sha256.json',{n:hashes[n] for n in c['documents']})
            pending=(out/'checks.json').as_posix()
            links=local_links(c['documents'],revision,pending)
            write(out/'checks.json',dict(all_checks_pass=True,canonical_studies=13,
                resolved_document_findings=10,selected_quantitative_claims_match=True,
                panel_head_status_table_matches=True,citation_identities_match=True,
                prior_scientific_and_draft_audit_records_preserved=True,
                original_figures_unchanged=True,document_bytes_match_git=True,
                all_local_links_resolve=True,own_paper_author_information_added=False,
                new_model_runs=0,new_model_fits=0,new_historical_neural_replays=0,
                external_peer_review=False,external_submission=False,
                automated_prose_entailment=False))
            for entry in links:
                if entry['current_audit_artifact']:
                    assert Path(entry['path']).is_file()
                    entry['materialized']=True
            write(out/'links.json',links)
            assert git('rev-parse','HEAD') == revision and not git('diff','--name-only','HEAD')
            assert all(sha(n) == h for n,h in hashes.items())
            resources.check()
        seal(out,dict(complete=True,kind='internal manuscript scientific review/document audit'))
    finally:
        blobs.close()
    print(read(out/'checks.json'))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    audit(parser.parse_args().out)
