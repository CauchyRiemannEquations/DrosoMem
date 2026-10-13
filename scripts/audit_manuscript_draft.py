"""Document/provenance audit; no model execution, fitting or historical replay.

Numerical checks concern selected reported values, not automatic proof of every
prose claim. Literature entailment was reviewed manually at the recorded access
scope; this audit checks citation identity, classification and connectivity.
"""
import argparse
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
import hashlib
from pathlib import Path
import re

from audit_research_synthesis import GitBlobs, OVERVIEWS, local_links
from counterfactual_support import tree, materialized_checks
from tdc_support import read, write, sha, git, environment, attempt, seal

BASELINE = '5e78dc5fec885234ae591ba2a56c45c870210ab9'
SCOPE_COMMIT = '087c72975ee62ffa6c4f9ac06c080a51200b560f'
SCOPE = 'configs/manuscript_scope.json'
DRAFT = 'docs/manuscript-draft.md'
BIB = 'docs/manuscript-references.bib'
SCRIPT = 'scripts/audit_manuscript_draft.py'


def authenticate_manifest(root):
    root = Path(root)
    m = read(root / 'manifest.json')
    assert m['complete'], root
    for name, digest in m['artifacts'].items():
        p = root / name
        assert p.resolve().is_relative_to(root.resolve())
        assert sha(p) == digest, (root, name)
    return dict(path=root.as_posix(), manifest_sha256=sha(root / 'manifest.json'),
                listed_artifacts_checked=len(m['artifacts']))


def history():
    old, now = tree(BASELINE), tree('HEAD')
    protected = {n: oid for n, oid in old.items() if n not in OVERVIEWS}
    assert all(now.get(n) == oid for n, oid in protected.items())
    assert not set(git('diff', '--name-only', 'HEAD').splitlines()).intersection(protected)
    assert all(n.startswith('results/manuscript_draft_v1/')
               for n in now.keys() - old.keys() if n.startswith('results/'))
    return dict(baseline_commit=BASELINE, protected_paths=len(protected),
                preserved_result_git_identities=sum(n.startswith('results/') for n in protected),
                historical_scientific_and_failure_records_unchanged=True,
                **materialized_checks(protected))


def evidence(c, draft):
    root = Path(c['parent_inventory'])
    assert sha(root / 'manifest.json') == c['parent_manifest_sha256']
    manifests = [authenticate_manifest(root)]
    assert read(root / 'checks.json')['all_checks_pass']
    facts = read(root / 'study-evidence.json')
    assert {f['id']: f['resolved_status'] for f in facts} == c['expected_statuses']
    for f in facts:
        main, validation = Path(f['main']), Path(f['validation'])
        assert read(main / 'summary.json') == f['archived_summary']
        assert read(validation / 'checks.json') == f['archived_checks']
        assert sha(main / 'manifest.json') == f['main_manifest_sha256']
        assert sha(validation / 'manifest.json') == f['validation_manifest_sha256']
        assert f['archived_checks']['all_checks_pass']
        assert f['archived_checks']['result_manifest_sha256'] == f['main_manifest_sha256']
        assert f['report'].removeprefix('docs/') in draft, f['id']
    by_id = {f['id']: f for f in facts}
    assert by_id['M1']['archived_summary']['primary_pass'] is None
    assert '**assay-invalid**' in draft and 'remaining null' in draft
    assert by_id['M4']['archived_summary']['levels']['brain5']['degree_role_status'] == 'UNRESOLVED'
    assert 'degree-only local tests remain UNRESOLVED' in draft
    assert by_id['M4']['archived_summary']['neural_runs'] == by_id['M4']['archived_summary']['decoder_fits'] == 0
    for level in by_id['M9']['archived_summary']['levels'].values():
        assert level['current_access_valid'] and level['real_score'] == level['null_mean'] == 1.
        assert not level['registered_rule_pass'] and level['real_minus_null_mean_exact'] == '0'
    assert 'ceiling ties do not establish universal equivalence' in draft
    assert by_id['M10']['producer_raw_status'] == 'READY_PENDING_INDEPENDENT_VERIFICATION'
    assert by_id['M10']['archived_checks']['outcome'] == 'READY'
    assert "producer summary's pending label is preserved" in draft
    m11 = by_id['M11']
    assert m11['archived_checks']['independent_svd_refits'] == m11['archived_summary']['fits'] == 216
    assert m11['archived_checks']['full_state_trajectories_exact'] == 427680
    assert not m11['archived_summary']['primary_pass'] and not m11['archived_summary']['secondary_pass']
    assert '427,680' in draft and 'Acquisition already fails in the training data' in draft
    manifests.append(authenticate_manifest(m11['main']))
    return by_id, manifests


def numerical_table(facts, draft):
    f = facts['M11']
    rows = read(Path(f['main']) / 'metrics.json')
    cohorts = f['archived_summary']['cohorts']
    table = {}
    for cohort, seeds in [('discovery', range(2710001, 2710004)),
                          ('confirmation', range(2710004, 2710007))]:
        def selected(split, group):
            v = [r for r in rows if r['family'] == 'affine' and r['model'] == 'real'
                 and r['input_seed'] in seeds and r['split'] == split and r['group'] == group]
            assert len(v) == 3
            return v
        train, test = selected('train', 'all'), selected('parameter_test', 'nonperiod2')
        pairs = {
            'train': (sum(r['teacher_correct'] for r in train), sum(r['teacher_predictions'] for r in train)),
            'first': (sum(r['first_correct'] for r in test), sum(r['episodes'] for r in test)),
            'teacher8': (sum(r['teacher_correct'] for r in test), sum(r['teacher_predictions'] for r in test)),
            'completion': (sum(r['completed'] for r in test), sum(r['episodes'] for r in test)),
        }
        assert Fraction(*pairs['train']) == Fraction(cohorts[cohort]['real_train_teacher'])
        assert Fraction(*pairs['first']) == Fraction(cohorts[cohort]['real_nonperiod2_first'])
        assert Fraction(*pairs['completion']) == Fraction(cohorts[cohort]['nonperiod2_completions']['real'])
        for name, (a, b) in pairs.items():
            cell = f'{a}/{b} ({100*a/b:.2f}%)'
            assert cell in draft, (cohort, name, cell)
        for key in ['real_minus_shift_completion', 'real_minus_null_completion']:
            value = Fraction(cohorts[cohort][key])
            cell = f'{abs(100*float(value)):.3f}pp'
            cell = ('−' if value < 0 else '+') + cell
            assert cell in draft, (cohort, key, cell)
        assert not any(cohorts[cohort]['primary_gates'].values())
        assert not cohorts[cohort]['real_strictly_above_every_null']
        table[cohort] = dict(raw_count_pairs=pairs, all_reported_table_cells_match=True,
                             gates=cohorts[cohort]['primary_gates'])
    # Two other primary manuscript anchors recomputed from archived exact cells.
    selected = {}
    for level, expected in [('legacy5', '255/384 (66.4063%)'), ('brain5', '109/192 (56.7708%)')]:
        cells = facts['M7']['archived_summary']['levels'][level]['cells']
        a, b = sum(x['joint_correct'] for x in cells), sum(x['samples'] for x in cells)
        percentage = (Decimal(a)*100/Decimal(b)).quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)
        actual = f'{a}/{b} ({percentage}%)'
        assert actual == expected and actual in draft
        assert all(not x['scientific_rule_pass'] for x in cells)
        selected[level] = dict(joint_correct=a, pairs=b)
    m2 = facts['M2']['archived_summary']['levels']['legacy5']['primary']
    raw_history_pp = 90 * Fraction(m2['historical_gain']['mean_exact'])
    current_pp = 100 * Fraction(m2['current_difference']['mean_exact'])
    assert f'+{float(raw_history_pp):.4f}pp' in draft
    assert f'−{abs(float(current_pp)):.4f}pp' in draft
    gains = facts['M5']['archived_summary']['levels']['legacy5']['cells']
    mean_gain = sum(x['relative_gain'] for x in gains) / len(gains)
    assert f'{mean_gain:.6f}' in draft and f'{mean_gain*100:.4f}%' in draft
    return dict(m11=table, m7=selected, m2_raw_history_pp=str(raw_history_pp),
                m2_current_pp=str(current_pp), m5_relative_gain_mean=mean_gain,
                scope='selected exact/rounded manuscript anchors; not every prose numeral')


def citations(c, draft):
    ledger = read(c['literature'])
    refs = ledger['references']
    ids = {r['id'] for r in refs}
    assert len(refs) == len(ids) == 9 and ids == {f'L{i}' for i in range(1, 10)}
    assert len({r.get('doi') or r['canonical_url'] for r in refs}) == 9
    bib = Path(BIB).read_text(encoding='utf-8')
    entries = {key: (kind, dict(re.findall(r'^\s+(\w+) = \{(.*)\},?$', body, re.M)))
               for kind, key, body in re.findall(r'@(\w+)\{(L\d+),\n(.*?)\n\}', bib, re.S)}
    assert set(entries) == ids
    counts = {}
    checked = []
    for r in refs:
        kind, fields = entries[r['id']]
        assert fields['year'] == str(r['year']) and fields['author'].startswith(r['first_author'] + ',')
        if r.get('title_sha256'):
            assert hashlib.sha256(fields['title'].encode()).hexdigest() == r['title_sha256']
        else:
            # These institutional/arXiv records were verified directly, not
            # assigned Crossref-derived title hashes by the literature ledger.
            assert fields['title'] == {
                'L5': 'Short term memory in echo state networks',
                'L9': 'Determinants of hyperparameter robustness in connectome reservoir computing',
            }[r['id']]
        if r.get('doi'):
            assert fields['doi'] == r['doi']
        assert r['access_scope'] and r['locator'] and r['supported_claim']
        assert r['review_date'] == c['date'] and r['new_replication'] is False
        if kind == 'article':
            assert r['status'] == 'peer-reviewed journal article'
            assert fields['journal'] == r['venue'] and fields['volume'] == r['volume']
            assert fields['pages'].replace('--', '-') == r['pages']
        elif kind == 'techreport':
            assert r['id'] == 'L5' and 'technical report' in r['status']
        else:
            assert kind == 'misc' and r['id'] == 'L9' and 'preprint' in r['status'].lower()
            assert fields['version'] == '1' and 'not treated as peer-reviewed' in fields['note']
        citations = re.findall(r'\[' + r['id'] + r'\]\(#ref-' + r['id'].lower() + r'\)', draft)
        assert citations and draft.count(f'<a id="ref-{r["id"].lower()}"></a>') == 1
        assert f'**{r["id"]}.**' in draft
        locators = [r['canonical_url']] + ([r['doi']] if r.get('doi') else [])
        assert any(locator.lower() in draft.lower() for locator in locators)
        counts[kind] = counts.get(kind, 0) + 1
        checked.append(dict(id=r['id'], kind=kind, status=r['status'], citation_uses=len(citations),
                            identity_and_recorded_access_fields_match=True,
                            source_url=r['canonical_url'], fresh_remote_fetch=False))
    assert counts == dict(article=7, techreport=1, misc=1)
    assert set(re.findall(r'\[(L\d+)\]\(#ref-l\d+\)', draft)) == ids
    assert 'not a systematic' in draft and 'preprint' in draft
    return dict(references=checked, publication_classes=counts,
                automated_claim_entailment=False, manual_access_notes='docs/literature-context.md',
                systematic_review_or_novelty_certification=False)


def audit(out):
    c = read(SCOPE)
    assert c['baseline_commit'] == BASELINE and c['namespace'] == 'results/manuscript_draft_v1'
    assert c['new_model_runs'] == c['new_model_fits'] == 0
    assert not c['external_submission'] and not c['third_party_figure_reuse']
    assert not git('diff', '--name-only', 'HEAD'), 'Commit audited documents/source before running'
    assert out.resolve().is_relative_to(Path(c['namespace']).resolve())
    revision, blobs = git('rev-parse', 'HEAD'), GitBlobs()
    paths = c['documents'] + sorted(OVERVIEWS) + [SCOPE, c['literature'], SCRIPT,
            'scripts/audit_research_synthesis.py', 'scripts/counterfactual_support.py', 'scripts/tdc_support.py']
    try:
        for n in [SCOPE, c['literature'], BIB]:
            assert blobs.blob(SCOPE_COMMIT, n) == sha(n), ('frozen scope/ledger/bibliography', n)
        hashes = {n: sha(n) for n in paths}
        assert all(blobs.blob(revision, n) == h for n, h in hashes.items())
        draft = Path(DRAFT).read_text(encoding='utf-8')
        assert all(heading in draft for heading in c['required_draft_sections'])
        assert len(draft.split()) >= 4000 and c['scientific_cutoff'] in draft
        with attempt(out, c, 'manuscript-document-provenance-audit') as resources:
            write(out / 'source.json', dict(source_commit=revision, scope_commit=SCOPE_COMMIT,
                  hashes=hashes, environment=environment(), new_model_runs=0, new_model_fits=0))
            write(out / 'history.json', history())
            facts, manifests = evidence(c, draft)
            write(out / 'parent-checks.json', manifests)
            write(out / 'study-statuses.json', {k: v['resolved_status'] for k, v in facts.items()})
            write(out / 'numeric-checks.json', numerical_table(facts, draft))
            write(out / 'citation-checks.json', citations(c, draft))
            figures = []
            for n in c['figure_paths']:
                assert n in draft and sha(n) == blobs.blob(BASELINE, n)
                figures.append(dict(path=n, sha256=sha(n), existing_figure_unmodified=True))
            write(out / 'figure-checks.json', figures)
            method_paths = ['src/flying/data/connectome.py', 'src/flying/data/mushroom_body.py',
                            'src/flying/brain/timed_reservoir.py', 'src/flying/models/ridge.py',
                            'scripts/temporal_mechanism.py']
            write(out / 'methods-source-notes.json', dict(
                source_hashes={n: blobs.blob(BASELINE, n) for n in method_paths},
                manual_review=['postsynaptic rows/incoming-L1 normalization',
                    'carry and drive separate; same-step KC to MBON schedule',
                    'centered affine ridge; train-only moments and nominal budgets',
                    'teacher forcing separate from autonomous continuation'],
                numerical_execution=False, automatic_equation_entailment=False))
            documents = c['documents'] + sorted(OVERVIEWS)
            pending = (out / 'checks.json').as_posix()
            links = local_links(documents, revision, pending)
            write(out / 'document-sha256.json', {n: hashes[n] for n in documents})
            checks = dict(all_checks_pass=True, canonical_studies=13, primary_literature_references=9,
                peer_reviewed_articles=7, technical_reports=1, preprints=1,
                manuscript_words=len(draft.split()), selected_numeric_anchors_match=True,
                all_local_document_links_resolve=True, citation_anchors_and_bib_metadata_match=True,
                document_bytes_match_git=True, existing_figures_unchanged=True,
                historical_evidence_preserved=True, new_model_runs=0, new_model_fits=0,
                new_historical_neural_replays=0, external_submission=False,
                doi_or_journal_publication_created=False, audit_is_external_peer_review=False)
            write(out / 'checks.json', checks)
            for entry in links:
                if entry['current_audit_artifact']:
                    assert Path(entry['path']).is_file()
                    entry['materialized'] = True
            write(out / 'links.json', links)
            registry = Path(c['namespace']) / 'known-incomplete-attempts.json'
            failures = read(registry) if registry.is_file() else []
            for entry in failures:
                assert sha(entry['manifest_path']) == entry['manifest_sha256']
                assert not read(entry['manifest_path'])['complete'] and Path(entry['failure_path']).is_file()
            write(out / 'preserved-attempts.json', failures)
            assert git('rev-parse', 'HEAD') == revision and not git('diff', '--name-only', 'HEAD')
            assert all(sha(n) == h for n, h in hashes.items())
            resources.check()
        seal(out, dict(complete=True, kind='manuscript document/provenance audit', scientifically_valid=True))
    finally:
        blobs.close()
    print(read(out / 'checks.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    audit(parser.parse_args().out)
