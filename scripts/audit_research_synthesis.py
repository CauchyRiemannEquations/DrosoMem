"""Read-only M1-M11 synthesis: archived identities, verdicts and static replay APIs.

No old runner/verifier, trajectory, model fitter or scientific scorer executes.
"""
import argparse
import ast
import hashlib
from pathlib import Path
import re
import subprocess
from counterfactual_support import tree, materialized_checks
from tdc_support import read, write, sha, git, environment, attempt, seal

BASELINE = 'c48e0c43836e640ec1a15b1ac1152f7e3ec4e2a4'
SCOPE_COMMIT = '23a6f9d719c551499f3af3a75977548d52c39e91'
SCOPE = Path('configs/research_synthesis.json')
OVERVIEWS = {'README.md', 'docs/research-status.md', 'docs/research-roadmap.md', 'docs/next-work.md'}


def scope():
    c = read(SCOPE)
    assert c['baseline_commit'] == BASELINE and c['new_model_runs'] == c['new_model_fits'] == 0
    assert sha(SCOPE) == hashlib.sha256(subprocess.check_output(['git', 'show', SCOPE_COMMIT + ':' + SCOPE.as_posix()])).hexdigest()
    assert [s['id'] for s in c['studies']] == ['P1', 'P2'] + [f'M{i}' for i in range(1, 12)]
    return c


class GitBlobs:
    def __init__(self):
        self.trees, self.digests, self.text = {}, {}, {}
        self.process = subprocess.Popen(['git', 'cat-file', '--batch'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    def inventory(self, revision):
        if revision not in self.trees:
            self.trees[revision] = tree(revision)
        return self.trees[revision]
    def blob(self, revision, name, text=False):
        oid = self.inventory(revision)[name]
        if oid not in self.digests or (text and oid not in self.text):
            self.process.stdin.write((oid + '\n').encode())
            self.process.stdin.flush()
            header = self.process.stdout.readline().decode().split()
            assert len(header) == 3 and header[1] == 'blob'
            size, digest, chunks = int(header[2]), hashlib.sha256(), []
            remaining = size
            while remaining:
                part = self.process.stdout.read(min(1048576, remaining))
                assert part
                digest.update(part)
                if text:
                    chunks.append(part)
                remaining -= len(part)
            assert self.process.stdout.read(1) == b'\n'
            self.digests[oid] = digest.hexdigest()
            if text:
                self.text[oid] = b''.join(chunks).decode('utf-8')
        return self.text[oid] if text else self.digests[oid]
    def close(self):
        self.process.stdin.close()
        assert self.process.wait() == 0
        self.process.stdout.close()
        self.process.stderr.close()


def history(bytes_check=False):
    old, now = tree(BASELINE), tree('HEAD')
    protected = {n: h for n, h in old.items() if n not in OVERVIEWS}
    assert all(now.get(n) == h for n, h in protected.items())
    assert not set(git('diff', '--name-only', 'HEAD').splitlines()).intersection(protected)
    answer = dict(baseline_commit=BASELINE, prior_result_git_identities=sum(n.startswith('results/') for n in protected),
                  protected_paths=len(protected), protected_git_identities_preserved=True, new_model_runs=0,
                  old_models_freshly_replayed=False)
    if bytes_check:
        answer.update(materialized_checks(protected))
    return answer


def source():
    assert not git('diff', '--name-only', 'HEAD'), 'Commit synthesis source before audit'
    paths = [SCOPE, Path(__file__).resolve(), Path('scripts/tdc_support.py'), Path('scripts/counterfactual_support.py')]
    root = Path.cwd()
    names = [p.relative_to(root).as_posix() if p.is_absolute() else p.as_posix() for p in paths]
    revision = git('rev-parse', 'HEAD')
    hashes = {n: sha(n) for n in names}
    blobs = GitBlobs()
    try:
        assert all(blobs.blob(revision, n) == h for n, h in hashes.items())
    finally:
        blobs.close()
    return dict(source_commit=revision, scope_commit=SCOPE_COMMIT, hashes=hashes, environment=environment(), numerical_model_runs=0)


def unchanged(record):
    assert git('rev-parse', 'HEAD') == record['source_commit'] and not git('diff', '--name-only', 'HEAD')
    assert all(sha(n) == h for n, h in record['hashes'].items())


def manifest_checks(roots, resources):
    checked, invalid, missing, extra, digests = [], [], [], [], {}
    for p in sorted({p for root in roots for p in Path(root).rglob('manifest.json')}):
        m = read(p)
        assert isinstance(m.get('artifacts'), dict), p
        for name, expected in m['artifacts'].items():
            target = p.parent / name
            assert target.resolve().is_relative_to(p.parent.resolve()), (p, name)
            if not target.is_file():
                missing.append(dict(manifest=p.as_posix(), artifact=name))
                continue
            key = target.as_posix()
            if key not in digests:
                digests[key] = sha(target)
            assert digests[key] == expected, (p, name)
        files = {q.relative_to(p.parent).as_posix() for q in p.parent.rglob('*') if q.is_file() and q != p}
        unlisted = sorted(files - set(m['artifacts']))
        if unlisted:
            extra.append(dict(manifest=p.as_posix(), paths=unlisted))
        checked.append(dict(path=p.as_posix(), complete=m.get('complete'), scientifically_valid=m.get('scientifically_valid'),
                            kind=m.get('kind'), entries=len(m['artifacts']), sha256=sha(p)))
        if m.get('complete') is False or m.get('scientifically_valid') is False:
            invalid.append(checked[-1])
        resources.check()
    assert not missing, 'Selected materialized archive has missing referenced artifact'
    return dict(manifests=checked, unique_materialized_artifacts_checked=len(digests), missing_artifacts=missing,
                unlisted_paths=extra, retained_incomplete_or_invalid_records=invalid,
                all_listed_materialized_checksums_pass=True)


def cli_signature(blobs, revision, path):
    code = ast.parse(blobs.blob(revision, path, text=True), filename=path)
    arguments = []
    for node in ast.walk(code):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument':
            arguments += [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
    assert '--out' in arguments
    return dict(revision=revision, script=path, git_sha256=blobs.blob(revision, path), arguments=sorted(set(arguments)),
                static_ast_checked=True, executed=False)


def inspect_study(s, blobs, external):
    main, val, audit = Path(s['main']), Path(s['validation']), Path(s['audit'])
    summary, checks = read(main / 'summary.json'), read(val / 'checks.json')
    assert read(main / 'manifest.json')['complete'] and read(val / 'manifest.json')['complete']
    assert checks['all_checks_pass'] and checks['result_manifest_sha256'] == sha(main / 'manifest.json')
    assert read(audit / 'audit.json')['all_checks_pass'] and read(audit / 'manifest.json')['complete']
    if s['id'] in ('P1', 'P2'):
        outcome = 'PASS' if summary['primary_pass'] else 'FAIL'
    elif s['id'] == 'M10':
        assert summary['outcome'] == 'READY_PENDING_INDEPENDENT_VERIFICATION'
        outcome = checks['outcome']
    else:
        outcome = summary['outcome']
    assert outcome == s['status'], (s['id'], outcome)
    if s['id'] == 'M1':
        assert summary['primary_pass'] is None
        assert all(not v['input_access_valid'] for v in summary['levels'].values())
    if s['id'] == 'M4':
        assert summary['levels']['brain5']['degree_role_status'] == 'UNRESOLVED'
        assert summary['decoder_fits'] == summary['neural_runs'] == 0
    if s['id'] in ('M6', 'M7'):
        assert summary['new_trained_parameters'] == 0
    if s['id'] == 'M8':
        assert summary['new_decoder_fits'] == 0 and not summary['performance_experiment_executed']
    if s['id'] == 'M9':
        assert all(v['current_access_valid'] and v['real_score'] == v['null_mean'] == 1. and not v['registered_rule_pass'] for v in summary['levels'].values())
    if s['id'] == 'M11':
        assert checks['independent_svd_refits'] == summary['fits'] == 216
        assert checks['full_state_trajectories_exact'] == 427680
        assert not summary['primary_pass'] and not summary['secondary_pass']
        assert all(not c['primary_gates']['train_acquisition'] for c in summary['cohorts'].values())
    producer = read(main / 'source.json')
    producer_revision = producer['source_commit']
    records = [dict(path=(main / 'source.json').as_posix(), data=producer)]
    if (val / 'source.json').is_file():
        verifier_record = read(val / 'source.json')
        verifier_revision = verifier_record['source_commit']
        records.append(dict(path=(val / 'source.json').as_posix(), data=verifier_record))
    elif 'verifier_pin' in s:
        verifier_revision = read(s['verifier_pin'])['verifier_commit']
    else:
        verifier_revision = checks['verifier_commit']
    authenticated, external_refs = 0, []
    for row in records:
        record = row['data']
        revision = record['source_commit']
        inventory = blobs.inventory(revision)
        for name, h in record['hashes'].items():
            if name in inventory:
                assert blobs.blob(revision, name) == h, (s['id'], row['path'], name)
                authenticated += 1
            else:
                key = (name, h)
                if key not in external:
                    p = Path(name)
                    external[key] = dict(path=name, expected_sha256=h, locally_available=p.is_file(),
                                         current_sha256=sha(p) if p.is_file() else None)
                assert not external[key]['locally_available'] or external[key]['current_sha256'] == h, (s['id'], name)
                external_refs.append(dict(path=name, expected_sha256=h))
    protocol_revision = git('rev-parse', s['protocol_ref'])
    assert subprocess.run(['git', 'merge-base', '--is-ancestor', protocol_revision, producer_revision]).returncode == 0
    registered_config = read(s['config'])
    protocol = registered_config['protocol']
    assert blobs.blob(protocol_revision, protocol) == blobs.blob(producer_revision, protocol)
    assert blobs.blob(protocol_revision, s['config']) == blobs.blob(producer_revision, s['config'])
    producer_cli = cli_signature(blobs, producer_revision, s['runner'])
    verifier_cli = cli_signature(blobs, verifier_revision, s['verifier'])
    required = '--result' if s['verify_style'] == 'result_flag' else 'target' if s['verify_style'] == 'command_positional' else 'result'
    assert required in verifier_cli['arguments']
    if 'verifier_sha256' in checks:
        assert blobs.blob(verifier_revision, s['verifier']) == checks['verifier_sha256']
    return dict(**s, resolved_status=outcome, producer_raw_status=summary.get('outcome'),
                protocol=protocol, protocol_revision=protocol_revision, producer_revision=producer_revision,
                verifier_revision=verifier_revision, main_manifest_sha256=sha(main / 'manifest.json'),
                validation_manifest_sha256=sha(val / 'manifest.json'), audit_manifest_sha256=sha(audit / 'manifest.json'),
                archived_checks=checks, archived_summary=summary, source_records=[r['path'] for r in records],
                authenticated_tracked_source_references=authenticated, external_input_references=external_refs,
                producer_cli=producer_cli, verifier_cli=verifier_cli,
                verifier_pin_record=s.get('verifier_pin', (val / 'source.json').as_posix() if (val / 'source.json').exists() else (val / 'checks.json').as_posix()))


def inspect_failures(c, archive):
    reported = []
    for s in c['studies']:
        paths = [p for p in [Path(s['root']) / 'known-incomplete-attempts.json', Path(s.get('failure_registry', 'unavailable'))] if p.is_file()]
        if paths:
            for p in paths:
                for record in read(p):
                    reported.append(dict(study=s['id'], registry=p.as_posix(), record=record))
    return dict(retained_manifest_records=archive['retained_incomplete_or_invalid_records'],
                registries=reported, technical_failure_records_are_not_scientific_outcomes=True,
                m4_complete_failure_snapshot_is_scientifically_invalid=True)


def inventory(out):
    c, record = scope(), source()
    with attempt(out, c, 'synthesis-read-only-inventory') as resources:
        write(out / 'source.json', record)
        write(out / 'scope.json', c)
        write(out / 'baseline-git-tree.json', tree(BASELINE))
        write(out / 'history.json', history(True))
        roots = sorted({s['root'] for s in c['studies']})
        archive = manifest_checks(roots, resources)
        write(out / 'archive-checks.json', archive)
        blobs, external = GitBlobs(), {}
        try:
            facts = [inspect_study(s, blobs, external) for s in c['studies']]
        finally:
            blobs.close()
        write(out / 'study-evidence.json', facts)
        write(out / 'external-inputs.json', list(external.values()))
        write(out / 'retained-technical-records.json', inspect_failures(c, archive))
        write(out / 'checks.json', dict(all_checks_pass=True, canonical_studies=13, new_model_runs=0, new_model_fits=0,
                                        all_statuses_and_validation_manifest_links_match=True,
                                        successful_source_records_authenticated=True, all_replay_clis_statically_checked=True,
                                        actual_replay_clis_executed=False, older_models_replayed=False,
                                        archived_source_references=sum(s['authenticated_tracked_source_references'] for s in facts),
                                        current_external_input_files_available=sum(v['locally_available'] for v in external.values()),
                                        unavailable_external_inputs=sum(not v['locally_available'] for v in external.values()),
                                        manifests_checked=len(archive['manifests']), retained_manifest_records=len(archive['retained_incomplete_or_invalid_records'])))
        unchanged(record)
    seal(out, dict(complete=True, kind='read-only research synthesis inventory'))
    print(read(out / 'checks.json'))


def link(path, label):
    return f'[{label}](../{path})'


def matrix_text(facts, c):
    text = ['# DrosoMem M1–M11 claim, evidence and limitation matrix', '',
            f'Date: {c["date"]}. Evidence cutoff `{BASELINE}`. This is an archived-evidence synthesis with **zero new model runs/fits**.', '',
            'P1/P2 are anchors for M1–M11; the earlier '+link(c['v1_anchor'], 'v1 closeout')+' remains sealed. P3 physiology/spiking/plasticity is outside this programme.', '',
            '**Outcome legend:** PASS = the named registered gate; FAIL = a valid completed negative hypothesis test; assay-invalid = an undefined conditional endpoint; INFEASIBLE = a strict design obstruction; READY = tools/task preparation. Verification PASS authenticates the recorded result, including FAIL. These heterogeneous rows must not be combined into a success percentage.', '',
            '| Study / question | Learned budget in that study | Final registered status | Supported observation | Interpretation limit |',
            '| --- | --- | --- | --- | --- |']
    for s in facts:
        text.append(f'| {s["id"]}: {s["question"]} | {s["learning"]} | **{s["resolved_status"]}** | {s["supported"]} | {s["limit"]} |')
    text += ['', '## Canonical evidence and provenance', '',
             'All links below address the immutable study records. The synthesis verifies saved decisions and checksum/source links; it does not freshly execute the archived model validations. Detailed source versions, cache/parent dependencies and static CLI checks are in the '+link('docs/research-reproduction-index.md', 'reproduction index')+'.', '']
    for s in facts:
        text += [f'### {s["id"]} — {s["title"]}', '',
                 f'Endpoint kind: **{s["kind"]}**. Final status: **{s["resolved_status"]}**.', '',
                 ' · '.join([link(s['protocol'], 'Protocol'), link(s['config'], 'Config'), link(s['report'], 'Result report'),
                             link(s['main']+'/manifest.json', 'Raw archive manifest'), link(s['validation']+'/checks.json', 'Independent checks'),
                             link(s['audit']+'/audit.json', 'Canonical final audit')]), '',
                 f'Protocol `{s["protocol_revision"]}`; producer `{s["producer_revision"]}`; verifier `{s["verifier_revision"]}`.', '',
                 f'Main manifest SHA256: `{s["main_manifest_sha256"]}`.', '']
        if s['id'] == 'M1':
            text += ['The original primary flag remains JSON `null`; the positive descriptive contrasts are not a PASS or a valid mechanism FAIL. Use `main_validation_retry1`, retaining the original memory/test failures.', '']
        if s['id'] == 'M4':
            text += ['The whole degree-only subquestion remains UNRESOLVED. The archived source-validation failure snapshot is complete as an archive but scientifically invalid; it is retained separately from successful M4 checks.', '']
        if s['id'] == 'M10':
            text += ['The producer summary intentionally remains `READY_PENDING_INDEPENDENT_VERIFICATION`. Final READY is supplied by the matching independent checks/final audit; the old producer file is not edited to change that label.', '']
    text += ['## Cross-study claim boundaries', '',
             '- Historical-input access: supported by P1 and bounded M3/M6 observations; it is neither next-symbol prediction nor autonomous string recall.',
             '- Pure feedback-cycle attribution: unresolved. M3 is sufficiency under a broader mask; M4 blocks one exact ideal control.',
             '- Fine source-wiring superiority: no registered material support in P2, M9 or M11; finite constraints and saturation limit wider inference.',
             '- Robust frozen tracking: M6 clean success does not rescind M7 one-dose failure.',
             '- Rule learning/transfer: M10 task READY and its privileged formula oracle do not establish neural learning; M11 acquisition itself misses its gate.',
             '- Internal synaptic learning, physiology, biological storage sites, living-fly arithmetic and formal capacity remain unestablished in this programme.', '',
             'See '+link('docs/research-synthesis.md', 'research synthesis')+' and '+link('docs/manuscript-outline.md', 'internal manuscript outline')+'.', '']
    return '\n'.join(text)


def replay_text(facts, c):
    text = ['# M1–M11 historical reproduction index', '', f'Date: {c["date"]}; evidence cutoff `{BASELINE}`.', '',
            'The synthesis audits **static replay entry points, recorded revisions, environment/parent references and archived checksums**. It does not execute these commands, install packages, fetch data or freshly replay old neural outcomes. Static consistency is not a guarantee of current-machine or cross-platform bitwise replay.', '',
            'Use separate disposable clones/checkouts at the producer and verifier pins below; keep the full repository history and required parent artifacts. Leave the primary research checkout and all sealed result directories intact. A producer pin can predate its final result commit, so supply the recorded input/parent archives rather than assuming its own completed result is present at that older pin.', '',
            'Use each recorded Python/package environment. These studies document Python3.12 and `requirements-act1-lock.txt`, but the archived environment record is authoritative. Set `PYTHONPATH=src;scripts` and numerical threads to1. Restore only hash-matched source/cache inputs at their configured paths; the JSON external-input inventory reports local availability at audit time. Whole-brain runners/verifiers can need ignored `outputs/tdc_v2/whole_cache` and pinned raw inputs; M10 and partial-only M11 differ. No data download was performed by this synthesis.', '',
            'Use fresh destinations. If producer and verifier revisions differ, keep the newly produced result available in both disposable checkouts (copying new replay output is acceptable; do not relocate/overwrite old evidence). M1 uses the memory-repaired verifier pin; P1 also has a later verifier pin. Never update an old checksum or lower a validation tolerance to accommodate a changed checkout.', '',
            '| Study | Producer source | Successful verifier source | Parent/protocol lineage |', '| --- | --- | --- | --- |']
    for s in facts:
        text.append(f'| {s["id"]} | `{s["producer_revision"]}` | `{s["verifier_revision"]}` | '+(', '.join(s['dependencies']) or 'v1/source-cache anchor')+' |')
    text += ['', '## Statically checked full-run entry points', '',
             'Commands below are templates, **not executed by this synthesis**. Complete each study’s recorded safeguards/smoke first and obey its unchanged resource/stopping rules. The result reports supply the detailed stage order; these two command lines alone are not a complete protocol. For P2, the committed retry bank is reused explicitly to avoid its old failed default bank paths.', '']
    for s in facts:
        id = s['id']
        destination = f'outputs/synthesis_replay/{id}/main'
        val_destination = f'outputs/synthesis_replay/{id}/validation'
        if id == 'P2':
            producer = f'python {s["runner"]} run --graphs results/tdc_v2/p2_graphs_retry --graph-validation results/tdc_v2/p2_graph_validation_retry --out {destination}'
            verifier = f'python {s["verifier"]} run {destination} --out {val_destination}'
        else:
            producer = f'python {s["runner"]} --out {destination}'
            result_flag = '--result ' if s['verify_style'] == 'result_flag' else ''
            verifier = f'python {s["verifier"]} {result_flag}{destination} --out {val_destination}'
        text += [f'### {id}', '', link(s['report'], 'Study’s detailed reproduction/evidence')+' · '+link(s['verifier_pin_record'], 'Verifier pin record'), '',
                 '```powershell', producer, verifier, '```', '',
                 f'Producer CLI arguments at its pin: `{", ".join(s["producer_cli"]["arguments"])}`.',
                 f'Verifier CLI arguments at its pin: `{", ".join(s["verifier_cli"]["arguments"])}`.', '',
                 f'Archived manifest to compare: `{s["main_manifest_sha256"]}`. A fresh replay has new timestamps/resources/manifests; compare the registered scientific arrays/counts and source/parent validation rules, not an expectation that every new metadata file hashes identically.', '']
    text += ['## Preserved technical records and availability', '',
             'The new inventory lists incomplete manifests and complete-but-scientifically-invalid snapshots separately. Active successful validations never use those failed stages. In particular P2 uses the retry graph bank, M1 uses `main_validation_retry1`, M6/M7 use their final-audit retries, and M4 retains its captured source mismatch. M8 retains the interrupted root-ID smoke and successful retry. None is removed or relabeled by this index.', '',
             'The baseline Git inventory protects sparse-excluded paths. Fresh byte/hash checks concern materialized files, and external-input availability is stated separately. Existing recorded independent replay counts remain historical evidence; they are not new replay counts for this synthesis.', '',
             link('docs/historical-reproduction.md', 'General historical replay discipline')+' · '+link('results/research_synthesis_v1/inventory/checks.json', 'Static inventory checks')+' · '+link('results/research_synthesis_v1/inventory/external-inputs.json', 'External inputs and local availability')+' · '+link('results/research_synthesis_v1/inventory/retained-technical-records.json', 'Retained technical records'), '']
    return '\n'.join(text)


def render(inventory_path):
    c = scope()
    assert read(inventory_path / 'checks.json')['all_checks_pass']
    facts = read(inventory_path / 'study-evidence.json')
    for name, content in [('docs/research-evidence-matrix.md', matrix_text(facts, c)), ('docs/research-reproduction-index.md', replay_text(facts, c))]:
        p = Path(name)
        assert not p.exists(), 'Never overwrite generated synthesis documents'
        p.write_text(content, encoding='utf-8')


def local_links(documents, revision):
    inventory = tree(revision)
    links = []
    for name in documents:
        text = Path(name).read_text(encoding='utf-8')
        for target in re.findall(r'\]\(([^)]+)\)', text):
            if '://' in target or target.startswith('#'):
                continue
            path = (Path(name).parent / target.split('#')[0]).resolve().relative_to(Path.cwd()).as_posix()
            assert path in inventory or Path(path).is_file(), (name, target)
            links.append(dict(document=name, path=path, present_in_git=path in inventory, materialized=Path(path).is_file()))
    return links


def final(out, inventory_path):
    c, record = scope(), source()
    with attempt(out, c, 'synthesis-final-document-audit'):
        write(out / 'source.json', record)
        inventory_manifest = read(inventory_path / 'manifest.json')
        assert inventory_manifest['complete']
        assert all(sha(inventory_path / n) == h for n, h in inventory_manifest['artifacts'].items())
        assert read(inventory_path / 'checks.json')['all_checks_pass']
        facts = read(inventory_path / 'study-evidence.json')
        assert Path('docs/research-evidence-matrix.md').read_text(encoding='utf-8') == matrix_text(facts, c)
        assert Path('docs/research-reproduction-index.md').read_text(encoding='utf-8') == replay_text(facts, c)
        write(out / 'history.json', history(True))
        docs = c['documents'] + sorted(OVERVIEWS)
        write(out / 'links.json', local_links(docs, git('rev-parse', 'HEAD')))
        write(out / 'document-sha256.json', {n: sha(n) for n in docs})
        write(out / 'checks.json', dict(all_checks_pass=True, canonical_studies=13,
                                        matrix_and_reproduction_index_match_frozen_evidence=True,
                                        inventory_manifest_sha256=sha(inventory_path / 'manifest.json'),
                                        all_local_document_links_resolve=True, baseline_evidence_preserved=True,
                                        new_model_runs=0, new_model_fits=0, external_submission=False,
                                        publication_or_doi_claimed=False))
        unchanged(record)
    seal(out, dict(complete=True, kind='synthesis final audit'))
    print(read(out / 'checks.json'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['inventory', 'render', 'final'])
    p.add_argument('--out', type=Path)
    p.add_argument('--inventory', type=Path, default=Path('results/research_synthesis_v1/inventory'))
    args = p.parse_args()
    if args.mode == 'render':
        render(args.inventory)
    else:
        assert args.out is not None
        inventory(args.out) if args.mode == 'inventory' else final(args.out, args.inventory)
