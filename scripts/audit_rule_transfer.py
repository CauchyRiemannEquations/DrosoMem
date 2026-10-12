"""M11 final immutable-history, manifests, exact outcomes and report audit."""
import argparse
from pathlib import Path
from rule_transfer_support import (config, source, unchanged, history, coverage, authenticate_source,
                                   read, write, sha, attempt, seal, OVERVIEWS, artifact_limit)


def audit(out):
    c, record = config(), source(config())
    root = Path(c['namespace'])
    with attempt(out, c, 'm11-final-audit'):
        write(out / 'source.json', record)
        write(out / 'history.json', history(True))
        manifests, incomplete = [], []
        for p in sorted(root.rglob('manifest.json')):
            if out.resolve() in p.resolve().parents:
                continue
            m = read(p)
            coverage(p.parent, m['complete'])
            manifests.append(dict(path=p.as_posix(), complete=m['complete'], entries=len(m['artifacts']), sha256=sha(p)))
            if not m['complete']:
                incomplete.append(p.as_posix())
                assert (p.parent / 'failure.json').is_file()
        failures = read(root / 'known-incomplete-attempts.json')
        assert sorted(f['manifest_path'] for f in failures) == sorted(incomplete)
        for stage in ['baseline_audit', 'guards', 'smoke', 'smoke_validation', 'main', 'main_validation']:
            r = root / stage
            assert read(r / 'manifest.json')['complete']
            authenticate_source(read(r / 'source.json'))
            artifact_limit(r, c)
        assert read(root / 'guards/checks.json')['tests_passed'] == 9
        stage_checks = {}
        for stage, validation, fits in [('smoke', 'smoke_validation', 3), ('main', 'main_validation', 216)]:
            r, v = root / stage, root / validation
            checked = read(v / 'checks.json')
            assert checked['all_checks_pass'] and checked['fits'] == checked['independent_svd_refits'] == fits
            assert checked['result_manifest_sha256'] == sha(r / 'manifest.json')
            assert read(r / 'metrics.json') == read(v / 'recomputed-metrics.json')
            assert read(r / 'summary.json') == read(v / 'recomputed-summary.json')
            cases = sorted((r / 'cases').iterdir())
            assert len(cases) == fits
            for case in cases:
                check = read(v / 'cases' / case.name / 'checks.json')
                assert check['all_checks_pass'] and check['result_manifest_sha256'] == sha(case / 'manifest.json')
                assert check['independent_svd_refits'] == 1 and check['all_predictions_and_metrics_exact']
            stage_checks[stage] = dict(fits=fits, trajectories=checked['full_state_trajectories_exact'], all_checks_pass=True)
        assert read(root / 'main/fixed-lag-reference-metrics.json') == read(root / 'main_validation/recomputed-fixed-lag-metrics.json')
        docs = {n: sha(n) for n in sorted(OVERVIEWS | {'docs/rule-transfer-results.md'})}
        write(out / 'document-sha256.json', docs)
        outcome = read(root / 'main/summary.json')
        write(out / 'audit.json', dict(all_checks_pass=True, outcome=outcome['outcome'], primary_pass=outcome['primary_pass'],
                                       secondary_pass=outcome['secondary_pass'], historical_scientific_evidence_preserved=True,
                                       independent_verification_complete=True, stages=stage_checks,
                                       manifests=manifests, recorded_incomplete_attempts=len(failures)))
        unchanged(record)
    seal(out, dict(complete=True, study='M11 final audit'))
    print(dict(all_checks_pass=True, outcome=outcome['outcome'], independent_verification_complete=True))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    audit(p.parse_args().out)
