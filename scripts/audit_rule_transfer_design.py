"""M10 final artifact, decision, current report and historical identity audit."""
import argparse
from pathlib import Path
from rule_transfer_design_support import (config, source, unchanged, history, budget, attempt,
                                          write, read, seal, sha, authenticate, OVERVIEWS)


def audit(out):
    c, record = config(), source()
    root = Path(c['namespace'])
    with attempt(out, budget(c), 'm10-final-audit'):
        write(out / 'source.json', record)
        preserved = history(check_bytes=True)
        write(out / 'history.json', preserved)
        coverage = []
        for name in ('guards', 'main', 'independent_validation'):
            stage = root / name
            authenticate(stage)
            manifest = read(stage / 'manifest.json')
            coverage.append(dict(stage=name, entries=len(manifest['artifacts']), manifest_sha256=sha(stage / 'manifest.json')))
        assert read(root / 'known-incomplete-attempts.json') == []
        assert not list(root.rglob('failure.json'))
        checks = read(root / 'independent_validation/checks.json')
        task = read(root / 'main/design-checks.json')
        assert checks['all_checks_pass'] and checks['result_manifest_sha256'] == sha(root / 'main/manifest.json')
        assert checks['all_task_gates_pass'] == task['all_task_gates_pass']
        assert read(root / 'main/metrics.json') == read(root / 'independent_validation/recomputed-metrics.json')
        assert task == read(root / 'independent_validation/recomputed-design-checks.json')
        expected = 'READY' if task['all_task_gates_pass'] else 'INVALID'
        assert checks['outcome'] == expected
        assert read(root / 'guards/checks.json')['tests_passed'] == 7
        documents = {n: sha(n) for n in sorted(OVERVIEWS | {'docs/rule-transfer-design-results.md'})}
        write(out / 'document-sha256.json', documents)
        write(out / 'audit.json', dict(all_checks_pass=True, outcome=expected, neural_model_fits=0,
                                      historical_git_identities_preserved=True, manifest_coverage=coverage,
                                      independent_verification_complete=True, tests_passed=7,
                                      future_neural_performance_registered=False, incomplete_attempts=0))
        unchanged(record)
    seal(out, dict(complete=True, study='M10 final artifact audit'))
    print(dict(all_checks_pass=True, outcome=expected, historical_git_identities_preserved=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    audit(parser.parse_args().out)
