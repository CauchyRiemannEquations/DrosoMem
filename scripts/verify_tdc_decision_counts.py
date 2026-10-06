"""Exact rational check of the preregistered >= criterion, using correct counts."""
import argparse
import csv
from fractions import Fraction
from pathlib import Path

from tdc_support import read, write, sha, attempt, seal, git, check_manifest


def cell_excess(row, k):
    n = int(row['samples'])
    baseline = max(Fraction(1, k), Fraction(int(row['frequency_correct']), n), Fraction(int(row['current_correct']), n))
    return Fraction(int(row['correct']), n) - baseline


def verify(result, out):
    m = check_manifest(result)
    assert not m['smoke']
    c = read(result/'config.json')
    with attempt(out, c, 'p1-exact-count-decision-validation'):
        with (result/'raw-lags.csv').open(newline='', encoding='utf-8') as f:
            raw = list(csv.DictReader(f))
        selected = [r for r in raw if r['level'] == 'legacy5' and int(r['lag']) in c['primary_lags']]
        threshold = Fraction(str(c['minimum_excess']))
        checks = []
        for row in selected:
            excess = cell_excess(row, c['alphabet_size'])
            checks.append(dict(seed=int(row['seed']), lag=int(row['lag']), excess_numerator=excess.numerator,
                               excess_denominator=excess.denominator, exact_cell_pass=excess >= threshold))
        assert len(checks) == 30
        exact = all(r['exact_cell_pass'] for r in checks)
        original = read(result/'summary.json')['primary_pass']
        assert exact is original, 'Original endpoint does not match exact preregistered count criterion'
        write(out/'checks.json', dict(all_checks_pass=True, primary_pass=exact, source_commit=git('rev-parse', 'HEAD'),
                                    verifier_sha256=sha(__file__), result_manifest_sha256=sha(result/'manifest.json'),
                                    threshold=str(threshold), cells=checks))
    seal(out, dict(complete=True, result_manifest_sha256=sha(result/'manifest.json')))
    print(dict(primary_pass=exact, exact_rational_cells=len(checks)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('result', type=Path)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    verify(a.result, a.out)
