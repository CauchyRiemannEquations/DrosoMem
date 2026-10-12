"""Archive M10 synthetic pre-outcome safeguards and exact test output."""
import argparse
from pathlib import Path
import subprocess
import sys
from rule_transfer_design_support import config, source, unchanged, budget, attempt, write, seal


def run(out):
    c, record = config(), source()
    with attempt(out, budget(c), 'm10-synthetic-guards'):
        write(out / 'source.json', record)
        result = subprocess.run([sys.executable, '-m', 'pytest', 'tests/test_rule_transfer_design.py', '-q'],
                                text=True, capture_output=True, timeout=60)
        (out / 'pytest.txt').write_text(result.stdout + result.stderr, encoding='utf-8')
        assert result.returncode == 0 and '7 passed' in result.stdout
        write(out / 'checks.json', dict(all_checks_pass=True, tests_passed=7, neural_model_fits=0))
        unchanged(record)
    seal(out, dict(complete=True, study='M10 synthetic guards'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args().out)
