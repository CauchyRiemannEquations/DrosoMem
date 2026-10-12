"""Archive all nine prospective M11 synthetic safeguards."""
import argparse
from pathlib import Path
import subprocess
import sys
from rule_transfer_support import config, source, unchanged, attempt, write, seal


def run(out):
    c, record = config(), source(config())
    with attempt(out, c, 'm11-synthetic-guards'):
        write(out / 'source.json', record)
        process = subprocess.run([sys.executable, '-m', 'pytest', 'tests/test_rule_transfer.py', '-q'], text=True,
                                 capture_output=True, timeout=60)
        (out / 'pytest.txt').write_text(process.stdout + process.stderr, encoding='utf-8')
        assert process.returncode == 0 and '9 passed' in process.stdout
        write(out / 'checks.json', dict(all_checks_pass=True, tests_passed=9))
        unchanged(record)
    seal(out, dict(complete=True))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    run(p.parse_args().out)
