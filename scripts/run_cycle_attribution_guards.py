"""Archive synthetic safeguards before M4 structural source-graph outcomes."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path
import re

import pytest

from m4_support import config, source_record, history_preserved, attempt, seal, write, sha, git


TESTS = ['tests/test_temporal_mechanism.py', 'tests/test_temporal_cycles.py',
         'tests/test_cycle_attribution_feasibility.py', 'tests/test_cycle_attribution_verifier.py']


def run(out):
    c = config()
    with attempt(out, c, 'm4-synthetic-guards'):
        write(out/'source.json', source_record(c))
        write(out/'historical-preservation.json', history_preserved(c))
        captured = StringIO()
        with redirect_stdout(captured), redirect_stderr(captured):
            status = pytest.main(['-q', *TESTS])
        output = captured.getvalue()
        (out/'pytest.txt').write_text(output, encoding='utf-8')
        assert status == 0, 'Synthetic safeguard failure; retain this attempt'
        counts = re.findall(r'(\d+) passed', output)
        assert counts, 'Missing pytest passed-test count'
        checks = {'all_checks_pass': True, 'tests_passed': int(counts[-1]),
                  'source_commit': git('rev-parse', 'HEAD'),
                  'test_hashes': {name: sha(name) for name in TESTS},
                  'parameters_changed': False, 'source_graph_outcomes_executed': False}
        write(out/'checks.json', checks)
    seal(out, {'complete': True, 'kind': 'm4-synthetic-guards'})
    print(checks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args().out)
