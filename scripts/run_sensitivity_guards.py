"""Archive frozen synthetic M5 safeguards before source-graph outcomes."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path
import re

import pytest
from sensitivity_support import (config, source_record, history_preserved,
                                 assert_source_unchanged, attempt, seal, write, sha)

TESTS = ['tests/test_temporal_mechanism.py', 'tests/test_temporal_cycles.py',
         'tests/test_temporal_functional_sensitivity.py', 'tests/test_functional_sensitivity_verifier.py']


def run(out):
    c = config()
    with attempt(out, c, 'm5-synthetic-guards'):
        source = source_record(c)
        write(out/'source.json', source)
        write(out/'historical-preservation.json', history_preserved(c))
        log = StringIO()
        with redirect_stdout(log), redirect_stderr(log):
            status = pytest.main(['-q', *TESTS])
        output = log.getvalue()
        (out/'pytest.txt').write_text(output, encoding='utf-8')
        assert status == 0, 'Synthetic safeguard failure; retain attempt'
        counts = re.findall(r'(\d+) passed', output)
        assert counts
        assert_source_unchanged(source)
        checks = dict(all_checks_pass=True, tests_passed=int(counts[-1]),
                      source_commit=source['source_commit'],
                      test_hashes={name: sha(name) for name in TESTS},
                      parameters_changed=False, source_graph_outcomes_executed=False,
                      source_freeze_rechecked=True)
        write(out/'checks.json', checks)
    seal(out, dict(complete=True, kind='m5-synthetic-guards'))
    print(checks, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args().out)
