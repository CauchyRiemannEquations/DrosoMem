"""Seal M8 synthetic safeguards before source graph/prototype outcomes."""
import argparse
from pathlib import Path
from contextlib import redirect_stdout,redirect_stderr
from io import StringIO
import re
import pytest
from specificity_design_support import config,source_record,source_unchanged,history_preserved,attempt,write,sha,seal

TESTS=['tests/test_structural_specificity_design.py']
def run(out):
    c=config()
    with attempt(out,c,'m8-guards'):
        captured=source_record(c);write(out/'source.json',captured);write(out/'historical-preservation.json',history_preserved())
        buffer=StringIO()
        with redirect_stdout(buffer),redirect_stderr(buffer):status=pytest.main(['-q',*TESTS])
        txt=buffer.getvalue();(out/'pytest.txt').write_text(txt,encoding='utf-8');assert status==0,'Preserve failed safeguards'
        source_unchanged(captured)
        write(out/'checks.json',dict(all_checks_pass=True,tests_passed=int(re.findall(r'(\d+) passed',txt)[-1]),source_graph_outcomes_executed=False,
            source_freeze_rechecked=True,test_hashes={n:sha(n) for n in TESTS}))
    seal(out,dict(complete=True));print(txt,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);run(p.parse_args().out)
