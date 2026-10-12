"""Seal M9 synthetic safeguards on committed source before empirical outcomes."""
import argparse
from pathlib import Path
from contextlib import redirect_stdout,redirect_stderr
from io import StringIO
import re
import pytest
from structural_tracking_support import config,source_record,source_unchanged,history_preserved,parent_audit,attempt,write,sha,seal

TESTS=['tests/test_structural_tracking.py']
def run(out):
    c=config()
    with attempt(out,c,'m9-guards'):
        captured=source_record(c);write(out/'source.json',captured);write(out/'historical-preservation.json',history_preserved());write(out/'parent-audit.json',parent_audit(c))
        buffer=StringIO()
        with redirect_stdout(buffer),redirect_stderr(buffer):status=pytest.main(['-q',*TESTS])
        txt=buffer.getvalue();(out/'pytest.txt').write_text(txt,encoding='utf-8');assert status==0,'Retain safeguard failure'
        source_unchanged(captured);write(out/'checks.json',dict(all_checks_pass=True,tests_passed=int(re.findall(r'(\d+) passed',txt)[-1]),source_freeze_rechecked=True,
            source_graph_outcomes_executed=False,test_hashes={n:sha(n) for n in TESTS}))
    seal(out,dict(complete=True));print(txt,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);run(p.parse_args().out)
