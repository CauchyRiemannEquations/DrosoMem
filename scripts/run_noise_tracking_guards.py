"""Run and seal pre-outcome M7 synthetic safeguards on frozen source."""
import argparse
from pathlib import Path
from contextlib import redirect_stdout,redirect_stderr
from io import StringIO
import re
import pytest
from noise_tracking_support import config,source_record,assert_source_unchanged,history_preserved,attempt,write,sha,seal

TESTS=['tests/test_counterfactual_tracking.py','tests/test_counterfactual_tracking_verifier.py','tests/test_counterfactual_noise.py']
def run(out):
    c=config()
    with attempt(out,c,'m7-guards'):
        src=source_record(c);write(out/'source.json',src);write(out/'historical-preservation.json',history_preserved(c))
        capture=StringIO()
        with redirect_stdout(capture),redirect_stderr(capture):status=pytest.main(['-q',*TESTS])
        txt=capture.getvalue();(out/'pytest.txt').write_text(txt,encoding='utf-8')
        assert status==0,'Synthetic failure; retain attempt'
        assert_source_unchanged(src)
        write(out/'checks.json',dict(all_checks_pass=True,tests_passed=int(re.findall(r'(\d+) passed',txt)[-1]),test_hashes={p:sha(p) for p in TESTS},
            source_commit=src['source_commit'],source_freeze_rechecked=True,source_graph_outcomes_executed=False))
    seal(out,dict(complete=True));print(txt,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);run(p.parse_args().out)
