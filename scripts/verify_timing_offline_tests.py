"""Explicit targeted checks when pytest is unavailable; not the full suite."""
import argparse,json,runpy,unittest
from pathlib import Path


def run(output):
    files=['tests/test_constrained_bptt.py','tests/test_mushroom_body.py','tests/test_reward_plasticity.py',
           'tests/test_plasticity.py','tests/test_timed_reservoir.py']
    passed=[]
    for path in files:
        module=runpy.run_path(path)
        for name,function in module.items():
            if name.startswith('test_'):
                function();passed.append(path+'::'+name)
    suite=unittest.defaultTestLoader.discover('tests',pattern='test_pi_fallback.py')
    result=unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful():raise AssertionError('π fallback checks failed')
    report=dict(direct_assertion_tests=passed,unittest_pi_checks=result.testsRun,total_passed=len(passed)+result.testsRun,
                full_pytest_suite_run=False,reason='pytest not installed and package download timed out; no pytest emulation used')
    Path(output).write_text(json.dumps(report,indent=2)+'\n');print('Targeted checks passed:',report['total_passed'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();run(a.output)
