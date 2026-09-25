"""Run the new readout unit tests, explicitly without claiming full pytest."""
import argparse,json,unittest
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
suite=unittest.defaultTestLoader.discover('tests',pattern='test_nonlinear_readout.py')
result=unittest.TextTestRunner(verbosity=2).run(suite)
if not result.wasSuccessful():raise AssertionError('Nonlinear readout checks failed')
Path(a.output).write_text(json.dumps(dict(nonlinear_readout_tests_passed=result.testsRun,full_pytest_suite_run=False),indent=2)+'\n')
