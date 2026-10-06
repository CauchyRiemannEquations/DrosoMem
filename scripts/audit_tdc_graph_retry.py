"""Verify that the console-only repair changed no graph or scientific setting."""
import argparse
from pathlib import Path
import subprocess

from tdc_support import read,write,sha,attempt,seal,git,check_manifest


def audit(failed,retry,out):
    original = read(failed/'manifest.json')
    assert original['complete'] is False
    for name,digest in original['artifacts'].items():
        assert sha(failed/name) == digest,name
    assert read(failed/'failure.json')['exception'] == 'UnicodeEncodeError'
    complete = check_manifest(retry)
    old_c,new_c = read(failed/'config.json'),read(retry/'config.json')
    assert {k:v for k,v in old_c.items() if k!='graphs'} == {k:v for k,v in new_c.items() if k!='graphs'}
    old_s,new_s = read(failed/'source.json'),read(retry/'source.json')
    old_pool = subprocess.check_output(['git','show',old_s['source_commit']+':scripts/tdc_pool.py'])
    new_pool = subprocess.check_output(['git','show',new_s['source_commit']+':scripts/tdc_pool.py'])
    assert old_pool.replace('\u2014'.encode('utf-8'),b'-') == new_pool
    for name,digest in old_s['hashes'].items():
        if name != 'scripts/tdc_pool.py':
            assert sha(name) == digest,name
    with attempt(out,new_c,'p2-console-retry-identity-audit'):
        members = []
        for root in sorted(failed.iterdir()):
            if root.is_dir() and (root/'manifest.json').exists():
                check_manifest(root)
                other = retry/root.name
                check_manifest(other)
                assert sha(root/'raw.npz') == sha(other/'raw.npz')
                assert sha(root/'audit.json') == sha(other/'audit.json')
                members.append(dict(identity=root.name,raw_file_sha256=sha(root/'raw.npz'),audit_sha256=sha(root/'audit.json')))
        assert len(members) == 24
        write(out/'checks.json',dict(all_checks_pass=True,scientific_settings_unchanged=True,
                                    only_code_change='progress printer em dash replaced with ASCII hyphen',
                                    identical_completed_graphs=24,members=members,
                                    original_manifest_sha256=sha(failed/'manifest.json'),retry_manifest_sha256=sha(retry/'manifest.json'),
                                    original_source_commit=old_s['source_commit'],retry_source_commit=new_s['source_commit'],
                                    verifier_commit=git('rev-parse','HEAD'),verifier_sha256=sha(__file__),
                                    failed_attempt_aggregate_peak='not exported after console failure; parent and completed-worker resource files preserved'))
    seal(out,dict(complete=True,retry_manifest_sha256=sha(retry/'manifest.json')))
    print('Retry identity audit: 24/24 completed graphs and their structural audits byte-identical; settings unchanged',flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('failed',type=Path)
    p.add_argument('retry',type=Path)
    p.add_argument('--out',type=Path,required=True)
    a = p.parse_args()
    audit(a.failed,a.retry,a.out)
