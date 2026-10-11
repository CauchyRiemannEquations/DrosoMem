"""M7 fixed one-dose protocol and frozen source/parent/history authentication."""
import argparse
import hashlib
from pathlib import Path
import subprocess
from counterfactual_support import tree, materialized_checks, authenticate_git_sources
from tdc_support import read,write,sha,array_sha,git,environment,attempt,seal,check_manifest

BASELINE='191e8edbcb7b90e066c85bfead24f00d02d04c6e'
PREREGISTRATION='2d07a9d303bd700ab69fa29797f8d6a712624d08'
CONFIG=Path('configs/counterfactual_noise.json')
NAMESPACE=Path('results/counterfactual_noise_v1')
OVERVIEWS={'README.md','docs/research-status.md','docs/research-roadmap.md','docs/next-work.md'}

def config(smoke=False):
    c=read(CONFIG)
    frozen=subprocess.check_output(['git','show',PREREGISTRATION+':'+CONFIG.as_posix()])
    assert hashlib.sha256(frozen).hexdigest()==sha(CONFIG)
    assert c['baseline_commit']==BASELINE and c['namespace']==NAMESPACE.as_posix()
    assert c['blocks']==[dict(seed=1700001+i,parent_seed=1500001+i,input_seed=1510001+i,stream_seed=1710001+i,noise_seed=1720001+i) for i in range(6)]
    assert c['blocks_by_level']=={'legacy5':list(range(1700001,1700007)),'brain5':list(range(1700001,1700004))}
    assert (c['noise_sigma'],c['probe_count'],c['primary_lag'],c['minimum_joint_accuracy'])==(.1,64,2,.9)
    assert not c['noise_clipping'] and not c['noise_feedback'] and not c['performance_eligibility_gate']
    assert (c['gain'],c['leak'],c['carry_multiplier'],c['test_samples'],c['warmup'])==(.9,.6,0.,2000,200)
    assert c['new_trained_parameters']==0 and c['neural_workers']==4 and c['max_rss_bytes']==4294967296 and c['max_seconds']==7200
    if smoke:
        c['test_samples']=c['smoke']['test_samples']
        c['blocks_by_level']={k:v[:1] for k,v in c['blocks_by_level'].items()}
    return c

def history_preserved(c=None):
    old=read(NAMESPACE/'baseline_audit/baseline-git-blobs.json')
    assert old==tree(BASELINE)
    protected={n:h for n,h in old.items() if n not in OVERVIEWS}
    now=tree('HEAD')
    assert all(now.get(n)==h for n,h in protected.items())
    assert not set(git('diff','--name-only','HEAD').splitlines()).intersection(protected)
    return dict(baseline_commit=BASELINE,prior_result_identities_preserved=sum(n.startswith('results/') for n in protected),
        protected_baseline_paths=len(protected),historical_scientific_source_data_protocol_config_tests_preserved=True,
        p2_material_gates_unchanged='FAIL',m1_outcome_unchanged='assay-invalid',m4_outcome_unchanged='INFEASIBLE',m6_outcome_unchanged='PASS',
        **materialized_checks(protected))

def source_record(c):
    assert subprocess.run(['git','diff','--quiet','HEAD']).returncode==0
    rev=git('rev-parse','HEAD')
    paths=sorted(Path('src/flying').rglob('*.py'))+[CONFIG,Path(c['protocol']),Path('docs/counterfactual-noise-prior-audit.md'),Path('requirements-act1-lock.txt')]
    paths += [Path('scripts')/n for n in ['noise_tracking_support.py','counterfactual_noise.py','verify_counterfactual_noise.py','audit_counterfactual_noise.py','run_noise_tracking_guards.py',
        'counterfactual_tracking.py','verify_counterfactual_tracking.py','counterfactual_support.py','temporal_mechanism.py','temporal_memory_curve.py','verify_temporal_memory_curve.py','tdc_support.py','tdc_pool.py']]
    paths += [Path('tests/test_counterfactual_noise.py')]
    paths += sorted(Path('data/flywire_783_mb_left_kc512_s701').glob('*'))
    cache=Path(c['cache']);paths += [cache/n for n in ['nodes.npz','brain5.npz','provenance.json']]+sorted((cache.parent/'raw').glob('*'))
    parent=Path(c['parent_namespace']);paths += [parent/n for n in ['manifest.json','source.json','config.json']]
    for level in c['levels']:
        for i,b in enumerate(c['blocks']):
            if level=='brain5' and i>=3:continue
            p=parent/f'{level}_s{b["parent_seed"]}'
            paths += [p/n for n in ['manifest.json','case.npz','graph.json']]
    assert all(p.is_file() for p in paths)
    names=[p.as_posix() for p in paths]; assert len(set(names))==len(names)
    hashes={n:sha(n) for n in names}; tracked=[n for n in names if not n.startswith('outputs/')]
    authenticate_git_sources(tracked,hashes,rev)
    for n in [CONFIG.as_posix(),c['protocol']]:
        assert hashlib.sha256(subprocess.check_output(['git','show',PREREGISTRATION+':'+n])).hexdigest()==hashes[n]
    assert sha(parent/'manifest.json')==c['parent_manifest_sha256']
    assert git('rev-parse','HEAD')==rev and subprocess.run(['git','diff','--quiet','HEAD']).returncode==0
    return dict(source_commit=rev,source_tree=git('rev-parse',rev+'^{tree}'),protocol_commit=PREREGISTRATION,
        every_tracked_hash_authenticated_at_capture=True,tracked_changes=False,parent_bank_frozen=True,new_trained_parameters=0,hashes=hashes,environment=environment())

def assert_source_unchanged(source):
    assert git('rev-parse','HEAD')==source['source_commit'] and subprocess.run(['git','diff','--quiet','HEAD']).returncode==0
    assert all(sha(n)==h for n,h in source['hashes'].items())

def baseline_audit(out):
    c=config()
    with attempt(out,c,'m7-baseline-audit'):
        out=Path(out);old=tree(BASELINE);write(out/'baseline-git-blobs.json',old)
        roots=[Path('results/counterfactual_tracking_v1'),Path('results/functional_sensitivity_v1')]
        checks=[]
        for root in roots:
            for p in sorted(root.rglob('manifest.json')):
                m=read(p)
                assert all(sha(p.parent/n)==h for n,h in m['artifacts'].items())
                checks.append(dict(path=p.as_posix(),complete=m.get('complete'),entries=len(m['artifacts'])))
        assert read(roots[0]/'final_audit_retry1/audit.json')['all_checks_pass']
        assert sha(Path(c['parent_namespace'])/'manifest.json')==c['parent_manifest_sha256']
        write(out/'manifest-checks.json',checks)
        write(out/'audit.json',dict(all_checks_pass=True,baseline_commit=BASELINE,prior_result_identities=sum(n.startswith('results/') for n in old),
            protected_baseline_paths=len(old)-4,prior_manifests_checked=len(checks),retained_prior_failures_preserved=True,
            source_based_old_noise_scope_audit=True,no_historical_neural_replay_performed=True,**materialized_checks({n:h for n,h in old.items() if n not in OVERVIEWS})))
        write(out/'environment.json',environment())
    seal(out,dict(complete=True,kind='m7-baseline-audit'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);baseline_audit(p.parse_args().baseline)
