"""M3 fixed inputs, source provenance and immutable history guards."""
from pathlib import Path
import subprocess
from tdc_support import read,write,sha,array_sha,git,environment,attempt,seal,check_manifest

CONFIG=Path('configs/temporal_cycles.json')
BASELINE='eb571bd83107e2b108bf270003a918f8eb21e797'

def config(smoke=False):
    c=read(CONFIG)
    assert c['baseline_commit']==BASELINE and c['namespace']=='results/tdc_cycles_v1'
    assert c['levels']==['legacy5','brain5'] and c['circuit_seeds']==[701]
    assert c['arms']=={'intact_synaptic':{'carry':False,'synaptic_history':True,'dag':False},'dag_synaptic':{'carry':False,'synaptic_history':True,'dag':True},'instantaneous':{'carry':False,'synaptic_history':False,'dag':False}}
    for name,start in [('seed',1300001),('input_seed',1310001),('train_seed',1320001),('test_seed',1330001),
        ('initial_a_seed',1350001),('initial_b_seed',1350101),('prefix_a_seed',1360001),('prefix_b_seed',1360101),('suffix_seed',1370001)]:
        assert [b[name] for b in c['blocks']]==list(range(start,start+6))
    assert c['blocks_by_level']=={'legacy5':list(range(1300001,1300007)),'brain5':list(range(1300001,1300004))}
    assert (c['gain'],c['leak'],c['alpha'],c['input_fraction'],c['input_amplitude'])==(.9,.6,1.,.1,.5)
    assert c['carry_multiplier']==0 and c['normalization']=='incoming_l1' and c['schedule']=='mbon_after_kc'
    assert (c['alphabet_size'],c['warmup'],c['train_samples'],c['test_samples'])==(10,200,4000,2000)
    assert c['lags']==list(range(21)) and (c['primary_arm'],c['primary_lag'],c['minimum_excess'],c['reference_current_min'])==('dag_synaptic',2,.1,.99)
    assert c['expected_depth']=={'legacy5':11,'brain5':153} and c['certificate_prefix_steps']==64
    assert (c['bootstrap_seed'],c['bootstrap_draws'],c['neural_workers'],c['max_seconds'],c['max_rss_bytes'])==(1340001,10000,4,7200,4294967296)
    if smoke:
        c.update({n:c['smoke'][n] for n in ['train_samples','test_samples']})
        c['blocks_by_level']={level:seeds[:1] for level,seeds in c['blocks_by_level'].items()}
    return c

def history_preserved(c):
    old=read(Path(c['namespace'])/'baseline_audit/baseline-git-blobs.json');now={}
    for line in git('ls-tree','-r','HEAD').splitlines():
        meta,name=line.split('\t',1);now[name]=meta.split()[2]
    editable={'README.md','docs/research-status.md','docs/research-roadmap.md','docs/next-work.md'}
    protected={n:h for n,h in old.items() if n not in editable and n.startswith(('results/','src/','data/','configs/','scripts/','tests/','docs/'))}
    assert all(now.get(n)==h for n,h in protected.items()),'Previously sealed scientific source/results changed'
    assert not set(git('diff','--name-only','HEAD').splitlines()).intersection(protected)
    assert not git('diff','--name-only','HEAD','--','results/'),'Modified tracked evidence bytes'
    return dict(baseline_commit=BASELINE,prior_result_blobs_unchanged=sum(n.startswith('results/') for n in protected),
                historical_numerical_source_data_config_protocols_tests_unchanged=True,m1_outcome_unchanged='assay-invalid',m2_outcome_unchanged='PASS')

def source_record(c):
    assert subprocess.run(['git','diff','--quiet','HEAD']).returncode==0,'Tracked source edits before execution'
    paths=sorted(Path('src/flying').rglob('*.py'))
    paths += [CONFIG,Path(c['protocol']),Path('requirements-act1-lock.txt')]
    paths += [Path('scripts')/n for n in ['temporal_cycles.py','verify_temporal_cycles.py','cycle_support.py','cycle_structure.py',
        'temporal_mechanism.py','verify_temporal_mechanism.py','mechanism_support.py','tdc_support.py','tdc_pool.py',
        'temporal_memory_curve.py','verify_temporal_memory_curve.py']]
    paths += sorted(Path('data/flywire_783_mb_left_kc512_s701').glob('*'))
    cache=Path(c['cache']);paths += [cache/n for n in ['nodes.npz','brain5.npz','provenance.json']]
    paths += sorted((cache.parent/'raw').glob('*'))
    for p in paths:
        if not p.as_posix().startswith('outputs/'):
            subprocess.run(['git','ls-files','--error-unmatch',p.as_posix()],check=True,stdout=subprocess.DEVNULL)
    return dict(source_commit=git('rev-parse','HEAD'),source_tree=git('rev-parse','HEAD^{tree}'),tracked_changes=False,
                hashes={p.as_posix():sha(p) for p in paths},environment=environment())
