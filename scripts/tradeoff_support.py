"""M2 fixed configuration and preservation/provenance guards, no dynamics."""
from pathlib import Path
import subprocess
from tdc_support import read,write,sha,array_sha,git,environment,attempt,seal,check_manifest

CONFIG=Path('configs/temporal_tradeoff.json')
BASELINE='e52d1ff959d4426ce61315ad17788aa7c66fcb88'

def config(smoke=False):
    c=read(CONFIG)
    assert c['baseline_commit']==BASELINE and c['namespace']=='results/tdc_tradeoff_v1'
    assert c['levels']==['legacy5','brain5'] and c['circuit_seeds']==[701]
    assert c['arms']=={'full':{'carry':True,'synaptic_history':True},'carry_only':{'carry':True,'synaptic_history':False},'synaptic_only':{'carry':False,'synaptic_history':True},'instantaneous':{'carry':False,'synaptic_history':False}}
    for key,start in [('seed',1200001),('input_seed',1210001),('train_seed',1220001),('test_seed',1230001)]:
        assert [b[key] for b in c['blocks']]==list(range(start,start+6))
    assert c['blocks_by_level']=={'legacy5':list(range(1200001,1200007)),'brain5':list(range(1200001,1200004))}
    assert c['lags']==list(range(21)) and c['score_lags']==list(range(1,6))
    assert c['primary_contrast']==['carry_only','instantaneous'] and c['primary_level']=='legacy5'
    assert (c['gain'],c['leak'],c['alpha'],c['input_fraction'],c['input_amplitude'])==(.9,.6,1.,.1,.5)
    assert (c['alphabet_size'],c['warmup'],c['train_samples'],c['test_samples'])==(10,200,4000,2000)
    assert (c['historical_gain_min'],c['current_difference_max'],c['reference_current_min'])==(.03,-.05,.99)
    assert c['schedule']=='mbon_after_kc' and c['normalization']=='incoming_l1'
    assert (c['bootstrap_seed'],c['bootstrap_draws'],c['neural_workers'])==(1240001,10000,4)
    assert (c['max_seconds'],c['max_rss_bytes'])==(7200,4294967296)
    if smoke:
        c.update({n:c['smoke'][n] for n in ['train_samples','test_samples']})
        c['blocks_by_level']={level:seeds[:1] for level,seeds in c['blocks_by_level'].items()}
    return c

def history_preserved(c):
    old=read(Path(c['namespace'])/'baseline_audit/baseline-git-blobs.json')
    now={}
    for line in git('ls-tree','-r','HEAD').splitlines():
        meta,name=line.split('\t',1);now[name]=meta.split()[2]
    editable={'README.md','docs/research-status.md','docs/research-roadmap.md','docs/next-work.md'}
    protected={n:h for n,h in old.items() if n not in editable and n.startswith(('results/','src/','data/','configs/','scripts/','tests/','docs/'))}
    assert all(now.get(n)==h for n,h in protected.items()),'Previously sealed result/scientific source changed'
    dirty=git('diff','--name-only','HEAD').splitlines()
    assert not set(dirty).intersection(protected),'Uncommitted changes to sealed scientific source'
    assert not git('diff','--name-only','HEAD','--','results/'),'Modified tracked evidence bytes'
    return dict(baseline_commit=BASELINE,prior_result_blobs_unchanged=sum(n.startswith('results/') for n in protected),
                historical_numerical_source_data_config_protocols_tests_unchanged=True,m1_outcome_unchanged='assay-invalid')

def source_record(c):
    assert subprocess.run(['git','diff','--quiet','HEAD']).returncode==0,'Tracked edits before execution'
    paths=sorted(Path('src/flying').rglob('*.py'))
    paths += [CONFIG,Path(c['protocol']),Path('requirements-act1-lock.txt')]
    paths += [Path('scripts')/n for n in ['temporal_tradeoff.py','verify_temporal_tradeoff.py','tradeoff_support.py',
        'temporal_mechanism.py','verify_temporal_mechanism.py','mechanism_support.py','tdc_support.py','tdc_pool.py',
        'temporal_memory_curve.py','verify_temporal_memory_curve.py']]
    paths += sorted(Path('data/flywire_783_mb_left_kc512_s701').glob('*'))
    cache=Path(c['cache']);paths += [cache/n for n in ['nodes.npz','brain5.npz','provenance.json']]
    paths += sorted((cache.parent/'raw').glob('*'))
    for p in paths:
        if not p.as_posix().startswith('outputs/'):
            subprocess.run(['git','ls-files','--error-unmatch',p.as_posix()],check=True,stdout=subprocess.DEVNULL)
    return dict(source_commit=git('rev-parse','HEAD'),source_tree=git('rev-parse','HEAD^{tree}'),
                tracked_changes=False,hashes={p.as_posix():sha(p) for p in paths},environment=environment())
