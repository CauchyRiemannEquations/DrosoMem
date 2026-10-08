"""New study IO/config/history guards; all prior numerical modules stay frozen."""
from pathlib import Path
import subprocess

from tdc_support import read,write,sha,array_sha,git,environment,attempt,seal,check_manifest

CONFIG=Path('configs/temporal_mechanism.json')


def config(smoke=False):
    c=read(CONFIG)
    assert c['baseline_commit']=='28248fc7bbf306bcc4af708343c5cb4c6017856e'
    assert c['namespace']=='results/tdc_mechanism_v1'
    assert c['levels']==['legacy5','brain5'] and c['circuit_seeds']==[701]
    assert c['arms']=={'full':{'carry':True,'synaptic_history':True},'carry_only':{'carry':True,'synaptic_history':False},'synaptic_only':{'carry':False,'synaptic_history':True},'instantaneous':{'carry':False,'synaptic_history':False}}
    assert [b['seed'] for b in c['blocks']]==list(range(1100001,1100007))
    for key,start in [('input_seed',1110001),('train_seed',1120001),('test_seed',1130001)]:
        assert [b[key] for b in c['blocks']]==list(range(start,start+6))
    assert c['blocks_by_level']=={'legacy5':list(range(1100001,1100007)),'brain5':list(range(1100001,1100004))}
    assert c['lags']==list(range(21)) and c['score_lags']==list(range(1,21))
    assert c['primary_contrast']==['full','carry_only'] and c['primary_level']=='legacy5'
    assert (c['gain'],c['leak'],c['alpha'],c['input_fraction'],c['input_amplitude'])==(.9,.6,1.,.1,.5)
    assert (c['alphabet_size'],c['warmup'],c['train_samples'],c['test_samples'])==(10,200,4000,2000)
    assert c['minimum_score_difference']==.03 and c['minimum_current_accuracy']==.90
    assert c['schedule']=='mbon_after_kc' and c['normalization']=='incoming_l1'
    assert c['bootstrap_seed']==1140001 and c['bootstrap_draws']==10000 and c['neural_workers']==4
    if smoke:
        c.update({n:c['smoke'][n] for n in ['train_samples','test_samples']})
        c['blocks_by_level']={level:seeds[:1] for level,seeds in c['blocks_by_level'].items()}
    return c


def history_preserved(c):
    baseline=read(Path(c['namespace'])/'baseline_audit/baseline-git-blobs.json')
    now={}
    for line in git('ls-tree','-r','HEAD').splitlines():
        meta,name=line.split('\t',1)
        now[name]=meta.split()[2]
    protected={n:h for n,h in baseline.items() if n.startswith(('results/','src/','data/','configs/','scripts/'))}
    assert all(now.get(n)==h for n,h in protected.items()),'Previously committed evidence/numerical source changed'
    assert not git('diff','--name-only','HEAD','--','results/'),'Modified tracked evidence bytes'
    return dict(baseline_commit=c['baseline_commit'],prior_result_blobs_unchanged=sum(n.startswith('results/') for n in protected),prior_numerical_source_data_config_scripts_unchanged=True)


def source_record(c):
    assert subprocess.run(['git','diff','--quiet','HEAD']).returncode==0,'Tracked edits before trajectory generation'
    paths=sorted(Path('src/flying').rglob('*.py'))
    paths += [CONFIG,Path(c['protocol']),Path('requirements-act1-lock.txt')]
    paths += [Path('scripts')/n for n in ['temporal_mechanism.py','verify_temporal_mechanism.py','mechanism_support.py','tdc_support.py','tdc_pool.py','temporal_memory_curve.py','verify_temporal_memory_curve.py']]
    paths += sorted(Path('data/flywire_783_mb_left_kc512_s701').glob('*'))
    cache=Path(c['cache'])
    paths += [cache/n for n in ['nodes.npz','brain5.npz','provenance.json']]
    paths += sorted((cache.parent/'raw').glob('*'))
    for p in paths:
        if not p.as_posix().startswith('outputs/'):
            subprocess.run(['git','ls-files','--error-unmatch',p.as_posix()],check=True,stdout=subprocess.DEVNULL)
    return dict(source_commit=git('rev-parse','HEAD'),source_tree=git('rev-parse','HEAD^{tree}'),
                tracked_changes=False,hashes={p.as_posix():sha(p) for p in paths},environment=environment())
