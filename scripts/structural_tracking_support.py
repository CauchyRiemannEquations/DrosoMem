"""M9 frozen protocol/parent/source and historical provenance guards."""
import argparse
import hashlib
from pathlib import Path
import subprocess
from counterfactual_support import tree,materialized_checks,authenticate_git_sources
from tdc_support import read,write,sha,array_sha,git,environment,attempt,seal,check_manifest

BASELINE='b2ef8412832ce64e208287fe063abe8068b5df5c'
PREREGISTRATION='6bb55fb7c534fe4d29ab366bf9a04eec3ac4b81e'
CONFIG=Path('configs/structural_tracking.json');ROOT=Path('results/structural_tracking_v1')
OVERVIEWS={'README.md','docs/research-status.md','docs/research-roadmap.md','docs/next-work.md'}

def config(smoke=False):
    c=read(CONFIG)
    assert sha(CONFIG)==hashlib.sha256(subprocess.check_output(['git','show',PREREGISTRATION+':'+CONFIG.as_posix()])).hexdigest()
    assert c['baseline_commit']==BASELINE and c['namespace']==ROOT.as_posix()
    assert c['graph_ids']=={'legacy5':['real']+[f'g{s}' for s in range(1840001,1840011)]+['delay'],'brain5':['real']+[f'g{s}' for s in range(1841001,1841004)]+['delay']}
    assert c['blocks']==[dict(seed=1900001+i,input_seed=1810001+i,train_seed=1910001+i,test_seed=1920001+i) for i in range(6)]
    assert c['blocks_by_level']=={'legacy5':list(range(1900001,1900007)),'brain5':list(range(1900001,1900004))}
    assert (c['train_samples'],c['test_samples'],c['warmup'],c['lags'],c['probe_count'])==(4000,2000,200,[0,2],64)
    assert (c['gain'],c['leak'],c['carry_multiplier'],c['alpha'],c['scale_floor'])==(.9,.6,0.,1.,1e-5)
    assert (c['minimum_current_accuracy'],c['minimum_tracking_accuracy'],c['minimum_real_null_gap'])==(.9,.9,.03)
    assert not c['delay_primary_comparator'] and not c['performance_eligibility_filter'] and not c['observation_noise'] and c['new_graphs']==0
    assert c['neural_workers']==4 and c['max_seconds']==7200 and c['max_rss_bytes']==4294967296
    if smoke:
        c.update(train_samples=300,test_samples=200)
        c['graph_ids']={l:[ids[0],ids[1],ids[-1]] for l,ids in c['graph_ids'].items()}
        c['blocks_by_level']={l:seeds[:1] for l,seeds in c['blocks_by_level'].items()}
    return c

def parent_audit(c):
    p=Path(c['parent']);v=Path(c['parent_validation'])
    assert sha(p/'manifest.json')==c['parent_manifest_sha256'] and sha(v/'manifest.json')==c['parent_validation_manifest_sha256']
    check_manifest(p);check_manifest(v)
    checks=read(v/'checks.json');assert checks['all_checks_pass'] and checks['primary_ready'] and checks['graphs']==13 and checks['result_manifest_sha256']==c['parent_manifest_sha256']
    return dict(all_checks_pass=True,graphs_authenticated=13,parent_main_sha256=sha(p/'manifest.json'),parent_validation_sha256=sha(v/'manifest.json'),no_new_graphs=True)

def parent_paths(c):
    p=Path(c['parent']);v=Path(c['parent_validation']);paths=[p/n for n in ['manifest.json','source.json','config.json']]+[v/n for n in ['manifest.json','checks.json']]
    for l,seeds in [('legacy5',range(1840001,1840011)),('brain5',range(1841001,1841004))]:
        for s in seeds:paths += [p/'graphs'/f'{l}_g{s}'/n for n in ['manifest.json','audit.json','raw.npz']]
        d=p/'definitions'/l;paths += [d/n for n in ['manifest.json','definition.json','definition.npz','delay-transition.npz','cells.json']]
        paths += [d/f'prototypes-s{s}.npz' for s in range(1810001,1810007 if l=='legacy5' else 1810004)]
    return paths

def history_preserved():
    old=read(ROOT/'baseline_audit/baseline-git-blobs.json');assert old==tree(BASELINE)
    protected={n:h for n,h in old.items() if n not in OVERVIEWS};now=tree('HEAD')
    assert all(now.get(n)==h for n,h in protected.items()) and not set(git('diff','--name-only','HEAD').splitlines()).intersection(protected)
    return dict(baseline_commit=BASELINE,prior_result_identities_preserved=sum(n.startswith('results/') for n in protected),protected_baseline_paths=len(protected),
        old_scientific_files_and_failures_preserved=True,p2_outcome='FAIL',m7_outcome='FAIL',m1_outcome='assay-invalid',m4_outcome='INFEASIBLE',m8_outcome='READY',**materialized_checks(protected))

def source_record(c):
    rev=git('rev-parse','HEAD');assert subprocess.run(['git','diff','--quiet','HEAD']).returncode==0
    paths=sorted(Path('src/flying').rglob('*.py'))+[CONFIG,Path(c['protocol']),Path('docs/structural-tracking-prior-audit.md'),Path('requirements-act1-lock.txt')]
    paths += [Path('scripts')/n for n in ['structural_tracking_support.py','structural_tracking.py','verify_structural_tracking.py','audit_structural_tracking.py','run_structural_tracking_guards.py',
        'counterfactual_support.py','counterfactual_tracking.py','temporal_memory_curve.py','temporal_mechanism.py','mechanism_support.py','verify_temporal_memory_curve.py',
        'verify_structural_specificity_design.py','specificity_design_support.py','tdc_support.py','tdc_pool.py']]
    paths += [Path('tests/test_structural_tracking.py')]+sorted(Path('data/flywire_783_mb_left_kc512_s701').glob('*'))+parent_paths(c)
    cache=Path(c['cache']);paths += [cache/n for n in ['nodes.npz','brain5.npz','provenance.json']]+sorted((cache.parent/'raw').glob('*'))
    assert all(p.is_file() for p in paths)
    hashes={p.as_posix():sha(p) for p in paths};assert len(hashes)==len(paths)
    authenticate_git_sources([n for n in hashes if not n.startswith('outputs/')],hashes,rev)
    for n in [CONFIG.as_posix(),c['protocol']]:assert hashes[n]==hashlib.sha256(subprocess.check_output(['git','show',PREREGISTRATION+':'+n])).hexdigest()
    assert git('rev-parse','HEAD')==rev and subprocess.run(['git','diff','--quiet','HEAD']).returncode==0
    return dict(source_commit=rev,source_tree=git('rev-parse',rev+'^{tree}'),protocol_commit=PREREGISTRATION,all_tracked_hashes_authenticated=True,
        graph_bank_frozen=True,new_recurrent_learning=False,hashes=hashes,environment=environment())

def source_unchanged(record):
    assert git('rev-parse','HEAD')==record['source_commit'] and subprocess.run(['git','diff','--quiet','HEAD']).returncode==0
    assert all(sha(n)==h for n,h in record['hashes'].items())

def baseline_audit(out):
    c=config()
    with attempt(out,c,'m9-baseline-audit'):
        write(out/'baseline-git-blobs.json',tree(BASELINE));write(out/'parent-audit.json',parent_audit(c));write(out/'environment.json',environment())
        write(out/'audit.json',dict(all_checks_pass=True,new_neural_outcomes_generated=False,**history_preserved()))
    seal(out,dict(complete=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);baseline_audit(p.parse_args().baseline)
