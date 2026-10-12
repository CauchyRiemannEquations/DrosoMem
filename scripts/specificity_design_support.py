"""M8 preregistration, source-byte authentication and historical preservation."""
import argparse
import hashlib
from pathlib import Path
import subprocess
from counterfactual_support import tree,materialized_checks,authenticate_git_sources
from tdc_support import read,write,sha,array_sha,git,environment,attempt,seal,check_manifest

BASELINE='4185b214ae831499b556764fa288295b1dbd917b'
PREREGISTRATION='8f36a891f3b572aabc5e6865e8683cc5cbcdf1a4'
CONFIG=Path('configs/structural_specificity_design.json')
ROOT=Path('results/structural_specificity_design_v1')
OVERVIEWS={'README.md','docs/research-status.md','docs/research-roadmap.md','docs/next-work.md'}

def config(smoke=False):
    c=read(CONFIG)
    assert sha(CONFIG)==hashlib.sha256(subprocess.check_output(['git','show',PREREGISTRATION+':'+CONFIG.as_posix()])).hexdigest()
    assert c['baseline_commit']==BASELINE and c['namespace']==ROOT.as_posix()
    assert c['graph_seeds']=={'legacy5':list(range(1840001,1840011)),'brain5':list(range(1841001,1841004))}
    assert c['input_seeds']=={'legacy5':list(range(1810001,1810007)),'brain5':list(range(1810001,1810004))}
    assert c['swaps_per_delayed_edge']=={'legacy5':5,'brain5':1}
    assert (c['maximum_delayed_edge_overlap'],c['max_proposals_per_requested_swap'])==(.9,200)
    assert (c['gain'],c['drive'],c['input_fraction'],c['input_amplitude'])==(.9,.6,.1,.5)
    assert (c['input_pool_coordinates'],c['observed_coordinates'],c['delay_active_coordinates'],c['delay_lag'])==(512,48,608,2)
    assert c['rank_relative_tolerance']==1e-12 and c['new_decoder_fits']==c['source_graph_neural_trajectories']==0
    assert c['graph_workers']==1 and c['max_seconds']==7200 and c['max_rss_bytes']==4294967296
    if smoke:
        c['levels']=['legacy5'];c['graph_seeds']={'legacy5':c['graph_seeds']['legacy5'][:1]};c['input_seeds']={'legacy5':c['input_seeds']['legacy5'][:1]}
    return c

def history_preserved():
    old=read(ROOT/'baseline_audit/baseline-git-blobs.json');assert old==tree(BASELINE)
    protected={n:h for n,h in old.items() if n not in OVERVIEWS};current=tree('HEAD')
    assert all(current.get(n)==h for n,h in protected.items())
    assert not set(git('diff','--name-only','HEAD').splitlines()).intersection(protected)
    return dict(baseline_commit=BASELINE,prior_result_identities_preserved=sum(n.startswith('results/') for n in protected),
        protected_baseline_paths=len(protected),old_scientific_files_and_failures_preserved=True,p2_outcome='FAIL',m7_outcome='FAIL',m1_outcome='assay-invalid',m4_outcome='INFEASIBLE',**materialized_checks(protected))

def source_record(c):
    revision=git('rev-parse','HEAD');assert subprocess.run(['git','diff','--quiet','HEAD']).returncode==0
    paths=sorted(Path('src/flying').rglob('*.py'))+[CONFIG,Path(c['protocol']),Path('docs/structural-specificity-design-prior-audit.md'),Path('requirements-act1-lock.txt')]
    paths += [Path('scripts')/n for n in ['specificity_design_support.py','structural_specificity_design.py','verify_structural_specificity_design.py','audit_structural_specificity_design.py','run_specificity_design_guards.py','counterfactual_support.py','tdc_support.py','tdc_pool.py','verify_temporal_memory_curve.py']]
    paths += [Path('tests/test_structural_specificity_design.py')]+sorted(Path('data/flywire_783_mb_left_kc512_s701').glob('*'))
    cache=Path(c['cache']);paths += [cache/n for n in ['nodes.npz','brain5.npz','provenance.json']]+sorted((cache.parent/'raw').glob('*'))
    paths += [Path('results/counterfactual_noise_v1/final_audit_retry1/manifest.json')]
    assert all(p.is_file() for p in paths)
    hashes={p.as_posix():sha(p) for p in paths};assert len(hashes)==len(paths)
    authenticate_git_sources([n for n in hashes if not n.startswith('outputs/')],hashes,revision)
    for n in [CONFIG.as_posix(),c['protocol']]:assert hashes[n]==hashlib.sha256(subprocess.check_output(['git','show',PREREGISTRATION+':'+n])).hexdigest()
    assert git('rev-parse','HEAD')==revision and subprocess.run(['git','diff','--quiet','HEAD']).returncode==0
    return dict(source_commit=revision,source_tree=git('rev-parse',revision+'^{tree}'),protocol_commit=PREREGISTRATION,
        all_tracked_hashes_authenticated=True,new_decoder_fits=0,source_graph_neural_trajectories=0,hashes=hashes,environment=environment())

def source_unchanged(record):
    assert git('rev-parse','HEAD')==record['source_commit'] and subprocess.run(['git','diff','--quiet','HEAD']).returncode==0
    assert all(sha(n)==h for n,h in record['hashes'].items())

def baseline_audit(out):
    c=config()
    with attempt(out,c,'m8-baseline-audit'):
        write(out/'baseline-git-blobs.json',tree(BASELINE))
        paths=['results/tdc_v2/p2_main/summary.json','results/tdc_cycles_v1/main/summary.json','results/cycle_attribution_v1/main/summary.json','results/counterfactual_noise_v1/main/summary.json','results/counterfactual_noise_v1/final_audit_retry1/audit.json']
        records={n:dict(sha256=sha(n),summary=read(n)) for n in paths}
        assert records[paths[0]]['summary']['primary_pass'] is False and records[paths[3]]['summary']['primary_pass'] is False
        assert records[paths[4]]['summary']['all_checks_pass']
        check_manifest(Path('results/counterfactual_noise_v1/final_audit_retry1'))
        write(out/'prior-closed-outcomes.json',records);write(out/'environment.json',environment())
        write(out/'audit.json',dict(all_checks_pass=True,prior_neural_replay_performed=False,**history_preserved()))
    seal(out,dict(complete=True,kind='m8-baseline-audit'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);baseline_audit(p.parse_args().baseline)
