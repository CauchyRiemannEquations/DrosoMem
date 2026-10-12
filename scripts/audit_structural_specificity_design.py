"""M8 artifact coverage, historical/source identities and exact readiness audit."""
import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy import sparse
from specificity_design_support import config,ROOT,read,write,sha,git,tree,authenticate_git_sources,history_preserved,source_record,source_unchanged,attempt,seal,PREREGISTRATION

def csv_check(actual,expected):
    assert set(actual.columns)==set(expected.columns) and actual.columns.is_unique
    actual=actual.sort_values(['level','seed']).reset_index(drop=True);expected=expected.sort_values(['level','seed']).reset_index(drop=True)
    assert len(actual)==len(expected)==len(actual.drop_duplicates(['level','seed']))
    for n in expected:
        if expected[n].dtype.kind=='f':np.testing.assert_allclose(actual[n],expected[n],atol=1e-10,rtol=1e-10)
        else:np.testing.assert_array_equal(actual[n],expected[n])

def readiness(stage,validation,c,smoke):
    reports=read(stage/'structural-audits.json');summary=read(stage/'summary.json');checks=read(validation/'checks.json')
    assert checks['all_checks_pass'] and checks['result_manifest_sha256']==sha(stage/'manifest.json')
    assert checks['source_graph_neural_trajectories']==checks['new_decoder_fits']==0
    csv_check(pd.read_csv(stage/'graph-summary.csv'),pd.DataFrame(reports).drop(columns=['role_labels','role_block_counts']))
    levels={}
    for level in c['levels']:
        selected=sorted([a for a in reports if a['level']==level],key=lambda x:x['seed']);assert [a['seed'] for a in selected]==c['graph_seeds'][level]
        gates=[]
        for a in selected:
            key=f'{level}_g{a["seed"]}';assert read(stage/'graphs'/key/'audit.json')==a
            assert read(validation/'graphs'/key/'audit.json')==a and read(validation/'graphs'/key/'checks.json')['all_checks_pass']
            assert a['requested_swaps']==c['swaps_per_delayed_edge'][level]*a['delayed_edges']
            assert a['finite_budget_complete']==(a['accepted_swaps']==a['requested_swaps'])
            assert a['attempts']<=200*a['requested_swaps']
            gate=a['finite_budget_complete'] and 10*a['common_delayed_edges']<=9*a['delayed_edges'] and a['null_raw_sha256']!=a['original_raw_sha256']
            assert a['construction_rule_pass']==gate
            for n in ['exact_degrees_and_both_raw_weight_multisets','exact_role_blocks','exact_incoming_strength_and_factors','exact_protected_raw_and_normalized_coefficients','no_invalid_edges']:assert a[n]
            gates.append(gate)
        distinct=len({a['null_raw_sha256'] for a in selected})==len(selected)
        root=stage/'definitions'/level;cells=read(root/'cells.json');assert [x['seed'] for x in cells]==c['input_seeds'][level]
        assert cells==read(validation/'definitions'/level/'cells.json')
        d=sparse.load_npz(root/'delay-transition.npz').tocsr();assert d.nnz==48 and (d@d).nnz==24 and (d@d@d).nnz==0
        info=read(root/'definition.json');assert info['active_reference_coordinates']==608 and info['source_neurons']==d.shape[0]>=608
        assert info['idle_reference_coordinates']==info['source_neurons']-608 and info['observed_coordinates']==48 and info['decoder_parameter_budget_per_lag']==490
        map_gates=[]
        for x in cells:
            with np.load(root/f'prototypes-s{x["seed"]}.npz',allow_pickle=False) as a:
                for name in ['current','past']:
                    s=a[name+'_singular_values'];rank=int(np.sum(s>1e-12*s[0]));assert rank==x[name+'_rank']
            gate=x['current_rank']==x['past_rank']==10;assert x['numerical_affine_certificate_pass']==gate;map_gates.append(gate)
        ready=None if smoke else bool(distinct and all(gates) and all(map_gates))
        assert summary['levels'][level]['registered_ready']==ready;levels[level]=ready
    primary=levels[c['primary_level']]
    assert summary['primary_ready']==checks['primary_ready']==primary
    assert summary['outcome']==checks['outcome']==('smoke' if smoke else 'READY' if primary else 'NOT_READY')
    assert summary['new_decoder_fits']==summary['source_graph_neural_trajectories']==0 and not summary['real_wiring_superiority_tested']
    assert summary==read(validation/'recomputed-summary.json')
    return dict(outcome=summary['outcome'],primary_ready=primary,levels=levels,graph_constructions=len(reports),new_decoder_fits=0,source_graph_neural_trajectories=0)

def audit(out):
    c=config();captured_source=source_record(c)
    with attempt(out,c,'m8-closeout'):
        write(out/'closeout-source.json',captured_source);history=history_preserved();write(out/'history.json',history)
        checked=[];incomplete=[]
        for p in sorted(ROOT.rglob('manifest.json')):
            if out.resolve() in p.resolve().parents:continue
            m=read(p);present={q.relative_to(p.parent).as_posix() for q in p.parent.rglob('*') if q.is_file() and q!=p}
            assert present==set(m['artifacts']),p
            for name,h in m['artifacts'].items():assert sha(p.parent/name)==h,(p,name)
            checked.append(dict(path=p.as_posix(),entries=len(present),complete=m.get('complete')))
            if m.get('complete') is False:incomplete.append(p.as_posix())
        registry=ROOT/'known-incomplete-attempts.json';failures=read(registry) if registry.exists() else[]
        assert sorted(f['manifest_path'] for f in failures)==sorted(incomplete)
        for f in failures:assert not read(f['failure_path'])['complete']
        sources=[]
        for p in sorted(ROOT.rglob('source.json')):
            if out.resolve() in p.resolve().parents:continue
            record=read(p);assert record['protocol_commit']==PREREGISTRATION and record['all_tracked_hashes_authenticated']
            inventory=tree(record['source_commit']);tracked=[n for n in record['hashes'] if n in inventory]
            authenticate_git_sources(tracked,record['hashes'],record['source_commit'])
            for n,h in record['hashes'].items():
                if n not in inventory:assert sha(n)==h
            sources.append(dict(path=p.as_posix(),source_commit=record['source_commit'],hashes_checked=len(record['hashes'])))
        replay={}
        for stage,validation in [('smoke','smoke_validation'),('main','main_validation')]:
            sc=config(stage=='smoke');assert read(ROOT/stage/'config.json')==sc
            replay[stage]=readiness(ROOT/stage,ROOT/validation,sc,stage=='smoke')
            for directory in [stage,validation]:
                resources=read(ROOT/directory/'resources.json');usage=read(ROOT/directory/'process-tree-resources.json')
                assert resources['seconds']<=7200 and usage['peak_sampled_process_tree_rss_bytes']<=4294967296
        guards=read(ROOT/'guard_tests/checks.json');assert guards['all_checks_pass'] and guards['source_freeze_rechecked']
        for n,h in guards['test_hashes'].items():assert sha(n)==h
        arrays=0
        for p in ROOT.rglob('*.npz'):
            with np.load(p,allow_pickle=False) as a:
                for name in a.files:
                    arrays+=1
                    if a[name].dtype.kind in 'fc':assert np.isfinite(a[name]).all()
        docs=[c['protocol'],'docs/structural-specificity-design-prior-audit.md','docs/structural-specificity-design-results.md','README.md','docs/research-status.md','docs/research-roadmap.md','docs/next-work.md']
        write(out/'document-sha256.json',{n:sha(n) for n in docs})
        write(out/'manifest-checks.json',checked);write(out/'recorded-source-checks.json',sources);write(out/'exact-readiness-replay.json',replay)
        write(out/'audit.json',dict(all_checks_pass=True,**history,outcome=replay['main']['outcome'],primary_ready=replay['main']['primary_ready'],
            manifests_checked=len(checked),checksum_entries=sum(x['entries'] for x in checked),arrays_inspected=arrays,retained_incomplete_attempts=failures,
            exact_integer_overlap=True,new_decoder_fits=0,source_graph_neural_trajectories=0,source_commit=captured_source['source_commit']))
        inventory=tree('HEAD');links=[]
        for name in docs[:3]:
            for link in re.findall(r'\[[^\]]+\]\(([^)]+)\)',Path(name).read_text(encoding='utf-8')):
                if '://' in link or link.startswith('#'):continue
                target=(Path(name).parent/link.split('#')[0]).resolve();relative=target.relative_to(Path.cwd().resolve()).as_posix()
                assert target.is_file() or relative in inventory,(name,link)
                links.append(dict(document=name,link=link,target=relative))
        write(out/'document-link-checks.json',links);source_unchanged(captured_source)
    seal(out,dict(complete=True,kind='m8-final-audit'));print(read(out/'audit.json'),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);audit(p.parse_args().out)
