"""M7 artifact/provenance audit and independent integer/noisy score replay."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import re
import numpy as np
import pandas as pd
from noise_tracking_support import config,read,write,sha,array_sha,git,tree,history_preserved,attempt,seal,PREREGISTRATION,source_record,assert_source_unchanged

WORLDS=('original','counterfactual','instantaneous_original','instantaneous_counterfactual')

def compare_csv(actual,expected,keys):
    assert actual.columns.is_unique and expected.columns.is_unique
    assert set(actual.columns)==set(expected.columns)
    assert len(actual.drop_duplicates(keys))==len(actual)
    actual=actual.sort_values(keys).reset_index(drop=True)
    expected=expected.sort_values(keys).reset_index(drop=True)
    assert len(actual)==len(expected)
    for name in expected.columns:
        np.testing.assert_array_equal(actual[name].isna(),expected[name].isna())
        present=~expected[name].isna()
        if expected[name].dtype.kind=='f':
            np.testing.assert_allclose(actual.loc[present,name],expected.loc[present,name],atol=1e-10,rtol=1e-10)
        else:np.testing.assert_array_equal(actual.loc[present,name],expected.loc[present,name])

def stage_replay(root,validation,c,smoke):
    cells=read(root/'primary-cells.json');summary=read(root/'summary.json');checks=read(validation/'checks.json')
    assert checks['all_checks_pass'] and checks['result_manifest_sha256']==sha(root/'manifest.json')
    assert len(cells)==(2 if smoke else 9) and checks['cases']==len(cells)
    raw=pd.read_csv(root/'raw-trials.csv');assert len(raw)==64*len(cells)
    blocks=pd.read_csv(root/'block-scores.csv');assert len(blocks)==len(cells)
    compare_csv(blocks,pd.DataFrame(cells),['level','seed'])
    expected_rows=[row for cell in cells for row in read(root/'noisy'/f'{cell["level"]}_s{cell["seed"]}'/'trials.json')]
    compare_csv(raw,pd.DataFrame(expected_rows),['level','seed','probe_index'])
    replay={}
    for l in c['levels']:
        selected=[]
        for b in c['blocks']:
            if b['seed'] not in c['blocks_by_level'][l]:continue
            key=f'{l}_s{b["seed"]}';noisy=root/'noisy'/key;clean=root/'clean'/key
            cell=read(noisy/'primary-cell.json')
            with np.load(clean/'case.npz',allow_pickle=False) as a,np.load(noisy/'noise.npz',allow_pickle=False) as n:
                order=np.argsort(a['root_ids'][a['observed_indices']])
                canonical=np.random.default_rng(b['noise_seed']).standard_normal((64,48))
                np.testing.assert_array_equal(n['canonical_standard_normal'],canonical)
                z=np.empty_like(canonical);z[:,order]=canonical
                np.testing.assert_array_equal(n['standard_normal'],z)
                epsilon=.1*a['scale']*z;np.testing.assert_array_equal(n['raw_noise'],epsilon)
                hashes=read(noisy/'array-hashes.json')
                assert set(n.files)==set(hashes)
                for name in n.files:assert array_sha(n[name])==hashes[name]
                predictions={}
                for w in WORLDS:
                    observed=a['trial_'+w+'_features'][:,-1]+epsilon
                    np.testing.assert_array_equal(n['observed_'+w],observed)
                    scores=((observed-a['mean'])/a['scale'])@a['coefficients_lag2']+a['intercept_lag2']
                    np.testing.assert_allclose(n['scores_'+w],scores,atol=1e-10,rtol=1e-10)
                    predictions[w]=scores.argmax(axis=1);np.testing.assert_array_equal(n['predictions_'+w],predictions[w])
                np.testing.assert_array_equal(predictions['instantaneous_original'],predictions['instantaneous_counterfactual'])
                s,r=a['original_symbols'],a['alternative_symbols'];assert np.all(s!=r)
                joint=(predictions['original']==s)&(predictions['counterfactual']==r)
                orig=predictions['original']==s;cf=predictions['counterfactual']==r
                assert cell['samples']==64 and cell['joint_correct']==int(joint.sum())
                assert cell['original_correct']==int(orig.sum()) and cell['replacement_correct']==int(cf.sum())
                passed=None if smoke else 10*int(joint.sum())>=576
                assert cell['scientific_rule_pass']==passed
                for k in ['frequency_joint_correct','current_joint_correct','instantaneous_joint_correct']:assert cell[k]==0
                clean_j=(a['predictions_original']==s)&(a['predictions_counterfactual']==r)
                assert cell['clean_joint_correct']==int(clean_j.sum())
                rows=raw[(raw.level==l)&(raw.seed==b['seed'])].sort_values('probe_index')
                np.testing.assert_array_equal(rows.joint_correct,joint)
                np.testing.assert_array_equal(rows.original_correct,orig);np.testing.assert_array_equal(rows.replacement_correct,cf)
                for w in WORLDS:np.testing.assert_array_equal(rows['prediction_'+w],predictions[w])
                np.testing.assert_array_equal(rows.clean_joint_correct,clean_j)
            independent=read(validation/'noisy'/key/'primary-cell.json');assert independent==cell
            clean_checks=read(validation/'clean'/key/'checks.json');assert clean_checks['all_checks_pass']
            assert clean_checks['old_head_independent_svd_authentications']==1
            assert clean_checks['exact_integer_primary_recalculated']
            assert read(validation/'noisy'/key/'checks.json')['all_checks_pass']
            selected.append(cell)
        assert [x['seed'] for x in selected]==c['blocks_by_level'][l]
        passed=None if smoke else all(10*x['joint_correct']>=576 for x in selected)
        assert summary['levels'][l]['registered_rule_pass']==passed
        replay[l]=dict(joint_correct=[x['joint_correct'] for x in selected],registered_rule_pass=passed)
    primary=replay[c['primary_level']]['registered_rule_pass']
    assert summary['primary_pass']==primary and checks['primary_pass']==primary
    assert summary['outcome']==('smoke' if smoke else 'PASS' if primary else 'FAIL')
    assert summary['new_trained_parameters']==0 and not summary['noise_feedback'] and not summary['noise_clipping']
    assert checks['baseline_full_state_hashes']==len(cells) and checks['four_world_window_hash_entries']==256*len(cells)
    return dict(outcome=summary['outcome'],primary_pass=primary,levels=replay,cases=len(cells),probe_pairs=64*len(cells),new_trained_parameters=0,all_csv_columns_replayed=True)

def audit(out):
    c=config();root=Path(c['namespace'])
    source=source_record(c)
    with attempt(out,c,'m7-closeout'):
        write(out/'closeout-source.json',source)
        preservation=history_preserved(c);write(out/'history.json',preservation)
        manifests=[];incomplete=[]
        for p in sorted(root.rglob('manifest.json')):
            if out.resolve() in p.resolve().parents:continue
            m=read(p);files={x.relative_to(p.parent).as_posix() for x in p.parent.rglob('*') if x.is_file() and x!=p}
            assert files==set(m['artifacts']),p
            for n,h in m['artifacts'].items():assert sha(p.parent/n)==h,(p,n)
            manifests.append(dict(path=p.as_posix(),entries=len(m['artifacts']),complete=m.get('complete')))
            if m.get('complete') is False:incomplete.append(p.as_posix())
        registry=root/'known-incomplete-attempts.json';failures=read(registry) if registry.exists() else[]
        assert sorted(x['manifest_path'] for x in failures)==sorted(incomplete)
        for f in failures:assert read(f['failure_path'])['complete'] is False
        records=[];blob_cache={}
        for p in sorted(root.rglob('source.json')):
            if out.resolve() in p.resolve().parents:continue
            source=read(p);revision=source['source_commit'];inventory=tree(revision)
            assert source['protocol_commit']==PREREGISTRATION and source['every_tracked_hash_authenticated_at_capture']
            for n,h in source['hashes'].items():
                if n in inventory:
                    blob=inventory[n]
                    if blob not in blob_cache:blob_cache[blob]=hashlib.sha256(subprocess.check_output(['git','cat-file','blob',blob])).hexdigest()
                    assert blob_cache[blob]==h
                else:assert sha(n)==h
                records.append(dict(record=p.as_posix(),path=n,source_commit=revision,sha256=h))
        replay={}
        for stage,validation in [('smoke','smoke_validation'),('main','main_validation')]:
            sc=config(stage=='smoke');assert read(root/stage/'config.json')==sc
            replay[stage]=stage_replay(root/stage,root/validation,sc,stage=='smoke')
            for directory in [stage,validation]:
                r=read(root/directory/'resources.json');t=read(root/directory/'process-tree-resources.json')
                assert r['seconds']<=7200 and t['peak_sampled_process_tree_rss_bytes']<=4294967296
        guards=read(root/'guard_tests/checks.json');assert guards['all_checks_pass'] and guards['source_freeze_rechecked']
        for n,h in guards['test_hashes'].items():assert sha(n)==h
        total_arrays=0
        for p in root.rglob('*.npz'):
            with np.load(p,allow_pickle=False) as a:
                for n in a.files:
                    total_arrays+=1
                    if a[n].dtype.kind in 'fc':assert np.isfinite(a[n]).all()
        docs=[c['protocol'],'docs/counterfactual-noise-prior-audit.md','docs/counterfactual-noise-results.md','README.md','docs/research-status.md','docs/research-roadmap.md','docs/next-work.md']
        write(out/'document-sha256.json',{n:sha(n) for n in docs})
        write(out/'manifest-checks.json',manifests);write(out/'recorded-source-checks.json',records);write(out/'exact-primary-replay.json',replay)
        write(out/'audit.json',dict(**preservation,all_checks_pass=True,outcome=replay['main']['outcome'],primary_pass=replay['main']['primary_pass'],
            manifests_checked=len(manifests),checksum_entries=sum(x['entries'] for x in manifests),recorded_source_entries=len(records),arrays_inspected=total_arrays,
            retained_incomplete_attempts=failures,exact_integer_primary=True,noise_sigma=.1,new_trained_parameters=0,physiological_noise_claim=False,source_commit=git('rev-parse','HEAD')))
        inventory=tree('HEAD');links=[]
        for name in docs[:3]:
            for link in re.findall(r'\[[^\]]+\]\(([^)]+)\)',Path(name).read_text(encoding='utf-8')):
                if '://' in link or link.startswith('#'):continue
                target=(Path(name).parent/link.split('#')[0]).resolve()
                relative=target.relative_to(Path.cwd().resolve()).as_posix()
                assert target.is_file() or relative in inventory,(name,link)
                links.append(dict(document=name,link=link,target=relative))
        write(out/'document-link-checks.json',links)
        assert_source_unchanged(source)
    seal(out,dict(complete=True,kind='m7-final-audit'))
    print(read(out/'audit.json'),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);audit(p.parse_args().out)
