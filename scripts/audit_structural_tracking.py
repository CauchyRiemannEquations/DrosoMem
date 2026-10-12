"""M9 artifacts/provenance and separate integer criterion replay."""
import argparse
from fractions import Fraction
from pathlib import Path
import re
import numpy as np
import pandas as pd
from structural_tracking_support import config,ROOT,read,write,sha,array_sha,git,tree,authenticate_git_sources,history_preserved,source_record,source_unchanged,attempt,seal,PREREGISTRATION

def compare_csv(actual,expected,keys):
    assert actual.columns.is_unique and set(actual.columns)==set(expected.columns) and not actual.duplicated(keys).any()
    actual=actual.sort_values(keys).reset_index(drop=True);expected=expected[actual.columns].sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual,expected,check_exact=False,check_dtype=False,atol=1e-9,rtol=1e-9)

def replay_stage(root,validation,c,smoke):
    cells=read(root/'cells.json');summary=read(root/'summary.json');checks=read(validation/'checks.json')
    expected={(l,g,s) for l in c['levels'] for g in c['graph_ids'][l] for s in c['blocks_by_level'][l]}
    assert len(cells)==len(expected) and {(r['level'],r['graph_id'],r['seed']) for r in cells}==expected
    assert checks['all_checks_pass'] and checks['result_manifest_sha256']==sha(root/'manifest.json') and checks['all_csv_columns_replayed']
    trials=[];metrics=[]
    for row in cells:
        key=f'{row["level"]}_{row["graph_id"]}_s{row["seed"]}';case=root/'cases'/key
        assert row==read(case/'cell.json')==read(validation/'cases'/key/'cell.json')
        verify=read(validation/'cases'/key/'checks.json');assert verify['all_checks_pass'] and verify['predictions_exact'] and verify['independently_refit_heads']==2
        assert verify['prestates']==64 and verify['four_world_windows']==verify['four_world_finals']==256
        info=read(case/'graph.json');assert info['heads_frozen_before_test'] and not info['new_recurrent_learning'] and info['new_fitted_affine_parameters']==980
        b=next(b for b in c['blocks'] if b['seed']==row['seed'])
        with np.load(case/'case.npz',allow_pickle=False) as a:
            assert set(a.files)==set(info['array_hashes'])
            for name in a.files:
                assert array_sha(a[name])==info['array_hashes'][name]
                if a[name].dtype.kind in 'fc':assert np.isfinite(a[name]).all()
            for split in ['train','test']:
                symbols=np.random.default_rng(b[split+'_seed']).integers(0,10,c['warmup']+c[split+'_samples'],dtype=np.int64)
                np.testing.assert_array_equal(symbols,a[split+'_symbols'])
                for j,lag in enumerate([0,2]):np.testing.assert_array_equal(a['y'+split][:,j],symbols[a[split+'_times']-lag])
            assert int(np.sum(a['predictions'][:,0]==a['ytest'][:,0]))==row['current_correct']
            s,r=a['original_symbols'],a['alternative_symbols'];np.testing.assert_array_equal(r,(s+1)%10)
            anchors=c['warmup']+np.floor(np.linspace(0,c['test_samples']-3,64)).astype(np.int64);np.testing.assert_array_equal(a['anchors'],anchors)
            np.testing.assert_array_equal(s,a['test_symbols'][anchors]);np.testing.assert_array_equal(a['evaluation_times'],anchors+2)
            p=a['predictions_original'];q=a['predictions_counterfactual'];joint=(p==s)&(q==r)
            assert row['samples']==64 and row['joint_correct']==int(joint.sum()) and row['original_correct']==int((p==s).sum()) and row['replacement_correct']==int((q==r).sum())
            assert row['tracking90_pass']==(None if smoke else 10*int(joint.sum())>=576)
            for world in ['original','counterfactual','instantaneous_original','instantaneous_counterfactual']:
                scores=((a['trial_'+world+'_features'][:,-1]-a['mean'])/a['scale'])@a['coefficients'][:,10:20]+a['intercept'][10:20]
                np.testing.assert_allclose(scores,a['scores_'+world],atol=1e-9,rtol=1e-9);np.testing.assert_array_equal(scores.argmax(axis=1),a['predictions_'+world])
            for name in ['frequency','current','instantaneous']:assert row[name+'_joint_correct']==0
        trials += read(case/'trials.json');metrics += read(case/'stream-metrics.json')
    compare_csv(pd.read_csv(root/'raw-trials.csv'),pd.DataFrame(trials),['level','graph_id','seed','probe_index'])
    compare_csv(pd.read_csv(root/'stream-metrics.csv'),pd.DataFrame(metrics),['level','graph_id','seed','lag'])
    compare_csv(pd.read_csv(root/'block-scores.csv'),pd.DataFrame(cells),['level','graph_id','seed'])
    outcomes={};scores=[];paired=[]
    for l in c['levels']:
        groups={g:sorted([r for r in cells if r['level']==l and r['graph_id']==g],key=lambda r:r['seed']) for g in c['graph_ids'][l]}
        score={g:Fraction(sum(r['joint_correct'] for r in rows),64*len(rows)) for g,rows in groups.items()}
        null=[score[g] for g in groups if g not in ['real','delay']];real=score['real'];gap=real-sum(null,Fraction(0))/len(null)
        valid=all(10*r['current_correct']>=9*r['test_samples'] for g,rows in groups.items() if g!='delay' for r in rows)
        passed=None if smoke or not valid else all(real>x for x in null) and gap>=Fraction(3,100)
        outcome='smoke' if smoke else 'assay-invalid' if not valid else 'PASS' if passed else 'FAIL';entry=summary['levels'][l]
        assert entry['registered_rule_pass']==passed and entry['current_access_valid']==valid and entry['outcome']==outcome
        assert entry['real_minus_null_mean_exact']==str(gap) and entry['real_score']==float(real)
        dv=all(10*r['current_correct']>=9*r['test_samples'] for r in groups['delay']);dt=None if smoke or not dv else all(10*r['joint_correct']>=576 for r in groups['delay'])
        assert entry['delay_current_access_valid']==dv and entry['delay_tracking90_pass']==dt
        outcomes[l]=dict(outcome=outcome,registered_rule_pass=passed,real_score_exact=str(real),real_null_gap_exact=str(gap),current_access_valid=valid)
        scores += [dict(level=l,graph_id=g,score=float(x),numerator=x.numerator,denominator=x.denominator,blocks=len(groups[g])) for g,x in score.items()]
        null_ids=[g for g in groups if g not in ['real','delay']]
        paired += [dict(level=l,seed=r['seed'],real_score=r['joint_correct']/64,real_minus_null_mean=float(Fraction(r['joint_correct'],64)-sum((Fraction(groups[g][i]['joint_correct'],64) for g in null_ids),Fraction(0))/len(null_ids)),
            real_minus_delay=(r['joint_correct']-groups['delay'][i]['joint_correct'])/64) for i,r in enumerate(groups['real'])]
    compare_csv(pd.read_csv(root/'graph-scores.csv'),pd.DataFrame(scores),['level','graph_id']);compare_csv(pd.read_csv(root/'paired-differences.csv'),pd.DataFrame(paired),['level','seed'])
    assert summary==read(validation/'recomputed-summary.json') and summary['primary_pass']==outcomes[c['primary_level']]['registered_rule_pass']
    assert summary['cases']==len(cells) and summary['lag_heads']==2*len(cells) and summary['new_fitted_affine_parameters']==980*len(cells)
    assert not summary['new_recurrent_learning'] and not summary['observation_noise'] and summary['new_graphs']==0
    assert checks['train_test_full_state_hashes']==2*len(cells) and checks['independently_refit_heads']==2*len(cells)
    return dict(outcome=summary['outcome'],primary_pass=summary['primary_pass'],levels=outcomes,cases=len(cells),paired_probes=64*len(cells),lag_heads=2*len(cells),all_csv_columns_checked=True)

def audit(out):
    c=config();captured_source=source_record(c)
    with attempt(out,c,'m9-closeout'):
        write(out/'closeout-source.json',captured_source);history=history_preserved();write(out/'history.json',history)
        manifests=[];incomplete=[]
        for p in sorted(ROOT.rglob('manifest.json')):
            if out.resolve() in p.resolve().parents:continue
            m=read(p);files={q.relative_to(p.parent).as_posix() for q in p.parent.rglob('*') if q.is_file() and q!=p};assert files==set(m['artifacts']),p
            for n,h in m['artifacts'].items():assert sha(p.parent/n)==h,(p,n)
            manifests.append(dict(path=p.as_posix(),complete=m.get('complete'),entries=len(files)))
            if m.get('complete') is False:incomplete.append(p.as_posix())
        failures=read(ROOT/'known-incomplete-attempts.json');assert sorted(f['manifest_path'] for f in failures)==sorted(incomplete)
        for f in failures:assert not read(f['failure_path'])['complete']
        sources=[]
        for p in sorted(ROOT.rglob('source.json')):
            if out.resolve() in p.resolve().parents:continue
            record=read(p);assert record['protocol_commit']==PREREGISTRATION and record['all_tracked_hashes_authenticated'];inventory=tree(record['source_commit'])
            authenticate_git_sources([n for n in record['hashes'] if n in inventory],record['hashes'],record['source_commit'])
            for n,h in record['hashes'].items():
                if n not in inventory:assert sha(n)==h
            sources.append(dict(path=p.as_posix(),source_commit=record['source_commit'],hashes_checked=len(record['hashes'])))
        replay={}
        for stage,validation in [('smoke','smoke_validation'),('main','main_validation')]:
            sc=config(stage=='smoke');assert read(ROOT/stage/'config.json')==sc;replay[stage]=replay_stage(ROOT/stage,ROOT/validation,sc,stage=='smoke')
            for directory in [stage,validation]:
                resource=read(ROOT/directory/'resources.json');tree_resource=read(ROOT/directory/'process-tree-resources.json')
                assert resource['seconds']<=7200 and tree_resource['peak_sampled_process_tree_rss_bytes']<=4294967296
        guards=read(ROOT/'guard_tests/checks.json');assert guards['all_checks_pass'] and guards['source_freeze_rechecked']
        for n,h in guards['test_hashes'].items():assert sha(n)==h
        docs=[c['protocol'],'docs/structural-tracking-prior-audit.md','docs/structural-tracking-results.md','README.md','docs/research-status.md','docs/research-roadmap.md','docs/next-work.md']
        write(out/'document-sha256.json',{n:sha(n) for n in docs});write(out/'manifest-checks.json',manifests);write(out/'recorded-source-checks.json',sources);write(out/'exact-primary-replay.json',replay)
        write(out/'audit.json',dict(all_checks_pass=True,**history,outcome=replay['main']['outcome'],primary_pass=replay['main']['primary_pass'],manifests_checked=len(manifests),
            checksum_entries=sum(x['entries'] for x in manifests),cases=replay['main']['cases'],lag_heads=replay['main']['lag_heads'],paired_probes=replay['main']['paired_probes'],
            retained_incomplete_attempts=failures,exact_integer_primary=True,new_graphs=0,new_recurrent_learning=False,source_commit=captured_source['source_commit']))
        links=[];inventory=tree('HEAD')
        for name in docs[:3]:
            for link in re.findall(r'\[[^\]]+\]\(([^)]+)\)',Path(name).read_text(encoding='utf-8')):
                if '://' in link or link.startswith('#'):continue
                target=(Path(name).parent/link.split('#')[0]).resolve();relative=target.relative_to(Path.cwd().resolve()).as_posix();assert target.is_file() or relative in inventory,(name,link)
                links.append(dict(document=name,link=link,target=relative))
        write(out/'document-link-checks.json',links);source_unchanged(captured_source)
    seal(out,dict(complete=True,kind='m9-final-audit'));print(read(out/'audit.json'),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);audit(p.parse_args().out)
