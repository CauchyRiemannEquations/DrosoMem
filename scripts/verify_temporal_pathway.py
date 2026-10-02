"""Independent crossed-state, ridge, statistics and autonomous replay audit."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from flying.training import whole_brain_memory as core
from alphabet_memory import read, check
from frozen_state_probe import Budget, fit_fold, labels, measures, verify_fold
from pathway_memory import seal
from relative_noise import read_table
from verify_allocation_noise import independent_head
import temporal_pathway as run
import feedback_noise as feedback
import fresh_coordinate_sd as fresh
import research_suite as suite


def exact_files(root):
    m = check(root)
    listed = set(m['artifacts'])
    present = {p.relative_to(root).as_posix() for p in root.rglob('*')
               if p.is_file() and p != root/'manifest.json'}
    assert listed == present
    return m


def replay_states(base, cut, symbols, union, roles):
    base.reset(); cut.reset()
    one_cut = core.TimedReservoir(cut.weights, cut.encoder, roles, cut.leak, cut.schedule)
    one_base = core.TimedReservoir(base.weights, base.encoder, roles, base.leak, base.schedule)
    states = np.empty((4, len(symbols), len(union)))
    distance = np.empty((len(symbols), 2))
    for t, sym in enumerate(symbols):
        before_base = base.state.copy(); before_cut = cut.state.copy()
        intact = base.step(int(sym)); persistent = cut.step(int(sym))
        one_cut.state = before_base.copy(); one_base.state = before_cut.copy()
        current = one_cut.step(int(sym)); history = one_base.step(int(sym))
        for j, x in enumerate([intact, current, history, persistent]):
            states[j,t] = x[union]
        distance[t] = [np.linalg.norm(intact-persistent),
                       np.linalg.norm(intact[union]-persistent[union])]
    return states, distance


def autonomous_reference(base, cut, cp, arm, horizon, roles):
    base.reset(); cut.reset()
    one_cut = core.TimedReservoir(cut.weights, cut.encoder, roles, cut.leak, cut.schedule)
    one_base = core.TimedReservoir(base.weights, base.encoder, roles, base.leak, base.schedule)
    for sym in cp['symbols'][:2]:
        base.step(int(sym)); cut.step(int(sym))
    before_base=base.state.copy(); before_cut=cut.state.copy()
    intact=base.step(int(cp['symbols'][2])); persistent=cut.step(int(cp['symbols'][2]))
    one_cut.state=before_base; one_base.state=before_cut
    current=one_cut.step(int(cp['symbols'][2])); history=one_base.step(int(cp['symbols'][2]))
    result={k:[] for k in ['features','probabilities','prediction']}
    for t in range(horizon):
        state={'intact':intact,'current_only':current,'history_only':history,'persistent':persistent}[arm]
        x=state[base.mbon].copy()
        p=independent_head(x[None,:],cp)[0]
        digit=int(p.argmax())
        result['features'].append(x); result['probabilities'].append(p); result['prediction'].append(digit)
        if t+1<horizon:
            before_base=base.state.copy(); before_cut=cut.state.copy()
            intact=base.step(digit); persistent=cut.step(digit)
            one_cut.state=before_base; one_base.state=before_cut
            current=one_cut.step(digit); history=one_base.step(digit)
    return {k:np.asarray(v) for k,v in result.items()}


def verify_case(c, root, cohort, block, ci, smoke):
    path=root/run.stem(cohort,block,ci)
    exact_files(path)
    meta=read(path/'case.json')
    assert meta['cohort']==cohort and meta['block']==block and meta['circuit_seed']==ci
    assert meta['smoke']==smoke and meta['independent_refits']==8
    base,cut,sites,union,graph=run.build(c,ci,block)
    assert graph==meta['graph']
    assert meta['site_indices']=={k:v.tolist() for k,v in sites.items()}
    folder=Path(f'data/flywire_783_mb_left_kc512_s{ci}')
    _,ids,_=core.load_connectome(folder)
    roles,_=core.load_roles(folder,ids)
    ntrain=c['smoke_train_samples'] if smoke else c['train_samples']
    ntest=c['smoke_test_samples'] if smoke else c['test_samples']
    streams={}
    for split,n in [('train',ntrain),('test',ntest)]:
        with np.load(path/f'{split}-states.npz') as z: archive=dict(z)
        expected_symbols=np.random.default_rng(block[split+'_seed']).integers(
            0,c['alphabet_size'],c['warmup']+n).astype(np.uint8)
        np.testing.assert_array_equal(archive['symbols'],expected_symbols)
        states,distance=replay_states(base,cut,expected_symbols,union,roles)
        np.testing.assert_allclose(archive['states'],states,atol=1e-12,rtol=1e-10)
        np.testing.assert_allclose(archive['distance'],distance,atol=1e-12,rtol=1e-10)
        streams[split]=archive
    ytrain=labels(streams['train']['symbols'],np.arange(c['warmup'],len(streams['train']['symbols'])),c['lags'])
    ytest=labels(streams['test']['symbols'],np.arange(c['warmup'],len(streams['test']['symbols'])),c['lags'])
    rows=[]; fits={}
    for site in c['sites']:
        for arm in c['arms']:
            xtrain=run.features_for(streams['train']['states'],arm,site,c,c['warmup'])
            xtest=run.features_for(streams['test']['states'],arm,site,c,c['warmup'])
            with np.load(path/f'{site}_{arm}_refit.npz') as z: saved=dict(z)
            rebuilt=fit_fold(xtrain,xtest,ytrain,ytest,c['alpha'])
            verify_fold(saved,c['alpha'])
            assert saved.keys()==rebuilt.keys()
            for key in saved:
                np.testing.assert_allclose(saved[key],rebuilt[key],atol=1e-12,rtol=1e-10,err_msg=key)
            fits[(site,arm)]=saved
            for result in measures([saved],c['lags']):
                rows.append(dict(cohort=cohort,seed=block['seed'],circuit_seed=ci,
                                 site=site,arm=arm,mode='refit',**result))
        intact=fits[(site,'intact')]
        for arm in c['arms']:
            x=run.features_for(streams['test']['states'],arm,site,c,c['warmup'])
            scores=(((x-intact['mean'])/intact['scale'])@intact['weights']+intact['bias']).reshape(
                len(x),len(c['lags']),c['alphabet_size'])
            prediction=scores.argmax(2)
            with np.load(path/f'{site}_{arm}_frozen.npz') as z:
                np.testing.assert_allclose(z['scores'],scores,atol=1e-12,rtol=1e-10)
                np.testing.assert_array_equal(z['prediction'],prediction)
            for j,lag in enumerate(c['lags']):
                rows.append(dict(cohort=cohort,seed=block['seed'],circuit_seed=ci,
                                 site=site,arm=arm,mode='frozen',lag=lag,
                                 test_accuracy=float(np.mean(prediction[:,j]==ytest[:,j]))))
    expected=pd.DataFrame(rows)
    saved=read_table(path/'lag-metrics.csv')
    pd.testing.assert_frame_equal(saved[expected.columns],expected,check_exact=False,atol=1e-12,rtol=1e-10)
    return rows


def verify_summary(frame,c,root):
    raw=read_table(root/'raw-lag-metrics.csv')
    pd.testing.assert_frame_equal(raw[frame.columns],frame,check_exact=False,atol=1e-12,rtol=1e-10)
    selected=frame[frame.lag.isin(c['primary_lags'])]
    circuit=selected.groupby(['cohort','seed','circuit_seed','site','arm','mode']).test_accuracy.mean().reset_index()
    blocks=circuit.groupby(['cohort','seed','site','arm','mode']).test_accuracy.mean().reset_index()
    for name,expected,keys in [('circuit-cells.csv',circuit,['cohort','seed','circuit_seed','site','arm','mode']),
                               ('seed-blocks.csv',blocks,['cohort','seed','site','arm','mode'])]:
        saved=read_table(root/name).sort_values(keys).reset_index(drop=True)
        expected=expected.sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(saved[expected.columns],expected,check_exact=False,atol=1e-12,rtol=1e-10)
    pairs=read_table(root/'paired-differences.csv')
    summary=read(root/'summary.json')
    gates={}
    for co in c['cohorts']:
        subset=pairs[(pairs.cohort==co)&(pairs.site=='MBON')&(pairs['mode']=='refit')]
        assert len(subset)==3
        for row in subset.itertuples():
            v=blocks[(blocks.cohort==co)&(blocks.seed==row.seed)&(blocks.site=='MBON')&(blocks['mode']=='refit')].set_index('arm').test_accuracy
            for arm in ['persistent','current_only','history_only']:
                np.testing.assert_allclose(getattr(row,arm+'_loss'),v['intact']-v[arm],atol=1e-12,rtol=1e-12)
        p=subset.persistent_loss.to_numpy();d=subset.current_only_loss.to_numpy();h=subset.history_only_loss.to_numpy()
        gates[co]=dict(persistent=bool(p.mean()>=c['minimum_persistent_loss'] and (p>0).all()),
                       current=bool(d.mean()>=c['minimum_current_loss'] and (d>0).all()),
                       dominance=bool(d.mean()>=c['minimum_current_fraction']*p.mean()),
                       history=bool(h.mean()<=c['maximum_history_loss']))
    assert gates==summary['gates']
    assert summary['primary_confirmed']==all(all(z.values()) for z in gates.values())
    assert summary['primary_site']=='MBON' and summary['primary_mode']=='refit'
    assert not summary['cohorts_pooled']


def verify_autonomous(c,root,budget):
    parent_root=Path(c['parent'])
    assert check(parent_root)
    frame=read_table(root/'autonomous-rollouts.csv')
    count=0;maxerr=0.
    for _,case in run.parent_cases():
        budget.check()
        parent=parent_root/suite.stem(case)
        check(parent)
        with np.load(parent/'checkpoint.npz') as z:cp=dict(z)
        head=feedback.load_head(cp,case['seed'])
        bc=read('configs/fresh_coordinate_sd.json')
        base,_,_,_,_=fresh.build(bc,case)
        folder=Path(f'data/flywire_783_mb_left_kc512_s{case["circuit_seed"]}')
        _,ids,_=core.load_connectome(folder);roles,_=core.load_roles(folder,ids)
        cut,weight=run.cut_model(base,np.asarray(roles))
        path=root/suite.stem(case)
        exact_files(path)
        target=cp['symbols'][3:]
        for arm in c['arms']:
            row=frame[(frame.seed==case['seed'])&(frame.circuit_seed==case['circuit_seed'])&(frame.arm==arm)]
            assert len(row)==1
            row=row.iloc[0]
            assert row.parent_manifest_sha256==core.sha256(parent/'manifest.json')
            assert row.parent_head_sha256==head.digest()
            assert row.cut_weight_sha256==core.weight_hash(weight)
            actual=autonomous_reference(base,cut,cp,arm,len(target),roles)
            with np.load(path/(arm+'.npz')) as z:saved=dict(z)
            np.testing.assert_array_equal(saved['prediction'],actual['prediction'])
            for key in ['features','probabilities']:
                np.testing.assert_allclose(saved[key],actual[key],atol=1e-12,rtol=1e-10)
            maxerr=max(maxerr,float(np.max(np.abs(saved['probabilities']-actual['probabilities']))))
            assert int(row.exact_prefix_symbols)==core.prefix_score(target,actual['prediction'])
            np.testing.assert_allclose(row.accuracy,np.mean(actual['prediction']==target),atol=1e-12)
            count+=1
    assert count==64==len(frame)
    return count,maxerr


@threadpool_limits.wrap(limits=1)
def verify(root,out):
    m=exact_files(root);c=m['config'];assert m['context']==run.context(c)==read(root/'source-hashes.json')
    out.mkdir(parents=True,exist_ok=False)
    budget=Budget(dict(max_seconds=c['verification_seconds'],max_rss_bytes=c['max_rss_bytes']))
    counts=dict(cases=0,state_streams=0,ridge_refits=0,autonomous_paths=0)
    try:
        status=read(root/'verification.json');smoke=status['smoke'];assert status['complete']
        panel=[('smoke',c['smoke'],701)] if smoke else [
            (co,b,ci) for co,blocks in c['cohorts'].items() for b in blocks for ci in c['circuit_seeds']]
        rows=[]
        for co,block,ci in panel:
            budget.check()
            rows.extend(verify_case(c,root,co,block,ci,smoke))
            counts['cases']+=1;counts['state_streams']+=2;counts['ridge_refits']+=8
            print(f"Verified crossed panel {counts['cases']}/{len(panel)}",flush=True)
        if not smoke:
            verify_summary(pd.DataFrame(rows),c,root)
            count,maxerr=verify_autonomous(c,root,budget)
            counts['autonomous_paths']=count
        else:maxerr=0.
        assert counts['cases']==status['cases'] and counts['ridge_refits']==status['refits']
        assert counts['autonomous_paths']==status['autonomous_paths']
        budget.check()
        core.write_json(out/'checks.json',dict(all_checks_pass=True,smoke=smoke,counts=counts,
                        result_manifest_sha256=core.sha256(root/'manifest.json'),
                        max_autonomous_probability_error=maxerr,
                        environment=core.environment(),budget=budget.close()))
        seal(out,c,run.context(c),purpose='Independent temporal-pathway audit')
    except Exception as exc:
        core.write_json(out/'failure.json',dict(error=repr(exc),counts=counts,budget=budget.close()))
        seal(out,c,run.context(c),purpose='Preserved incomplete temporal audit')
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('result',type=Path)
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args();verify(args.result,args.out)
