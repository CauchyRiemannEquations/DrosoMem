"""Prospective cycle-free, zero-carry delayed feedforward sufficiency test."""
import argparse
from fractions import Fraction
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from flying.models.ridge import RidgeDecoder
from flying.brain.timed_reservoir import TimedReservoir
from flying.data.mushroom_body import load_roles
from flying.training import whole_brain_memory as core
from temporal_memory_curve import build,collect,controls
from temporal_mechanism import FactorReservoir,zero_prototypes,estimates,exact_score
from cycle_structure import rank_dag,graph_structure
from cycle_support import config,read,write,sha,array_sha,attempt,seal,history_preserved,source_record
from tdc_pool import pooled

def identifiers(c,level):
    if level=='brain5':
        with np.load(Path(c['cache'])/'nodes.npz',allow_pickle=False) as a:return a['ids'].copy(),a['roles'].copy()
    folder=Path('data/flywire_783_mb_left_kc512_s701')
    _,ids,_=core.load_connectome(folder);roles,_=load_roles(folder,ids)
    return np.asarray(ids,dtype=np.int64),np.asarray(roles)

def finite_certificate(model,block,depth,c,resources):
    n=model.weights.shape[0]
    a={name:np.random.default_rng(block['initial_'+name+'_seed']).standard_normal(n) for name in ['a','b']}
    suffix=np.random.default_rng(block['suffix_seed']).integers(0,10,depth+1,dtype=np.int64)
    arrays=dict(initial_a=a['a'],initial_b=a['b'],suffix_symbols=suffix);hashes={}
    for name in ['a','b']:
        prefix=np.random.default_rng(block['prefix_'+name+'_seed']).integers(0,10,c['certificate_prefix_steps'],dtype=np.int64)
        model.reset();model.state=a[name].copy();digest=hashlib.sha256()
        for j,s in enumerate(prefix):
            digest.update(model.step(int(s)).tobytes())
            if j%100==0:resources.check()
        arrays['prefix_'+name+'_symbols']=prefix;arrays['prefix_'+name+'_final_state']=model.state.copy()
        for j,s in enumerate(suffix):
            digest.update(model.step(int(s)).tobytes())
            if j%100==0:resources.check()
        arrays['final_'+name+'_state']=model.state.copy();hashes[name+'_full_state_trajectory_sha256']=digest.hexdigest()
    assert not np.array_equal(arrays['prefix_a_final_state'],arrays['prefix_b_final_state'])
    np.testing.assert_array_equal(arrays['final_a_state'],arrays['final_b_state'])
    meta=dict(all_checks_pass=True,global_dependency_depth=depth,common_suffix_steps=depth+1,prefix_steps=c['certificate_prefix_steps'],
              distinct_prefix_states=True,exact_terminal_equality=True,**hashes,array_hashes={n:array_sha(x) for n,x in arrays.items()})
    model.reset();return arrays,meta

def execute(c,block,level,arm,resources):
    base,obs,info=build(c,level,block);ids,roles=identifiers(c,level)
    original=base.weights;original_hash=core.weight_hash(original);s=c['arms'][arm];topology=None
    if s['dag']:
        used,rank=rank_dag(original,roles,ids)
        topology=graph_structure(original,used,roles,ids,obs)
        depth=topology['global_dependency_depth'];assert depth==c['expected_depth'][level]
        selected=TimedReservoir(used,base.encoder,roles,c['leak'],c['schedule'])
    else:used=original;rank=None;depth=None;selected=base
    model=FactorReservoir(selected,False,s['synaptic_history'])
    prototypes,observed=zero_prototypes(base,model,obs)
    w,k=c['warmup'],c['alphabet_size']
    arrays=dict(input_patterns=model.encoder.patterns.copy(),observed_indices=obs,zero_state_observed_prototypes=observed)
    if s['dag']:arrays['dag_rank']=rank
    if arm=='instantaneous':arrays['instant_state_prototypes']=prototypes
    traces={}
    for split in ['train','test']:
        symbols=np.random.default_rng(block[split+'_seed']).integers(0,k,w+c[split+'_samples'],dtype=np.int64)
        x,norm,last,h=collect(model,obs,symbols,resources);times=np.arange(w,len(symbols))
        arrays.update({split+'_symbols':symbols,split+'_features':x,split+'_norms':norm,split+'_final_state':last,split+'_times':times,
                       'y'+split:symbols[times[:,None]-np.asarray(c['lags'])[None,:]]})
        traces[split+'_full_state_trajectory_sha256']=h
        if arm=='instantaneous':
            np.testing.assert_array_equal(x,observed[symbols]);np.testing.assert_array_equal(last,prototypes[symbols[-1]])
    targets=np.eye(k)[arrays['ytrain']].reshape(c['train_samples'],-1)
    head=RidgeDecoder(c['alpha']).fit(arrays['train_features'][w:],targets)
    scores=head.scores(arrays['test_features'][w:]).reshape(c['test_samples'],21,k)
    arrays.update(mean=head.mean,scale=head.scale,coefficients=head.weights,intercept=head.target_mean,scores=scores,predictions=scores.argmax(axis=2))
    arrays['frequency_predictions'],arrays['current_predictions'],arrays['current_tables'],arrays['majority']=controls(arrays['train_symbols'],arrays['test_symbols'],arrays['ytrain'],arrays['ytest'],w,k)
    rows=[]
    for j,lag in enumerate(c['lags']):
        y=arrays['ytest'][:,j];n=len(y);correct=int(np.sum(arrays['predictions'][:,j]==y));freq=int(np.sum(arrays['frequency_predictions'][:,j]==y));current=int(np.sum(arrays['current_predictions'][:,j]==y))
        acc=correct/n;baseline=max(.1,freq/n,current/n)
        rows.append(dict(level=level,arm=arm,seed=block['seed'],lag=lag,samples=n,correct=correct,frequency_correct=freq,current_correct=current,
            accuracy=acc,frequency_accuracy=freq/n,current_accuracy=current/n,chance=.1,chance_adjusted=(acc-.1)/.9,baseline=baseline,baseline_excess=acc-baseline))
    info.update(arm=arm,switches=s,original_weight_sha256=original_hash,weight_sha256=core.weight_hash(used),direct_carry_multiplier=0.,
        drive_multiplier=.6,same_step_kc_to_mbon_retained=True,global_dependency_depth=depth,decoder_parameters_per_lag=490,
        zero_state_all_arm_response_equal=True,instantaneous_certificate=arm=='instantaneous',**traces,array_hashes={n:array_sha(x) for n,x in arrays.items()})
    certificate=finite_certificate(model,block,depth,c,resources) if s['dag'] else None
    assert core.weight_hash(original)==original_hash and core.weight_hash(model.weights)==info['weight_sha256']
    return arrays,rows,info,topology,certificate

def case_job(job):
    c,b,l,a,out=job['config'],job['block'],job['level'],job['arm'],Path(job['out'])
    with threadpool_limits(1):
        with attempt(out,c,'cycles-case') as resources:
            arrays,rows,graph,topology,certificate=execute(c,b,l,a,resources)
            np.savez_compressed(out/'case.npz',**arrays);write(out/'metrics.json',rows);write(out/'graph.json',graph)
            if topology is not None:write(out/'topology.json',topology)
            if certificate is not None:
                np.savez_compressed(out/'certificate.npz',**certificate[0]);write(out/'certificate.json',certificate[1])
        seal(out,dict(complete=True,block=b,level=l,arm=a))
    return dict(identity=f'{l}/{a}/s{b["seed"]}',rows=rows,resources=read(out/'resources.json'),certificate_pairs=int(certificate is not None))

def primary_rows(part):
    cells=[];excess=[]
    for row in part.sort_values('seed').itertuples():
        baseline=max(Fraction(1,10),Fraction(int(row.frequency_correct),int(row.samples)),Fraction(int(row.current_correct),int(row.samples)))
        value=Fraction(int(row.correct),int(row.samples))-baseline;excess.append(value)
        cells.append(dict(level=row.level,seed=int(row.seed),lag=2,correct=int(row.correct),samples=int(row.samples),baseline_numerator=baseline.numerator,
            baseline_denominator=baseline.denominator,excess=float(value),excess_numerator=value.numerator,excess_denominator=value.denominator,meets_margin=bool(value>=Fraction(1,10))))
    return cells,excess

def summarize(f,c,smoke):
    score_rows=[];long={};levels={};cells=[];diffs=[];curves=[]
    for (level,arm,seed),part in f[f.lag>0].groupby(['level','arm','seed']):
        score=exact_score(part);long[(level,arm,int(seed))]=score
        score_rows.append(dict(level=level,arm=arm,seed=int(seed),score=float(score),numerator=score.numerator,denominator=score.denominator))
    for level in c['levels']:
        seeds=c['blocks_by_level'][level]
        reference=f[(f.level==level)&(f.arm=='instantaneous')&(f.lag==0)]
        valid=bool(len(reference)==len(seeds) and all(Fraction(int(r.correct),int(r.samples))>=Fraction(99,100) for r in reference.itertuples()))
        rows,excess=primary_rows(f[(f.level==level)&(f.arm=='dag_synaptic')&(f.lag==2)]);cells+=rows
        assert len(rows)==len(seeds);passed=bool(all(x>=Fraction(1,10) for x in excess))
        primary=dict(lag=2,arm='dag_synaptic',excess=estimates(list(map(float,excess)),c),mean_exact=str(sum(excess,Fraction(0))/len(excess)),
            minimum_excess=float(min(excess)),minimum_excess_exact=str(min(excess)),all_blocks_meet_margin=passed,registered_rule_pass=None if smoke or not valid else passed)
        levels[level]=dict(reference_valid=valid,primary=primary,arm_long_scores={arm:estimates([float(long[(level,arm,s)]) for s in seeds],c) for arm in c['arms']})
        for lag in c['lags']:
            p=f[(f.level==level)&(f.lag==lag)].pivot(index='seed',columns='arm',values='accuracy').loc[seeds]
            diffs.append(dict(level=level,lag=lag,**estimates(p.intact_synaptic-p.dag_synaptic,c)))
    for (level,arm,lag),part in f.groupby(['level','arm','lag']):
        row=dict(level=level,arm=arm,lag=int(lag))
        for metric in ['accuracy','baseline_excess','chance_adjusted','frequency_accuracy','current_accuracy']:
            row.update({metric+'_'+name:value for name,value in estimates(part.sort_values('seed')[metric],c).items()})
        curves.append(row)
    primary=levels['legacy5']['primary']['registered_rule_pass'];outcome='smoke' if smoke else 'assay-invalid' if not levels['legacy5']['reference_valid'] else 'PASS' if primary else 'FAIL'
    summary=dict(smoke=smoke,outcome=outcome,primary_pass=primary,levels=levels,cases=len(score_rows),lag_heads=len(f),certificate_pairs=sum(len(s) for s in c['blocks_by_level'].values()),
                 criterion_unchanged=True,m1_outcome_unchanged='assay-invalid',m2_outcome_unchanged='PASS',cycle_effect_isolation_claim=False,biological_plasticity_performed=False)
    return summary,pd.DataFrame(score_rows),pd.DataFrame(cells),pd.DataFrame(diffs),pd.DataFrame(curves)

def pair_checks(out,c):
    for block in c['blocks']:
        graphs=[];streams=None
        for level in c['levels']:
            if block['seed'] not in c['blocks_by_level'][level]:continue
            anchor=out/f'{level}_intact_synaptic_s{block["seed"]}'
            with np.load(anchor/'case.npz',allow_pickle=False) as a:
                if streams is None:streams={n:a[n] for n in ['train_symbols','test_symbols','ytrain','ytest']}
                else:
                    for n,x in streams.items():np.testing.assert_array_equal(a[n],x)
                for arm in c['arms']:
                    root=out/f'{level}_{arm}_s{block["seed"]}';graphs.append(read(root/'graph.json'))
                    with np.load(root/'case.npz',allow_pickle=False) as b:
                        for n in ['input_patterns','observed_indices','train_symbols','test_symbols','ytrain','ytest','zero_state_observed_prototypes']:np.testing.assert_array_equal(a[n],b[n])
        for n in ['input_root_ids','observation_root_ids','input_mapping_sha256']:
            assert all(g[n]==graphs[0][n] for g in graphs)

def plot(f,out,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.8));colors={'intact_synaptic':'#17679e','dag_synaptic':'#188c72','instantaneous':'#8a6b9c'}
    for ax,level in zip(axes,['legacy5','brain5']):
        for arm,color in colors.items():
            part=f[(f.level==level)&(f.arm==arm)&(f.lag>0)]
            for _,b in part.groupby('seed'):ax.plot(b.lag,b.accuracy,color=color,alpha=.18,lw=.8)
            mean=part.groupby('lag').accuracy.mean();ax.plot(mean.index,mean,'o-',color=color,label=arm,markersize=3)
        ax.axhline(.1,color='gray',ls=':',label='chance')
        if level=='legacy5':ax.axvline(11.5,color='gray',ls='--',label='DAG dependency upper bound')
        ax.set(title=level,xlabel='Historical symbol lag',ylabel='Independent test accuracy',ylim=(0,1.02),xticks=[1,2,3,4,5,8,12,16,20]);ax.legend(fontsize=8)
    fig.suptitle('M3 cycle-free feedforward sufficiency - registered outcome: '+summary['outcome']);fig.tight_layout();fig.savefig(out/'cycles-tdc.png',dpi=180);plt.close(fig)

def run(out,smoke):
    c=config(smoke);source=source_record(c)
    with attempt(out,c,'cycles-smoke' if smoke else 'cycles-main'):
        write(out/'config.json',c);write(out/'source.json',source);write(out/'historical-preservation.json',history_preserved(c))
        jobs=[dict(config=c,level=l,arm=a,block=b,out=str(out/f'{l}_{a}_s{b["seed"]}')) for l in c['levels'] for b in c['blocks'] if b['seed'] in c['blocks_by_level'][l] for a in c['arms']]
        results,usage=pooled(case_job,jobs,c['neural_workers'],c,'M3 feedforward')
        f=pd.DataFrame([r for result in results for r in result['rows']]).sort_values(['level','arm','seed','lag']).reset_index(drop=True)
        assert len(f)==(126 if smoke else 567)
        summary,scores,cells,diffs,curves=summarize(f,c,smoke)
        for name,frame in [('raw-lags',f),('seed-scores',scores),('primary-cells',cells),('descriptive-differences',diffs),('curve-summary',curves)]:frame.to_csv(out/(name+'.csv'),index=False)
        write(out/'summary.json',summary);write(out/'process-tree-resources.json',usage);write(out/'job-resources.json',[dict(identity=r['identity'],resources=r['resources']) for r in results])
        assert sum(r['certificate_pairs'] for r in results)==summary['certificate_pairs']
        pair_checks(out,c);plot(f,out,summary)
        for n,h in source['hashes'].items():assert sha(n)==h,n
        history_preserved(c)
    seal(out,dict(complete=True,smoke=smoke,config=c,source_commit=source['source_commit']))
    print(dict(outcome=summary['outcome'],primary_pass=summary['primary_pass'],cases=summary['cases']),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    with threadpool_limits(1):run(a.out,a.smoke)
