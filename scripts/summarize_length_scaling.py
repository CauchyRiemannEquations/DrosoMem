"""Predeclared paired-block length curves; no condition or seed selection."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flying.training import whole_brain_memory as core
from flying.training.sequence_memory import score_metrics
from run_length_scaling import length_config,validate_spec


def estimate(values,spec):
    x=np.asarray(values,dtype=float)
    rng=np.random.default_rng(spec['bootstrap_seed'])
    draws=x[rng.integers(0,len(x),size=(spec['bootstrap_resamples'],len(x)))].mean(axis=1)
    sd=float(x.std(ddof=1)) if len(x)>1 else 0.
    return dict(n=len(x),mean=float(x.mean()),median=float(np.median(x)),variance=sd**2,
        bootstrap95=np.quantile(draws,[.025,.975]).tolist(),paired_dz=float(x.mean()/sd) if sd else None,
        wins=int((x>0).sum()),ties=int((x==0).sum()),losses=int((x<0).sum()))


def analyze(source,out):
    root=json.loads((source/'manifest.json').read_text());assert root['complete']
    spec=root['config'];validate_spec(spec);lengths=spec['lengths'];levels=spec['base']['conditions']
    for path,digest in root['artifacts'].items():assert core.sha256(source/path)==digest,path
    frames=[];reference={};mappings={};diagnostics=[]
    for n in lengths:
        c=length_config(spec,n)
        frame=pd.read_csv(source/f'n{n}'/'evaluations.csv')
        assert len(frame)==len(c['blocks'])*len(c['circuit_seeds'])*len(levels)
        assert not frame.duplicated(['level','seed','circuit_seed']).any()
        frame['length']=n;frame['horizon']=n-3
        frame['normalized_prefix']=frame.exact_prefix_symbols/(n-3)
        frame['prefix_weight_mass']=4*c['prefix_window']/((n-1)+3*c['prefix_window'])
        frame['weight_sum']=(n-1)+3*c['prefix_window']
        frame['best_simple_prefix']=frame[['majority_prefix','markov1_prefix']].max(axis=1)
        frame['control_advantage']=frame.exact_prefix_symbols-frame.best_simple_prefix
        for row in frame.to_dict('records'):
            path=source/f'n{n}'/row['checkpoint'];m=json.loads((path.parent/'manifest.json').read_text())
            assert m['config']==c
            key=(row['seed'],row['circuit_seed'])
            mapping={k:m['graph'][k] for k in ['input_root_ids','observation_root_ids','input_mapping_sha256']}
            assert mappings.setdefault(key,mapping)==mapping
            key=(*key,row['level'])
            with np.load(path,allow_pickle=False) as a:
                scores=score_metrics(a['symbols'],a['prediction'],a['teacher'],a['probabilities'],c)
                assert scores['exact_prefix_symbols']==row['exact_prefix_symbols']
                assert scores['censored']==row['censored'] and pd.isna(row['pi_memory_score'])
                np.testing.assert_array_equal(a['position_accuracy'],a['prediction']==a['symbols'][3:])
                assert a['features'].shape==(n-1,48)
                if key in reference:
                    previous=reference[key]
                    np.testing.assert_array_equal(a['symbols'][:len(previous['symbols'])],previous['symbols'])
                    np.testing.assert_array_equal(a['features'][:len(previous['features'])],previous['features'])
                reference[key]=dict(symbols=a['symbols'],features=a['features'])
                z=(a['features']-a['mean'])/a['scale'];s=np.linalg.svd(z,compute_uv=False);p=s[s>0]/s.sum()
                diagnostics.append(dict(length=n,level=row['level'],seed=row['seed'],circuit_seed=row['circuit_seed'],
                    standardized_effective_rank=float(np.exp(-np.sum(p*np.log(p)))),
                    std_floor_features=int((a['features'].std(axis=0)<1e-5).sum())))
        frame['checkpoint']=[f'n{n}/{p}' for p in frame.checkpoint];frames.append(frame)
    f=pd.concat(frames,ignore_index=True)
    blocks=f.groupby(['length','level','seed'])[['exact_prefix_symbols','normalized_prefix','control_advantage']].mean()
    required=3 if spec['base']['study'].endswith('confirmation') else 4
    conditions={};contrasts={};rows=[]
    for n in lengths:
        conditions[str(n)]={}
        for level in levels:
            part=f[(f.length==n)&(f.level==level)];b=blocks.loc[(n,level)]
            conditions[str(n)][level]=dict(prefix=estimate(b.exact_prefix_symbols,spec),normalized=estimate(b.normalized_prefix,spec),
                raw_median=float(part.exact_prefix_symbols.median()),raw_variance=float(part.exact_prefix_symbols.var()) if len(part)>1 else None,
                raw_min=int(part.exact_prefix_symbols.min()),raw_max=int(part.exact_prefix_symbols.max()),
                completed_runs=int(part.censored.sum()),total_runs=len(part),
                operational_collapse=bool(b.normalized_prefix.mean()<.5 and (b.normalized_prefix<.5).sum()>=required),
                collapse_blocks=int((b.normalized_prefix<.5).sum()),
                teacher_forced_accuracy=float(part.teacher_forced_accuracy.mean()),
                next_digit_accuracy=float(part.next_digit_accuracy.mean()),autonomous_accuracy=float(part.autonomous_accuracy.mean()),
                mean_bits=float(part.autonomous_prefix_bits.mean()),control_advantage=estimate(b.control_advantage,spec),
                markov1_mean=float(part.markov1_prefix.mean()),majority_mean=float(part.majority_prefix.mean()),
                prefix_weight_mass=float(part.prefix_weight_mass.iloc[0]),weight_sum=int(part.weight_sum.iloc[0]))
        pivot=blocks.loc[n,'exact_prefix_symbols'].unstack('level');delta=pivot.brain1-pivot.legacy5
        contrasts[str(n)]=estimate(delta,spec)
        contrasts[str(n)]['qualifies']=bool(delta.mean()>=5 and (delta>0).sum()>=required)
        rows.extend(dict(length=n,seed=int(seed),brain1_minus_legacy5=float(v)) for seed,v in delta.items())
    paired=pd.DataFrame(rows);aggregate=estimate(paired.groupby('seed').brain1_minus_legacy5.mean(),spec)
    adjacent=[[a,b] for a,b in zip(lengths[:-1],lengths[1:]) if contrasts[str(a)]['qualifies'] and contrasts[str(b)]['qualifies']]
    graph_pass=bool(aggregate['mean']>=5 and aggregate['wins']>=required and adjacent)
    length_effect={}
    for level in levels:
        drop=blocks.loc[(lengths[0],level),'normalized_prefix']-blocks.loc[(lengths[-1],level),'normalized_prefix']
        collapse=[n for n in lengths if conditions[str(n)][level]['operational_collapse']]
        first=collapse[0] if collapse else None
        length_effect[level]=dict(drop_first_to_last=estimate(drop,spec),
            supports_length_drop=bool(drop.mean()>=.25 and (drop>0).sum()>=required),
            collapse_lengths=collapse,first_collapse_length=first,left_censored=first==lengths[0],
            later_noncollapse_lengths=[n for n in lengths if first is not None and n>first and n not in collapse])
    result=dict(conditions=conditions,graph_contrasts=contrasts,aggregate_graph_contrast=aggregate,
        qualifying_adjacent_lengths=adjacent,graph_criterion_passed=graph_pass,length_effect=length_effect,
        nested_raw_features_exact=True,matched_input_observation_ids=True,
        worker_wall_seconds_sum=float(f.wall_seconds.sum()),maximum_sampled_rss_bytes=int(f.peak_process_tree_rss_bytes.max()))
    out.mkdir(parents=True,exist_ok=False)
    f.to_csv(out/'all-metrics.csv',index=False)
    f.pivot(index=['length','seed','dataset_seed','circuit_seed'],columns='level',values='exact_prefix_symbols').to_csv(out/'raw-seed-table.csv')
    blocks.to_csv(out/'seed-blocks.csv');paired.to_csv(out/'paired-differences.csv',index=False)
    diagnostic=pd.DataFrame(diagnostics);diagnostic.to_csv(out/'representation.csv',index=False)
    f.groupby(['length','level'])[['effective_rank','active_neurons_mean','activity_sparsity','mean_observed_cosine',
        'observed_decay_ratio_32','train_loss','runtime_seconds','peak_process_tree_rss_bytes']].agg(['mean','min','max']).to_csv(out/'diagnostics.csv')
    core.write_json(out/'summary.json',result)
    colors={'legacy5':'#1976b5','brain1':'#e17430'}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(15,4.4))
    for level in levels:
        for ax,key in zip(axes[:2],['exact_prefix_symbols','normalized_prefix']):
            means=[];low=[];high=[]
            for i,n in enumerate(lengths):
                v=blocks.loc[(n,level),key];e=estimate(v,spec)
                means.append(e['mean']);low.append(e['bootstrap95'][0]);high.append(e['bootstrap95'][1])
                raw=f[(f.length==n)&(f.level==level)][key]
                ax.scatter(np.full(len(raw),i+(-.07 if level=='legacy5' else .07)),raw,color=colors[level],alpha=.25,s=16)
            ax.plot(range(len(lengths)),means,'o-',color=colors[level],label=level)
            ax.fill_between(range(len(lengths)),low,high,color=colors[level],alpha=.12)
    axes[0].plot(range(len(lengths)),[n-3 for n in lengths],'--',color='gray',label='evaluation horizon')
    axes[0].set(ylabel='Exact-prefix symbols',title='All runs; paired-block mean and 95% interval')
    axes[0].legend(frameon=False,fontsize=8)
    axes[1].axhline(.5,color='gray',ls=':');axes[1].set(ylabel='Prefix / evaluation horizon',ylim=(-.04,1.06),title='Normalized recall (censored at 1)')
    for i,n in enumerate(lengths):
        e=contrasts[str(n)];axes[2].plot([i,i],e['bootstrap95'],color='black');axes[2].scatter(i,e['mean'],color='black',marker='D')
        axes[2].scatter(np.full(len(spec['base']['blocks']),i),paired[paired.length==n].brain1_minus_legacy5,alpha=.4)
    axes[2].axhline(0,color='gray',ls='--');axes[2].axhline(5,color='gray',ls=':')
    axes[2].set(ylabel='brain1 - legacy5 (symbols)',title='Paired graph differences')
    for ax in axes:ax.set(xticks=range(len(lengths)),xticklabels=lengths,xlabel='Training sequence length')
    fig.tight_layout();fig.savefig(out/'length-scaling.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(len(lengths),1,figsize=(12,2.2*len(lengths)),squeeze=False)
    for ax,n in zip(axes[:,0],lengths):
        for level in levels:
            accuracy=[]
            for p in f[(f.length==n)&(f.level==level)].checkpoint:
                with np.load(source/p,allow_pickle=False) as a:accuracy.append(a['position_accuracy'])
            ax.plot(np.arange(1,n-2),np.mean(accuracy,axis=0),color=colors[level],label=level)
        ax.axvline(min(32,n-3),color='gray',ls=':');ax.set(title=f'N={n}',ylabel='Autonomous accuracy',ylim=(-.03,1.03))
    axes[0,0].legend(frameon=False);axes[-1,0].set_xlabel('Generated position')
    fig.tight_layout();fig.savefig(out/'position-accuracy.png',dpi=180);plt.close(fig)
    core.write_json(out/'manifest.json',dict(source=str(source),source_manifest_sha256=core.sha256(source/'manifest.json'),
        script_sha256=core.sha256(__file__),files={p.name:core.sha256(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(dict(graph_criterion_passed=graph_pass,length_effect=length_effect),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();analyze(a.source,a.out)
