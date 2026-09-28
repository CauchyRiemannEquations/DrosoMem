"""Locked ACT II paired-block analysis and prediction-control comparisons."""
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


def estimate(x):
    x=np.asarray(x,dtype=float);rng=np.random.default_rng(9299)
    draws=x[rng.integers(0,len(x),size=(10000,len(x)))].mean(axis=1)
    sd=float(x.std(ddof=1)) if len(x)>1 else 0.
    return dict(n=len(x),mean=float(x.mean()),median=float(np.median(x)),variance=sd*sd,
                bootstrap95=np.quantile(draws,[.025,.975]).tolist(),paired_dz=float(x.mean()/sd) if sd else None,
                wins=int((x>0).sum()),ties=int((x==0).sum()),losses=int((x<0).sum()))


def analyze(source,out):
    m=json.loads((source/'manifest.json').read_text());c=m['config']
    for path,digest in m['artifacts'].items():
        if core.sha256(source/path)!=digest:raise ValueError('Corrupt artifact: '+path)
    f=pd.read_csv(source/'evaluations.csv')
    assert len(f)==len(c['families'])*len(c['conditions'])*len(c['blocks'])*len(c['circuit_seeds'])
    assert not f.duplicated(['family','level','seed','circuit_seed']).any()
    diagnostic=[];dataset_ids={}
    for row in f.to_dict('records'):
        path=source/row['checkpoint'];meta=json.loads((path.parent/'dataset.json').read_text())
        dataset_ids.setdefault(row['family'],set()).add(meta['sha256'])
        with np.load(path,allow_pickle=False) as a:
            scores=score_metrics(a['symbols'],a['prediction'],a['teacher'],a['probabilities'],c)
            assert scores['exact_prefix_symbols']==row['exact_prefix_symbols']
            np.testing.assert_array_equal(a['prediction']==a['symbols'][c['prompt_length']:],a['position_accuracy'])
            assert a['features'].shape==(c['dataset']['length']-1,48)
            np.testing.assert_allclose(a['probabilities'].sum(axis=1),1,atol=1e-14)
            if row['family']=='periodic':assert row['markov1_prefix']==c['eval_length']
            if row['family']!='pi':assert pd.isna(row['pi_memory_score'])
            if row['family']=='shuffled_pi':
                pi=core.SequenceDataset(length=c['dataset']['length'],offset=c['dataset']['offset']).symbols()
                np.testing.assert_array_equal(np.sort(pi),np.sort(a['symbols']))
                np.testing.assert_array_equal(a['symbols'][:c['prompt_length']],pi[:c['prompt_length']])
            z=(a['features']-a['mean'])/a['scale'];s=np.linalg.svd(z,compute_uv=False);p=s[s>0]/s.sum()
            diagnostic.append(dict(family=row['family'],level=row['level'],seed=row['seed'],circuit_seed=row['circuit_seed'],
                   standardized_effective_rank=float(np.exp(-np.sum(p*np.log(p)))),
                   std_floor_features=int(np.count_nonzero(a['features'].std(axis=0)<1e-5))))
    for family,count in [('pi',1),('periodic',1),('random',len(c['blocks'])),('shuffled_pi',len(c['blocks']))]:
        if family in dataset_ids:assert len(dataset_ids[family])==count
    out.mkdir(parents=True,exist_ok=False)
    f.pivot(index=['family','seed','dataset_seed','circuit_seed'],columns='level',values='exact_prefix_symbols').to_csv(out/'raw-seed-table.csv')
    pd.DataFrame(diagnostic).to_csv(out/'representation.csv',index=False)
    f['best_simple_prefix']=f[['markov1_prefix','majority_prefix']].max(axis=1)
    f['control_advantage']=f.exact_prefix_symbols-f.best_simple_prefix
    blocks=f.groupby(['family','level','seed'])[['exact_prefix_symbols','control_advantage','autonomous_prefix_bits']].mean()
    blocks.to_csv(out/'seed-blocks.csv')
    condition={};contrasts={};family_contrasts={};generalization={};paired=[]
    for family in c['families']:
        condition[family]={}
        for level in c['conditions']:
            part=f[(f.family==family)&(f.level==level)];b=blocks.loc[(family,level),'exact_prefix_symbols']
            summary=estimate(b)
            for k in ['paired_dz','wins','ties','losses']:summary.pop(k)
            summary.update(raw_mean=float(part.exact_prefix_symbols.mean()),raw_median=float(part.exact_prefix_symbols.median()),
                 raw_variance=float(part.exact_prefix_symbols.var()),prefix16_fraction=float((part.exact_prefix_symbols>=16).mean()),
                 prefix32_fraction=float((part.exact_prefix_symbols>=32).mean()),full_fraction=float(part.censored.mean()),
                 mean_bits=float(part.autonomous_prefix_bits.mean()),teacher_forced_accuracy=float(part.teacher_forced_accuracy.mean()),
                 autonomous_accuracy=float(part.autonomous_accuracy.mean()),next_digit_accuracy=float(part.next_digit_accuracy.mean()),
                 markov1_mean=float(part.markov1_prefix.mean()),majority_mean=float(part.majority_prefix.mean()),
                 effective_rank=float(part.effective_rank.mean()))
            condition[family][level]=summary
        pivot=blocks.loc[family,'exact_prefix_symbols'].unstack('level')
        delta=pivot.brain1-pivot.legacy5;contrasts[family]=estimate(delta)
        required_wins = 3 if c['study'].startswith('act2-confirmation') else 4
        contrasts[family]['passes_graph_criterion']=bool(family!='periodic' and delta.mean()>=5 and (delta>0).sum()>=required_wins)
        paired.extend(dict(family=family,seed=int(seed),brain1_minus_legacy5=float(value)) for seed,value in delta.items())
    for level in c['conditions']:
        checks={}
        for family in ['random','shuffled_pi']:
            if family not in c['families']:continue
            b=blocks.loc[(family,level),'exact_prefix_symbols'];a=blocks.loc[(family,level),'control_advantage']
            checks[family]=dict(prefix16_blocks=int((b>=16).sum()),mean=float(b.mean()),control_advantage=estimate(a),
                 passed=bool(b.mean()>=16 and (b>=16).sum()>=4 and a.mean()>=5 and (a>0).sum()>=4))
        generalization[level]=dict(families=checks,limited_beyond_pi_evidence=len(checks)==2 and all(x['passed'] for x in checks.values()))
        family_contrasts[level]={}
        if 'pi' in c['families']:
            pi=blocks.loc[('pi',level),'exact_prefix_symbols']
            for family in c['families']:
                if family!='pi':family_contrasts[level][family+'-pi']=estimate(blocks.loc[(family,level),'exact_prefix_symbols']-pi)
    pd.DataFrame(paired).to_csv(out/'paired-differences.csv',index=False)
    result=dict(conditions=condition,graph_contrasts=contrasts,family_contrasts=family_contrasts,
                generalization=generalization,dataset_realizations={k:len(v) for k,v in dataset_ids.items()},
                qualifying_confirmation_families=[k for k,v in contrasts.items() if v['passes_graph_criterion']],
                general_nonpi_graph_superiority=all(contrasts.get(k,{}).get('passes_graph_criterion',False) for k in ['random','shuffled_pi']))
    core.write_json(out/'summary.json',result)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,4.6));colors={'legacy5':'#1976b5','brain1':'#e17430'}
    for i,family in enumerate(c['families']):
        for level,shift in [('legacy5',-.12),('brain1',.12)]:
            part=f[(f.family==family)&(f.level==level)]
            axes[0].scatter(np.full(len(part),i+shift),part.exact_prefix_symbols,color=colors[level],alpha=.55,s=26)
            axes[0].scatter(i+shift,part.exact_prefix_symbols.mean(),color=colors[level],marker='D',s=60,label=level if i==0 else None)
        control=condition[family]['legacy5']['markov1_mean']
        axes[0].scatter(i,control,color='black',marker='x',s=60,label='first-order control' if i==0 else None)
        est=contrasts[family];axes[1].plot([i,i],est['bootstrap95'],color='black')
        axes[1].scatter(i,est['mean'],color='black',marker='D')
        axes[1].scatter(np.full(len(c['blocks']),i),[r['brain1_minus_legacy5'] for r in paired if r['family']==family],alpha=.5)
    axes[0].set(xticks=range(len(c['families'])),xticklabels=c['families'],ylabel='Exact generated prefix (symbols)',title='Every run; diamonds show means')
    axes[0].legend(frameon=False);axes[1].axhline(0,color='gray',linestyle='--');axes[1].axhline(5,color='gray',linestyle=':')
    axes[1].set(xticks=range(len(c['families'])),xticklabels=c['families'],ylabel='brain1 − legacy5 (symbols)',title='Five paired blocks; bootstrap 95% intervals')
    fig.tight_layout();fig.savefig(out/'four-families.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,6.5),sharex=True,sharey=True)
    for ax,family in zip(axes.ravel(),c['families']):
        for level in c['conditions']:
            arrays=[]
            for relative in f[(f.family==family)&(f.level==level)].checkpoint:
                with np.load(source/relative,allow_pickle=False) as a:arrays.append(a['position_accuracy'])
            ax.plot(np.arange(1,c['eval_length']+1),np.mean(arrays,axis=0),label=level,color=colors[level],alpha=.85)
        ax.axvline(32,color='gray',linestyle=':',label='weighted prefix boundary')
        ax.set(title=family,ylim=(-.03,1.05),xlabel='Generated position',ylabel='Autonomous accuracy')
    axes[0,0].legend(frameon=False,fontsize=8)
    fig.tight_layout();fig.savefig(out/'position-accuracy.png',dpi=180);plt.close(fig)
    f.groupby(['family','level'])[['runtime_seconds','peak_process_tree_rss_bytes','active_neurons_mean','activity_sparsity',
              'mean_observed_cosine','observed_decay_ratio_32','effective_rank']].agg(['mean','min','max']).to_csv(out/'diagnostics.csv')
    core.write_json(out/'manifest.json',dict(source=str(source),source_manifest_sha256=core.sha256(source/'manifest.json'),
             script_sha256=core.sha256(__file__),files={p.name:core.sha256(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();analyze(a.source,a.out)
