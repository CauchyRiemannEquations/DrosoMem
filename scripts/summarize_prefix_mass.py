"""Paired loss-mass diagnostic, including predeclared late-decoding costs."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from flying.training import whole_brain_memory as core
from run_prefix_mass import check_artifacts,validate_pair


def estimate(values):
    x=np.asarray(values,dtype=float);rng=np.random.default_rng(13399)
    draws=x[rng.integers(0,len(x),size=(10000,len(x)))].mean(axis=1)
    sd=float(x.std(ddof=1)) if len(x)>1 else 0.
    return dict(n=len(x),mean=float(x.mean()),median=float(np.median(x)),variance=sd**2 if len(x)>1 else None,
        bootstrap95=np.quantile(draws,[.025,.975]).tolist(),paired_dz=float(x.mean()/sd) if sd else None,
        wins=int((x>0).sum()),ties=int((x==0).sum()),losses=int((x<0).sum()))


def measures(a):
    symbols=a['symbols'];truth=symbols[3:];correct=a['prediction']==truth
    teacher_correct=a['teacher'][2:]==truth
    head=core.NonlinearReadout(np.arange(48),8,0)
    head.mean=a['mean'];head.scale=a['scale'];head.parameters={k:a[k] for k in ['w1','b1','w2','b2']}
    logits=head.logits(a['features']);labels=symbols[1:]
    losses=logsumexp(logits,axis=1)-logits[np.arange(len(labels)),labels]
    weights=core.sample_weights(511,3,32,4)
    return dict(exact_prefix_symbols=core.prefix_score(truth,a['prediction']),
        early_teacher_accuracy=float(teacher_correct[:32].mean()),later_teacher_accuracy=float(teacher_correct[32:].mean()),
        early_autonomous_accuracy=float(correct[:32].mean()),later_autonomous_accuracy=float(correct[32:].mean()),
        unweighted_cross_entropy=float(losses.mean()),common_4x_cross_entropy=float(weights@losses/weights.sum()),
        early_cross_entropy=float(losses[2:34].mean()),later_cross_entropy=float(losses[34:].mean()))


def analyze(source,treatment,out):
    b=check_artifacts(source);t=check_artifacts(treatment);validate_pair(b['config'],t['config'])
    assert t['source_manifest_sha256']==core.sha256(source/'manifest.json')
    rows=[];c=t['config'];paired_values={};traces={}
    for block in c['blocks']:
        for circuit in c['circuit_seeds']:
            for level in c['conditions']:
                stem=f'random_{level}_c{circuit}_s{block["model_seed"]}_d{block["dataset_seed"]}'
                with np.load(source/stem/'checkpoint.npz',allow_pickle=False) as base,np.load(treatment/stem/'checkpoint.npz',allow_pickle=False) as changed:
                    for key in ['features','symbols','mean','scale','observed_indices','singular_values','decay_full','decay_observed','active_counts','observed_cosine']:
                        np.testing.assert_array_equal(base[key],changed[key])
                    bm=json.loads((source/stem/'manifest.json').read_text());tm=json.loads((treatment/stem/'manifest.json').read_text())
                    assert bm['graph']==tm['graph'] and bm['dataset']==tm['dataset']
                    assert tm['cached_source']['manifest_sha256']==core.sha256(source/stem/'manifest.json')
                    assert tm['cached_source']['checkpoint_sha256']==core.sha256(source/stem/'checkpoint.npz')
                    for label,a,meta in [('baseline',base,bm),('restored',changed,tm)]:
                        r=dict(level=level,seed=block['model_seed'],dataset_seed=block['dataset_seed'],circuit_seed=circuit,treatment=label,**measures(a))
                        assert r['exact_prefix_symbols']==meta['metrics']['exact_prefix_symbols']
                        r.update(teacher_forced_accuracy=meta['metrics']['teacher_forced_accuracy'],autonomous_accuracy=meta['metrics']['autonomous_accuracy'],
                            weighted_training_loss=meta['metrics']['train_loss'],prefix_weight=meta['config']['prefix_weight'],
                            majority_prefix=meta['metrics']['majority_prefix'],markov1_prefix=meta['metrics']['markov1_prefix'])
                        rows.append(r);traces.setdefault((level,label),[]).append(a['position_accuracy'])
    f=pd.DataFrame(rows);metrics=['exact_prefix_symbols','early_teacher_accuracy','later_teacher_accuracy','early_autonomous_accuracy',
        'later_autonomous_accuracy','unweighted_cross_entropy','common_4x_cross_entropy','early_cross_entropy','later_cross_entropy']
    blocks=f.groupby(['level','seed','treatment'])[metrics].mean();contrasts={};conditions={};deltas=[]
    required=3 if c['study'].endswith('confirmation') else 4
    for level in c['conditions']:
        conditions[level]={}
        for label in ['baseline','restored']:
            part=f[(f.level==level)&(f.treatment==label)]
            conditions[level][label]={k:estimate(blocks.loc[level].xs(label,level='treatment')[k]) for k in metrics}
            conditions[level][label].update(raw_prefix_median=float(part.exact_prefix_symbols.median()),
                raw_prefix_variance=float(part.exact_prefix_symbols.var()) if len(part)>1 else None,
                raw_prefix_min=int(part.exact_prefix_symbols.min()),raw_prefix_max=int(part.exact_prefix_symbols.max()))
        wide=blocks.loc[level].unstack('treatment');delta=pd.DataFrame({k:wide[k]['restored']-wide[k]['baseline'] for k in metrics})
        contrasts[level]={k:estimate(delta[k]) for k in metrics}
        prefix=delta.exact_prefix_symbols;late=delta.later_teacher_accuracy
        passes=bool(prefix.mean()>=5 and (prefix>0).sum()>=required)
        cost_free=bool(passes and late.mean()>=-.02 and (late>=-.02).sum()>=required)
        contrasts[level].update(prefix_criterion=passes,cost_free_criterion=cost_free,later_within_margin_blocks=int((late>=-.02).sum()))
        deltas.extend(dict(level=level,seed=int(seed),**row) for seed,row in delta.to_dict('index').items())
    differences=pd.DataFrame(deltas)
    bygraph=differences.pivot(index='seed',columns='level',values='exact_prefix_symbols')
    result=dict(conditions=conditions,contrasts=contrasts,qualifying_graphs=[k for k,v in contrasts.items() if v['prefix_criterion']],
        intervention_effect_brain1_minus_legacy5=estimate(bygraph.brain1-bygraph.legacy5),all_cached_invariants_exact=True,
        source_manifest_sha256=core.sha256(source/'manifest.json'),treatment_manifest_sha256=core.sha256(treatment/'manifest.json'))
    out.mkdir(parents=True,exist_ok=False);f.to_csv(out/'all-metrics.csv',index=False);blocks.to_csv(out/'seed-blocks.csv')
    f.pivot(index=['level','seed','dataset_seed','circuit_seed'],columns='treatment',values='exact_prefix_symbols').to_csv(out/'raw-seed-table.csv')
    differences.to_csv(out/'paired-differences.csv',index=False);core.write_json(out/'summary.json',result)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(13,4.3));colors={'legacy5':'#1976b5','brain1':'#e17430'}
    for i,level in enumerate(c['conditions']):
        for circuit in c['circuit_seeds']:
            for seed in [b['model_seed'] for b in c['blocks']]:
                pair=f[(f.level==level)&(f.seed==seed)&(f.circuit_seed==circuit)].set_index('treatment')
                axes[0].plot([i*3,i*3+1],pair.loc[['baseline','restored'],'exact_prefix_symbols'],'o-',color=colors[level],alpha=.4)
        for ax,metric in [(axes[1],'exact_prefix_symbols'),(axes[2],'later_teacher_accuracy')]:
            e=contrasts[level][metric];scale=100 if metric=='later_teacher_accuracy' else 1
            ax.plot([i,i],np.array(e['bootstrap95'])*scale,color=colors[level]);ax.scatter(i,e['mean']*scale,color=colors[level],marker='D')
            ax.scatter(np.full(len(c['blocks']),i),differences[differences.level==level][metric]*scale,color=colors[level],alpha=.5)
    axes[0].set(xticks=[0,1,3,4],xticklabels=['partial4x','partial restored','whole4x','whole restored'],ylabel='Exact-prefix symbols',title='Every paired run')
    axes[0].tick_params(axis='x',rotation=20)
    for ax in axes[1:]:ax.axhline(0,color='gray',ls='--');ax.set(xticks=[0,1],xticklabels=c['conditions'])
    axes[1].axhline(5,color='gray',ls=':');axes[1].set(ylabel='Restored - baseline (symbols)',title='Paired prefix gain; 95% interval')
    axes[2].axhline(-2,color='gray',ls=':');axes[2].set(ylabel='Later teacher accuracy change (pp)',title='Later-decoding cost')
    fig.tight_layout();fig.savefig(out/'prefix-mass.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(12,6),sharex=True)
    for ax,level in zip(axes,c['conditions']):
        for label in ['baseline','restored']:ax.plot(np.arange(1,510),np.mean(traces[(level,label)],axis=0),label=label)
        ax.axvline(32,color='gray',ls=':');ax.set(title=level,ylabel='Autonomous accuracy',ylim=(-.03,1.03));ax.legend(frameon=False)
    axes[-1].set_xlabel('Generated position');fig.tight_layout();fig.savefig(out/'position-accuracy.png',dpi=180);plt.close(fig)
    core.write_json(out/'manifest.json',dict(source=str(source),treatment=str(treatment),source_manifest_sha256=result['source_manifest_sha256'],
        treatment_manifest_sha256=result['treatment_manifest_sha256'],script_sha256=core.sha256(__file__),
        artifacts={p.name:core.sha256(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(dict(qualifying_graphs=result['qualifying_graphs'],contrasts={k:{m:v[m] for m in ['exact_prefix_symbols','later_teacher_accuracy','prefix_criterion','cost_free_criterion']} for k,v in contrasts.items()}),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--treatment',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();analyze(a.source,a.treatment,a.out)
