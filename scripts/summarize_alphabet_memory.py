"""Registered within-K paired comparisons; cross-K bits remain a score."""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits
from flying.training import whole_brain_memory as core
from alphabet_memory import read,check


def estimate(values):
    x=np.asarray(values,dtype=float);rng=np.random.default_rng(25399)
    draws=x[rng.integers(0,len(x),size=(10000,len(x)))].mean(axis=1);sd=float(x.std(ddof=1)) if len(x)>1 else 0.
    return dict(n=len(x),mean=float(x.mean()),median=float(np.median(x)),variance=sd**2 if len(x)>1 else None,
        bootstrap95=np.quantile(draws,[.025,.975]).tolist(),paired_dz=float(x.mean()/sd) if sd else None,
        wins=int((x>0).sum()),ties=int((x==0).sum()),losses=int((x<0).sum()))


@threadpool_limits.wrap(limits=1)
def analyze(source,out):
    m=check(source);c=m['config'];rows=[];interface={};sequences={};heads={};mapping={}
    for b in c['blocks']:
        for circuit in c['circuit_seeds']:
            for level in c['conditions']:
                stem=f'{level}_c{circuit}_s{b["model_seed"]}'
                for k in c['alphabet_sizes']:
                    p=source/stem/f'k{k}';meta=check(p);r=dict(meta['metrics']);g=meta['graph']
                    key=(b['model_seed'],circuit,k);mapped={n:g[n] for n in ['input_root_ids','observation_root_ids','input_mapping_sha256','eligible_input_root_ids','stimulated_per_symbol']}
                    assert interface.setdefault(key,mapped)==mapped
                    dkey=(b['dataset_seed'],k);assert sequences.setdefault(dkey,meta['dataset']['sha256'])==meta['dataset']['sha256']
                    assert r['readout_parameters']==392+9*k
                    with np.load(p/'checkpoint.npz',allow_pickle=False) as a:
                        score=core.prefix_score(a['symbols'][3:],a['prediction']);assert score==r['exact_prefix_symbols']
                        assert r['autonomous_prefix_bits']==score*np.log2(k)
                        assert a['probabilities'].shape==(125,k) and a['prediction'].min()>=0 and a['prediction'].max()<k
                        headkey=(level,circuit,b['model_seed']);heads[(headkey,k)]=a['prediction'].copy()
                    controls=read(p/'controls.json');best=max(controls['predictions'][n]['exact_prefix_symbols'] for n in ['majority','markov1'])
                    assert r['control_excess']==score-best
                    mapping.setdefault((b['model_seed'],circuit,level),{})[k]=g
                    r.update(stimulated_union_count=g['stimulated_union_count'],neurons=g['neurons'],edges=g['edges'])
                    rows.append(r)
    for group in mapping.values():
        for k in c['alphabet_sizes']:
            assert group[k]['input_root_ids']==group[16]['input_root_ids'][:k]
            assert group[k]['eligible_input_root_ids']==group[16]['eligible_input_root_ids']
    f=pd.DataFrame(rows);metrics=['exact_prefix_symbols','autonomous_prefix_bits','prefix_fraction','teacher_forced_accuracy','autonomous_accuracy','control_excess','full_completion','effective_rank','observed_mean_abs','mean_observed_cosine','observed_decay_ratio_32']
    blocks=f.groupby(['alphabet_size','level','seed'])[metrics].mean();conditions={};contrasts={};paired=[];qualified=[]
    required=3 if c['study'].endswith('confirmation') else 4
    for k in c['alphabet_sizes']:
        conditions[str(k)]={}
        for level in c['conditions']:
            b=blocks.loc[k,level];r=f[(f.alphabet_size==k)&(f.level==level)]
            conditions[str(k)][level]={metric:estimate(b[metric]) for metric in metrics}
            conditions[str(k)][level].update(raw_prefix_median=float(r.exact_prefix_symbols.median()),
                raw_prefix_variance=float(r.exact_prefix_symbols.var()) if len(r)>1 else None,
                recall_gate=bool(b.exact_prefix_symbols.mean()>=8 and b.control_excess.mean()>=5 and (b.control_excess>0).sum()>=required))
        delta=blocks.loc[k,'brain1']-blocks.loc[k,'legacy5'];contrasts[str(k)]={metric:estimate(delta[metric]) for metric in metrics}
        gate=bool(delta.exact_prefix_symbols.mean()>=5 and (delta.exact_prefix_symbols>0).sum()>=required)
        contrasts[str(k)]['whole_brain_gate']=gate
        if gate:qualified.append(k)
        paired.extend(dict(alphabet_size=k,seed=int(seed),**row) for seed,row in delta.to_dict('index').items())
    endpoint={}
    for level in c['conditions']:
        d=blocks.loc[16,level].prefix_fraction-blocks.loc[2,level].prefix_fraction
        endpoint[level]=dict(**estimate(d),load_drop_gate=bool(d.mean()<=-.1 and (d<0).sum()>=required))
    result=dict(study=c['study'],conditions=conditions,contrasts=contrasts,alphabet_endpoint=endpoint,qualifying_k=qualified,
        confirmation_triggered=bool(qualified and len(c['blocks'])==5),source_manifest_sha256=core.sha256(source/'manifest.json'),
        matched_interfaces_and_sequences=True,nested_symbol_patterns=True)
    out.mkdir(parents=True,exist_ok=False);f.to_csv(out/'raw-seed-table.csv',index=False);blocks.to_csv(out/'seed-blocks.csv')
    pd.DataFrame(paired).to_csv(out/'paired-differences.csv',index=False);core.write_json(out/'summary.json',result)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(14,4.2));x=np.arange(4);colors={'legacy5':'#1976b5','brain1':'#e17430'}
    for level,color in colors.items():
        for ax,metric in [(axes[0],'exact_prefix_symbols'),(axes[1],'autonomous_prefix_bits')]:
            means=[conditions[str(k)][level][metric]['mean'] for k in c['alphabet_sizes']]
            lo=[conditions[str(k)][level][metric]['bootstrap95'][0] for k in c['alphabet_sizes']];hi=[conditions[str(k)][level][metric]['bootstrap95'][1] for k in c['alphabet_sizes']]
            ax.plot(x,means,'o-',color=color,label=level);ax.fill_between(x,lo,hi,color=color,alpha=.13)
        for _,g in f[f.level==level].groupby(['seed','circuit_seed']):axes[0].plot(x,g.set_index('alphabet_size').loc[c['alphabet_sizes'],'exact_prefix_symbols'],color=color,alpha=.1)
    d=pd.DataFrame(paired)
    for i,k in enumerate(c['alphabet_sizes']):
        e=contrasts[str(k)]['exact_prefix_symbols'];axes[2].scatter(np.full(len(c['blocks']),i),d[d.alphabet_size==k].exact_prefix_symbols,color='#7055a2',alpha=.45)
        axes[2].plot([i,i],e['bootstrap95'],color='#7055a2');axes[2].scatter(i,e['mean'],color='#7055a2',marker='D')
    for ax in axes:ax.set(xticks=x,xticklabels=c['alphabet_sizes'],xlabel='Alphabet size K (categorical spacing)')
    axes[0].axhline(125,color='gray',ls=':');axes[0].set(ylabel='Exact-prefix symbols',title='All runs; mean and 95% block interval',ylim=(-2,130));axes[0].legend(frameon=False)
    axes[1].set(ylabel='Recalled-prefix bits: L log2(K)',title='A score, not formal memory capacity');axes[1].legend(frameon=False)
    axes[2].axhline(0,color='gray',ls='--');axes[2].axhline(5,color='gray',ls=':');axes[2].set(ylabel='Whole - partial (symbols)',title='Within-K paired differences')
    fig.tight_layout();fig.savefig(out/'alphabet-curve.png',dpi=180);plt.close(fig)
    core.write_json(out/'manifest.json',dict(source=str(source),source_manifest_sha256=result['source_manifest_sha256'],script_sha256=core.sha256(__file__),
        artifacts={p.name:core.sha256(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(dict(qualifying_k=qualified,confirmation_triggered=result['confirmation_triggered'],means={k:{g:p['exact_prefix_symbols']['mean'] for g,p in v.items()} for k,v in conditions.items()}),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();analyze(a.source,a.out)
