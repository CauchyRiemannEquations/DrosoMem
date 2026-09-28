"""Registered arm contrasts; graph contrasts remain descriptive."""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits
from alphabet_memory import read,check
from context_memory import estimate
from flying.training import whole_brain_memory as core


@threadpool_limits.wrap(limits=1)
def analyze(source,out):
    m=check(source); c=m['config']; seq=check(Path(c['sequences']))
    rows=[]; interfaces={}; datasets={}; resources=[]
    for b in c['blocks']:
        for circuit in c['circuit_seeds']:
            for level in c['conditions']:
                group=source/f'{level}_c{circuit}_s{b["model_seed"]}'
                resources.append(read(group/'resources.json'))
                for arm in c['arms']:
                    p=group/arm; meta=check(p); r=meta['metrics']; g=meta['graph']
                    interface={k:g[k] for k in ['input_root_ids','observation_root_ids','input_mapping_sha256','eligible_input_root_ids','stimulated_per_symbol']}
                    key=(b['model_seed'],circuit); assert interfaces.setdefault(key,interface)==interface
                    key=(b['dataset_seed'],arm); assert datasets.setdefault(key,meta['dataset'])==meta['dataset']
                    assert r['readout_parameters']==428
                    with np.load(p/'checkpoint.npz',allow_pickle=False) as a:
                        assert core.prefix_score(a['symbols'][3:],a['prediction'])==r['exact_prefix_symbols']
                        assert np.bincount(a['symbols'],minlength=4).tolist()==[32]*4
                        assert a['symbols'][:3].tolist()==[0,1,0]
                    controls=read(Path(c['sequences'])/f'd{b["dataset_seed"]}'/f'{arm}_m3.json')['metrics']
                    diagnostics=read(Path(c['sequences'])/f'd{b["dataset_seed"]}'/'diagnostics.json')[arm]
                    rows.append(dict(**r,context3_prefix=controls['exact_prefix_symbols'],context3_floor=controls['ambiguity_error_floor'],
                        early_floor=diagnostics['early_ambiguity']['ambiguity_error_floor'],late_floor=diagnostics['late_ambiguity']['ambiguity_error_floor']))
    f=pd.DataFrame(rows); metrics=['exact_prefix_symbols','autonomous_prefix_bits','teacher_forced_accuracy','autonomous_accuracy',
        'effective_rank','observed_mean_abs','observed_decay_ratio_32','context3_prefix','context3_floor','early_floor','late_floor']
    blocks=f.groupby(['level','arm','seed'])[metrics].mean()
    cells={}; contrasts={}; qualifying=[]; pairs=[]
    required=3 if c['study'].endswith('confirmation') else 4
    for level in c['conditions']:
        cells[level]={arm:{metric:estimate(blocks.loc[level,arm][metric],c) for metric in metrics} for arm in c['arms']}
        delta=blocks.loc[level,'low']-blocks.loc[level,'high']
        contrasts[level]={metric:estimate(delta[metric],c) for metric in metrics}
        passed=bool(delta.exact_prefix_symbols.mean()>=10 and (delta.exact_prefix_symbols>0).sum()>=required)
        contrasts[level]['neural_gate']=passed
        if passed:qualifying.append(level)
        pairs.extend(dict(level=level,seed=int(seed),**row) for seed,row in delta.to_dict('index').items())
    graph={arm:{metric:estimate(blocks.loc['brain1',arm][metric]-blocks.loc['legacy5',arm][metric],c) for metric in metrics} for arm in c['arms']}
    interaction={metric:estimate((blocks.loc['brain1','low'][metric]-blocks.loc['brain1','high'][metric])-(blocks.loc['legacy5','low'][metric]-blocks.loc['legacy5','high'][metric]),c) for metric in metrics}
    out.mkdir(parents=True,exist_ok=False); f.to_csv(out/'raw-seed-table.csv',index=False); blocks.to_csv(out/'seed-blocks.csv')
    pd.DataFrame(pairs).to_csv(out/'paired-differences.csv',index=False)
    summary=dict(study=c['study'],conditions=cells,low_minus_high=contrasts,whole_minus_partial=graph,interaction=interaction,
        qualifying_graphs=qualifying,confirmation_triggered=bool(qualifying and c['study'].endswith('main')),
        construction_pairs=seq['pairs'],matched_interfaces=True,
        resources=dict(group_wall_seconds_sum=sum(r['wall_seconds'] for r in resources),peak_group_rss_bytes=max(r['peak_process_tree_rss_bytes'] for r in resources)))
    core.write_json(out/'summary.json',summary)
    fig,axes=plt.subplots(1,3,figsize=(13.5,4.2))
    for level,color in [('legacy5','#1976b5'),('brain1','#e17430')]:
        for seed in sorted(f.seed.unique()):
            values=[blocks.loc[level,arm,seed]['exact_prefix_symbols'] for arm in ['high','low']]
            axes[0].plot([0,1],values,'o-',color=color,alpha=.3)
        axes[0].plot([0,1],[cells[level][arm]['exact_prefix_symbols']['mean'] for arm in ['high','low']],'o-',color=color,lw=3,label=level)
        e=contrasts[level]['exact_prefix_symbols'];x=c['conditions'].index(level)
        axes[1].scatter(np.full(len(c['blocks']),x),[r['exact_prefix_symbols'] for r in pairs if r['level']==level],color=color,alpha=.4)
        axes[1].plot([x,x],e['bootstrap95'],color=color); axes[1].scatter(x,e['mean'],marker='D',color=color)
    for order in [1,2,3,4,5,8]:
        values=[np.mean([read(Path(c['sequences'])/f'd{b["dataset_seed"]}'/f'{arm}_m{order}.json')['metrics']['exact_prefix_symbols'] for b in c['blocks']]) for arm in ['high','low']]
        axes[2].plot([0,1],values,'o-',label=f'context {order}')
    axes[0].set(xticks=[0,1],xticklabels=['High conflict','Low conflict'],ylabel='Exact-prefix symbols',title='Paired block means',ylim=(-2,130));axes[0].legend()
    axes[1].set(xticks=[0,1],xticklabels=c['conditions'],ylabel='Low minus high (symbols)',title='Mean and 95% block interval');axes[1].axhline(0,color='gray');axes[1].axhline(10,color='gray',ls=':')
    axes[2].set(xticks=[0,1],xticklabels=['High conflict','Low conflict'],ylabel='Exact-prefix symbols',title='All fixed context controls',ylim=(-2,130));axes[2].legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'intervention-curve.png',dpi=180);plt.close(fig)
    core.write_json(out/'manifest.json',dict(source_manifest_sha256=core.sha256(source/'manifest.json'),script_sha256=core.sha256(__file__),artifacts={p.name:core.sha256(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(dict(qualifying_graphs=qualifying,contrasts={g:r['exact_prefix_symbols'] for g,r in contrasts.items()}),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();analyze(a.source,a.out)
