"""Registered descriptive rho-minus-zero contrast, separate from allocation gains."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from flying.training import whole_brain_memory as core
from alphabet_memory import check,read
from context_memory import estimate
from pathway_memory import seal
from relative_noise import read_table
from verify_neuron_panel import audit_stats


def analyze(root,out):
    m=check(root);c=m['config'];vroot=Path(str(root)+'_validation');check(vroot)
    v=read(vroot/'checks.json')
    assert v['all_checks_pass'] and v['result_manifest_sha256']==core.sha256(root/'manifest.json')
    assert v['stage']=='correlation' and not v['smoke']
    b=read_table(root/'seed-blocks.csv');raw=read_table(root/'raw-certificates.csv')
    keys=['cohort','family','level','topology','seed','arm'];metrics=['exact_prefix_symbols','retention']
    base=b[b.rho==0].set_index(keys);rows=[]
    for x in b[b.rho!=0].to_dict('records'):
        original=base.loc[tuple(x[k] for k in keys)]
        row={k:x[k] for k in keys+['rho']}
        for metric in metrics:row[metric]=x[metric]-original[metric]
        # Independently reduce the raw circuit/noise/dose rows with no NaN dropping.
        selected=raw.copy()
        for k in keys:selected=selected[selected[k]==x[k]]
        for metric in metrics:
            a=selected[selected.rho==x['rho']][metric].to_numpy()
            z=selected[selected.rho==0][metric].to_numpy()
            assert len(a)==len(z)==18
            np.testing.assert_allclose(row[metric],np.mean(a)-np.mean(z),atol=1e-12,rtol=1e-12,equal_nan=True)
        rows.append(row)
    assert len(rows)==192
    f=pd.DataFrame(rows);statistics={}
    group=['cohort','family','level','topology','arm','rho']
    for values,part in f.groupby(group):
        key='/'.join(map(str,values));statistics[key]={}
        for metric in metrics:
            data=part[metric].to_numpy();s=estimate(data,c) if np.isfinite(data).all() else None
            if s is not None:audit_stats(data,s,c)
            statistics[key][metric]=s
    assert len(statistics)==48
    out.mkdir(parents=True,exist_ok=False)
    f.to_csv(out/'rho-minus-zero-seed-blocks.csv',index=False)
    core.write_json(out/'statistics.json',statistics)
    core.write_json(out/'checks.json',dict(all_checks_pass=True,seed_block_rows=len(f),groups=len(statistics),
        contrast='rho minus zero within the same allocation/model/task',negative_means_degradation=True,
        registered_descriptive_analysis=True,new_fits=0,new_trajectories=0,primary_criterion_changed=False,
        independently_reduced_from_raw_certificates=True,result_manifest_sha256=core.sha256(root/'manifest.json')))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    doses=read_table(root/'dose-seed-table.csv')
    fig,axes=plt.subplots(2,3,figsize=(14,7),layout='constrained')
    styles={0.:'-',.25:'--',.75:':'}
    for i,cohort in enumerate(c['cohorts']):
        for j,family in enumerate(c['families']):
            ax=axes[i,j];part=doses[(doses.cohort==cohort)&(doses.family==family)&(doses.level=='brain1')]
            for (rho,arm),cells in part.groupby(['rho','arm']):
                means=cells.groupby('strength').exact_prefix_symbols.mean()
                label=f"rho={float(rho):g} / {'allocated' if arm=='training_sd' else 'flat'}"
                ax.plot(means.index,means.values,marker='o',linestyle=styles[rho],
                    color='#c65b1c' if arm=='training_sd' else '#2369a0',label=label)
            ax.set_xscale('log');ax.set_xticks(c['relative_strengths'],['.0003','.03','.3'])
            ax.set_ylim(bottom=0);ax.set(title=cohort+' / '+family,xlabel='Relative dose (log scale)',ylabel='Exact prefix (symbols)')
            ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.savefig(out/'prefix-curves.png',dpi=160);plt.close(fig)
    seal(out,c,m['context'],analysis_script_sha256=core.sha256(__file__),purpose='Registered correlation versus independent-noise contrast')
    print('192 paired block contrasts and 48 descriptive summaries verified')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();analyze(a.source,a.out)
