"""Prespecified paired analysis; checks raw checkpoint-derived recall metrics."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flying.training.whole_brain_memory import write_json, sha256, prefix_score


def estimate(values):
    values=np.asarray(values,dtype=float)
    rng=np.random.default_rng(7199)
    bootstrap=values[rng.integers(0,len(values),size=(10000,len(values)))].mean(axis=1)
    sd=float(values.std(ddof=1)) if len(values)>1 else 0.
    return dict(n=len(values),mean=float(values.mean()),median=float(np.median(values)),
                sample_variance=sd*sd,ci95=np.quantile(bootstrap,[.025,.975]).tolist(),
                paired_dz=float(values.mean()/sd) if sd else None)


def summarize(source,out):
    manifest=json.loads((source/'manifest.json').read_text()); c=manifest['config']
    for name,digest in manifest['artifacts'].items():
        if sha256(source/name)!=digest: raise ValueError('Corrupt source: '+name)
    frame=pd.read_csv(source/'evaluations.csv')
    assert len(frame)==len(c['conditions'])*len(c['seeds'])*len(c['circuit_seeds'])
    assert not frame.duplicated(['level','circuit_seed','seed']).any()
    representation=[]
    for row in frame.to_dict('records'):
        with np.load(source/row['checkpoint'],allow_pickle=False) as a:
            truth=a['digits'][c['prompt_length']:]; correct=a['prediction']==truth
            assert prefix_score(truth,a['prediction'])==row['pi_memory_score']
            np.testing.assert_array_equal(correct,a['position_accuracy'])
            assert np.isclose(correct.mean(),row['autonomous_accuracy'])
            assert np.isclose(np.mean(a['teacher']==a['digits'][1:]),row['next_digit_accuracy'])
            assert a['features'].shape==(c['dataset']['length']-1,48)
            assert a['probabilities'].shape==(c['eval_length'],10)
            np.testing.assert_allclose(a['probabilities'].sum(axis=1),1,atol=1e-14)
            z=(a['features']-a['mean'])/a['scale']
            singular=np.linalg.svd(z,compute_uv=False); mass=singular[singular>0]/singular.sum()
            half=np.flatnonzero(a['decay_observed']<=a['decay_observed'][0]*.5)
            representation.append(dict(level=row['level'],circuit_seed=row['circuit_seed'],seed=row['seed'],
                raw_effective_rank=row['effective_rank'],standardized_effective_rank=float(np.exp(-sum(mass*np.log(mass)))),
                std_floor_features=int(np.count_nonzero(a['features'].std(axis=0)<1e-5)),
                zero_input_observed_half_decay_step=int(half[0]) if len(half) else None))
    out.mkdir(parents=True,exist_ok=False)
    pd.DataFrame(representation).to_csv(out/'representation-diagnostics.csv',index=False)
    raw=frame.pivot(index=['circuit_seed','seed'],columns='level',values='pi_memory_score')[c['conditions']]
    raw.to_csv(out/'raw-seed-table.csv')
    blocks=frame.groupby(['seed','level']).pi_memory_score.mean().unstack()[c['conditions']]
    blocks.to_csv(out/'seed-blocks.csv')
    condition={name:estimate(blocks[name]) for name in c['conditions']}
    # d_z applies only to paired differences, not individual condition means.
    for val in condition.values(): val.pop('paired_dz')
    contrasts={}
    delta_frame=pd.DataFrame(index=blocks.index)
    for a,b in [('brain5','legacy5'),('brain1','legacy5'),('brain1','brain5'),
                ('left5','legacy5'),('left1','left5'),('brain5','left5'),('brain1','left1')]:
        delta=blocks[a]-blocks[b]; key=f'{a}-{b}'; result=estimate(delta)
        result.update(wins=int((delta>0).sum()),ties=int((delta==0).sum()),losses=int((delta<0).sum()))
        contrasts[key]=result; delta_frame[key]=delta
    delta_frame.to_csv(out/'paired-differences.csv')
    decision={name:bool(contrasts[f'{name}-legacy5']['mean']>=5 and contrasts[f'{name}-legacy5']['wins']>=4)
              for name in ['brain5','brain1']}
    weak=contrasts['brain1-brain5']
    result=dict(conditions=condition,contrasts=contrasts,main_improvement_criteria=decision,
                weak_edge_criterion=bool(weak['mean']>=5 and weak['wins']>=4),
                scope='Five computational seed blocks; two input-map strata averaged within seed. Not animals.')
    write_json(out/'summary.json',result)
    diagnostic_columns=['effective_rank','observed_mean_abs','active_neurons_mean','active_neurons_max',
                        'activity_sparsity','mean_observed_cosine','full_decay_ratio_32','observed_decay_ratio_32',
                        'next_digit_accuracy','teacher_forced_accuracy','autonomous_accuracy',
                        'runtime_seconds','peak_process_tree_rss_bytes','neurons','edges']
    diag=frame.groupby('level')[diagnostic_columns].agg(['mean','min','max']).reindex(c['conditions'])
    diag.to_csv(out/'diagnostics.csv')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(15,4.6))
    for index,row in raw.iterrows():
        axes[0].plot(range(5),row.values,'o-',alpha=.4,linewidth=1,label=f'{index[0]}/{index[1]}')
    axes[0].plot(range(5),raw.mean().values,'ko-',linewidth=2.5,label='mean')
    axes[0].set(ylabel='Exact autonomous prefix (digits)',title='All 10 paired runs / condition',xticks=range(5),xticklabels=c['conditions'])
    axes[0].set_ylim(bottom=0)
    for k,key in enumerate(['brain5-legacy5','brain1-legacy5','brain1-brain5']):
        est=contrasts[key]
        axes[1].scatter(np.full(5,k),delta_frame[key],alpha=.5,s=28)
        axes[1].plot([k,k],est['ci95'],color='black',linewidth=2)
        axes[1].scatter(k,est['mean'],color='black',marker='D')
    axes[1].axhline(0,color='gray',linestyle='--'); axes[1].axhline(5,color='gray',linestyle=':')
    axes[1].set(xticks=range(3),xticklabels=['brain5\n− legacy5','brain1\n− legacy5','brain1\n− brain5'],
                ylabel='Paired prefix difference',title='5 seed blocks; bootstrap 95% interval')
    for level in c['conditions']:
        part=frame[frame.level==level]
        axes[2].scatter(part.effective_rank,part.pi_memory_score,label=level,s=35,alpha=.7)
    axes[2].set(xlabel='Centered 48-MBON effective rank',ylabel='Exact autonomous prefix',title='Representation vs decoding (descriptive)')
    axes[2].legend(frameon=False)
    fig.tight_layout(); fig.savefig(out/'comparison.png',dpi=180); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for level in c['conditions']:
        arrays=[]; decay=[]
        for path in frame[frame.level==level].checkpoint:
            with np.load(source/path,allow_pickle=False) as a:
                arrays.append(a['position_accuracy']); d=a['decay_observed']; decay.append(d/max(d[0],1e-30))
        correctness=np.mean(arrays,axis=0)
        # Normalize truncated boundary windows; zero padding must not imply errors.
        smoothed=np.convolve(correctness,np.ones(8),mode='same')/np.convolve(np.ones_like(correctness),np.ones(8),mode='same')
        axes[0].plot(np.arange(1,len(correctness)+1),smoothed,label=level)
        axes[1].semilogy(np.arange(len(decay[0])),np.mean(decay,axis=0),label=level)
    axes[0].set(xlabel='Generated position',ylabel='Accuracy (8-position visual smoothing)',title='All autonomous positions; raw arrays retained')
    axes[1].set(xlabel='Zero-input update steps',ylabel='Relative 48-MBON state norm',title='Post-training decay probe; no physiological units')
    axes[1].legend(frameon=False); fig.tight_layout();fig.savefig(out/'recall-and-decay.png',dpi=180);plt.close(fig)
    write_json(out/'manifest.json',dict(source=str(source),source_manifest_sha256=sha256(source/'manifest.json'),
               analysis_script_sha256=sha256(__file__),files={p.name:sha256(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();summarize(a.source,a.out)
