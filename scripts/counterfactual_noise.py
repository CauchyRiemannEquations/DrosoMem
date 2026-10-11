"""M7: ONE registered observation-noise dose; frozen M6 dynamics/M5 heads."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from counterfactual_tracking import execute as clean_execute
from tdc_pool import pooled
from noise_tracking_support import config,read,write,sha,array_sha,attempt,seal,history_preserved,source_record,assert_source_unchanged

WORLDS=('original','counterfactual','instantaneous_original','instantaneous_counterfactual')

def draw_noise(ids,scale,seed,count=64,sigma=.1):
    ids=np.asarray(ids,dtype=np.int64);scale=np.asarray(scale)
    assert ids.shape==scale.shape==(48,) and len(np.unique(ids))==48 and np.all(scale>0)
    order=np.argsort(ids,kind='stable')
    canonical=np.random.Generator(np.random.PCG64(seed)).standard_normal((count,48))
    z=np.empty_like(canonical);z[:,order]=canonical
    return dict(canonical_standard_normal=canonical,standard_normal=z,canonical_order=order,
                amplitude=sigma*scale,raw_noise=sigma*scale*z)

def score(observed,mean,scale,beta,intercept):
    return ((observed-mean)/scale)@beta+intercept

def counts(p0,p1,s,r):
    assert len(s)==64 and np.all(s!=r)
    a=p0==s;b=p1==r;j=int(np.sum(a&b))
    return dict(samples=64,original_correct=int(a.sum()),replacement_correct=int(b.sum()),joint_correct=j,
        original_accuracy=float(a.mean()),replacement_accuracy=float(b.mean()),joint_accuracy=j/64,
        prediction_changes=int(np.sum(p0!=p1)))

def panel(clean,c,block,level,smoke=False):
    ids=clean['root_ids'][clean['observed_indices']]
    a=draw_noise(ids,clean['scale'],block['noise_seed'],64,c['noise_sigma'])
    for world in WORLDS:
        x=clean['trial_'+world+'_features'][:,-1]
        a['observed_'+world]=x+a['raw_noise']
        a['scores_'+world]=score(a['observed_'+world],clean['mean'],clean['scale'],clean['coefficients_lag2'],clean['intercept_lag2'])
        a['predictions_'+world]=a['scores_'+world].argmax(axis=1)
        expected=clean['scores_'+world]+c['noise_sigma']*a['standard_normal']@clean['coefficients_lag2']
        np.testing.assert_allclose(a['scores_'+world],expected,atol=c['score_atol'],rtol=c['score_rtol'])
    for field in ['observed_','scores_','predictions_']:
        np.testing.assert_array_equal(a[field+'instantaneous_original'],a[field+'instantaneous_counterfactual'])
    clean_delta=clean['scores_counterfactual']-clean['scores_original']
    np.testing.assert_allclose(a['scores_counterfactual']-a['scores_original'],clean_delta,atol=c['score_atol'],rtol=c['score_rtol'])
    s,r=clean['original_symbols'],clean['alternative_symbols']
    cell=dict(level=level,seed=block['seed'],noise_seed=block['noise_seed'],**counts(a['predictions_original'],a['predictions_counterfactual'],s,r))
    cc=counts(clean['predictions_original'],clean['predictions_counterfactual'],s,r)
    cell.update(clean_joint_correct=cc['joint_correct'],clean_joint_accuracy=cc['joint_accuracy'],
        clean_minus_noisy_joint=cc['joint_accuracy']-cell['joint_accuracy'],
        frequency_joint_correct=0,current_joint_correct=0,instantaneous_joint_correct=counts(a['predictions_instantaneous_original'],a['predictions_instantaneous_counterfactual'],s,r)['joint_correct'],
        scientific_rule_pass=None if smoke else 10*cell['joint_correct']>=9*64,
        technical_checks_pass=True,scale_floor_value_count=int(np.sum(clean['scale']==1e-5)),
        mean_raw_noise_energy=float(np.mean(np.sum(a['raw_noise']**2,axis=1))),
        mean_standardized_noise_energy=float(np.mean(np.sum((c['noise_sigma']*a['standard_normal'])**2,axis=1))))
    assert cell['instantaneous_joint_correct']==0
    rows=[]
    for i in range(64):
        row=dict(level=level,seed=block['seed'],probe_index=i,anchor=int(clean['anchors'][i]),evaluation_time=int(clean['evaluation_times'][i]),
                 original_symbol=int(s[i]),replacement_symbol=int(r[i]),current_symbol=int(clean['current_symbols'][i]),
                 original_correct=bool(a['predictions_original'][i]==s[i]),replacement_correct=bool(a['predictions_counterfactual'][i]==r[i]),
                 joint_correct=bool(a['predictions_original'][i]==s[i] and a['predictions_counterfactual'][i]==r[i]),
                 clean_joint_correct=bool(clean['predictions_original'][i]==s[i] and clean['predictions_counterfactual'][i]==r[i]),
                 raw_noise_energy=float(a['raw_noise'][i]@a['raw_noise'][i]),standardized_noise_energy=float(np.sum((c['noise_sigma']*a['standard_normal'][i])**2)))
        for world in WORLDS:row['prediction_'+world]=int(a['predictions_'+world][i])
        rows.append(row)
    return a,cell,rows

def case_job(job):
    c,b,l,out=job['config'],job['block'],job['level'],Path(job['out'])
    with threadpool_limits(1):
        cleanroot=out/'clean'/f'{l}_s{b["seed"]}'
        with attempt(cleanroot,c,'m7-clean-case') as resources:
            a,metrics,cell,rows,graph,probes=clean_execute(c,b,l,resources,job['smoke'])
            np.savez_compressed(cleanroot/'case.npz',**a)
            for name,value in [('metrics',metrics),('primary-cell',cell),('trial-metrics',dict(counts=cell,rows=rows)),('graph',graph),('trial-info',probes)]:write(cleanroot/(name+'.json'),value)
        seal(cleanroot,dict(complete=True,block=b,level=l,smoke=job['smoke']))
        noisyroot=out/'noisy'/f'{l}_s{b["seed"]}'
        with attempt(noisyroot,c,'m7-noisy-case'):
            na,nc,nr=panel(a,c,b,l,job['smoke'])
            np.savez_compressed(noisyroot/'noise.npz',**na)
            write(noisyroot/'array-hashes.json',{n:array_sha(v) for n,v in na.items()})
            write(noisyroot/'primary-cell.json',nc);write(noisyroot/'trials.json',nr)
        seal(noisyroot,dict(complete=True,block=b,level=l,clean_case_manifest_sha256=sha(cleanroot/'manifest.json')))
    return dict(identity=f'{l}/s{b["seed"]}',cell=nc,rows=nr,clean_cell=cell)

def estimates(values,c):
    x=np.asarray(values,dtype=float);ix=np.random.default_rng(c['bootstrap_seed']).integers(len(x),size=(c['bootstrap_draws'],len(x)))
    lo,hi=np.percentile(x[ix].mean(axis=1),[2.5,97.5])
    return dict(mean=float(x.mean()),median=float(np.median(x)),sd=float(x.std(ddof=1)) if len(x)>1 else 0.,bootstrap_lo=float(lo),bootstrap_hi=float(hi))

def summarize(cells,c,smoke):
    levels={}
    for level in c['levels']:
        rows=sorted([x for x in cells if x['level']==level],key=lambda x:x['seed'])
        assert [x['seed'] for x in rows]==c['blocks_by_level'][level] and all(x['samples']==64 for x in rows)
        passed=None if smoke else all(10*x['joint_correct']>=9*64 for x in rows)
        levels[level]=dict(registered_rule_pass=passed,outcome='smoke' if smoke else 'PASS' if passed else 'FAIL',cells=rows,
            **{k:estimates([x[k] for x in rows],c) for k in ['joint_accuracy','original_accuracy','replacement_accuracy','clean_joint_accuracy','clean_minus_noisy_joint']})
    primary=levels[c['primary_level']]
    return dict(smoke=smoke,outcome=primary['outcome'],primary_pass=primary['registered_rule_pass'],levels=levels,cases=len(cells),probe_pairs=64*len(cells),
        noise_sigma=.1,new_trained_parameters=0,decoder_parameters=490,noise_feedback=False,noise_clipping=False,performance_eligibility_gate=False,physiological_noise_claim=False,criterion_unchanged=True)

def plot(cells,out,summary):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,level in zip(axes,['legacy5','brain5']):
        rows=sorted([x for x in cells if x['level']==level],key=lambda x:x['seed']);idx=np.arange(len(rows))
        ax.plot(idx,[x['clean_joint_accuracy'] for x in rows],'o-',label='Fresh clean joint tracking')
        ax.plot(idx,[x['joint_accuracy'] for x in rows],'s-',label='Shared noise sigma=0.10 (fixed)')
        ax.axhline(.9,color='gray',ls=':',label='Registered threshold')
        ax.axhline(0,color='gray',ls='--',label='Unchanged paired controls')
        ax.set(title=level,xticks=idx,xticklabels=[x['seed'] for x in rows],ylabel='Fraction of all64 pairs',ylim=(-.04,1.04));ax.tick_params(axis='x',labelrotation=25);ax.legend(fontsize=8)
    fig.suptitle('M7 artificial observation-noise tracking: '+summary['outcome']);fig.tight_layout();fig.savefig(out/'counterfactual-noise.png',dpi=180);plt.close(fig)

def run(out,smoke=False):
    c=config(smoke);source=source_record(c)
    with attempt(out,c,'m7-smoke' if smoke else 'm7-main'):
        write(out/'config.json',c);write(out/'source.json',source);write(out/'historical-preservation.json',history_preserved(c))
        jobs=[dict(config=c,block=b,level=l,out=str(out),smoke=smoke) for l in c['levels'] for b in c['blocks'] if b['seed'] in c['blocks_by_level'][l]]
        results,usage=pooled(case_job,jobs,c['neural_workers'],c,'M7 fixed-dose tracking')
        cells=sorted([r['cell'] for r in results],key=lambda x:(x['level'],x['seed']))
        raw=[x for r in results for x in r['rows']]
        summary=summarize(cells,c,smoke)
        pd.DataFrame(raw).sort_values(['level','seed','probe_index']).to_csv(out/'raw-trials.csv',index=False)
        pd.DataFrame(cells).to_csv(out/'block-scores.csv',index=False)
        write(out/'primary-cells.json',cells);write(out/'summary.json',summary);write(out/'process-tree-resources.json',usage)
        for b in c['blocks'][:(1 if smoke else 3)]:
            with np.load(out/'noisy'/f'legacy5_s{b["seed"]}'/'noise.npz') as a,np.load(out/'noisy'/f'brain5_s{b["seed"]}'/'noise.npz') as d:
                np.testing.assert_array_equal(a['canonical_standard_normal'],d['canonical_standard_normal'])
        plot(cells,out,summary);assert_source_unchanged(source)
    seal(out,dict(complete=True,smoke=smoke,source_commit=source['source_commit']))
    print(dict(outcome=summary['outcome'],cases=len(cells)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    with threadpool_limits(1):run(a.out,a.smoke)
