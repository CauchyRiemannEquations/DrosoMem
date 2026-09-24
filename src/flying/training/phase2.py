"""Predeclared exploratory structure/dynamics/length study, CPU only."""
import argparse
from datetime import datetime,timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits
from flying.data.connectome import load_connectome,sha256
from flying.data.pi_digits import pi_digits
from flying.encoding.digit_encoder import DigitEncoder
from flying.brain.reservoir import Reservoir
from flying.brain.diagnostics import normalize_condition,graph_diagnostics,state_diagnostics
from flying.brain.shuffled_network import shuffled_network
from flying.brain.random_network import random_network
from flying.models.readout import Readout
from flying.evaluation.free_recall import evaluate_recall
from flying.evaluation.next_digit import next_digit_accuracy


def validate(c):
    if not c['graphs'] or not c['seeds'] or not c['models'] or not c['normalizations'] or not c['train_lengths']:
        raise ValueError('Empty experimental factor')
    if not set(c['models']) <= {'fly','shuffled','random','leaky_only','memoryless'}:
        raise ValueError('Unknown model')
    if not set(c['normalizations']) <= {'spectral','incoming_l1'}:
        raise ValueError('Unknown normalization')
    if not 1<=c['prompt_length']<min(c['train_lengths']):
        raise ValueError('Invalid prompt/length')
    if c['heldout_start']<=max(c['train_lengths']) or c['heldout_digits']<1:
        raise ValueError('Heldout block must be strictly beyond every training prefix')
    if not 0<c['gain']<1 or not 0<c['leak']<=1:
        raise ValueError('Invalid gain/leak')


def run_phase2(c,out,base=Path('.')):
    validate(c)
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    (out/'config.json').write_text(json.dumps(c,indent=2))
    h=c['heldout_start']; end=h+c['heldout_digits']
    digits=pi_digits(end)
    (out/'pi_digits.txt').write_text(''.join(map(str,digits))+'\n')
    manifest=dict(time_utc=datetime.now(timezone.utc).isoformat(),python=platform.python_version(),
                  packages={p:importlib.metadata.version(p) for p in ['numpy','scipy','pandas','mpmath','matplotlib','threadpoolctl']},
                  threads=1,split=f'train target indices 1..L-1; heldout target indices {h}..{end-1}',
                  source_code_sha256={str(p.relative_to(Path(__file__).parents[1])):sha256(p)
                                      for p in sorted(Path(__file__).parents[1].rglob('*.py'))},
                  config_sha256=sha256(out/'config.json'),pi_sha256=sha256(out/'pi_digits.txt'))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    rows=[]; structure=[]; curves=[]
    provenance={}
    recalls=[]
    for graph,path in c['graphs'].items():
        a,ids,meta=load_connectome(Path(base)/path)
        if a.shape[0]>3000:
            raise ValueError('Phase 2 CPU study limit is 3000; explicitly select a smaller subset')
        provenance[graph]=meta
        structure.append(dict(graph=graph,**graph_diagnostics(a)))
        for seed in c['seeds']:
            encoder=DigitEncoder(a.shape[0],seed,**c['encoding'])
            raw_graphs={};shuffle_log={}
            for model in c['models']:
                if model=='fly':raw_graphs[model]=a
                elif model=='shuffled':raw_graphs[model],shuffle_log=shuffled_network(a,seed+10000)
                elif model=='random':raw_graphs[model]=random_network(a,seed+20000)
                else:raw_graphs[model]=sparse.csr_matrix(a.shape,dtype=float)
            for norm in c['normalizations']:
                for model,raw in raw_graphs.items():
                    if model in {'leaky_only','memoryless'}:
                        w=raw.copy()
                    else:
                        w=normalize_condition(raw,norm,c['gain'])
                    leak=1. if model=='memoryless' else c['leak']
                    reservoir=Reservoir(w,encoder,leak)
                    states=reservoir.states(digits[:-1])
                    for length in c['train_lengths']:
                        tic=time.perf_counter()
                        key=dict(graph=graph,normalization=norm,model=model,seed=seed,train_digits=length)
                        x=states[:length-1]; y=digits[1:length]
                        ro=Readout();history=ro.fit(x,y,**c['training'])
                        recall=evaluate_recall(reservoir,ro,digits[:length],c['prompt_length'],length-c['prompt_length'])
                        row=dict(**key,neurons=a.shape[0],edges=raw.nnz,
                                 train_accuracy=next_digit_accuracy(ro,x,y),
                                 heldout_accuracy=next_digit_accuracy(ro,states[h-1:end-1],digits[h:end]),
                                 pi_memory_score=recall['pi_memory_score'],horizon=recall['horizon'],
                                 censored=recall['censored'],recall_fraction=recall['pi_memory_score']/recall['horizon'],
                                 incoming_l1_max=float(abs(w).sum(axis=1).max()),
                                 **state_diagnostics(x),seconds=time.perf_counter()-tic)
                        if model=='shuffled':row.update(shuffle_log)
                        rows.append(row)
                        recalls.append(dict(**key,**recall))
                        # Sparse epoch logging keeps the versioned experiment compact.
                        for entry in history:
                            if entry['epoch']==1 or entry['epoch']%25==0 or entry['epoch']==c['training']['epochs']:
                                curves.append(dict(**key,**entry))
            print(f'Completed {graph}, seed {seed}: {len(rows)} runs',flush=True)
    # Publish complete snapshots atomically: do not expose in-place append files
    # to shared-filesystem synchronization during a long-running experiment.
    for name,records in [('metrics',rows),('recall',recalls)]:
        temporary=out/f'{name}.jsonl.tmp'
        temporary.write_text(''.join(json.dumps(record)+'\n' for record in records))
        temporary.replace(out/f'{name}.jsonl')
    expected=len(c['graphs'])*len(c['normalizations'])*len(c['models'])*len(c['seeds'])*len(c['train_lengths'])
    if len(rows)!=expected or len(recalls)!=expected:
        raise RuntimeError('Incomplete experimental design')
    frame=pd.DataFrame(rows)
    frame.to_csv(out/'results.csv',index=False)
    pd.DataFrame(structure).to_csv(out/'structure.csv',index=False)
    pd.DataFrame(curves).to_csv(out/'training.csv',index=False)
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2))
    frame.groupby(['graph','normalization','model','train_digits'])[['pi_memory_score','recall_fraction','train_accuracy','heldout_accuracy','effective_rank']].agg(['mean','std','min','max']).to_csv(out/'summary.csv')
    plot_phase2(frame,out)
    (out/'runtime.json').write_text(json.dumps(dict(total_seconds=time.perf_counter()-start,experiments=len(rows)),indent=2))
    return frame


def plot_phase2(frame,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    graphs=list(frame.graph.unique()); norms=list(frame.normalization.unique())
    colors={'fly':'#0f766e','shuffled':'#d97706','random':'#6366f1','leaky_only':'#be185d','memoryless':'#64748b'}
    fig,axes=plt.subplots(len(graphs),len(norms),figsize=(12,3*len(graphs)),squeeze=False)
    for i,graph in enumerate(graphs):
        for j,norm in enumerate(norms):
            ax=axes[i,j]
            selected=frame[(frame.graph==graph)&(frame.normalization==norm)]
            for model,d in selected.groupby('model',sort=False):
                g=d.groupby('train_digits').pi_memory_score.agg(['mean','min','max'])
                ax.plot(g.index,g['mean'],marker='o',label=model,color=colors[model])
                ax.fill_between(g.index,g['min'],g['max'],color=colors[model],alpha=.12)
            ax.set(title=f'{graph} / {norm}',xlabel='Training digits (including 3)',ylabel='Pi Memory Score')
            ax.grid(alpha=.15)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.978),ncol=5)
    fig.suptitle('Exploratory length curves: mean and seed range; scores capped by horizon',y=.998,fontsize=12)
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(out/'length_curves.png',dpi=140);plt.close(fig)
    longest=frame[frame.train_digits==frame.train_digits.max()]
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for model,d in longest.groupby('model',sort=False):
        axes[0].scatter(d.effective_rank,d.pi_memory_score,label=model,color=colors[model],alpha=.6,s=24)
        index=list(longest.model.unique()).index(model)
        axes[1].scatter(index+np.linspace(-.18,.18,len(d)),d.heldout_accuracy,color=colors[model],s=18,alpha=.65)
    axes[0].set(xlabel='State effective rank',ylabel='Pi Memory Score',title='Association, not a causal test');axes[0].legend(fontsize=8)
    axes[1].set_xticks(range(len(longest.model.unique())),longest.model.unique(),rotation=20)
    axes[1].axhline(.1,color='gray',linestyle='--');axes[1].set(ylabel='Heldout accuracy',title='Same unseen 100-digit block / longest training prefix')
    fig.tight_layout();fig.savefig(out/'diagnostics.png',dpi=150);plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/phase2.json')
    p.add_argument('--output',default=None)
    a=p.parse_args()
    out=a.output or 'outputs/phase2_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    with threadpool_limits(limits=1):
        run_phase2(json.loads(Path(a.config).read_text()),out)
    print(f'Saved {out}')

if __name__=='__main__':main()
