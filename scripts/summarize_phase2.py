"""Produce a compact paired-seed overview from already recorded phase-2 results."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def overview(directory):
    directory=Path(directory)
    frame=pd.read_csv(directory/'results.csv')
    selected=frame[(frame.model=='fly')&(frame.train_digits==200)]
    graphs=list(selected.graph.unique())
    fig,ax=plt.subplots(figsize=(10,4.6))
    for offset,norm,color in [(-.18,'spectral','#64748b'),(.18,'incoming_l1','#0f766e')]:
        for i,graph in enumerate(graphs):
            vals=selected[(selected.graph==graph)&(selected.normalization==norm)].pi_memory_score.to_numpy()
            ax.bar(i+offset,vals.mean(),width=.32,color=color,alpha=.75,label=norm if i==0 else None)
            ax.scatter(i+offset+np.linspace(-.05,.05,len(vals)),vals,color=color,edgecolor='black',s=28,zorder=3)
    ax.set_xticks(range(len(graphs)),graphs)
    ax.set(ylabel='Pi Memory Score (prompt excluded)',title='Real Fly subsets: 200 training digits, three paired fresh seeds')
    ax.axhline(197,color='gray',linestyle='--',linewidth=1)
    ax.text(.99,.94,'197 = evaluation horizon',ha='right',va='top',transform=ax.transAxes,fontsize=9)
    ax.legend(loc='upper right',bbox_to_anchor=(1,.9));ax.grid(axis='y',alpha=.15)
    fig.tight_layout();fig.savefig(directory/'overview.png',dpi=160);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',nargs='?',default='results/phase2')
    overview(p.parse_args().directory)
