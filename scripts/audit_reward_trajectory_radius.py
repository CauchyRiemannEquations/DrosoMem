"""Post-hoc geometry audit; explains radius limits without changing endpoints."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from alphabet_memory import read,check
from flying.training import whole_brain_memory as core
from pathway_memory import seal


def audit(root,out):
    m=check(root);c=m['config'];out.mkdir(parents=True,exist_ok=False);rows=[]
    for dest in sorted(p.parent for p in root.glob('*/*/radius.json')):
        cm=check(dest);i=cm['identity'];original=sparse.load_npz(Path(cm['source_case'])/'initial-weights.npz');r=read(dest/'radius.json');best=None
        for epoch in cm['epochs']:
            stage=dest/f'epoch{epoch}';w=sparse.load_npz(stage/'baseline-weights.npz')
            with np.load(stage/'checkpoint.npz') as a:
                mask=a['plastic_mask'];unit=a['unit_directions'];mag=abs(w.data[mask]);orig=abs(original.data[mask])
                limits=np.divide(mag[None,:],-unit,out=np.full_like(unit,np.inf),where=unit<0)
                j,k=np.unravel_index(np.argmin(limits),limits.shape);value=float(limits[j,k])
                if best is None or value<best['positivity_limit']:
                    best=dict(epoch=epoch,variant='local' if j==0 else f'random{j-1}',plastic_edge_index=int(k),positivity_limit=value,current_magnitude=float(mag[k]),original_magnitude=float(orig[k]),current_to_original=float(mag[k]/orig[k]),direction_component=float(unit[j,k]))
        expected=0. if r['zero_direction'] else min(r['nominal_radius'],c['boundary_fraction']*best['positivity_limit']);assert expected==r['radius']
        rows.append(dict(**{k:i[k] for k in ['cohort','seed','circuit_seed']},**best,**r,nominal_fraction=r['radius']/r['nominal_radius']))
    pd.DataFrame(rows).to_csv(out/'radius-limiters.csv',index=False)
    seal(out,c,m['context'],analysis='Post-hoc radius limiter audit; endpoints unchanged',analysis_script_sha256=core.sha256(__file__),source_manifest_sha256=core.sha256(root/'manifest.json'))
    print(pd.DataFrame(rows)[['cohort','seed','circuit_seed','epoch','variant','nominal_fraction','current_to_original']].to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args();audit(a.root,a.out)
