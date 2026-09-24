"""Exact numerical comparison of two complete Phase 4 runs."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
p=argparse.ArgumentParser();p.add_argument('--reference',default='results/phase4');p.add_argument('--repeat',default='outputs/phase4_repeat');a=p.parse_args()
ref=Path(a.reference);repeat=Path(a.repeat)
for name in ['config.json','results.csv','recalls.jsonl','plastic_training.csv','teacher_codes.json','mixing.json']:
    assert (ref/name).read_bytes()==(repeat/name).read_bytes(),name
checkpoints=list((ref/'checkpoints').glob('*.npz'))
for path in checkpoints:
    other=repeat/'checkpoints'/path.name
    if path.name.endswith('_weights.npz'):
        x=sparse.load_npz(path);y=sparse.load_npz(other)
        for attr in ['data','indices','indptr']:assert np.array_equal(getattr(x,attr),getattr(y,attr)),path.name
    else:
        with np.load(path) as x,np.load(other) as y:
            for key in x.files:assert np.array_equal(x[key],y[key]),path.name
report=dict(repeated_runs=len(pd.read_csv(ref/'results.csv')),all_numeric_results_exact=True,all_recall_sequences_exact=True,
    all_plastic_training_history_exact=True,checkpoint_arrays_exact=True,checkpoint_files=len(checkpoints))
(ref/'reproducibility.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
