"""Compare two complete runs without relying on ZIP timestamps."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
p=argparse.ArgumentParser();p.add_argument('--reference',default='results/phase3b');p.add_argument('--repeat',default='outputs/phase3b_repeat');a=p.parse_args()
ref=Path(a.reference);repeat=Path(a.repeat)
for name in ['config.json','pi.csv','memory.csv','recalls.jsonl','observations.json','mixing.json']:
    assert (ref/name).read_bytes()==(repeat/name).read_bytes(),f'Mismatch: {name}'
with np.load(ref/'delayed_predictions.npz') as first,np.load(repeat/'delayed_predictions.npz') as second:
    for key in ['predictions','targets']:assert np.array_equal(first[key],second[key]),key
report=dict(pi_runs=len(pd.read_csv(ref/'pi.csv')),delay_measurements=len(pd.read_csv(ref/'memory.csv')),
    all_results_exact=True,all_pi_recalls_exact=True,all_delay_predictions_and_targets_exact=True)
(ref/'reproducibility.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
