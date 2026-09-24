"""Exact repeat verification for reward signals, learned weights and recall."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
p=argparse.ArgumentParser();p.add_argument('--reference',default='results/phase5');p.add_argument('--repeat',default='outputs/phase5_repeat');a=p.parse_args();ref=Path(a.reference);repeat=Path(a.repeat)
for name in ['config.json','results.csv','recalls.jsonl','reward_training.csv','event_keys.json','mixing.json']:
    assert (ref/name).read_bytes()==(repeat/name).read_bytes(),name
for path in [ref/'reward_events.npz',*sorted((ref/'checkpoints').glob('*.npz'))]:
    other=repeat/path.relative_to(ref)
    with np.load(path) as x,np.load(other) as y:
        assert set(x.files)==set(y.files)
        for key in x.files:assert np.array_equal(x[key],y[key]),str(path)+':'+key
report=dict(evaluations=len(pd.read_csv(ref/'results.csv')),all_numeric_results_exact=True,all_recall_sequences_exact=True,
    all_reward_events_exact=True,all_training_history_exact=True,all_checkpoint_arrays_exact=True)
(ref/'reproducibility.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
