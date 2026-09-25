"""Compare two completed numerical runs; ignore ZIP timestamps in NPZ files."""
import argparse
import json
from pathlib import Path
import numpy as np


def verify(reference, repeat, output):
    reference,repeat=Path(reference),Path(repeat)
    text_names=['config.json','results.csv','paired.csv','recalls.jsonl','endpoints.csv','event_keys.json','reward_training.csv']
    for name in text_names:
        assert (reference/name).read_bytes()==(repeat/name).read_bytes(),name
    arrays=0
    for name in ['endpoints.npz','reward_events.npz']:
        a,b=np.load(reference/name),np.load(repeat/name)
        assert set(a.files)==set(b.files)
        for key in a.files:
            assert np.array_equal(a[key],b[key]),(name,key)
            arrays+=1
    result=dict(text_files_exact=text_names,checkpoint_and_reward_arrays_exact=arrays,
                reference_note='First run completed all numerical work but failed while serializing selection metadata; compared all saved numerical artifacts after a metadata-type-only fix.')
    Path(output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reference',required=True);p.add_argument('--repeat',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();verify(args.reference,args.repeat,args.output)
