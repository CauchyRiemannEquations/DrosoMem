"""Replay/refit the first archived fixed-32 condition, all initializations."""
import json
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits, threadpool_info
from flying.training.phase5_prefix import conditions, fit_head, metrics, NonlinearReadout, evaluate_recall, sha256

out = Path('results/act1_baseline'); out.mkdir(exist_ok=False)
source = Path('results/phase5_prefix_confirmation_windows')
manifest = json.loads((source/'manifest.json').read_text())
cfg = manifest['context']['config']
path = source/'checkpoints/condition_000.npz'
assert sha256(path) == manifest['file_sha256']['checkpoints/condition_000.npz']
rows = []
with threadpool_limits(1), np.load(path, allow_pickle=False) as saved:
    payload = json.loads(saved['payload'].item()); digits = np.array(list(map(int,payload['segment'])))
    key, reservoir, mbon = next(conditions(cfg)); states = reservoir.states(digits[:-1])
    for row, expected in zip(payload['rows'], payload['recalls']):
        if row['treatment'] != 'weighted': continue
        ix = row['head_index']; head = NonlinearReadout(mbon, cfg['hidden_units'])
        head.mean = saved[f'mean_{ix}']; head.scale = saved[f'scale_{ix}']
        head.parameters = {k:saved[f'{k}_{ix}'] for k in ['w1','b1','w2','b2']}
        got = evaluate_recall(reservoir,head,digits,3,197)
        assert got == {k:v for k,v in expected.items() if k!='head_index'}
        fitted, history = fit_head(states,digits[1:],mbon,key,row['initialization'],'weighted',cfg)
        refit = evaluate_recall(reservoir,fitted,digits,3,197)
        rows.append(dict(initialization=row['initialization'], expected=row['pi_memory_score'],
                         replay=got['pi_memory_score'], refit=refit['pi_memory_score'],
                         exact_saved_replay=True, exact_refit_parameters=fitted.digest()==head.digest(),
                         expected_readout_hash=head.digest(), actual_readout_hash=fitted.digest(),
                         refit_recall=refit, metrics=metrics(fitted,states,digits[1:],cfg)))
        np.savez_compressed(out/f'refit_{ix}.npz',mean=fitted.mean,scale=fitted.scale,**fitted.parameters)
    environment = dict(python=platform.python_version(), packages={n:version(n) for n in ['numpy','scipy','pandas','mpmath','threadpoolctl']},blas=threadpool_info())
report = dict(source_checkpoint=str(path), source_sha256=sha256(path), config=cfg, selected_condition=payload['key'],
              environment=environment, rows=rows, scope='Exact saved-head replay and independent refit of first condition, all three weighted initializations; not the full historical context verifier.')
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
(out/'environment.txt').write_text(subprocess.check_output([__import__('sys').executable,'-m','pip','freeze'],text=True))
print(json.dumps(rows,indent=2))
