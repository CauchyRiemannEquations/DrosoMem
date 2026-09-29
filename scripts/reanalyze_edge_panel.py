"""Correct the between/betweenness name collision without rewriting raw runs."""
from pathlib import Path
import subprocess
import pandas as pd
from alphabet_memory import read,check
from flying.training import whole_brain_memory as core
from edge_panel import summarize,seal


def main():
    root=Path('results/edge_ablation');out=Path('results/edge_ablation_analysis_v2')
    original=check(root);before=core.sha256(root/'manifest.json');out.mkdir(exist_ok=False)
    rows=pd.read_csv(root/'raw-lag-table.csv').to_dict('records')
    summary=summarize(rows,original['config'],out);old=read(root/'summary.json')
    for key in old['statistics']:
        if not key.endswith('/between'):
            # CSV roundtripping can perturb last bits; compare metrics numerically.
            import numpy as np
            for metric in old['statistics'][key]:
                a=old['statistics'][key][metric];b=summary['statistics'][key][metric]
                if isinstance(a,dict):
                    for k in a:
                        if a[k] is None:assert b[k] is None
                        else:np.testing.assert_allclose(a[k],b[k],atol=1e-10,rtol=1e-10)
                else:assert a==b
    assert old['primary_DAN_refit']==summary['primary_DAN_refit'] and old['confirmed']==summary['confirmed']
    core.write_json(out/'correction.json',dict(reason='between prefix also matched betweenness; use exact arm membership',
        original_derived_between_statistics_and_plots_superseded=True,original_raw_arrays_and_metrics_unchanged=True,
        decisions_unchanged=True,thresholds_and_seeds_unchanged=True,source_manifest_sha256=before))
    check(root);assert core.sha256(root/'manifest.json')==before
    seal(out,original['config'],dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        analysis_source_sha256={p:core.sha256(p) for p in ['scripts/edge_panel.py',__file__]}),source_manifest_sha256=before)
    print('Corrected derived analysis only; raw artifacts unchanged; confirmed:',summary['confirmed'])


if __name__=='__main__':main()
