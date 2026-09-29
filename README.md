# Flying

Flying studies sequential recall in **computational models using actual Drosophila
connectome structure**. Pi is a trained-sequence benchmark; this is not evidence
that a living fly memorized pi or that the model predicts unseen pi digits.

[한국어 설명](docs/README-ko.md) · [Research status / audit](docs/research-status.md) ·
[Roadmap](docs/research-roadmap.md) · [ACT I results](docs/whole-brain-memory-results.md)

## Current research

Latest: [ACT III-A population-panel closeout](docs/neuron-panel-results.md).
In the current two partial circuits, DAN-group removal impairs refit decoding
beyond count/input-matched controls in both seed cohorts; APL removal shows a
confirmed frozen-head sensitivity with substantial refit recovery. Edge count
and strength are not matched for these groups, limiting cell-type attribution.

Current position: ACT III-A current-model panel complete, including input-only
controls and fresh-seed confirmation. Whole-brain/downstream localization remains
untested. Next: one DAN-associated edge-count/weight-matched control study (III-B).

The established models freeze recurrent connectivity and train an external
readout. Distinguish **representation**, **decoding** and **internal learning**.
Teacher-forced accuracy, autonomous recall and robustness are separate measurements.

## Reproduce the latest diagnostic

Use Python3.12 and a clean environment; commands run from the repository root.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-act1-lock.txt
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe scripts/neuron_panel.py --package panel --out outputs/neuron_panel_new
.\.venv\Scripts\python.exe scripts/verify_neuron_panel.py outputs/neuron_panel_new --out outputs/neuron_panel_check_new
```

Use fresh directories. The runner verifies an archived baseline, executes the locked
discovery and confirmation cohorts, and checks frozen/refit decoding. See the report for source hashes,
per-seed results, pairing checks and limitations. Historical results remain intact.
[ACT I protocol](docs/whole-brain-memory-protocol.md) · [Data and attribution](data/README.md) ·
[Primary-source review](docs/research.md) · [Historical reproduction](docs/historical-reproduction.md) · [License](LICENSE)
