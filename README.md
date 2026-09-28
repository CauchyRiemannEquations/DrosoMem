# Flying

Flying studies sequential recall in **computational models using actual Drosophila
connectome structure**. Pi is a trained-sequence benchmark; this is not evidence
that a living fly memorized pi or that the model predicts unseen pi digits.

[한국어 설명](docs/README-ko.md) · [Research status / audit](docs/research-status.md) ·
[Roadmap](docs/research-roadmap.md) · [ACT I results](docs/whole-brain-memory-results.md)

## Current research

Latest: [fixed-K context-conflict intervention](docs/context-intervention-results.md),
with a [locked protocol](docs/context-intervention-protocol.md). Main40 fits
compare two constructed orderings with K4, N128, the same symbol multiset,
prompt, input map,48 MBONs and428-parameter head. The partial model's discovery
gain (+12.3) failed fresh confirmation (+7.0, only2/3 positive seed blocks).
No registered low-conflict benefit is confirmed.
All seeds and negative differences are retained; reordering changes other
sequence statistics too. This is not isolated biological causality.

Next: one frozen-state representation/decoding diagnostic on these saved pairs, not yet executed. ACT III structural attribution and ACT IV
internal learning remain separate. [Prior context diagnostic](docs/context-memory-results.md).

The established models freeze recurrent connectivity and train an external
readout. Distinguish **representation**, **decoding** and **internal learning**.
Teacher-forced accuracy, autonomous recall and robustness are separate measurements.

## Reproduce the latest diagnostic

Use Python3.12 and a clean environment; commands run from the repository root.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-act1-lock.txt
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe -m flying.training.phase6 prepare --cache outputs/act1-graphs
.\.venv\Scripts\python.exe scripts/run_context_intervention.py run --config configs/context_intervention_main.json --out outputs/intervention_new
.\.venv\Scripts\python.exe scripts/run_context_intervention.py verify --source outputs/intervention_new --out outputs/intervention_verify_new
.\.venv\Scripts\python.exe scripts/summarize_context_intervention.py --source outputs/intervention_new --out outputs/intervention_analysis_new
```

Use fresh output directories. The config reuses committed sequence pairs;
see the result document for independent sequence reconstruction. Raw downloads and sparse graph caches are excluded
from Git; provenance and SHA-256 validation are mandatory.
[ACT I protocol](docs/whole-brain-memory-protocol.md) · [Data and attribution](data/README.md) ·
[Primary-source review](docs/research.md) · [Historical reproduction](docs/historical-reproduction.md) · [License](LICENSE)
