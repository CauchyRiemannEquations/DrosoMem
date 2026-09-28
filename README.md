# Flying

Flying studies sequential recall in **computational models using actual Drosophila
connectome structure**. Pi is a trained-sequence benchmark; this is not evidence
that a living fly memorized pi or that the model predicts unseen pi digits.

[한국어 설명](docs/README-ko.md) · [Research status / audit](docs/research-status.md) ·
[Roadmap](docs/research-roadmap.md) · [ACT I results](docs/whole-brain-memory-results.md)

## Current research

Latest: [ACT II-A alphabet comparison](docs/alphabet-memory-results.md), with a
[locked protocol](docs/alphabet-memory-protocol.md). Eighty fits at fixed N128
compare K2/4/10/16 using matched48-MBON observations within each K. Both models
pass scoped trained-prefix criteria, but no K establishes whole-brain superiority.
Mean recall rises from K2 toK16; this is not evidence of increased intrinsic capacity.

The next diagnostic asks whether reduced short-context ambiguity helps explain
that curve. It is not yet executed. ACT III structural controls and ACT IV
internal learning remain separate. Historical findings, including negative
whole-brain, plasticity and robustness results, remain in the research-status index.

The established models freeze recurrent connectivity and train an external
readout. Distinguish **representation**, **decoding** and **internal learning**.
Teacher-forced accuracy, autonomous recall and robustness are separate measurements.

## Reproduce the alphabet study

Use Python3.12 and a clean environment; commands run from the repository root.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-act1-lock.txt
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe -m flying.training.phase6 prepare --cache outputs/act1-graphs
.\.venv\Scripts\python.exe scripts/run_alphabet_memory.py run --config configs/alphabet_memory.json --out outputs/alphabet_new
.\.venv\Scripts\python.exe scripts/run_alphabet_memory.py verify --source outputs/alphabet_new --out outputs/alphabet_verify_new
.\.venv\Scripts\python.exe scripts/summarize_alphabet_memory.py --source outputs/alphabet_new --out outputs/alphabet_analysis_new
```

Use fresh output directories. Raw downloads and sparse graph caches are excluded
from Git; provenance and SHA-256 validation are mandatory.
[ACT I protocol](docs/whole-brain-memory-protocol.md) · [Data and attribution](data/README.md) ·
[Primary-source review](docs/research.md) · [Historical reproduction](docs/historical-reproduction.md) · [License](LICENSE)
