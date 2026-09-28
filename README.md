# Flying

Flying studies sequential recall in **computational models using actual Drosophila
connectome structure**. Pi is a trained-sequence benchmark; this is not evidence
that a living fly memorized pi or that the model predicts unseen pi digits.

[한국어 설명](docs/README-ko.md) · [Research status / audit](docs/research-status.md) ·
[Roadmap](docs/research-roadmap.md) · [ACT I results](docs/whole-brain-memory-results.md)

## Current research

Latest: [finite-context diagnostic](docs/context-memory-results.md), with a
[locked protocol](docs/context-memory-protocol.md). All120 context-table fits
were independently verified on20 unique saved sequences. Both diagnostic
endpoints pass: K2→K16 reduces order2 ambiguity and increases order3 recall.
Context5 completes all saved K10/K16 instances without a connectome.

This supports a task-level explanation candidate for the
[alphabet curve](docs/alphabet-memory-results.md), not a causal account of the
neural model. The next single study is a fixed-K context-conflict intervention;
it is not yet executed. ACT III controls and ACT IV internal learning remain
separate. Historical findings, including negative
whole-brain, plasticity and robustness results, remain in the research-status index.

The established models freeze recurrent connectivity and train an external
readout. Distinguish **representation**, **decoding** and **internal learning**.
Teacher-forced accuracy, autonomous recall and robustness are separate measurements.

## Reproduce the latest diagnostic

Use Python3.12 and a clean environment; commands run from the repository root.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-act1-lock.txt
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe scripts/context_memory.py --config configs/context_memory.json --out outputs/context_new
```

Use fresh output directories. This diagnostic uses saved checkpoints and needs
no graph download. Raw downloads and sparse graph caches are excluded
from Git; provenance and SHA-256 validation are mandatory.
[ACT I protocol](docs/whole-brain-memory-protocol.md) · [Data and attribution](data/README.md) ·
[Primary-source review](docs/research.md) · [Historical reproduction](docs/historical-reproduction.md) · [License](LICENSE)
