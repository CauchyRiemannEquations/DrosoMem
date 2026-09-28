# Flying

Flying studies sequential recall in **computational models using actual Drosophila
connectome structure**. Pi is a trained-sequence benchmark; this is not evidence
that a living fly memorized pi or that the model predicts unseen pi digits.

[한국어 설명](docs/README-ko.md) · [Research status / audit](docs/research-status.md) ·
[Roadmap](docs/research-roadmap.md) · [ACT I results](docs/whole-brain-memory-results.md)

## Current research

Latest: [train-only moment alignment](docs/moment-alignment-results.md) raises
frozen-head transfer from25–27% to49–53% without target-label fitting. It still
fails the registered access/retention gates: target refits reach79–80%, and
aligned regression R² remains negative. Improvement is distinct from portability.

Current position: scoped ACT III structure, normalization and alignment diagnostics
complete. Next proposed: decompose residual state mismatch and fixed-head
amplification. Broader ablations and internal learning remain open.

The established models freeze recurrent connectivity and train an external
readout. Distinguish **representation**, **decoding** and **internal learning**.
Teacher-forced accuracy, autonomous recall and robustness are separate measurements.

## Reproduce the latest diagnostic

Use Python3.12 and a clean environment; commands run from the repository root.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-act1-lock.txt
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe scripts/moment_alignment.py --out outputs/moment_new
.\.venv\Scripts\python.exe scripts/verify_moment_alignment.py --out outputs/moment_check_new outputs/moment_new
```

Use fresh directories. This diagnostic reuses committed checkpoints; no new
graph download or simulation is needed. See the report for source hashes,
per-seed results, pairing checks and limitations. Historical results remain intact.
[ACT I protocol](docs/whole-brain-memory-protocol.md) · [Data and attribution](data/README.md) ·
[Primary-source review](docs/research.md) · [Historical reproduction](docs/historical-reproduction.md) · [License](LICENSE)
