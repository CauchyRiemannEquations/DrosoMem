# Flying

Flying studies sequential recall in **computational models using actual Drosophila
connectome structure**. Pi is a trained-sequence benchmark; this is not evidence
that a living fly memorized pi or that the model predicts unseen pi digits.

[한국어 설명](docs/README-ko.md) · [Research status / audit](docs/research-status.md) ·
[Roadmap](docs/research-roadmap.md) · [ACT I results](docs/whole-brain-memory-results.md)

## Current research

Latest: [fresh-seed alignment confirmation](docs/fresh-alignment-results.md)
replicates partial improvement in both directions:25–26% unaligned becomes45–47%
aligned, versus78–80% target refit. All three new paired seeds improve; the
registered access/retention gates still fail. Existing negative findings remain.

Current position: scoped ACT III normalization/readout diagnostics and fresh-seed
confirmation complete. Next proposed: a matched KCγ ablation with condition-specific
readout refits. Broader ablations and internal learning remain open.

The established models freeze recurrent connectivity and train an external
readout. Distinguish **representation**, **decoding** and **internal learning**.
Teacher-forced accuracy, autonomous recall and robustness are separate measurements.

## Reproduce the latest diagnostic

Use Python3.12 and a clean environment; commands run from the repository root.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-act1-lock.txt
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe scripts/fresh_alignment.py --out outputs/fresh_alignment_new
.\.venv\Scripts\python.exe scripts/verify_fresh_alignment.py outputs/fresh_alignment_new --out outputs/fresh_alignment_check_new
```

Use fresh directories. The runner checks archived baselines, then simulates new
paired seeds using committed circuit data; no new graph download is needed. See the report for source hashes,
per-seed results, pairing checks and limitations. Historical results remain intact.
[ACT I protocol](docs/whole-brain-memory-protocol.md) · [Data and attribution](data/README.md) ·
[Primary-source review](docs/research.md) · [Historical reproduction](docs/historical-reproduction.md) · [License](LICENSE)
