# Flying

Flying studies sequential recall in **computational models using actual Drosophila
connectome structure**. Pi is a trained-sequence benchmark; this is not evidence
that a living fly memorized pi or that the model predicts unseen pi digits.

[한국어 설명](docs/README-ko.md) · [Research status / audit](docs/research-status.md) ·
[Roadmap](docs/research-roadmap.md) · [ACT I results](docs/whole-brain-memory-results.md)

## Current research

Latest: [ACT III-B edge-ablation closeout](docs/edge-panel-results.md).
The raw-weight-bin control replicated a DAN-associated refit loss of about7pp.
A separately preregistered effective-normalized-weight control on fresh cohorts
found2.59/1.77pp, below the fixed5pp criterion. Greater target/control overlap
and changed cohorts limit attributing this difference to normalization alone.

Current position: ACT III-B current-model panel complete:286 conditions with
frozen/refit separation and independent verification. Whole-brain,full dose curves
and autonomous-recall localization remain untested. Next: one role/degree and
postsynaptic incoming-weight-preserving rewired control (ACT III-C).

The established models freeze recurrent connectivity and train an external
readout. Distinguish **representation**, **decoding** and **internal learning**.
Teacher-forced accuracy, autonomous recall and robustness are separate measurements.

## Reproduce the latest diagnostic

Use Python3.12 and a clean environment; commands run from the repository root.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-act1-lock.txt
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe scripts/edge_strength.py --out outputs/edge_strength_new
.\.venv\Scripts\python.exe scripts/verify_edge_strength.py outputs/edge_strength_new --out outputs/edge_strength_check_new
```

Use fresh directories. The runner verifies an archived baseline, executes the locked
discovery and confirmation cohorts, and checks frozen/refit decoding. See the report for source hashes,
per-seed results, pairing checks and limitations. Historical results remain intact.
[ACT I protocol](docs/whole-brain-memory-protocol.md) · [Data and attribution](data/README.md) ·
[Primary-source review](docs/research.md) · [Historical reproduction](docs/historical-reproduction.md) · [License](LICENSE)
