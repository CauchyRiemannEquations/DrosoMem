# Flying

Flying studies sequential recall in **computational models using actual Drosophila
connectome structure**. Pi is a trained-sequence benchmark; this is not evidence
that a living fly memorized pi or that the model predicts unseen pi digits.

[한국어 설명](docs/README-ko.md) · [Research status / audit](docs/research-status.md) ·
[Roadmap](docs/research-roadmap.md) · [ACT I results](docs/whole-brain-memory-results.md)

## Current research

Latest: [ACT IV-A readout-dependency comparison](docs/readout-dependency-results.md).
The bounded A/B/C/D comparison is complete:117 network/training cases across
real,random and role-preserving graphs, with trained ridge and fixed-code decoding.
On real graphs, artificial internal teaching raises fixed-code accuracy from
21.92/27.83% to49.58/51.52% in discovery/confirmation. Ridge already reads lag2
at99.93%, so increased memory capacity is not established. Random controls also
reach100% with ridge; this endpoint has a ceiling and shows no original-wiring advantage.

This is supervised internal recoding with an artificial target, not biological
learning. Next one proposed study: scalar reward and local eligibility without
the target-code vector. Earlier negative findings and all raw artifacts remain.

Historical recall baselines freeze recurrent connectivity and train an external
readout; the latest diagnostic additionally trains existing KC→MBON weights. Distinguish **representation**, **decoding** and **internal learning**.
Teacher-forced accuracy, autonomous recall and robustness are separate measurements.

## Reproduce the latest diagnostic

Use Python3.12 and a clean environment; commands run from the repository root.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-act1-lock.txt
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe scripts/readout_dependency.py --out outputs/readout_dependency_new
.\.venv\Scripts\python.exe scripts/verify_readout_dependency.py outputs/readout_dependency_new --out outputs/readout_dependency_check_new
```

Use the committed source artifacts and fresh output directories. The runner verifies an archived baseline, executes the locked
discovery and confirmation cohorts, and checks internal updates plus trained/fixed decoding. See the report for source hashes,
per-seed results, pairing checks and limitations. Historical results remain intact.
[ACT I protocol](docs/whole-brain-memory-protocol.md) · [Data and attribution](data/README.md) ·
[Primary-source review](docs/research.md) · [Historical reproduction](docs/historical-reproduction.md) · [License](LICENSE)
