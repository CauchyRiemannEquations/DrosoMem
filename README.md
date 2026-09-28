# Flying

Flying studies sequential recall in **computational models using actual Drosophila
connectome structure**. Pi is a trained-sequence benchmark; this is not evidence
that a living fly memorized pi or that the model predicts unseen pi digits.

[한국어 설명](docs/README-ko.md) · [Research status / audit](docs/research-status.md) ·
[Roadmap](docs/research-roadmap.md) · [ACT I results](docs/whole-brain-memory-results.md)

## Current research

Latest: the [current K4 structural comparison](docs/structural-k4-results.md)
completed20 graph runs with exact replay. Independent past-symbol decoding is
76.965% for the real partial graph versus78.603% for role-/degree-preserving
rewiring. Real-wiring superiority fails the preregistered criterion. Both decode
past inputs above controls; this is not an autonomous-recall result.

Current position: a scoped ACT III structural control is complete. Broader
ablations and internal learning remain open. Next proposed control separates
the contribution of renormalization on the same rewired graphs.

The established models freeze recurrent connectivity and train an external
readout. Distinguish **representation**, **decoding** and **internal learning**.
Teacher-forced accuracy, autonomous recall and robustness are separate measurements.

## Reproduce the latest diagnostic

Use Python3.12 and a clean environment; commands run from the repository root.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-act1-lock.txt
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe scripts/structural_k4.py --cohort smoke --out outputs/structural_smoke_new
.\.venv\Scripts\python.exe scripts/structural_k4.py --cohort main --out outputs/structural_main_new
```

Use fresh directories. This comparison uses committed partial graphs and needs
no whole-brain graph download. See the result report for independent verification,
source hashes, checkpoints and limitations. Historical results remain intact.
[ACT I protocol](docs/whole-brain-memory-protocol.md) · [Data and attribution](data/README.md) ·
[Primary-source review](docs/research.md) · [Historical reproduction](docs/historical-reproduction.md) · [License](LICENSE)
