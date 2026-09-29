# Flying

Flying studies sequential recall in **computational models using actual Drosophila
connectome structure**. Pi is a trained-sequence benchmark; this is not evidence
that a living fly memorized pi or that the model predicts unseen pi digits.

[한국어 설명](docs/README-ko.md) · [Research status / audit](docs/research-status.md) ·
[Roadmap](docs/research-roadmap.md) · [ACT I results](docs/whole-brain-memory-results.md)

## Current research

The [bounded ACT IV current-model programme](docs/act4-results.md) is complete:
A/B/C/D readout controls, scalar/local learning and its preregistered diagnostics.
The final [margin sensitivity study](docs/reward-margin-results.md) includes three
fresh paired seeds, 74 checkpoints and independent reverse-adjoint verification.
Past coding failures and inconclusive finite probes remain preserved.

Completion means the specified experiments and checks finished. It does not mean
physiological learning, whole-brain local memory or literal decoder-free biology
has been established. Next is a separately preregistered ACT V noise-robustness
curve comparison; no further ACT IV parameter search is queued.

Historical recall baselines freeze recurrent connectivity and train an external
readout; the latest diagnostic additionally trains existing KC→MBON weights. Distinguish **representation**, **decoding** and **internal learning**.
Teacher-forced accuracy, autonomous recall and robustness are separate measurements.

## Reproduce the latest diagnostic

Use Python3.12 and a clean environment; commands run from the repository root.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-act1-lock.txt
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe scripts/reward_margin.py --out outputs/reward_margin_new
.\.venv\Scripts\python.exe scripts/verify_reward_margin.py outputs/reward_margin_new --out outputs/reward_margin_check_new
```

Use the committed source artifacts and fresh output directories. The runner reuses archived checkpoints, executes fixed fresh-seed training and
computes analytic local/random margin sensitivities without changing probe weights. See the report for source hashes,
per-seed results, pairing checks and limitations. Historical results remain intact.
[ACT I protocol](docs/whole-brain-memory-protocol.md) · [Data and attribution](data/README.md) ·
[Primary-source review](docs/research.md) · [Historical reproduction](docs/historical-reproduction.md) · [License](LICENSE)
