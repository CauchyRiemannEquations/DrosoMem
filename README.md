# Flying

Flying studies sequential recall in **computational models using actual Drosophila
connectome structure**. Pi is a trained-sequence benchmark; this is not evidence
that a living fly memorized pi or that the model predicts unseen pi digits.

[한국어 설명](docs/README-ko.md) · [Research status / audit](docs/research-status.md) ·
[Roadmap](docs/research-roadmap.md) · [ACT I results](docs/whole-brain-memory-results.md)

## Current research

Latest ACT IV diagnostics: [local update directions](docs/reward-direction-results.md)
and [training/evaluation noise matching](docs/reward-noise-results.md), both complete.
Initial local directions improve noisy correctness in all six seed blocks, but one
misses the random-direction comparison, so the full confirmation criterion fails.
Re-evaluating the older final weights with training noise yields31.47/30.52%
fixed-code accuracy; coding and noise-interaction criteria still fail. Neither
study establishes reliable internal memory learning or increased memory capacity.

Next: inspect local update directions at fixed points along the same learning
trajectory. No new learning rule, parameter search or best-epoch selection.
Prior vector-teacher positives and scalar-reward negatives remain preserved.

Historical recall baselines freeze recurrent connectivity and train an external
readout; the latest diagnostic additionally trains existing KC→MBON weights. Distinguish **representation**, **decoding** and **internal learning**.
Teacher-forced accuracy, autonomous recall and robustness are separate measurements.

## Reproduce the latest diagnostic

Use Python3.12 and a clean environment; commands run from the repository root.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-act1-lock.txt
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe scripts/reward_noise.py --out outputs/reward_noise_new
.\.venv\Scripts\python.exe scripts/verify_reward_noise.py outputs/reward_noise_new --out outputs/reward_noise_check_new
```

Use the committed source artifacts and fresh output directories. The runner exactly replays archived clean scores and evaluates the locked
noise conditions without retraining. See the report for source hashes,
per-seed results, pairing checks and limitations. Historical results remain intact.
[ACT I protocol](docs/whole-brain-memory-protocol.md) · [Data and attribution](data/README.md) ·
[Primary-source review](docs/research.md) · [Historical reproduction](docs/historical-reproduction.md) · [License](LICENSE)
