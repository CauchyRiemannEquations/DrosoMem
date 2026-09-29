# Flying

Flying studies sequential recall in **computational models using actual Drosophila
connectome structure**. Pi is a trained-sequence benchmark; this is not evidence
that a living fly memorized pi or that the model predicts unseen pi digits.

[한국어 설명](docs/README-ko.md) · [Research status / audit](docs/research-status.md) ·
[Roadmap](docs/research-roadmap.md) · [ACT I results](docs/whole-brain-memory-results.md)

## Current research

Latest ACT IV study: [local directions through learning](docs/reward-trajectory-results.md),
complete with **inconclusive temporal endpoints**. The original 13 learning runs,
50 checkpoints and 11,130 evaluation trajectories replay exactly. Tiny learned
connections constrain the common probe radius below the preregistered resolution
threshold. This does not show that useful learning signals disappear.

Prior [initial-direction](docs/reward-direction-results.md) and
[noise-match](docs/reward-noise-results.md) negatives remain preserved.
Next: diagnose infinitesimal changes in fixed-code score margins on the same
checkpoints; predeclare a distinct sensitivity metric, without changing learning.

Historical recall baselines freeze recurrent connectivity and train an external
readout; the latest diagnostic additionally trains existing KC→MBON weights. Distinguish **representation**, **decoding** and **internal learning**.
Teacher-forced accuracy, autonomous recall and robustness are separate measurements.

## Reproduce the latest diagnostic

Use Python3.12 and a clean environment; commands run from the repository root.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-act1-lock.txt
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe scripts/reward_trajectory.py --out outputs/reward_trajectory_new
.\.venv\Scripts\python.exe scripts/verify_reward_trajectory.py outputs/reward_trajectory_new --out outputs/reward_trajectory_check_new
```

Use the committed source artifacts and fresh output directories. The runner exactly replays original training, captures fixed epochs and evaluates
locked local/random directional probes. See the report for source hashes,
per-seed results, pairing checks and limitations. Historical results remain intact.
[ACT I protocol](docs/whole-brain-memory-protocol.md) · [Data and attribution](data/README.md) ·
[Primary-source review](docs/research.md) · [Historical reproduction](docs/historical-reproduction.md) · [License](LICENSE)
