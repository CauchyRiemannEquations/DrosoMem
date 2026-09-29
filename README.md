# Flying

Flying studies sequential recall in **computational models using actual Drosophila
connectome structure**. Pi is a trained-sequence benchmark; this is not evidence
that a living fly memorized pi or that the model predicts unseen pi digits.

[한국어 설명](docs/README-ko.md) · [Research status / audit](docs/research-status.md) ·
[Roadmap](docs/research-roadmap.md) · [ACT I results](docs/whole-brain-memory-results.md)

## Current research

Latest: [scalar reward with local eligibility](docs/local-reward-results.md), an
ACT IV follow-up to the completed [IV-A comparison](docs/readout-dependency-results.md).
39 cases were independently replayed. Fixed-code accuracy is 32.45/29.83% with
contingent reward versus 27.15/26.13% without learning in discovery/confirmation.
Some seeds improve, but neither cohort passes the full preregistered primary.
Frozen trained-ridge decoding already reaches 100/99.93%; representation improvement
is not established. The prior artificial vector-teacher success remains separate.

Next: one diagnostic of whether local updates point toward higher fixed-code reward,
with weights frozen during measurement and norm-matched direction controls.
No post-outcome learning-rate search. All previous failures and artifacts remain.

Historical recall baselines freeze recurrent connectivity and train an external
readout; the latest diagnostic additionally trains existing KC→MBON weights. Distinguish **representation**, **decoding** and **internal learning**.
Teacher-forced accuracy, autonomous recall and robustness are separate measurements.

## Reproduce the latest diagnostic

Use Python3.12 and a clean environment; commands run from the repository root.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-act1-lock.txt
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe scripts/local_reward.py --out outputs/local_reward_new
.\.venv\Scripts\python.exe scripts/verify_local_reward.py outputs/local_reward_new --out outputs/local_reward_check_new
```

Use the committed source artifacts and fresh output directories. The runner verifies an archived baseline, executes the locked
discovery and confirmation cohorts, and checks internal updates plus trained/fixed decoding. See the report for source hashes,
per-seed results, pairing checks and limitations. Historical results remain intact.
[ACT I protocol](docs/whole-brain-memory-protocol.md) · [Data and attribution](data/README.md) ·
[Primary-source review](docs/research.md) · [Historical reproduction](docs/historical-reproduction.md) · [License](LICENSE)
