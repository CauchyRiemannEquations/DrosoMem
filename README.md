# Flying

Flying studies sequential recall in **computational models using actual Drosophila
connectome structure**. Pi is a trained-sequence benchmark; this is not evidence
that a living fly memorized pi or that the model predicts unseen pi digits.

[한국어 설명](docs/README-ko.md) · [Research status / audit](docs/research-status.md) ·
[Roadmap](docs/research-roadmap.md) · [ACT I results](docs/whole-brain-memory-results.md)

## Current research

The [bounded ACT V programme](docs/act5-robustness-results.md) and two follow-ups
are complete. [Relative-noise calibration](docs/relative-noise-results.md) still
does **not confirm a whole-brain recall advantage**: brain1 minus partial capped
retention is −2.07 percentage points in discovery and −5.71 in fresh confirmation.
The discovery gap is smaller than under the earlier absolute-dose definition.

The [observation-noise diagnostic](docs/observation-noise-results.md) reuses those
48 frozen heads. Corrupting only current observations of otherwise clean states
also sharply lowers teacher-forced decoding accuracy. Accumulated state noise adds
0.54/0.76 percentage points of mean loss in brain1. This is not evidence of lost
internal memory or recovered autonomous recall. All seeds and failures are retained.

The [two-study audit](results/observation_noise_closeout/audit.json) distinguishes
completed computation from physiological validation. Next proposal, not executed:
redistribute observation noise by each MBON's clean training variation while matching
expected total squared noise and keeping the readout fixed.

Historical recall baselines freeze recurrent connectivity and train an external
readout. ACT IV also studied internal KC→MBON learning; ACT V reuses frozen ACT I
reservoirs. Distinguish **representation**, **decoding** and **internal learning**.
Teacher-forced accuracy, autonomous recall and robustness are separate measurements.

## Reproduce the latest diagnostic

Use Python3.12 and a clean environment; commands run from the repository root.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-act1-lock.txt
$env:PYTHONPATH = 'src'
# Prepare the pinned graph cache if it is not already present:
.\.venv\Scripts\python.exe -m flying.training.phase6 prepare --cache outputs/act1-graphs
.\.venv\Scripts\python.exe scripts/observation_noise.py run --out outputs/observation_noise_new
.\.venv\Scripts\python.exe scripts/verify_observation_noise.py outputs/observation_noise_new --out outputs/observation_noise_check_new
```

Use the committed source artifacts and fresh output directories. This diagnostic
reuses clean heads and saved states, computes 144 observation controls and performs
no new training or autonomous rollout. See both reports for the parent dynamical
experiment, source hashes, per-seed results and independent verification scope.
[ACT I protocol](docs/whole-brain-memory-protocol.md) · [Data and attribution](data/README.md) ·
[Primary-source review](docs/research.md) · [Historical reproduction](docs/historical-reproduction.md) · [License](LICENSE)
