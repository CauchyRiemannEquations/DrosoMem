# Flying

Flying studies sequential recall in **computational models using actual Drosophila
connectome structure**. Pi is a trained-sequence benchmark; this is not evidence
that a living fly memorized pi or that the model predicts unseen pi digits.

[한국어 설명](docs/README-ko.md) · [Research status / audit](docs/research-status.md) ·
[Roadmap](docs/research-roadmap.md) · [ACT I results](docs/whole-brain-memory-results.md)

## Current research

The [bounded ACT V programme is closed](docs/act5-final-results.md): five perturbation
curves and three diagnostics, with preserved per-seed results and independent checks.
The original robustness criteria and whole-brain recall advantage remain unconfirmed.

The final [coordinate-noise control](docs/allocation-noise-results.md) holds expected
total squared observation noise fixed and redistributes it by clean training SD.
All three graphs pass the registered teacher-decoding improvement criterion. With
three fresh noise draws, brain1 gains 21.00/20.19 percentage points in the two reused
model cohorts, but remains 18.46/18.00 points below clean accuracy. This does not
establish recovered autonomous recall or physiological robustness.

[Final audit](results/act5_final_closeout/audit.json) ·
[Original robustness](docs/act5-robustness-results.md) ·
[Relative noise](docs/relative-noise-results.md) ·
[Observation diagnostic](docs/observation-noise-results.md).
Future proposal: test the same coordinate allocations in autonomous feedback;
not registered or executed, and outside the completed scope.

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
.\.venv\Scripts\python.exe scripts/allocation_noise.py run --out outputs/allocation_new
.\.venv\Scripts\python.exe scripts/verify_allocation_noise.py outputs/allocation_new --out outputs/allocation_check_new
```

Use the committed source artifacts and fresh output directories. This diagnostic
reuses clean heads and saved states, computes 1152 nonzero observation evaluations
and performs no new training or neural rollout. The latest diagnostic needs no
graph-cache rebuild. Historical dynamical studies require their pinned cache and
source revision; see the reports for exact commands and verification scope.
[ACT I protocol](docs/whole-brain-memory-protocol.md) · [Data and attribution](data/README.md) ·
[Primary-source review](docs/research.md) · [Historical reproduction](docs/historical-reproduction.md) · [License](LICENSE)
