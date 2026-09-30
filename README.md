# Flying

Flying studies sequential recall in **computational models using actual Drosophila
connectome structure**. Pi is a trained-sequence benchmark; this is not evidence
that a living fly memorized pi or that the model predicts unseen pi digits.

[한국어 설명](docs/README-ko.md) · [Research status / audit](docs/research-status.md) ·
[Roadmap](docs/research-roadmap.md) · [ACT I results](docs/whole-brain-memory-results.md)

## Current research

The [bounded ACT V rate-model programme](docs/act5-robustness-results.md) is complete:
five perturbation curves in partial/whole graphs, 30 archived heads and 18 fresh
confirmation heads, under fixed 48-MBON observation and frozen readouts.
The main ongoing-noise brain1-versus-legacy5 criterion is **not confirmed**.
Brain1's mean capped retention was lower by 22.1 percentage points in discovery
and 21.0 in fresh confirmation. Equal absolute noise had unequal size relative
to each model's observed-state variation, limiting a topology-only interpretation.
All per-seed results, secondary comparisons and failures remain available.

The [completion audit](results/act5_closeout/audit.json) distinguishes execution
from biological validation. Computational noise is not physiologically calibrated.
Next proposal: compare ongoing noise at matched dose relative to clean observed-state
variation, retaining frozen heads. Preregister it before execution; it has not been run.

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
.\.venv\Scripts\python.exe scripts/act5_robustness.py run --out outputs/act5_new
.\.venv\Scripts\python.exe scripts/verify_act5_robustness.py outputs/act5_new --out outputs/act5_check_new
```

Use the committed source artifacts and fresh output directories. The runner reuses archived clean heads, fits fixed fresh-seed clean heads and
evaluates five registered perturbation families without refitting under perturbation. See the report for source hashes,
per-seed results, pairing checks and limitations. Historical results remain intact.
[ACT I protocol](docs/whole-brain-memory-protocol.md) · [Data and attribution](data/README.md) ·
[Primary-source review](docs/research.md) · [Historical reproduction](docs/historical-reproduction.md) · [License](LICENSE)
