# Flying

Flying studies sequential recall in **computational models using actual Drosophila
connectome structure**. Pi is a trained-sequence benchmark, not evidence of a living
fly memorizing pi or a model predicting unseen pi digits.

[한국어 설명](docs/README-ko.md) · [Research status / audit](docs/research-status.md) ·
[Roadmap](docs/research-roadmap.md) · [Latest results](docs/graph-relative-noise-results.md)

## Current research

The [graph-relative noise follow-up](docs/graph-relative-noise-results.md) is complete: its
random/degree gap-attenuation criterion is **not confirmed**. It reused 96 partial-graph
heads with fresh noise, reproduced every clean baseline and independently replayed
all 1,824 paths. This tests a noise calibration effect, not intact wiring superiority.

The bounded ACT I–V studies and the [three additional computational studies](docs/additional-research-results.md)
are complete within their registered scopes. Completion does not mean every hypothesis succeeded.

| Additional question | Outcome |
| --- | --- |
| Does observation-noise allocation help beyond pi? | Confirmed on random digits and shuffled pi, including fresh model/task seeds |
| Is intact partial wiring better than role-preserving controls? | Not confirmed; keep the negative result |
| Does allocation still help with synthetic spatially correlated observation noise? | Confirmed under the registered random/brain1 criterion |

All use 48 observed MBONs and a 482-parameter readout. These studies test noise allocation
and partial wiring; they establish no whole-brain superiority or internal learning. Strong-noise recall remains weak.
There are 8,640 certificate evaluations (6,336 distinct exposure settings), 1,144 actual evaluation
trajectories, 1,074 independent full replays and 100 new readout fits. Certificates stop at the first
error; they are not full autonomous sequences. Physiological validation remains open.

## Reproduce the studies

Use Python 3.12, a fresh environment and new output directories. Run from the repository root.
For the latest frozen-head calibration, see the [exact commands](docs/graph-relative-noise-results.md#10-reproducibility).
The commands below reproduce the preceding extensions.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-act1-lock.txt
$env:PYTHONPATH = 'src'
# Prepare the pinned graph cache only if absent:
.\.venv\Scripts\python.exe -m flying.training.phase6 prepare --cache outputs/act1-graphs
.\.venv\Scripts\python.exe scripts/research_suite.py run --stage sequence --out outputs/suite_sequence_new
.\.venv\Scripts\python.exe scripts/verify_research_suite.py outputs/suite_sequence_new --out outputs/suite_sequence_new_validation
.\.venv\Scripts\python.exe scripts/research_suite.py run --stage structure --parent outputs/suite_sequence_new --out outputs/suite_structure_new
.\.venv\Scripts\python.exe scripts/verify_research_suite.py outputs/suite_structure_new --out outputs/suite_structure_new_validation
.\.venv\Scripts\python.exe scripts/research_suite.py run --stage correlation --parent outputs/suite_sequence_new --structural-validation outputs/suite_structure_new_validation --out outputs/suite_correlation_new
.\.venv\Scripts\python.exe scripts/verify_research_suite.py outputs/suite_correlation_new --out outputs/suite_correlation_new_validation
```

Keep the committed parent artifacts. The [protocol](docs/additional-research-completion-plan.md)
defines seeds, budgets, stopping rules and replay subsets; individual reports contain raw tables,
checkpoints and plotting commands. Distinguish **representation**, **decoding** and **internal learning**.
Next single proposal: graph-relative noise calibration to separate signal scale from wiring effects;
not registered or executed. [Data and attribution](data/README.md) · [Primary-source review](docs/research.md) ·
[Historical reproduction](docs/historical-reproduction.md) · [License](LICENSE)
