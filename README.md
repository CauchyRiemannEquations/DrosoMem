# Flying

Flying studies sequential recall in **computational models using actual Drosophila
connectome structure**. Pi is a trained-sequence benchmark; this is not evidence
that a living fly memorized pi or that the model predicts unseen pi digits.

[한국어 설명](docs/README-ko.md) · [Research status / audit](docs/research-status.md) ·
[Roadmap](docs/research-roadmap.md) · [ACT I results](docs/whole-brain-memory-results.md)

## Current research

The [bounded ACT V programme remains complete](docs/act5-final-results.md).
The [additional-research briefing](docs/additional-research-brief.md) prioritizes
autonomous feedback, sequence generalization, structural controls and realistic noise.
Only the first extension has now been executed: [feedback results](docs/feedback-noise-results.md).

Brain1's registered joint prefix/retention improvement is **confirmed**.
Fresh-noise mean prefix gains are 3.700/6.167 symbols and retention gains 10.986/16.283
percentage points in the two reused model cohorts. This compares two observation
noise allocations within a graph; it is not a whole-brain superiority test.
Allocated brain1 retains only 45.55/50.76% of its clean prefix on average across
these doses, and still collapses at the strongest dose.

All 2016 first-error certificates are distinguished from 336 actual neural rollouts.
Every saved metric and pre-error match passes; 140 full paths are independently
replayed. Existing negative findings and the ACT V closeout remain unchanged.
Next proposal: the same contrast on fixed-seed random digits, not yet registered
or executed. Distinguish **representation**, **decoding** and **internal learning**.

## Reproduce the latest diagnostic

Use Python 3.12 and a clean environment; commands run from the repository root.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-act1-lock.txt
$env:PYTHONPATH = 'src'
# Prepare pinned graph cache only if absent:
.\.venv\Scripts\python.exe -m flying.training.phase6 prepare --cache outputs/act1-graphs
.\.venv\Scripts\python.exe scripts/feedback_noise.py run --out outputs/feedback_new
.\.venv\Scripts\python.exe scripts/verify_feedback_noise.py outputs/feedback_new --out outputs/feedback_check_new
```

Use the committed source artifacts, pinned graph cache and fresh output directories.
This extension reuses clean heads and performs no training. Three new noise streams
give certified exact prefixes; only the first has full autonomous outputs after
the first error. See the report for raw data and the independent replay scope.
[ACT I protocol](docs/whole-brain-memory-protocol.md) · [Data and attribution](data/README.md) ·
[Primary-source review](docs/research.md) · [Historical reproduction](docs/historical-reproduction.md) · [License](LICENSE)
