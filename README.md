# Flying

Flying studies sequential recall in **computational models using actual Drosophila
connectome structure**. Pi is a trained-sequence benchmark; this is not evidence
that a living fly memorized pi or that the model predicts unseen pi digits.

[한국어 설명](docs/README-ko.md) · [Research status / audit](docs/research-status.md) ·
[Roadmap](docs/research-roadmap.md) · [ACT I protocol](docs/whole-brain-memory-protocol.md) ·
[ACT I results](docs/whole-brain-memory-results.md)

Latest: [Random-sequence length scaling](docs/length-scaling-results.md),
with a [locked protocol](docs/length-scaling-protocol.md): 100 paired fits,
100 exact replays and 10 independent refits. Both models complete N32;
at N512, mean prefixes are partial14.6 / whole-brain9.5. No whole-brain
advantage is established under this fixed training procedure.

Earlier: [ACT II four-family protocol](docs/sequence-memory-protocol.md) and
[results](docs/sequence-memory-results.md). Eighty fits support limited trained
random/shuffled-sequence recall beyond pi, with no established whole-brain
advantage. Periodic recall is also solved by a simple first-order predictor.

## Current research

Does whole-brain connectivity improve autonomous recall when input root IDs,
48 observed MBONs, 482 readout parameters and training budgets are fixed?
ACT I compares the original partial graph, expanded left mushroom-body graphs
and whole-brain graphs at contact thresholds 5 and 1.

The established baseline freezes recurrent connectivity and trains an external
readout. Distinguish **representation**, **decoding** and **internal learning**.
Clean recall, teacher-forced accuracy and robustness are separate measurements.
[Historical negative results and completed studies](docs/research-status.md)
remain part of the evidence.

## Reproduce ACT I

Use Python 3.12 and a clean environment; commands run from the repository root.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-act1-lock.txt
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe -m flying.training.phase6 prepare --cache outputs/act1-graphs
.\.venv\Scripts\python.exe -m flying.training.whole_brain_memory run --out outputs/act1_new
.\.venv\Scripts\python.exe -m flying.training.whole_brain_memory verify --out outputs/act1_new --refit
.\.venv\Scripts\python.exe scripts/summarize_whole_brain_memory.py --source outputs/act1_new --out outputs/act1_analysis_new
```

Use fresh output directories. Raw source downloads and sparse graph caches are
excluded from Git; provenance and SHA-256 validation are mandatory.
[Data and attribution](data/README.md) · [Primary-source review](docs/research.md) ·
[Historical reproduction](docs/historical-reproduction.md) · [License](LICENSE)
