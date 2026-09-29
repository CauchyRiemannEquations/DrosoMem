# Flying

Flying studies sequential recall in **computational models using actual Drosophila
connectome structure**. Pi is a trained-sequence benchmark; this is not evidence
that a living fly memorized pi or that the model predicts unseen pi digits.

[한국어 설명](docs/README-ko.md) · [Research status / audit](docs/research-status.md) ·
[Roadmap](docs/research-roadmap.md) · [ACT I results](docs/whole-brain-memory-results.md)

## Current research

Latest: [ACT III-D pathway sensitivity](docs/pathway-memory-results.md).
DAN→MBON cuts impair refit past-symbol decoding more than matched cuts by
7.33/8.32 percentage points in discovery/confirmation. KC→MBON also passes
(5.49/5.71pp), but loses current-symbol access too. Both feed the observed MBONs:
these are model decoding-pathway candidates, not identified storage locations.

III-A/B/C/D are complete within their two partial-model diagnostic scopes.
III-D adds273 fits/replays and260 frozen evaluations, with independent verification.
Earlier negative structural-control results remain valid. Next proposed study:
readout dependence using a fixed48-neuron observation budget and different locations.
Whole-brain localization and autonomous-recall mechanisms remain untested.

The established models freeze recurrent connectivity and train an external
readout. Distinguish **representation**, **decoding** and **internal learning**.
Teacher-forced accuracy, autonomous recall and robustness are separate measurements.

## Reproduce the latest diagnostic

Use Python3.12 and a clean environment; commands run from the repository root.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-act1-lock.txt
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe scripts/pathway_memory.py --out outputs/pathway_memory_new
.\.venv\Scripts\python.exe scripts/verify_pathway_memory.py outputs/pathway_memory_new --out outputs/pathway_memory_check_new
```

Use the committed mask bank and source artifacts; use fresh output directories. The runner verifies an archived baseline, executes the locked
discovery and confirmation cohorts, and checks frozen/refit decoding. See the report for source hashes,
per-seed results, pairing checks and limitations. Historical results remain intact.
[ACT I protocol](docs/whole-brain-memory-protocol.md) · [Data and attribution](data/README.md) ·
[Primary-source review](docs/research.md) · [Historical reproduction](docs/historical-reproduction.md) · [License](LICENSE)
