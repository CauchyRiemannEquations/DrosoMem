# Flying

Flying studies sequential recall in **computational models using actual Drosophila
connectome structure**. Pi is a trained-sequence benchmark; this is not evidence
that a living fly memorized pi or that the model predicts unseen pi digits.

[한국어 설명](docs/README-ko.md) · [Research status / audit](docs/research-status.md) ·
[Roadmap](docs/research-roadmap.md) · [ACT I results](docs/whole-brain-memory-results.md)

## Current research

Latest: [Observation-location diagnostic](docs/observation-location-results.md).
After DAN→MBON removal, MBON decoding loses7.88/7.27 percentage points in fresh
discovery/confirmation tasks;48 non-directly-stimulated KCs retain past-symbol
access with losses−0.08/−0.04pp. Both cohorts pass the locked location-dependence
rule. KC baseline accuracy is lower (~50% versus77%): this does not locate storage.

III-A/B/C/D are complete within their partial-model scopes. This ACT IV-A entry
diagnostic adds130 fits/replays; it does not establish internal synaptic learning.
Next proposed analysis: compare impairment at jointly accessible delays, under
separate rules fixed before that analysis. Earlier negative findings remain intact.

The established models freeze recurrent connectivity and train an external
readout. Distinguish **representation**, **decoding** and **internal learning**.
Teacher-forced accuracy, autonomous recall and robustness are separate measurements.

## Reproduce the latest diagnostic

Use Python3.12 and a clean environment; commands run from the repository root.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-act1-lock.txt
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe scripts/observation_location.py --out outputs/observation_location_new
.\.venv\Scripts\python.exe scripts/verify_observation_location.py outputs/observation_location_new --out outputs/observation_location_check_new
```

Use the committed mask bank and source artifacts; use fresh output directories. The runner verifies an archived baseline, executes the locked
discovery and confirmation cohorts, and checks frozen/refit decoding. See the report for source hashes,
per-seed results, pairing checks and limitations. Historical results remain intact.
[ACT I protocol](docs/whole-brain-memory-protocol.md) · [Data and attribution](data/README.md) ·
[Primary-source review](docs/research.md) · [Historical reproduction](docs/historical-reproduction.md) · [License](LICENSE)
