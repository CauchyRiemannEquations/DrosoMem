# Flying

Flying studies sequential recall in **computational models using actual Drosophila
connectome structure**. Pi is a trained-sequence benchmark; this is not evidence
that a living fly memorized pi or that the model predicts unseen pi digits.

[한국어 설명](docs/README-ko.md) · [Research status / audit](docs/research-status.md) ·
[Roadmap](docs/research-roadmap.md) · [ACT I results](docs/whole-brain-memory-results.md)

## Current research

Latest: [frozen versus refitted KC lesion decoding](docs/kc-frozen-results.md)
finds a large decoder dependence: full KCγ lesion26.250% with the intact head,
79.219% after condition-specific refitting. Matched lesions show a similar gap.
The primary refit-benefit criterion passes; frozen portability and KCγ-specific
impairment criteria fail. These are past-symbol decoding results, not recall.

Current position: ACT III-A frozen/refit lesion comparison complete.
Next: an independently seeded cohort under the same full-lesion design.
Broader critical-subnetwork work and internal learning remain open.

The established models freeze recurrent connectivity and train an external
readout. Distinguish **representation**, **decoding** and **internal learning**.
Teacher-forced accuracy, autonomous recall and robustness are separate measurements.

## Reproduce the latest diagnostic

Use Python3.12 and a clean environment; commands run from the repository root.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-act1-lock.txt
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe scripts/kc_frozen.py --out outputs/kc_frozen_new
.\.venv\Scripts\python.exe scripts/verify_kc_frozen.py outputs/kc_frozen_new --out outputs/kc_frozen_check_new
```

Use fresh directories. The runner reuses committed states and heads; it creates no
new reservoir trajectories or target fits. See the report for source hashes,
per-seed results, pairing checks and limitations. Historical results remain intact.
[ACT I protocol](docs/whole-brain-memory-protocol.md) · [Data and attribution](data/README.md) ·
[Primary-source review](docs/research.md) · [Historical reproduction](docs/historical-reproduction.md) · [License](LICENSE)
