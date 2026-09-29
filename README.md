# Flying

Flying studies sequential recall in **computational models using actual Drosophila
connectome structure**. Pi is a trained-sequence benchmark; this is not evidence
that a living fly memorized pi or that the model predicts unseen pi digits.

[한국어 설명](docs/README-ko.md) · [Research status / audit](docs/research-status.md) ·
[Roadmap](docs/research-roadmap.md) · [ACT I results](docs/whole-brain-memory-results.md)

## Current research

Latest: [KCγ population and disjoint subset lesions](docs/kc-ablation-results.md)
do not support population-specific impairment after readout refitting.
Intact77.303%, whole-KCγ lesion79.219%, matched77.156%; disjoint subsets76.957%
versus77.072%. All registered conditions and negative results are preserved.

Current position: ACT III-A first population-lesion study and matched follow-up
complete. Next: frozen-head lesion decoding versus the existing refit comparison.
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
.\.venv\Scripts\python.exe scripts/kc_ablation.py --out outputs/kc_ablation_new
.\.venv\Scripts\python.exe scripts/verify_kc_ablation.py outputs/kc_ablation_new --out outputs/kc_ablation_check_new
```

Use fresh directories. The runner checks archived baselines, then simulates new
paired seeds using committed circuit data; no new graph download is needed. See the report for source hashes,
per-seed results, pairing checks and limitations. Historical results remain intact.
[ACT I protocol](docs/whole-brain-memory-protocol.md) · [Data and attribution](data/README.md) ·
[Primary-source review](docs/research.md) · [Historical reproduction](docs/historical-reproduction.md) · [License](LICENSE)
