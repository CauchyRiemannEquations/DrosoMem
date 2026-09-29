# Flying

Flying studies sequential recall in **computational models using actual Drosophila
connectome structure**. Pi is a trained-sequence benchmark; this is not evidence
that a living fly memorized pi or that the model predicts unseen pi digits.

[한국어 설명](docs/README-ko.md) · [Research status / audit](docs/research-status.md) ·
[Roadmap](docs/research-roadmap.md) · [ACT I results](docs/whole-brain-memory-results.md)

## Current research

Latest: [fresh-seed KC lesion confirmation](docs/kc-confirmation-results.md)
replicates the primary refit benefit: KCγ26.047% frozen versus79.367% refit;
matched26.962% versus77.323%. All3 new blocks pass the fixed criterion.
Frozen portability and5pp KCγ specificity still fail; cohorts were not pooled.

Current position: ACT III-A fresh-seed frozen/refit confirmation complete.
Next: input-only silencing with recurrent connections retained, compared with
full neuron lesions. Broader circuit localization and internal learning remain open.

The established models freeze recurrent connectivity and train an external
readout. Distinguish **representation**, **decoding** and **internal learning**.
Teacher-forced accuracy, autonomous recall and robustness are separate measurements.

## Reproduce the latest diagnostic

Use Python3.12 and a clean environment; commands run from the repository root.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-act1-lock.txt
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe scripts/kc_confirmation.py --out outputs/kc_confirmation_new
.\.venv\Scripts\python.exe scripts/verify_kc_confirmation.py outputs/kc_confirmation_new --out outputs/kc_confirmation_check_new
```

Use fresh directories. The runner verifies an archived baseline, creates the fixed
fresh cohort, and checks frozen/refit decoding. See the report for source hashes,
per-seed results, pairing checks and limitations. Historical results remain intact.
[ACT I protocol](docs/whole-brain-memory-protocol.md) · [Data and attribution](data/README.md) ·
[Primary-source review](docs/research.md) · [Historical reproduction](docs/historical-reproduction.md) · [License](LICENSE)
