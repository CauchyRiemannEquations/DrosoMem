# Flying

Flying studies sequential recall in **computational models using actual Drosophila
connectome structure**. Pi is a trained-sequence benchmark; this is not evidence
that a living fly memorized pi or that the model predicts unseen pi digits.

[한국어 설명](docs/README-ko.md) · [Research status / audit](docs/research-status.md) ·
[Roadmap](docs/research-roadmap.md) · [ACT I results](docs/whole-brain-memory-results.md)

## Current research

Latest: the [normalization-factor control](docs/normalization-control-results.md)
completed30 main and18 fresh-seed confirmation runs. Fixed original factors
improved independent past-symbol decoding by1.245pp in main and0.894pp in
confirmation. The preregistered1pp confirmation criterion **failed**.
This is a decoding diagnostic, not autonomous recall or internal learning.

Current position: scoped ACT III structure and normalization controls complete.
Next proposed diagnostic transfers frozen readouts between normalization
conditions using saved states; broader ablations and internal learning remain open.

The established models freeze recurrent connectivity and train an external
readout. Distinguish **representation**, **decoding** and **internal learning**.
Teacher-forced accuracy, autonomous recall and robustness are separate measurements.

## Reproduce the latest diagnostic

Use Python3.12 and a clean environment; commands run from the repository root.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-act1-lock.txt
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe scripts/normalization_control.py --cohort smoke --out outputs/norm_smoke_new
.\.venv\Scripts\python.exe scripts/normalization_control.py --cohort main --out outputs/norm_main_new
```

Use fresh directories. This comparison uses committed partial graphs and needs
no whole-brain graph download. The runner verifies archived structural graphs.
See the result report for conditional confirmation and independent verification,
source hashes, checkpoints and limitations. Historical results remain intact.
[ACT I protocol](docs/whole-brain-memory-protocol.md) · [Data and attribution](data/README.md) ·
[Primary-source review](docs/research.md) · [Historical reproduction](docs/historical-reproduction.md) · [License](LICENSE)
