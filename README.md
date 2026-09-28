# Flying

Flying studies sequential recall in **computational models using actual Drosophila
connectome structure**. Pi is a trained-sequence benchmark; this is not evidence
that a living fly memorized pi or that the model predicts unseen pi digits.

[한국어 설명](docs/README-ko.md) · [Research status / audit](docs/research-status.md) ·
[Roadmap](docs/research-roadmap.md) · [ACT I results](docs/whole-brain-memory-results.md)

## Current research

Latest: [blocked-state probes](docs/frozen-state-probe-results.md) and
[frozen cross-arm transfer](docs/cross-arm-probe-results.md). Both studies are
complete. Past symbols are decodable above controls in all registered conditions;
frozen heads retain this access across the paired orderings. Next-symbol gains
and improved autonomous recall are not established. Related sequences and
small seed cohorts limit generalization.

Next is one scoped ACT III real-versus-role/degree-rewired partial-graph control
on independent K4 streams under current dynamics. It is proposed, not executed;
earlier structural negative results remain in the research-status index.

The established models freeze recurrent connectivity and train an external
readout. Distinguish **representation**, **decoding** and **internal learning**.
Teacher-forced accuracy, autonomous recall and robustness are separate measurements.

## Reproduce the latest diagnostic

Use Python3.12 and a clean environment; commands run from the repository root.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-act1-lock.txt
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe scripts/frozen_state_probe.py --out outputs/frozen_probe_new
.\.venv\Scripts\python.exe scripts/cross_arm_probe.py --out outputs/cross_arm_new
```

Use fresh output directories. These diagnostics use archived states and need no
graph download. The cross-arm config defaults to the committed probe checkpoints;
to chain an independent reconstruction, copy it and point `source_probes` to
`outputs/frozen_probe_new`. See the detailed reports for hashes and validation.
[ACT I protocol](docs/whole-brain-memory-protocol.md) · [Data and attribution](data/README.md) ·
[Primary-source review](docs/research.md) · [Historical reproduction](docs/historical-reproduction.md) · [License](LICENSE)
