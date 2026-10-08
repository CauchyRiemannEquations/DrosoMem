# DrosoMem

*Sequence Memory in a Fly Connectome*

DrosoMem studies sequential recall in **computational models using actual Drosophila
connectome structure**. Pi is a trained-sequence benchmark, not evidence of a living
fly memorizing pi or a model predicting unseen pi digits.

[한국어 설명](docs/README-ko.md) · [Research status / audit](docs/research-status.md) ·
[Roadmap](docs/research-roadmap.md) · [Extension closeout](docs/additional-research-closeout.md)

## Current research — 2026-10-08

The [state-carry / previous-state synaptic-drive study](docs/temporal-mechanism-results.md)
is complete with registered outcome **assay-invalid**, not confirmed mechanism
advantage. Four fixed interventions retain current KC-to-MBON input access and
equal decoder budgets. Carry-only current-symbol decoding is84–86%, below the
prospective90% eligibility requirement; primary and secondary endpoints remain
undefined. Full−carry-only mean historical accuracy is descriptively+17.815 pp
partial and+15.970 pp whole, but those gaps do not rescue the failed assay.
All72 full-state trace hashes and756 independent refits validate exactly at
predictions/counts. The first verifier memory failure, technical indexing repair
and all42,015 pre-study result blobs are preserved. No settings or criteria were
tuned. [Protocol](docs/temporal-mechanism-protocol.md) ·
[Independent verification](results/tdc_mechanism_v1/main_validation_retry1/checks.json) ·
[Final audit](results/tdc_mechanism_v1/final_audit/audit.json).
P3 remains excluded; earlier P1/P2 conclusions below are unchanged.

## Closed v1/P1/P2 programme — 2026-10-06

[v1 closeout](docs/drosomem-v1-closeout.md) seals baseline `0cf0867` and its
negative results. The [Temporal Decodability Curve](docs/temporal-memory-curve-results.md)
uses independent iid K10 streams, fixed recurrent weights and equal-budget
ridge decoders of historical inputs. All six partial-graph blocks pass the
prospective lag 1–5 access criterion. Mean accuracy is83.11% at lag 2,43.60% at
lag 5 and 9.62% at lag 20; chance is10%. This supports decodable past information
in a source-derived computational reservoir, not sequence prediction or
biological learning.

The [matched null ensembles](docs/temporal-null-ensemble-results.md) contain20
partial and 10 whole-brain controls preserving each neuron's degrees, role blocks
and raw presynaptic weight multisets. The registered wiring-advantage criterion
fails in both analyses. Partial real−null mean is−2.285 pp raw accuracy; secondary
whole-brain real is above all 10 controls but only+0.985 pp, below the fixed2.7 pp
requirement. Preserve that small positive direction and the failed gate together.
All planned159 cases,318 complete state replays and 3,339 independent lag refits
are validated. Failed console-only generation and same-seed retry records remain.

[Full synthesis and Korean explanation](docs/temporal-memory-synthesis.md) ·
[P1 protocol](docs/temporal-memory-curve-protocol.md) ·
[P2 protocol](docs/temporal-null-ensemble-protocol.md) ·
[Final provenance audit](results/tdc_v2/final_audit/audit.json).
No general real-wiring/whole-brain superiority, storage site, physiological
dopamine learning, living-fly pi memory or formal memory capacity is established.
P3 spiking/plasticity/physiology is excluded from this completed work.

## Sealed v1 studies

The [length × alphabet × sequence-family follow-up](docs/followup-3-results.md)
completed 162 trained-sequence cases and 54 independent-stream probes. Its
registered lag-2 historical-input criterion passes for iid and Markov inputs;
periodic motif recall also reflects easy prediction. The
[whole-brain scale and pathway follow-up](docs/followup-4-results.md) finds no
`brain5` decoding advantage over the partial graph, while its matched
DAN→MBON cut criterion passes. A separate
[whole-brain rewiring control](docs/followup-4b-results.md) finds no registered
real-wiring advantage over one degree- and role-preserving graph. The
[physiology and local-learning follow-up](docs/followup-5-results.md) verifies
local synaptic changes, but its MBON response and fixed-decoder benefit criteria
fail. Each result retains its prospective protocol, raw cases and independent
verification. These are computational findings, not evidence of biological
memory storage or calibrated physiology.

The [DAN→MBON temporal-pathway study](docs/temporal-pathway-results.md) is
complete: 12 paired state panels, 96 new readout fits and 64 independently
replayed autonomous paths. Its preregistered current-path-dominance criterion
fails in both cohorts. A current-only cut improves refitted MBON delayed
decoding by 2.894/2.225 percentage points, while a persistent cut loses
8.408/8.186 points. The interaction does not identify an anatomical memory
storage site. [Protocol](docs/temporal-pathway-protocol.md) and
[independent audit](results/temporal_pathway_main_validation/checks.json).

The [fresh-model coordinate-SD structural confirmation](docs/fresh-coordinate-sd-results.md)
is complete: 96 new readout fits and 1,824 independently replayed autonomous
paths. The registered intact-over-degree criterion fails in both fresh cohorts.
Coordinate-SD versus own-median calibration changes the structural gap, but
does not establish an intact-wiring advantage. The prior bounded extension and
its negative results remain closed.

The [bounded observation-noise extension](docs/additional-research-closeout.md) is closed.
The final [coordinate-SD calibration](docs/coordinate-sd-noise-results.md) criterion is **not confirmed**:
96 reused partial heads, no new fits, all 2,688 paths independently replayed.
This does not establish biological memory, whole-brain superiority or physiological validity.

The bounded ACT I–V studies and the [three additional computational studies](docs/additional-research-results.md)
are complete within their registered scopes. Completion does not mean every hypothesis succeeded.

| Additional question | Outcome |
| --- | --- |
| Does observation-noise allocation help beyond pi? | Confirmed on random digits and shuffled pi, including fresh model/task seeds |
| Is intact partial wiring better than role-preserving controls? | Not confirmed; keep the negative result |
| Does allocation still help with synthetic spatially correlated observation noise? | Confirmed under the registered random/brain1 criterion |

All use 48 observed MBONs and a 482-parameter readout. These studies test noise allocation
and partial wiring; they establish no whole-brain superiority or internal learning. Strong-noise recall remains weak.
These three studies contain 8,640 certificate evaluations (6,336 distinct exposure settings), 1,144 actual evaluation
trajectories, 1,074 independent full replays and 100 new readout fits. Certificates stop at the first
error; they are not full autonomous sequences. Physiological validation remains open.

## Reproduce the studies

Use Python 3.12, a fresh environment and new output directories. Run from the repository root.
For the final coordinate-SD calibration, see the [exact commands](docs/coordinate-sd-noise-results.md#10-reproducibility).
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
The subsequent graph-relative and coordinate-SD calibrations are also complete.
The fresh-model coordinate-SD study and the subsequent DAN→MBON temporal-pathway
study are complete within their own fixed scopes. [Data and attribution](data/README.md) · [Primary-source review](docs/research.md) ·
[Historical reproduction](docs/historical-reproduction.md) · [License](LICENSE)


## Citation, data terms, and archival

For release and reuse metadata, see [`CITATION.cff`](CITATION.cff),
[Data sources and attribution](DATA_SOURCES.md),
[Limitations](LIMITATIONS.md), and the
[Zenodo release guide](docs/ZENODO.md).

DrosoMem source code is MIT-licensed. FlyWire-derived public data retain their
upstream CC BY-NC 4.0 terms and attribution requirements; the code license does
not relicense those data.
