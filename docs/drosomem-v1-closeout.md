# DrosoMem v1 closeout — 2026-10-06

Baseline main: `0cf0867736ca598ae233cdf3e185031667dd52be`.
This closeout seals the research at that revision; it does not retune it.

## Claim

> A fixed computational reservoir built from source-derived Drosophila connectome structure retains decodable information about past inputs across time.

The independent iid K10 lag-2 experiment supports this statement: 83.48%
held-out decoding, versus 10.70% current-symbol-only and 9.90% frequency
prediction. Recurrent weights are fixed; external decoders are trained.
The experiment is distinct from memorizing or predicting a particular sequence.

## Preserved negative and qualified findings

| Question | v1 finding | Evidence |
| --- | --- | --- |
| Independent historical-symbol access | Registered iid/Markov lag-2 gate passes | [3](followup-3-results.md) |
| Whole brain versus partial graph | No registered general advantage; K4 scale difference −3.36pp | [4](followup-4-results.md) |
| Real versus degree/role-preserving whole-brain wiring | Gate fails; +0.178pp mean and one negative seed, one null graph | [4B](followup-4b-results.md) |
| DAN→MBON intervention | Temporal information changes; matched pathway gate passes; current-only dominance fails | [4](followup-4-results.md), [temporal intervention](temporal-pathway-results.md) |
| Local dopamine-like update | Local weight-change controls pass; response and fixed-decoder benefit fail | [5](followup-5-results.md) |
| Fresh partial wiring confirmation | Intact-over-degree criterion fails in both cohorts | [fresh confirmation](fresh-coordinate-sd-results.md) |
| Earlier learning/robustness/calibration | Preserve their failed gates and inconclusive diagnostics | [status history](research-status.md), [bounded closeout](additional-research-closeout.md) |

No claim: flies memorize pi; real wiring is superior to matched rewiring;
whole-brain connectivity improves memory; an anatomical memory storage site is
identified; physiological dopamine learning is established; formal Shannon or
reservoir memory capacity has been measured. Computational seeds are not animals.

## Provenance and audit boundary

[Audit](../results/tdc_v2/v1_audit/audit.json),
[full baseline Git inventory](../results/tdc_v2/v1_audit/baseline-git-tree.json),
[materialized archive hashes](../results/tdc_v2/v1_audit/materialized-result-sha256.json),
[latest evidence links](../results/tdc_v2/v1_audit/latest-evidence.json), and
[recomputed gates](../results/tdc_v2/v1_audit/recomputed-gates.json) distinguish
the complete Git snapshot from the subset available in a sparse checkout.
Every historical result path is sealed by its Git blob ID. Available archive
bytes and manifest checksums are checked; excluded archives are not claimed to
have been replayed or inspected. Recent raw tables reproduce the published gate
directions, and their validators name the correct result manifest hashes.
Historical validators share some model constructors with runners: numerical
refits provide independent checks, but historical construction independence is
not universal. New TDC validation will reconstruct graph and dynamics separately.

Audit totals: 39,608 tracked historical result files, 4,719 materialized files
with zero Git-byte mismatches, 8,788 recognized artifact checks with zero
checksum mismatches or unresolved checks, 28,512 numerical arrays with zero
nonfinite arrays. The 34,889 excluded result files retain baseline Git blob IDs
but are outside the fresh byte/array audit. All links in the four requested
overview documents exist in the full Git tree; absence on sparse disk is not
a repository broken link. Audit took 57.46 seconds.

Source-derived graph edges, neuron IDs, roles, source revisions and upstream
SHA256 provenance remain unchanged. Historical protocols must be replayed at
their recorded revision, rather than rewriting old source hashes for later
editorial changes. Historical dated handoffs in roadmap/status are snapshots;
the new current section will take precedence over superseded proposals.

## New namespace and stopping discipline

All new evidence goes under `results/tdc_v2/`; graph caches under
`outputs/tdc_v2/`. Existing directories are rejected, never overwritten.
Attempt/failure records are retained. No historical result file is modified.
P1 and P2 each receive a separate prospective protocol commit. P2 begins only
after P1 has completed and passed validation, even if its scientific gate fails.
P3 spiking/physiological plasticity is excluded.
