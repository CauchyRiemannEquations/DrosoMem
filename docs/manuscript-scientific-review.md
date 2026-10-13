# Internal scientific review and manuscript corrections

Date: 2026-10-13. Reviewed baseline:
`f916fe65a66263230d229a31e22d14fbc41bae6f`.
Scientific cutoff stays `c48e0c43836e640ec1a15b1ac1152f7e3ec4e2a4`.
[Revised manuscript](manuscript-draft.md) ·
[Review audit](../results/manuscript_review_v1/audit/checks.json) ·
[Evidence matrix](research-evidence-matrix.md).

This is an internal scientific/editorial review, not external peer review. It
checks arguments and definitions against registered protocols, numerical source,
immutable results and primary literature. No new neural run, fit, replay, seed
selection or scientific decision is made. The original draft and its audit
remain available at the baseline revision.

## Findings and completed corrections

| ID | Finding | Correction and evidence |
| --- | --- | --- |
| R01 | “All 440 coefficients” confuses episodes with rules. | Section2.6 now states 440 episodes across 44 held-out coefficient pairs; the primary has 330 episodes across 33 pairs. M10 audit metadata enumerates all ten nonfixed starts per pair. |
| R02 | Partial-circuit selection and DAN roles were underdescribed. | Section2.1 states left-side KC sampling, MBON/APL inclusion and pre-sampling DAN selection. DANs supply ordinary signed edges here, without dopamine modulation. Source provenance links upstream URLs/hashes. |
| R03 | Drive b overlaps the affine offset b; “integrates once” obscures MBON replacement. | Section2.2 uses β for drive, explicitly discards provisional MBON values and retains one final state per step. Carry removal still leaves drive0.6; numerical source is unchanged. |
| R04 | “Relative RMS gain” omits normalization order and the validity floor. | Section2.5 defines per-probe input normalization, RMS pooling, R_l=G_l/G_0 and zero-contrast/G_0<10⁻⁸ invalidity. The exact lag5 test is unchanged. |
| R05 | Noise Z overlaps the ridge design matrix; the Gaussian law was unstated. | Section2.5 uses ξ∼N(0,I₄₈), paired across worlds, with ε_i=0.10 σ_old,i ξ_i. Noise remains an endpoint-observation intervention. |
| R06 | Cases, heads, blocks and pairs can be mistaken for independent replications. | Section2.7 adds thirteen panel/head/status rows, separating shared scalers, smoke/refits, trained tables, witnesses and paired worlds. P1 has six whole blocks; later whole panels generally have three. M6 explicitly reports 576 two-world pairs. |
| R07 | Figure1 does not display lag0; Figure2's gain curves and axis ranges needed clarification. | Captions identify partial/whole labels, absent lag0, logarithmic versus linear axes and the derivative-only threshold. Image bytes are unchanged. |
| R08 | “Null advantage” can reverse the conn2res comparison's direction. | Section4.2 states empirical-over-null peak-capacity advantages for mouse/rat/macaque and a nonsignificant fruit-fly contrast, without equating nonsignificance with equivalence. The unnecessary FlyCircuit assertion is removed. |
| R09 | Independent implementation can suggest independent laboratory/dataset replication. | Section2.7 specifies a separate numerical implementation on the same registered inputs. Historical checks retain their revisions and are not rerun here. |
| R10 | Own-paper authorship/declaration placeholders conflict with the requested scope. | Front/back placeholders and the author-input handoff are removed. Literature attribution remains; no own-paper byline/affiliation/correspondence block is introduced. |

## Scientific conclusions retained

- P1 supports operational history access within its registered endpoint, not
  Shannon/formal IPC capacity, a biological forgetting time or autonomous recall.
- M1 remains assay-invalid/null. M2's narrower fresh-stream tradeoff does not
  rescue M1 or establish a universal information-budget law.
- M3 supports delayed-feedforward sufficiency under a broader mask. M4 blocks
  one exact all-lag ideal control; whole degree-only stays UNRESOLVED. Neither
  isolates a functional anatomical-cycle contribution.
- M5 attenuation, M6 clean tracking and M7 one-dose noise failure retain separate
  meanings. Numerical derivative checks are not biological validation; small
  signals can remain decodable after coordinate scaling.
- M8 READY is construction; M9's valid ceiling ties fail the material advantage
  gate without proving general wiring equivalence.
- M10 READY is task preparation with a privileged known-form oracle. M11's
  primary/secondary FAIL persist. Training acquisition already fails, preventing
  an isolated-transfer-after-mastery interpretation or a broad impossibility claim.
- Seeds, controls, negative families, denominators, equality rules and learned
  budgets stay fixed. Shared episodes/heads/anchors are not biological samples.

## Evidence and literature verification

The review audits the thirteen-row scope table, P1 lag means, P2 raw contrasts,
M3 lag2 counts/topology bounds, M4 obstruction/witness counts, M6 pair counts,
M8 change ranges/proposal scope, M9 counts, M10 unique rules and M11 cohort
counts/gaps/per-rule/fixed-copy controls. Existing checks cover M2/M5/M7 anchors
and all nine citation/BibTeX identities. Selected result/config inputs are
authenticated against baseline Git blobs. Decimal half-up display rounding is
explicit, including the exact 50.525% P1 lag4 value displayed as 50.53%; gates
remain count/fraction-based. Automated consistency complements, rather than
replaces, the prose review.

The [conn2res primary article](https://www.nature.com/articles/s41467-024-44900-4)
was rechecked through publisher-indexed Example2/Figure5c text after direct
publisher access failed. It supports the corrected direction/significance wording.
The [Costi primary article](https://www.mdpi.com/2313-7673/10/5/341) was rechecked
through indexed discussion/conclusion text: its overfitting resilience concerns
forecasting conditions, not the present categorical task. The
[Churchland v1 record](https://arxiv.org/abs/2609.07355v1) still supplies preprint
context for coupled structure/rescaling under spectral-radius normalization,
not a peer-reviewed Drosophila performance result. Other access limits remain
in [the literature notes](literature-context.md); this is not an exhaustive search.

## Completion boundary

All ten identified document findings are addressed. Registered outcomes,
scientific source/protocols, bibliography/ledger, figures and past audits remain
unchanged. The new audit authenticates revised documents separately from the
original-draft audit. No new experiment or external submission is required to
complete this review. No own-paper author information is added, as requested.
