# Original research phases: completion audit — 2026-09-27

Current update (2026-09-30): the [relative-noise comparison](relative-noise-results.md)
and [current-observation diagnostic](observation-noise-results.md) are complete.
The first adds fresh confirmation seeds and does not confirm whole-brain advantage;
the second reuses those heads and does not execute new autonomous trajectories.
[Two-study completion audit](../results/observation_noise_closeout/audit.json).
The next matched-energy coordinate-noise proposal is unregistered and unexecuted.

Preceding update (2026-09-30): [ACT V five-family rate-model robustness](act5-robustness-results.md)
is complete within its preregistered partial/whole-graph scope. Fresh confirmation,
full metric/hash checks and the specified independent replay subset are complete.
This is not physiological validation. Earlier scope audits below are historical.

Current update (2026-09-29): the [bounded ACT IV programme](act4-results.md) is
closed after [margin diagnostics with fresh seeds](reward-margin-results.md).
This closes specified experiments, not physiological/whole-brain learning questions.
Earlier completion audits below are retained as historical scope records.

Latest follow-up (2026-09-29): bounded IV-A comparison, scalar/local learning,
[initial-direction](reward-direction-results.md), [noise-match](reward-noise-results.md)
and [same-trajectory](reward-trajectory-results.md) diagnostics are complete within
their registered scopes. The last study is resolution-limited and inconclusive;
full learning criteria remain unconfirmed. Biological ACT IV is open. Original audit retained.

Update 2026-09-28: the subsequent [ACT I matched-memory study](whole-brain-memory-results.md)
has also completed, with no established whole-brain recall improvement. Its scope
is distinct from the original Phase 6 feasibility row below. Use
[research-status](research-status.md) for the current question-based audit.

**The research-first execution pass is complete within the scopes below.** Every
listed original phase now has executed evidence, including whole-brain Phase 6
feasibility. Several learning and robustness criteria failed. This is not a claim
of complete biological validation or whole-brain pi learning. This repository
continues with research on learning, recall and biological modeling assumptions.

| Original stage | Status | Evidence / remaining acceptance work |
|---|---|---|
| Phase 0 | Completed | Source review and pinned data provenance: [research](research.md) |
| Phase 1 | Completed | Fixed reservoir, trained readout, free recall and controls in `results/mvp` |
| Phase 2 main study | Completed | 450 normalization/size/length/control runs: [results](phase2-results.md) |
| Phase 2 follow-up | Completed within the current MB baseline scope; robustness criteria failed | Delayed memory in Phase 3b; [robustness study](phase2-robustness-results.md): 72 frozen heads, two model/noise-seed cohorts, 16,128 exact recall replays. Neither cohort passes the primary noise/dropout criteria. This does not assert robustness or cover every original Phase 2 variant |
| Phase 3 / 3b | Completed | Anatomical input/output restrictions, control graphs, ablations, delayed-memory and timing diagnostics: [3](phase3-results.md), [3b](phase3b-results.md) |
| Stage B/C | Completed within sourced small-circuit scope; no recall gain | Brian2 LIF, analytic and independent-reference checks, three timestep grids, four graph conditions / eight readouts. Real-graph recall 3.0 versus rate 33.5. Uniform source defaults are not cell-specific physiological calibration; [results](stage-bc-results.md) |
| Phase 4 | Completed, no reliable recall gain established | 432 constrained-plasticity runs: [results](phase4-results.md) |
| Phase 5A | Completed, negative baseline | Scalar reward/eligibility-trace study: [results](phase5-results.md); subsequent diagnostics, BPTT, timing and readout studies are follow-ups |
| Phase 5 readout follow-up | Confirmation completed; criteria failed | 324 fresh fits: anchored weighting failed retention/completion at every offset and recall at offset 0. Keep fixed-32; [results](phase5-retention-confirmation-results.md) |
| Phase 5B | Completed in gamma1/pedc model scope; neural-response criterion failed | Exact PPL101/MBON11 mapping and alpha3 control; actual DAN-spike-gated LTD, 12 conditioning cases, all causal/null checks pass. Both isolated probe baselines are silent; pi recall 3 -> 2. This is not an all-compartment or physiologically calibrated model; [results](phase5b-results.md) |
| Phase 6 | Completed in whole-brain engineering feasibility scope | 138,639 neurons / 15,091,983 source edges; eight graph/input conditions / 24 exactly repeated probes; legacy Brian2 comparisons pass; full-edge worker peak 541 MiB. MBON11 baseline responses recover when weak edges return. No whole-brain pi training or physiological calibration; [results](phase6-results.md) |

Completed means the stated experiment was executed and checked, not that its
hypothesis succeeded. Negative Phase 4/5 results must remain visible.

## Research-first order

1. Completed: close anchored confirmation with unchanged rules and fresh seeds.
   Its criteria failed; do not restart a post-hoc tuning loop to hide that outcome.
2. Completed: scoped Phase 2 perturbation work, with matched graph controls and
   two computational confirmation cohorts. The frozen baseline failed robustness;
   preserve that limit instead of selecting a weaker perturbation after outcomes.
3. Completed: sourced Stage B/C small-circuit LIF implementation, numerical checks
   and paired feasibility comparison. Keep the rate baseline; low LIF recall is
   preserved rather than repaired with an unregistered parameter sweep.
4. Completed: scoped Phase 5B gamma1/pedc mapping, online modulation and targeted
   controls. The rule checks pass but selective probe suppression cannot be
   assessed from silent baseline responses. Do not equate this with validated
   biological learning or improved pi memory.
5. Completed: Phase 6 source-complete whole-brain execution and scaling audit.
   Original samples retain only 10.34–10.45% of incoming left-MBON contacts.
   All 24 probes repeat exactly; no resource limit was hit. Restoring weak edges
   recovers isolated MBON11 responses in both input maps, but does not revalidate
   Phase 5B learning. Whole-brain pi readout training remains a separate follow-up.
6. Next research: predeclare a matched whole-brain versus partial-brain pi-learning
   and autonomous-recall comparison. This follow-up is proposed, not yet executed.

Numerical portability, Windows/Linux CI, expanded-graph dopamine conditioning,
whole-brain pi training and broader physiological validation remain follow-up
work. This audit distinguishes executed scoped experiments from successful
scientific hypotheses; no negative result is retrospectively promoted.
