# Original research phases: completion audit — 2026-09-27

**The original research roadmap is not complete.** A usable pretrained opponent
does not imply completion of the scientific phases. The user has now asked to
prioritize the original research stages and defer game screen work.

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
| Phase 6 | Not implemented; prerequisite-dependent | Whole-brain data/graph, sparse memory/time benchmarks, validated dynamics, reproducible whole-brain run. The bundled 686-neuron MB circuits do not satisfy whole-brain scope |
| Game | Console core completed; UI deferred | Existing artifacts remain available. No new screen work while research is the priority |

Completed means the stated experiment was executed and checked, not that its
hypothesis succeeded. Negative Phase 4/5 results must remain visible.

## Research-first order

1. Completed: close anchored confirmation with unchanged rules and fresh seeds.
   Its criteria failed; do not restart a post-hoc tuning loop to hide that outcome.
2. Completed: scoped Phase 2 perturbation work, with matched graph controls and
   two computational confirmation cohorts. The frozen baseline failed robustness;
   preserve that limit instead of selecting a weaker perturbation after outcomes.
3. Completed: sourced Stage B/C small-circuit LIF implementation, numerical checks
   and paired feasibility comparison. Keep the rate opponent; low LIF recall is
   preserved rather than repaired with an unregistered parameter sweep.
4. Completed: scoped Phase 5B gamma1/pedc mapping, online modulation and targeted
   controls. The rule checks pass but selective probe suppression cannot be
   assessed from silent baseline responses. Do not equate this with validated
   biological learning or improved pi memory.
5. **Next:** Phase 6 prerequisites: audit retained input versus graph coverage,
   benchmark sparse memory/runtime, and test activity sensitivity before a
   reproducible whole-brain run. The current samples retain 512 of 2,580 left
   KCs; quantify actual input loss rather than assuming its effect. Record actual
   resource limits if encountered, not assumed blockers.
6. Return to the graphical game flow after the research-first pass, or when the
   user explicitly changes priorities.

Numerical portability and Windows/Linux CI are supporting work, not replacements
for the unfinished scientific stages. This audit does not mark deferred stages
complete or promise that all learning interventions will improve recall.
