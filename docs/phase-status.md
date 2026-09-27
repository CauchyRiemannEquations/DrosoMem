# Original research phases: completion audit — 2026-09-27

**The original research roadmap is not complete.** A usable pretrained opponent
does not imply completion of the scientific phases. The user has now asked to
prioritize the original research stages and defer game screen work.

| Original stage | Status | Evidence / remaining acceptance work |
|---|---|---|
| Phase 0 | Completed | Source review and pinned data provenance: [research](research.md) |
| Phase 1 | Completed | Fixed reservoir, trained readout, free recall and controls in `results/mvp` |
| Phase 2 main study | Completed | 450 normalization/size/length/control runs: [results](phase2-results.md) |
| Phase 2 follow-up | Partially completed | Delayed random-input memory is covered by Phase 3b; fresh-seed/segment weighting confirmations exist. Noise/perturbation robustness and a locked independent robustness confirmation still need their own study |
| Phase 3 / 3b | Completed | Anatomical input/output restrictions, control graphs, ablations, delayed-memory and timing diagnostics: [3](phase3-results.md), [3b](phase3b-results.md) |
| Stage B/C | Not implemented | Brian2 LIF dynamics and sourced neuron/synapse parameters. First lock equations, units and validation cases, then compare against the rate-model baseline; passing rate-model tests cannot close this stage |
| Phase 4 | Completed, no reliable recall gain established | 432 constrained-plasticity runs: [results](phase4-results.md) |
| Phase 5A | Completed, negative baseline | Scalar reward/eligibility-trace study: [results](phase5-results.md); subsequent diagnostics, BPTT, timing and readout studies are follow-ups |
| Phase 5 readout follow-up | Confirmation completed; criteria failed | 324 fresh fits: anchored weighting failed retention/completion at every offset and recall at offset 0. Keep fixed-32; [results](phase5-retention-confirmation-results.md) |
| Phase 5B | Not implemented | Source-supported compartment mapping and dopamine-dependent modulation, with targeted controls. Artificial global reward and ordinary DAN graph nodes do not satisfy this stage |
| Phase 6 | Not implemented; prerequisite-dependent | Whole-brain data/graph, sparse memory/time benchmarks, validated dynamics, reproducible whole-brain run. The bundled 686-neuron MB circuits do not satisfy whole-brain scope |
| Game | Console core completed; UI deferred | Existing artifacts remain available. No new screen work while research is the priority |

Completed means the stated experiment was executed and checked, not that its
hypothesis succeeded. Negative Phase 4/5 results must remain visible.

## Research-first order

1. Completed: close anchored confirmation with unchanged rules and fresh seeds.
   Its criteria failed; do not restart a post-hoc tuning loop to hide that outcome.
2. **Next:** complete the remaining Phase 2 robustness work with a locked perturbation
   design, model/control comparisons, noise seeds and independent confirmation.
3. Establish sourced parameter/compartment requirements for Stage B/C and Phase
   5B, implement and validate them on small circuits before scaling. Recheck
   primary literature/documentation at that point; do not invent biological constants.
4. Benchmark sparse scaling and execute Phase 6 only with justified dynamics and
   data. Record actual resource limits if encountered, not assumed blockers.
5. Return to the graphical game flow after the research-first pass, or when the
   user explicitly changes priorities.

Numerical portability and Windows/Linux CI are supporting work, not replacements
for the unfinished scientific stages. This audit does not mark deferred stages
complete or promise that all learning interventions will improve recall.
