# Prioritized work — updated 2026-10-08

## Current handoff — 2026-10-08

The proposed leak/carry versus previous-state synaptic transmission study is
now [closed](temporal-mechanism-results.md) under its
[precommitted protocol](temporal-mechanism-protocol.md). All36 main cases,
72 exact complete-state replays and756 independent readout refits are complete.
Registered outcome: **assay-invalid**. Carry-only current decoding84–86% fails
the90% eligibility requirement; the material-mechanism endpoint is undefined.
Large descriptive historical gaps do not substitute for the registered result.
The technical verifier memory failure and same-data retry remain archived.
All42,015 prior result blobs and all old scientific code/data/configs remain
unchanged. No execution or validation remains in M1 or the closed P0/P1/P2 scopes.

The next candidate computational question is the **joint current/history
decoding tradeoff**: how do these interventions change both kinds of access
under a prospectively validated assay? Any such study needs a distinct protocol,
fixed reference checks, seeds, budgets and interpretation before new outcomes.
It must not lower M1's90% gate or relabel M1 as confirmed. This is a proposal,
not a registered or executed experiment. Another distinct later question is
delayed feedforward propagation versus anatomical feedback cycles; M1's R
switch includes both and cannot resolve that distinction.

P3 physiology/spiking/plasticity remains outside this work. Do not automatically
start a carry/drive sweep, favorable-lag selection, new decoder or extra seed.
Retain the current invalid assay, earlier wiring/whole-brain/local-learning
negative results and every interruption. The dated handoffs below are history.

## Current handoff — 2026-10-06

The requested P0 closeout, P1 iid TDC and P2 matched null ensembles are complete.
[Synthesis](temporal-memory-synthesis.md), [P1 results](temporal-memory-curve-results.md),
[P2 results](temporal-null-ensemble-results.md),
[final audit](../results/tdc_v2/final_audit/audit.json).
No experiment execution or validation remains in these fixed scopes.

P1 confirms short-lag historical access; P2 fails its real-wiring material
criterion in both graph analyses. Partial real is below 20 nulls; whole-brain
real is above 10 nulls but only+0.985 pp, below the registered2.7 pp requirement.
Keep all paired values, the positive secondary direction, both failed gates,
earlier v1 negatives and the console-only failed attempt. No additional seed,
null graph, weaker threshold or different decoder is a completion step.

A distinct future computational question could separate leak-only history from
recurrent-network contributions under a new fixed TDC protocol. This is a
proposal, unregistered and unexecuted; it is not an automatic attempt to repair
the failed wiring hypothesis. P3 physiology/spiking/plasticity is outside the
current work. The dated handoffs below retain historical context.

## Historical handoff — 2026-10-02

The requested [sequence grid and novel-family study](followup-3-results.md),
[whole-brain scale/pathway study](followup-4-results.md), separate
[whole-brain rewiring control](followup-4b-results.md) and
[physiology/local-rule study](followup-5-results.md) are complete within their
registered budgets. Independent audits and negative results are preserved.
The next prospective questions, if pursued, are independent graph-null
realizations and response-bearing physiological targets; neither is an
unfinished execution step in these studies. Earlier handoff notes below are
retained as historical context.

The [DAN→MBON temporal-pathway study](temporal-pathway-results.md) is now
executed and independently verified. Its registered current-path-dominance
criterion fails in both cohorts: a current-only cut improves refitted MBON
delayed decoding, while a persistent cut impairs it. The descriptive
nonadditive interaction warrants a distinct prospective study if pursued;
there is no execution or verification left in this bounded study. Storage
localization, physiological dopamine and whole-brain generalization remain
open questions, not claims from this result. The notes below retain the
preceding handoff state.

The single [fresh-model coordinate-SD study](fresh-coordinate-sd-results.md)
proposed by the closed extension is now executed and independently verified.
Its registered intact-over-degree criterion fails in both fresh cohorts.
There is no remaining execution or verification in this bounded study. Any
whole-brain, physiological, new-dynamics or local-learning extension requires
a separate question, fixed budget and prospective criteria. The older handoff
notes below describe their historical state.

**The bounded observation-noise extension is closed, including coordinate-SD calibration.**
[Programme synthesis](additional-research-closeout.md), [last experiment](coordinate-sd-noise-results.md),
[prospective protocol](coordinate-sd-noise-protocol.md), [closeout audit](../results/coordinate_sd_noise_closeout/audit.json).
The final random/degree incremental-gap criterion is **not confirmed**.
96 existing partial heads, zero new fits; all 2,688 actual paths independently replayed.
2,592 labeled noisy settings include 288 common/own intact repeats (2,304 distinct fresh settings).
Six extension studies comprise 5,992 actual evaluation paths and 5,726 independent full replays;
these are execution counts, not distinct models. Earlier negative results remain unchanged.
No execution or verification remains in this bounded extension. Physiology, new dynamics,
whole-brain rewiring and fresh-model generalization remain separate unexecuted research.
One future proposal is fresh model/data/graph seeds under coordinate-SD calibration;
it is not registered or executed. No further calibration sweep is included in this closeout.

## Historical handoff notes (superseded by the result above)

**ACT I follow-up is now complete with a negative main result:** 50 matched fits,
50 exact replays, five exact independent refits; whole-brain means 31.8/34.1
versus partial 35.0. See [results](whole-brain-memory-results.md) and
[current research status](research-status.md). The table below records the prior
execution pass. Its whole-brain proposal is now superseded by those results.
The follow-up selected at that time was the small four-family ACT II comparison;
it is now complete. No new plasticity sweep is prioritized.

The repository had no open issues or pull requests when this work began. Priorities
come from the latest handoff, measured results and failures observed on a fresh
Windows checkout. This repository is research-only; see the
[research direction](research-direction.md). Verified work goes directly to main.

**The requested research-first execution pass is now complete within documented
scopes.** The next research question is whether whole-brain connectivity improves
autonomous pi recall under matched training budgets. See the [phase audit](phase-status.md) for the completed scoped
Phase 2 follow-up, Stage B/C, Phase 5B and whole-brain Phase 6 feasibility work.
Several scientific criteria failed; whole-brain pi learning remains untested.

| Priority | Status | Work | Acceptance evidence |
|---|---|---|---|
| P0 | Done | Repair Windows checksum and checkpoint-path failures | 84 tests pass in a fresh CRLF-enabled clone; archived hashes survive; old/new manifest keys replay |
| P1 | Done | Reproduce cross-segment weighting | 216 new fits, 216 replays, 18 exact refits; qualitative tradeoff confirmed; cross-environment numerical differences documented |
| P1 | Done, negative result | Test 32→64→128 weighting at 6,000 updates | 324 fits, 972 replays, 27 refits; real recall 27.19 versus fixed 33.87; all segment criteria fail |
| P1 | Done, promising discovery | Preserve first-32 normalized loss share during expansion | 108 fits, 324 replays, 9 independent refits; real recall 31.67→38.89, retention passed, later accuracy 81.58%→80.00% |
| P1 | Done, confirmation failed | Confirm anchored weighting on fresh seeds and other segments | 324 fits / 972 stage heads; all offsets fail retention/completion, offset 0 also fails recall; [results](phase5-retention-confirmation-results.md) |
| P1 | Done, robustness criteria failed | Complete scoped Phase 2 perturbation robustness | 72 frozen heads / 16,128 exact recall replays / 216 null checks; both cohorts fail all primary families; 148 tests; [results](phase2-robustness-results.md) |
| P1 | Done within small-circuit scope; weak recall | Stage B/C: sourced LIF implementation and validation | Four graph conditions, independent-reference and timestep checks pass; eight exact replays/refits; 166 tests; real recall 3.0 versus rate 33.5; [results](stage-bc-results.md) |
| P1 | Done in gamma1/pedc scope; response criterion failed | Phase 5B: compartment mapping and dopamine-dependent modulation | Twelve cases / eight full recalls exactly rebuilt; 181 tests; local causal/null checks pass, both probe baselines silent; pi recall 3 -> 2; [results](phase5b-results.md) |
| P1 | Done in whole-brain feasibility scope | Phase 6: input coverage, sparse scaling and whole-brain execution | 138,639 cells / 15,091,983 edges; 24 exact probe repeats, six Brian2 comparisons, 193 tests; full graph peak 541 MiB; [results](phase6-results.md) |
| P1 | Next research; protocol not yet locked | Whole-brain versus partial-brain pi learning and autonomous recall | Match input IDs, observed MBONs, head capacity, updates and seeds; separate graph coverage from edge threshold; predeclare resource and success criteria |
| P2 | Open | Isolate numerical portability of nonlinear training | Record Python/BLAS/CPU details, compare per-step gradients on identical states, then verify controlled builds across platforms |
| P2 | Open | Add Windows/Linux CI for the small deterministic test suite | Both systems validate data checksums, schedule behavior and checkpoint replay on main pushes and PRs |
| Research backlog | Not complete; distinct follow-up | Whole-brain pi training, expanded-graph Phase 5B and broader biological validation | Phase 6 restores baseline probe activity but fits no readout and performs no conditioning; freeze new protocols before testing learning claims |

See [Windows reproduction](phase5-windows-reproduction.md) and the
[locked curriculum protocol](phase5-curriculum-protocol.md) and
[negative curriculum result](phase5-curriculum-results.md). The current curriculum
should not replace fixed weighting. The retention intervention passed its separate
recall/retention discovery criteria but failed the stronger joint criterion;
[read the discovery](phase5-retention-results.md). The subsequent
[fresh-seed confirmation failed](phase5-retention-confirmation-results.md).
Keep fixed-32 as the clean reference. The [subsequent robustness study](phase2-robustness-results.md)
also failed its primary criteria. The [subsequent Stage B/C LIF study](stage-bc-results.md)
passes numerical validation but has weak recall. The [Phase 5B follow-up](phase5b-results.md)
implements a sourced local dopamine rule but fails its functional-response criterion.
The [Phase 6 coverage/scaling study](phase6-results.md) now executes the complete
source graph and restores isolated MBON11 baseline activity when weak edges are
included. It does not validate the earlier learning criterion or improve a
measured pi recall model. Do not present the rate baseline as robust, or the
spiking/plastic model as an improved memory model.
