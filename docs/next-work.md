# Prioritized work — updated 2026-09-29

**ACT III-D current-model pathway panel complete.**
[Results](pathway-memory-results.md):273 conditions,260 frozen evaluations,
546 independent trajectories and5863 metric rows. DAN→MBON refit primary
confirms7.3306/8.3204pp extra impairment; KC→MBON secondary5.4907/5.7083pp
also confirms but impairs current-symbol access. No frozen specificity confirms.
APL specificity is graph-control limited, not demonstrated absent.
All previous artifacts and graph/verification failures are preserved.

Next single proposed experiment: **ACT IV-A entry diagnostic — observation
location and readout dependence after DAN→MBON removal**. Keep48 observed
neurons and the same linear head size; predefine an alternative observation set.
Establish intact past-symbol access for each observation set, then compare the
same lesion's refit impairment. This tests a blocked MBON access route versus
broader loss of accessible information; observation composition is an explicit
covariate. Preregister first. This proposal has not been executed, and does not
launch a large plasticity sweep. Whole-brain and minimal-subnetwork work remain
separate open extensions. Preserve III-B/C negatives.

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
