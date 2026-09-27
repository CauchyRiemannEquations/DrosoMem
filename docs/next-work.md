# Prioritized work — 2026-09-27

The repository had no open issues or pull requests when this work began. Priorities
come from the latest handoff, measured results and failures observed on a fresh
Windows checkout. The product is a human-versus-connectome-model pi memory game;
see [game direction](game-direction.md). Verified work now goes directly to main.

**Latest user priority: original research phases first; graphical game work is
deferred.** The original roadmap is not complete. See the [phase audit and
research-first order](phase-status.md) for the completed scoped Phase 2 follow-up, Stage B/C, Phase 5B
and Phase 6 acceptance work.

| Priority | Status | Work | Acceptance evidence |
|---|---|---|---|
| P0 | Done | Repair Windows checksum and checkpoint-path failures | 84 tests pass in a fresh CRLF-enabled clone; archived hashes survive; old/new manifest keys replay |
| P1 | Done | Reproduce cross-segment weighting | 216 new fits, 216 replays, 18 exact refits; qualitative tradeoff confirmed; cross-environment numerical differences documented |
| P1 | Done, negative result | Test 32→64→128 weighting at 6,000 updates | 324 fits, 972 replays, 27 refits; real recall 27.19 versus fixed 33.87; all segment criteria fail |
| P1 | Done, promising discovery | Preserve first-32 normalized loss share during expansion | 108 fits, 324 replays, 9 independent refits; real recall 31.67→38.89, retention passed, later accuracy 81.58%→80.00% |
| P1 | Done | Verified opponent adapter, scoring and timed console game | 18 fixed-32 opponents reproduce full 197-digit rollouts without reference access; 600-second study gate; 125 tests; [prototype](game-prototype.md) |
| Deferred | User deferred | Graphical ten-minute study → recall → results flow | Return after the research-first pass; keep the console core |
| Deferred | User deferred | Actual fresh training during preparation | Return with game work; fixed budget and honest progress |
| P1 | Done, confirmation failed | Confirm anchored weighting on fresh seeds and other segments | 324 fits / 972 stage heads; all offsets fail retention/completion, offset 0 also fails recall; [results](phase5-retention-confirmation-results.md) |
| P1 | Done, robustness criteria failed | Complete scoped Phase 2 perturbation robustness | 72 frozen heads / 16,128 exact recall replays / 216 null checks; both cohorts fail all primary families; 148 tests; [results](phase2-robustness-results.md) |
| P1 | Next original-phase task | Stage B/C: sourced small-circuit LIF implementation and validation | Lock equations, physiological units, parameter sources and reference tests before outcomes; separate validation from any recall-improvement claim |
| P2 | Open | Isolate numerical portability of nonlinear training | Record Python/BLAS/CPU details, compare per-step gradients on identical states, then verify controlled builds across platforms |
| P2 | Open | Add Windows/Linux CI for the small deterministic test suite | Both systems validate data checksums, schedule behavior and checkpoint replay on main pushes and PRs |
| Research backlog | Not complete | Phase 5B dopamine and Phase 6 whole brain | Proceed in [prerequisite order](phase-status.md); biological grounding and measured scaling are required |

See [Windows reproduction](phase5-windows-reproduction.md) and the
[locked curriculum protocol](phase5-curriculum-protocol.md) and
[negative curriculum result](phase5-curriculum-results.md). The current curriculum
should not replace fixed weighting. The retention intervention passed its separate
recall/retention discovery criteria but failed the stronger joint criterion;
[read the discovery](phase5-retention-results.md). The subsequent
[fresh-seed confirmation failed](phase5-retention-confirmation-results.md).
Keep fixed-32 as the clean reference. The [subsequent robustness study](phase2-robustness-results.md)
also failed its primary criteria; proceed to sourced Stage B/C validation without
presenting the existing noise-free recall as perturbation-robust memory.
