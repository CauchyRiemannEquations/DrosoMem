# Prioritized work — 2026-09-27

The repository had no open issues or pull requests when this work began. Priorities
come from the latest handoff, measured results and failures observed on a fresh
Windows checkout. The product is a human-versus-connectome-model pi memory game;
see [game direction](game-direction.md). Verified work now goes directly to main.

| Priority | Status | Work | Acceptance evidence |
|---|---|---|---|
| P0 | Done | Repair Windows checksum and checkpoint-path failures | 84 tests pass in a fresh CRLF-enabled clone; archived hashes survive; old/new manifest keys replay |
| P1 | Done | Reproduce cross-segment weighting | 216 new fits, 216 replays, 18 exact refits; qualitative tradeoff confirmed; cross-environment numerical differences documented |
| P1 | Done, negative result | Test 32→64→128 weighting at 6,000 updates | 324 fits, 972 replays, 27 refits; real recall 27.19 versus fixed 33.87; all segment criteria fail |
| P1 | Done, promising discovery | Preserve first-32 normalized loss share during expansion | 108 fits, 324 replays, 9 independent refits; real recall 31.67→38.89, retention passed, later accuracy 81.58%→80.00% |
| P1 | Done | Verified opponent adapter, scoring and timed console game | 18 fixed-32 opponents reproduce full 197-digit rollouts without reference access; 600-second study gate; 125 tests; [prototype](game-prototype.md) |
| P1 | Next product task | Graphical ten-minute study → recall → results flow | Hide study digits at deadline, show pretrained status, accept digits, report scores and replay identity |
| P1 | Open product follow-up | Actual fresh training during preparation | Fixed declared 6,000-update budget, honest progress, freeze before recall, slow-device handling; local fit 2.62 seconds including preparation, browser unmeasured |
| P1 | Next research task | Confirm anchored weighting on fresh seeds and other segments | Separate locked protocol; repeat recall and retention criteria and report later accuracy cost; do not promote on offset-0 discovery alone |
| P2 | Open | Isolate numerical portability of nonlinear training | Record Python/BLAS/CPU details, compare per-step gradients on identical states, then verify controlled builds across platforms |
| P2 | Open | Add Windows/Linux CI for the small deterministic test suite | Both systems validate data checksums, schedule behavior and checkpoint replay on main pushes and PRs |
| Deferred | Deferred | Whole-brain scale, LIF and biological dopamine | Revisit after rate-model bottlenecks and biological assumptions are supported |

See [Windows reproduction](phase5-windows-reproduction.md) and the
[locked curriculum protocol](phase5-curriculum-protocol.md) and
[negative curriculum result](phase5-curriculum-results.md). The current curriculum
should not replace fixed weighting. The retention intervention passed its separate
recall/retention discovery criteria but failed the stronger joint criterion;
[read the result and limits](phase5-retention-results.md). It is a candidate, not the new default.
