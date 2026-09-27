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
| P1 | In progress | Preserve first-32 normalized loss share during expansion | Locked offset-0 discovery, fresh seeds, 108 fits; compare fixed/curriculum/anchored with a retention criterion |
| P1 | Next product task | Verified opponent adapter and shared game scoring | Load a validated head, generate without target access, use the same prompt/first-error rules for human and model; fixed-32 remains available regardless of research outcome |
| P2 | Open | Isolate numerical portability of nonlinear training | Record Python/BLAS/CPU details, compare per-step gradients on identical states, then verify controlled builds across platforms |
| P2 | Open | Add Windows/Linux CI for the small deterministic test suite | Both operating systems validate data checksums, schedule behavior and checkpoint replay on each PR |
| Deferred | Deferred | Whole-brain scale, LIF and biological dopamine | Revisit after rate-model bottlenecks and biological assumptions are supported |

See [Windows reproduction](phase5-windows-reproduction.md) and the
[locked curriculum protocol](phase5-curriculum-protocol.md) and
[negative curriculum result](phase5-curriculum-results.md). The current curriculum
should not replace fixed weighting. The new retention intervention has a separate
[locked protocol](phase5-retention-protocol.md); no improvement is assumed in advance.
