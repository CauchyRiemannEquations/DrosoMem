# Prioritized work — 2026-09-27

The repository had no open issues or pull requests when this work began. Priorities
come from the latest handoff, measured results and failures observed on a fresh
Windows checkout.

| Priority | Work | Acceptance evidence |
|---|---|---|
| P0 | Repair Windows checksum and checkpoint-path failures | Fresh checkout preserves data hashes; full tests pass; old and new manifest keys replay |
| P1 | Reproduce cross-segment weighting before the next intervention | 216 new fits; complete saved-head replay; 18 independent refits; report both recall gain and later cost |
| P1 | Test cumulative 32→64→128 weighting at a matched 6,000-update budget | Locked protocol; uniform/fixed/curriculum controls; all seeds and offsets; final-stage criteria; 972 replays and 27 independent refits |
| P2 | Isolate numerical portability of nonlinear training | Record Python/BLAS/CPU details, compare per-step gradients on identical states, then verify controlled builds across platforms |
| P2 | Add Windows/Linux CI for the small deterministic test suite | Both operating systems validate data checksums, schedule behavior and checkpoint replay on each PR |
| Deferred | Whole-brain scale, LIF and biological dopamine | Revisit only after the present rate-model bottlenecks and biological assumptions are supported |

See [Windows reproduction](phase5-windows-reproduction.md) and the
[locked curriculum protocol](phase5-curriculum-protocol.md). The curriculum
result report will determine whether a fresh-seed confirmation is justified;
do not retune the schedule or success criteria after seeing its outcome.
