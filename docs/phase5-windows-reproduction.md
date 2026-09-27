# Windows reproduction of the cross-segment confirmation

Re-executed base commit `17ef56bb567748f2463cbe7c17742f2e565b7af0`
on Windows with Python 3.12.10 and the repository's pinned direct dependencies.
The published run recorded Python 3.12.14. A separate unchanged checkout was
used while curriculum development proceeded in another checkout.

## Findings and repairs

1. Git's Windows CRLF conversion broke the bundled connectome checksums. Restore
   the exact committed LF bytes. The new `.gitattributes` keeps source and data
   in LF regardless of `core.autocrlf`; archived result bytes remain untouched.
2. The original verifier looked up `checkpoints/condition_000.npz`, while Windows
   manifests stored `checkpoints\\condition_000.npz`. New manifests use POSIX keys,
   and the verifier accepts older backslash keys. A regression test covers both.

After restoring LF and completing dependency installation, the unchanged base
passed 70/71 tests; only the manifest-path verifier test failed. The updated code
passes all 84 tests, including 13 curriculum tests and the existing 71 tests.

## Reproduction results

All 216 fits were rerun. The first condition was saved in one process and the
remaining 35 resumed in another. The original base verifier replayed all 216
heads and complete generated strings and independently refit 18 heads exactly.
To run that unchanged verifier on Windows, only the newly generated manifest's
path-key separators were normalized to `/`; source, checkpoints, metrics and
checksums were not altered. This compatibility step is now built into the verifier.

| Offset | Real uniform recall | Real weighted recall | Later accuracy, uniform → weighted |
|---|---:|---:|---:|
| 0 | 6.72 | 34.33 | 87.78% → 79.76% |
| 1000 | 7.61 | 33.89 | 82.39% → 75.22% |
| 2000 | 6.28 | 35.17 | 84.85% → 75.82% |

The direction of the original result replicates: increased early-prefix recall
with a later-position accuracy cost. Shuffled graphs show the same tradeoff.

This is **not bit-for-bit reproduction across environments**. Source and data
byte hashes match the original, as do all 216 reservoir-weight and teacher-state
hashes. Nevertheless, none of the trained head hashes match the original run;
184/216 recall scores match exactly. The maximum individual score difference is
52 digits. Training accuracy matches in 81/216 rows and later accuracy in 89/216.
Python patch versions differ; platform/BLAS effects on nonlinear optimization
are a possible explanation, not an isolated or proven cause. Within this Windows
environment all saved-head replays and the 18 independent refits match exactly.
Do not replace the original results or claim exact cross-platform equality.

The new curriculum experiment pairs all arms within the same Windows environment
and uses fresh seeds. Its main comparison does not depend on matching older head
parameters. Further numerical portability investigation remains separate work.
