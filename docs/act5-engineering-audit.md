# ACT V engineering correction before main outcomes

The 7cdba73 smoke passed all three graphs, 15 actual zero rollouts and independent
scalar verification (66 non-alias evaluations). However its supervisor measured
only the Windows venv launcher process (~4MiB), not its Python worker child.
The original `results/act5_smoke/resources.json` is preserved unchanged. **Those
three RSS values are not valid worker peak-memory measurements.** Numerical
outputs, durations and independent verification remain valid.

Before any main outcome, supervision was corrected to sum RSS over the complete
worker process tree and terminate that same tree if a budget is exceeded. No
simulation, random stream, metric, seed, strength or decision criterion changed.
One additional whole-brain threshold1 horizon8 resource probe repeats the same
smoke case, checks its saved arrays against the original, and records the actual
tree RSS. It is engineering-only, excluded from main statistics. The main uses
the corrected supervisor and the unchanged preregistered protocol/config.

The earlier smoke verifier was executed while its 7cdba73 implementation was
current; replaying its strict source check requires that historical revision.
New main source context records the corrected revision. An obsolete measurement
is not silently replaced by a new hash.
