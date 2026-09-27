# Locked cumulative-prefix curriculum — 2026-09-27

Base: 17ef56bb567748f2463cbe7c17742f2e565b7af0. Written before outcomes.

Test one intervention: expand the weighted target window at fixed epoch boundaries.
All arms receive 6,000 full-batch Adam updates, learning rate .03, L2=1e-5:

- uniform: all 199 targets have weight 1 throughout;
- fixed: the first 32 generated targets have weight 4 throughout;
- curriculum: first 32 / 64 / 128 generated targets have weight 4 during
  epochs 1–2000 / 2001–4000 / 4001–6000 respectively.

All remaining targets always retain weight 1. Normalize by the weight sum.
Keep head parameters, Adam moments and bias-correction step across stage boundaries;
do not call a fresh fit for each stage. Fit unweighted feature statistics once.
Save each arm at all three boundaries; ONLY epoch 6000 is the primary endpoint.
No best-epoch selection, early stopping or outcome-dependent schedule is allowed.

Keep incoming-L1, mbon_after_kc, 48 MBON observations, 482 head parameters,
frozen connectivity and encoding, 200 digits, prompt length 3, recall horizon 197.
Use subsets s701/s702; real and role-degree-shuffled topology; fresh seeds
2142/2143/2144; all initializations 0/1/2; offsets 0/1000/2000. Reset reservoir
and head separately for every segment. Pair all arms on identical states and
initialization. This gives 36 state conditions, 324 fits and 972 saved evaluations.
This is trained-segment recall, not prediction of unseen pi or animal memory.

Primary descriptive criterion, at EACH offset on real graphs: final curriculum
mean recall exceeds BOTH controls, with at least 4/6 initialization-averaged
state conditions beating EACH control. A stronger joint-improvement criterion
also requires final later-165 accuracy at least that of fixed weighting at EACH
offset. Report both criteria separately, including failures. These are descriptive
criteria, not significance tests. Do not pool overlapping subsets, initializations
or segments as independent biological replicates.

Report every arm, seed and initialization, plus shuffled controls. Include final
32/64/128/197-digit completion counts; accuracy over generated positions 1–32,
33–64, 65–128 and 129–197; all 199-target accuracy and unweighted cross entropy;
and counts of models that completed the first 32 at epoch 2000 but lost completion
at 4000 or 6000. Later-165 accuracy always refers to positions 33–197, even when
the curriculum window expands. Report intermediate checkpoints only as diagnostics.

Atomically checkpoint whole conditions and checksum their ledger. Strict resume
must reject configuration, source, protocol, data or package changes and missing
or corrupt committed checkpoints. Only one process may write an output directory.
Uncommitted checkpoints are recomputed. Resume occurs at condition boundaries.

Verify every saved head, teacher prediction, complete autoregressive string,
frozen reservoir, metric, aggregate and artifact hash. Independently refit all
three arms and three initializations in the first real circuit/seed at each offset:
27 fits, matching all 81 saved stage heads exactly. Cross-check all 2,200 pi digits
using mpmath and Decimal. Require the full test suite. Test schedule boundaries,
continuous Adam, constant-schedule equivalence, fixed statistics, validation,
interruption/resume and rejection of damaged or changed execution context.
