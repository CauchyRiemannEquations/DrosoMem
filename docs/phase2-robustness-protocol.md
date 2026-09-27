# Locked Phase 2 follow-up: perturbation robustness — 2026-09-27

Base: 155cd224bc91ad2a59c9a0775808d1bbbfd4f508. Commit this protocol and code before
the main perturbation outcomes. This closes a scoped robustness question on the
current fixed-32 MB baseline, not a rerun of all 450 original Phase 2 conditions.
The delayed-memory part of that follow-up was already measured in Phase 3b.

## Models and independent computational confirmation

Use all final fixed-32 heads at offset 0 from two existing studies, without
selecting on clean or noisy performance:

- Initial cohort: `results/phase5_retention`, model seeds 3142/3143/3144.
- Confirmation cohort: `results/phase5_retention_confirmation`, seeds 4142/4143/4144.

Each cohort has two circuits × three model seeds × real/role-degree-shuffled
graphs × initializations 0/1/2 = 36 heads, hence **72 heads total**. Connectivity,
encoder, normalization and 482-parameter readout were previously trained/fixed
using 200 digits, 6,000 updates, incoming-L1 and mbon_after_kc. There are **zero new
fits** here. The independent model seeds are from saved, already evaluated models;
this is a new perturbation test, not new training or unseen-pi prediction. Both
cohorts and all noise levels are fixed together; do not tune after cohort one.

Reconstruct source weights/states, verify source artifact/data checksums and head
hashes/predictions, then reproduce each entire clean 197-digit rollout. Only the
public prompt `314` is supplied to the generator. Score the continuation up to
its first error outside the generator. Keep all 197 generated digits, including
those after the first error. Head parameters and normalization remain frozen.

## Three separate perturbations

1. **Pulse:** after the three clean prompt updates and before the first prediction,
   add independent zero-mean Gaussian noise to all 686 state coordinates once.
2. **Ongoing:** after clean prompt warmup, add such noise before every prediction.
   Noise changes the recurrent state, not just the readout observation. Pulse and
   ongoing use the same first Gaussian vector for a matched seed.
3. **Edge dropout:** before prompt warmup, remove exactly `floor(p*nnz)` existing
   edges, selected without replacement from a seeded permutation of canonical CSR
   data positions. Use the altered graph through the entire recall. Surviving
   weights/signs are unchanged; do not renormalize or refit. Removal changes both
   topology and input strength, so this is not a pure topology intervention.

State noise standard deviations: **0, 1e-6, 1e-4, 1e-3, 1e-2** in absolute,
dimensionless rate-state units. Clip perturbed states to [-1,1] and count clipped
coordinates. These are computational stress levels, not measured biological
noise. Record clean training-state SD statistics to contextualize the scale;
do not rescale noise using test outcomes or change the trained readout scale.

Dropout fractions: **0, .01, .05, .10**. A frozen classifier under perturbation
can fail from feature/distribution mismatch; failure alone does not prove that
all memory information is absent from the reservoir.

Use **20 noise seeds** per nonzero strength: initial cohort 5100–5119, confirmation
6100–6119. RNG SeedSequence components are noise seed, model seed, circuit index
in the fixed config, then family code (1 for either state mode, 2 for dropout).
Within a circuit/model seed use matched perturbations across initializations,
strengths and graph controls. Matching CSR positions across different supports
does not mean the same anatomical edges are removed. Each strength resets the RNG,
so masks are nested and Gaussian vectors are paired across strengths.

For every head store one clean trial and one zero-strength trial for each family,
plus 220 nonzero trials. Total: **16,128 full recalls** = 72 clean + 216 null checks
+ 15,840 perturbed recalls. Null trials must equal the clean full string exactly.

## Prespecified decision

Primary strengths: **sigma=.001** for pulse/ongoing, **p=.01** for dropout. Other
levels describe the response curve and cannot replace a failed primary endpoint.
For real graphs, separately in each cohort and each primary family, require:

1. Mean noisy recall at least **80%** of the matched clean mean.
2. At least **5/6 circuit × model-seed blocks** retain at least 80% of their clean
   mean. Average all three initializations and all noise seeds within a block.
3. Among heads that cleanly complete the first 32 digits, at least **80%** of
   matched noisy trials still complete 32 digits. Zero eligible heads is a failed
   criterion, not automatic success. Zero clean mean also cannot pass by ratio.

Overall confirmation requires all three criteria for all three primary families
in both cohorts. Report family/cohort decisions even if overall fails. The 80%
and 5/6 thresholds are explicit engineering acceptance rules, not biological
capacity bounds or statistical significance tests. Report every control curve;
noise repetitions, initializations and overlapping circuits are not independent
animals. Do not infer biological wiring superiority from pooled trials.

## Integrity and verification

Keep original studies unchanged. New per-head atomic JSON checkpoints include
seed, perturbation, entire output, first-error score, altered-weight hash and
clipping count. Strict resume binds config, code, data, protocol and pinned source
manifests. Stop after the first actual head, then resume in a separate process.
Independently replay **all 16,128 recalls**, verify original weights/readouts are
unchanged and recompute scores, aggregated decisions and pi digits through 200
with a second implementation. Tests cover manual injection timing, zero-noise
equivalence, nested exact dropout, immutability, deterministic seeds, criteria
failure, separate confirmation, checkpoint corruption and resume equivalence.
Full pytest must pass before the main experiment. No post-outcome tuning.

Completion of this scoped experiment is distinct from passing robustness. Neither
outcome completes LIF, biological dopamine, whole-brain modeling or game screens.
