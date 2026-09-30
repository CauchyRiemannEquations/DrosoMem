# Matched expected energy: observation-noise allocation

Prospective computational protocol. Commit config, implementation, verifier and
this document before main outcomes. This question was selected after the earlier
observation diagnostic; archived model cohorts are reused and are not new holdouts.

## Question and baseline

Does allocating observation noise in proportion to individual MBON training SD
improve frozen-head teacher decoding compared with equal raw-coordinate noise,
while preserving expected total squared injected noise? The prior median scalar
calibration does not equalize coordinate-relative corruption. A positive result
is about coordinate allocation, not better internal storage or physiological noise.

Use all 48 parents in results/relative_noise_main and the saved observation draws
in results/observation_noise_main, locked by manifest SHA256 in config. Require
the parent independent verification to pass. Same clean 199 training rows, 48
observed root IDs in the same order, 482-parameter 48→8 tanh→10 head, fixed mean
and floored scale. No training, preprocessing adaptation or neural simulation.
Pi includes initial3, offset0, total200, prompt314, 197 teacher-evaluation positions.
Each position's previous symbol is the correct digit. Input graph and head
identities stay those of the parent; no new observation neurons are added.

## Intervention and pairing

From clean **training** features compute s_j=SD_t(x_tj), ddof0, q=median(s_j),
and a_j=s_j/sqrt(mean(s_j²)). Require finite s and positive q/RMS; abort otherwise,
without replacing zero by epsilon. Individual zero-SD coordinates get a_j=0.
No evaluation-state fitting. Preserve the original head's scale floor unchanged.

At each evaluation time use the same independent standard Gaussian z_j for both:

- flat: clip(x_clean + r q z, −1,1);
- training-SD allocation: clip(x_clean + r q a z, −1,1).

Thus sum_j (r q a_j)² = 48 (r q)² before clipping. This matches **expected** total
squared noise within a model, not each realized draw, not energy after clipping,
and not raw energy between graphs whose q differs. Save those realized energies,
clipping fractions, coordinate amplitudes and standardized feature errors.
Draws are nested across r=[.0003,.03,.3]; zero is an explicit implementation check,
excluded from the primary curve index. No feedback of perturbed observations.

Archived stage reuses noise372001 exactly, requiring flat scores/features to match
the prior observation artifacts. Fresh stage uses noise seeds408001/408002/408003
for all models, regardless of archived outcomes. For each (noise seed, model seed,
circuit) generate197×48 standard normals in ascending observed root-ID order
using PCG64(SeedSequence([noise_seed,model_seed,circuit,2])), then restore the head
order. Same neuron IDs/order and draws across the three graphs are checked.
Fresh draws concern this 48-coordinate observation scope, not a new all-brain
noise trajectory. Save every draw. Fresh noise realizations are repeated measures,
not independent model or biological replicates.

Models: discovery7142–7146, confirmation381142–381144; circuits701/702;
legacy5,brain5,brain1. Both cohort names are inherited; neither is newly sampled
for this diagnostic. There are48×4streams×3doses×2arms=1152 nonzero evaluations,
plus48×2zero checks. No new autonomous trajectories or refits.

## Endpoints and decisions

Primary: brain1 mean teacher-position-accuracy gain of SD allocation over flat.
For each model seed average equally across the two circuits and three nonzero
doses; fresh stage additionally averages all three new noise realizations.
Keep n5/n3 cohorts and archived/fresh stages separate. Confirm only if **all four**
cohort×stage cells have mean gain>=.05 and positive gain in>=4/5 or3/3model seeds.
Secondary legacy5 and brain5 use the same unchanged gate; report every result.
Failure of this criterion is not equivalence or absence of any small effect.

Save per-case/dose/stream accuracy, first32 and tail accuracy, predictions,
probabilities, target probability, per-position correctness, raw/standardized noise
energy and clipping. Do not call teacher exact-prefix length a Pi Memory Score.
Also show clean teacher accuracy and remaining clean-minus-allocated loss;
an improvement is not necessarily full recovery. Per-seed paired mean/median,
sample variance, 10000 bootstrap intervals(seed408399), paired dz and sign counts
are descriptive; no p-value gate. Compute paired accuracy gain from integer correct
counts before division to avoid floating-point pseudo-wins on exact ties.

## Verification, resources, stopping

Synthetic tests: expected-energy equality, heterogeneity versus uniform SD,
zero-SD behavior, no input mutation, clipping, canonical ID pairing, independent
head scoring, exact ties and four-cell confirmation. Smoke uses7142/c701 in all
three graphs, all streams/doses, first8 teacher positions. No inference from smoke.
Independently reconstruct both allocations, draws, decoder probabilities, every
saved metric and bootstrap/gate; predictions must match exactly, numerical arrays
at atol1e-12/rtol1e-10 (amplitude equality rtol1e-12/atol1e-18). Verify zero checks,
all archive-flat matches, head identity, report parent hashes and prior-file bytes.

One numerical thread, 1200s and2GiB sampled process RSS for each main/verification
invocation. Stop and preserve evidence on any integrity/nonfinite/resource failure;
do not shorten horizon, remove seeds or change thresholds. Original outputs are
never overwritten. After verification follow the fixed ACT V completion plan,
regardless of whether the scientific criterion succeeds.
