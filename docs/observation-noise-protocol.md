# ACT V diagnostic — current observation versus accumulated noise

Preregistered 2026-09-30 before computing counterfactual outcomes.
The relative-noise study's independent validation passed before execution.

## Question and motivation

The preceding relative-noise panel did not confirm whole-brain superiority.
At r=.03 and .3, teacher-forced decoding also deteriorated. Does accumulated
state disturbance add a material decoding cost beyond corrupting the current
observed feature vector alone? This is a decoding/dynamics diagnostic, not a
new autonomous-memory comparison or a new model-training experiment.

## Fixed sources and counterfactual

Reuse all48 models and clean/full-noise teacher trajectories from the pinned
results/relative_noise_main manifest in config. The source has30 archived heads
at7142–7146 and18 freshly fitted heads at381142–381144, circuits701/702 and all
legacy5/brain5/brain1 conditions. All are included. Their clean-trained482-parameter
heads,48 MBON order, pi200/prompt314/197 positions, and training preprocessing
remain fixed. The source study must pass verification first. No refitting.

For each registered r=[.0003,.03,.3], use sigma=r*median clean training MBON SD
and the EXACT same canonical current-step Gaussian draws as the source.
Construct x_instant[t]=clip(x_clean_teacher[t]+sigma*Z_observed[t],−1,1), then
apply the frozen head. These noisy observations never modify the clean reservoir
or the next input. Correct previous pi symbols are supplied throughout, as in
the saved full-noise teacher condition. Thus there is no autoregressive error
feedback in either condition. At the first prediction both feature vectors must
match because the prompt was clean; check this for every head and strength.

Compare against saved x_full_teacher[t], which includes current and prior
all-neuron noise and its propagation. This removes all noise history from the
counterfactual, including history of observed-neuron noise. It is NOT an
unobserved-neuron-only ablation, equal-total-energy exposure, or a new physiological
noise model. Hidden noise is not independently isolated from observed-noise history.

Compute144 new instantaneous-observation evaluations of197 positions; compare
with144 archived full-state teacher paths. Recheck48 clean head outputs. No new
autonomous trajectories and no new full-graph dynamics are executed. Save all
new predictions, probabilities, features, correctness and paired standard-normal
draws, plus source checkpoint/manifest identities. Keep parent bytes unchanged.

## Registered metrics and decisions

Position accuracy is the endpoint; teacher prefix is not promoted to autonomous
Pi Memory Score. Save per-head/dose clean, instantaneous and full accuracies,
first32/tail accuracies, target probabilities and standardized feature distances.
History accuracy cost = instantaneous accuracy − full accuracy. It can be negative.
Also report clean−instantaneous and clean−full accuracy drops. These are arithmetic
contrasts, not fractions of Shannon information or proof of where memory is stored.

Average all3 nonzero doses and both circuit strata within each seed. Keep source
discovery n5 and confirmation n3 separate. They are inherited cohorts; this
diagnostic creates no additional fresh model/sequence/noise seeds or biological
replicates. The source outcomes informed the diagnostic question, so this is
exploratory follow-up even though its new endpoints are fixed before evaluation.

Primary: a material additional history cost in brain1. Require mean cost>=.05
(5 percentage points) AND positive cost in>=4/5 discovery and3/3 confirmation
blocks, unchanged in both cohorts. Apply the same secondary gate to legacy5 and
brain5; do not substitute a secondary for a failed primary.

Separately, report whether the history contrast is within a fixed descriptive
near-tolerance: absolute cohort mean<=.02 AND every seed's absolute contrast<=.05,
in both cohorts. This is not a formal statistical equivalence test. A failed
material-cost gate does not automatically pass near-tolerance. Show every dose
and seed; do not claim all noise history irrelevant from a grid-mean criterion.

Report paired per-seed values, mean, median, sample variance,10000 bootstrap
intervals,paired dz and wins/ties/losses (bootstrap seed398399, absent from prior
configs). No p-value gate, dose search, changed head, exclusions or optional stopping.

## Verification, budget and completion

Synthetic tests cover clipping, zero perturbation, immutable inputs/head, canonical
ID/time pairing, independent forward scoring and decision thresholds. Use an
independent ID lookup/stream reconstruction and head implementation to check ALL
144 counterfactuals, all48 clean outputs, all144 first-step identities, saved
metrics, summary/gates and hashes. No new full-graph replay is claimed; the source
study's dynamical verification is explicitly inherited with its stated scope.

One numerical thread,1200s/2GiB for the diagnostic and1200s/2GiB for verification.
Record environment, wall time and .05s sampled process RSS. Stop on mismatch or
resource failure and preserve artifacts; never relax tolerance after outcomes.
Freeze code/config/protocol in a commit before new counterfactual outcomes.
After reporting this second study, select one future experiment and stop this
two-study work request. The earlier absolute-noise and relative-noise findings
remain separate and unchanged.
