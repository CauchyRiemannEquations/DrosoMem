# Graph-relative observation noise — prospective protocol

Registered before smoke/main outcomes. The motivating **post-hoc** observation in
[the completed structural study](structure-noise-results.md) was larger training
observation SD in degree controls (own/common q ratio median 2.596). Its negative
intact-over-role result remains unchanged. This is a new calibration experiment,
not a revised analysis gate for the old study.

## Question and hypotheses

Does calibrating observation noise to each graph's training signal scale attenuate
the apparent degree-control recall advantage under a common raw-noise scale?
H1: on random digits, the control-minus-intact gap decreases by at least 2 exact
prefix symbols and .10 own-clean retention, with positive changes in at least
4/5 original discovery model blocks and 3/3 original confirmation model blocks.
Both existing cohorts must pass separately. These are **reused models**, not fresh
model confirmation. Noise seeds are fresh. The practical thresholds follow the
previous suite's magnitude thresholds; they are not physiological standards.
Failure means the joint criterion is not met. Undefined retention makes that cell
inconclusive, never a dropped model or an epsilon-adjusted denominator.

The primary family/control is **random/degree**. Pi/degree and both role contrasts
are descriptive secondary checks, without a multiple-testing discovery claim.
Intact winning is not a success condition. No hyperparameter or checkpoint search.

## Frozen sources and conditions

[Config](../configs/graph_relative_noise.json) pins the parent structural manifest
and independent verification. Reuse all 96 parent heads: pi/random × intact/degree/
role × two circuits (701,702) × eight paired model/dataset blocks. Cohorts remain
9142–9146/data9242–9246 (n5) and 440142–440144/data440242–440244 (n3).
Two circuits are input/circuit constructions from one connectome, not animals.
32 unique rewired graphs are reused across families. Parent sampler properties
and limitations remain unchanged. No new head or graph is fitted/generated.

Length200, prompt314, horizon197, K10; same inputs and 48 observed MBONs;
482-parameter 48→8 tanh→10 readout; parent training 2000 Adam updates, LR.03,
L2=1e-5, fixed32×4; incoming L1 gain.9, leak.6, mbon_after_kc.
All recurrent weights and head parameters remain frozen. Parent training loss,
teacher accuracy, dataset, head, graph and environment identities are retained.

Let s_g be the coordinate SD over all 199 **training** observations, q_g its
median, and a_g=s_g/RMS(s_g). For relative strength r and canonical independent
standard normal vector z, noise is r q a_g z, then observation clipping to [-1,1].

- common: q=q_intact for all graphs in the paired task/model/circuit cell.
- own: q=q_g. Intact is therefore an exact null comparison.

Within the allocated arm, common matches expected raw noise energy across graphs;
own matches a median-SD-relative amplitude. Neither matches realized or clipped
energy, per-coordinate SNR, readout-standardized energy or covariance geometry.
The allocation vector is graph-specific and unchanged between calibrations.
No flat allocation arm is needed for this question; its earlier result is preserved.

Fresh noise seeds488001–488003, strengths .0003/.03/.3, canonical root-ID order,
PCG64 SeedSequence([noise_seed,model_seed,circuit_seed,2]). Draws paired across
families, topologies, doses and calibrations. No time correlation. Noise acts on
observations; generated symbols feed back, observation noise does not directly
overwrite the reservoir state. Autonomous execution receives no target sequence.

## Metrics and analysis

Primary metric: exact autonomous prefix excluding prompt (cap197; PMS only for pi).
Retention=min(prefix/own clean prefix,1), undefined at clean0. Prefix bits are
L log2(10), not formal memory capacity. Reproduce all clean parent paths exactly.
Compute teacher first-error certificates and **all** corresponding autonomous
paths; check equality up to and including first error. Certificate accuracy after
that point is teacher-forced, not autonomous accuracy.

Save generated symbols, probabilities, position/region accuracy, error position,
clean retention, raw/noisy MBON states, active counts, state norms, sparsity,
effective rank, temporal cosine, state deviation from clean, injected/clipped
energy and clipping. This is observation-noise calibration, not a new zero-input
state-decay experiment; parent state diagnostics remain references.

For each metric compute G_common=control_common−intact_common,
G_own=control_own−intact_own, and **attenuation=G_common−G_own**.
Average the 18 circuit×noise×dose observations within each model block first.
Report all seed blocks, both gaps and attenuation, mean/median/sample variance,
paired dz, signs and percentile95 bootstrap interval (10000 draws, seed488399).
Do not pool the n5/n3 cohorts or count noise repeats as independent models.
Retention NaN propagates through block reductions and statistics. Dose curves,
calibration q ratios and energy/clipping are descriptive. No p-value gate.

There are 1728 labeled noisy evaluations, of which 288 own-intact settings are
exact repeats of common-intact. Thus 1440 distinct fresh noisy exposure settings,
plus 96 clean parent repeats; 1824 actual path executions and independent replays.
Fresh noise tests calibration on these chosen models, not new graph populations.

## Stopping, smoke and verification

One process, one numerical thread; run≤3600s and sampled process RSS≤3GiB;
independent verifier has the same bounds. Budget checks occur between paths/cases,
not a hard real-time kill. Preserve failed/partial outputs, never replace old ones.
No early stop for a positive/negative result. If budget or validation fails, stop
and report incomplete; no selective missing-cell statistics.

First smoke: three parent random graphs for seed9142/circuit701, horizon8,
separate smoke seeds488901–488903; all calibration/dose/noise settings (57 paths).
Smoke verifies interfaces and saves artifacts, is excluded from main statistics,
and cannot change the registered conditions. Main waits for successful smoke audit.

Independently reconstruct training observations and autonomous recurrence/head
probabilities, noise, calibration, energy, metrics and statistics. Audit saved
control graph invariants/normalization and frozen head/graph hashes. Exact symbols
and clean replay; continuous tolerance atol1e-12/rtol1e-10 (metrics atol1e-11).
Record git revision, source/config/protocol hashes, parent checkpoint paths/hashes,
versions/hardware, runtime and sampled memory. Hash all prior results before/after.
Results and plots receive new manifests; no sealed historical report is edited.

## Interpretation boundary

A decrease would show that calibration accounts for part of a robustness contrast
in this computational model. The own intervention changes noise energy; it cannot
prove pure topology causation, anatomical superiority or removal of every signal
scale effect. Reused graph-specific heads embed decoding differences. This is
representation/decoding under synthetic noise, not internal learning or animal
memory. Whole-brain rewiring, physiological calibration and other dynamics remain
outside scope.
