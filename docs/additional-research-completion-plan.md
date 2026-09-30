# Bounded completion of the additional research programme

This is a new prospective plan following 6e237a6. The earlier briefing and
feedback report are archived evidence and remain byte-for-byte unchanged.
User requested the remaining additional research through completion. Execute
the following three studies **in order**, verifying each before the next.
Finish negative and inconclusive outcomes without tuning or replacing them.
Completion means these finite computational studies, not all future research.

## Shared design and boundaries

Use actual Drosophila connectome structure in computational rate models.
No claim about animal memory, physiological validation or internal learning.
Same original input IDs/patterns and 48 observed MBONs; 48→8 tanh→10 head,
482 parameters; incoming-L1 gain .9, leak .6, mbon_after_kc. Length 200,
K10, prompt 314, 199 teacher pairs and 197 autonomous symbols. Adam 2000
updates, LR .03, L2 1e-5, fixed-32 ×4; final checkpoint only. All internal
weights remain fixed during learning; only external heads are fitted.

Reuse ACT II model/dataset pairs 9142/9242 through 9146/9246 (discovery n5).
Unconditionally add fresh confirmation pairs 440142/440242 through
440144/440244 (n3); never pool the cohorts. Circuits 701/702 are samples
from one connectome, not independent animals. Model and dataset seeds are
paired rather than crossed; their variance components cannot be separated.
Pi is one shared trained realization; random suffixes are iid pseudo-random
conditional on prompt. Shuffled pi preserves prompt and digit multiset.
Confirmation pi still uses the same pi segment: new model seeds, not new pi.

Freeze training-coordinate SD s_j, q=median(s), a=s/RMS(s). Compare flat
noise rq z with allocated rq a_j z, clip observations to [-1,1]. Doses
r=[.0003,.03,.3]. Same expected preclip squared noise energy 48(rq)^2;
neither realized energy nor postclip energy is matched. Observation noise
never writes into neural state; generated symbols alone feed back.
Canonical sorted-root-ID PCG64 draws use SeedSequence([noise_seed,model_seed,
circuit,2]), seeds 448001–448003; paired across graphs/families/arms/doses.

Exact prefix excludes prompt and is capped at 197; only pi calls it PMS.
Recalled-prefix bits=L log2(10) is not Shannon/formal memory capacity.
Capped retention=min(prefix/clean_prefix,1); clean=0 makes it undefined.
No dropping zeros or adding epsilon. A gate needing undefined retention is
inconclusive. Average doses, circuits and noise draws **within model seed**.
Joint improvement gate: mean prefix gain >=2 AND retention gain >=.10,
both positive in >=4/5 discovery and 3/3 confirmation blocks. Confirm only
if both cohorts pass. Save mean, median, sample variance, 10,000 seed-block
bootstrap intervals (448399), paired dz and sign counts. No p-value gate.

First-error certificates use deterministic teacher states and identical
time-indexed observation noise. They certify autonomous outputs only through
the first error. Never invent autonomous outputs after that error. Actual
rollouts save symbols/probabilities, raw and noisy observations, activity,
clipping, energies, rank, temporal similarity and region/post-error accuracy.
Keep certificates, actual trajectories, refits and reused baselines separate.

## Study 2 — Sequence generalization

Pi, random and shuffled_pi × two graphs legacy5/brain1 × two circuits ×
eight blocks = **96 models**, including 60 reused ACT II heads and 36 fresh
fits. Reconstruct clean autonomous paths for every model. Fresh fits collect
new teacher states; archived heads/states/checkpoints are hash-pinned.
Three draws × three doses × two allocations = **1728 certificates**.
Execute all six noisy paths for first draw 448001 in every partial model,
and in whole models at the first block of each cohort/circuit701 for every
family: 54 selected models. Together with 96 clean paths: **420 actual
neural evaluation trajectories**, plus 36 new teacher training trajectories.

Primary: allocated-minus-flat joint gain in **random/brain1**. Other graph/
family gates are fixed secondary outcomes. Broader support requires both
random and shuffled_pi in both cohorts; pi cannot rescue a failed random gate.
Do not reinterpret these trained sequences as unseen prediction.

Independently audit all endpoints/head probabilities/metrics/statistics and
pre-error consistency. Full independent replay: all 48 partial models and
whole random models at first block of each cohort/circuit701 = **350 paths**.
Reconstruct their teacher states too. Independently refit all 36 fresh heads
from stored features and the two archived random first-block/circuit701 heads.
The other whole trajectories are not claimed independently replayed in full.
Smoke: random, circuit701, both graphs, first archived block and separate
449139/449239 block, horizon8: four models, 72 certificates, 28 paths,
two smoke fits. Verify all smoke trajectories; exclude all smoke from inference.

## Study 3 — Structural controls

After study2 verification, use its pi/random partial heads as intact parents.
Eight blocks × two circuits × two families × intact/degree/role = **96 models**.
Fit each of 64 rewired graph heads from scratch with the same budget. This is
graph-specific refit followed by frozen evaluation, **not frozen-head transfer**.
One deterministic graph draw per family/circuit/block: generator seed
468000 + 10*(model_seed-9142 for discovery, model_seed-440142+10 for confirmation)
+ 2*(circuit-701) + (0 for degree, 1 for role). Reuse that graph across pi/random.

Degree preserves each neuron's signed in/out degree and incoming weight multiset;
role additionally preserves each neuron's signed in/out counts to each annotation
role. Specific partners/motifs change; outgoing strengths need not be preserved.
10E accepted swaps, 200E proposal cap, no loops/duplicates; stop on failure.
Finite sampler, no uniform mixing claim. Save actual matrices and independent
invariant/overlap audits. No whole-brain rewiring or physiological-module claim.

Use q from the paired **intact** training trace for all three topologies;
allocation a still uses each topology's training SD. Thus expected raw noise
energy also matches across topologies, before clipping. Save both q values.
Every model has 18 certificates and seven actual paths: **1728 certificates,
672 actual evaluation paths**, 64 training paths. Independently replay all
672 paths and all teacher states; exact refit all 64 new heads.

Primary structural hypothesis: under allocated noise, intact-minus-role mean
prefix >=2 and retention >=.10 with the same sign rule, in BOTH pi and random
and both cohorts. Degree is prespecified secondary. Also report clean scores
and allocation-gain interactions descriptively. Negative/inconclusive does
not prove equal networks. Zero clean scores remain visible/inconclusive.
Smoke: first archived block/c701/random, all three topologies, horizon8;
54 certificates/21 paths/two fits, excluded from inference.

## Study 4 — Correlated observation noise

After study3 verification, reuse all 96 intact study2 models. Keep the same
independent draws, and draw common components g_t from separate PCG64 streams
SeedSequence([common_seed,model_seed,circuit,3]), common seeds 478001–478003.
z_j(rho)=sqrt(1-rho)z_j+sqrt(rho)g, rho=[0,.25,.75]. This is time-white
correlation shared across the fixed 48 observed MBON coordinates, not calibrated
biological noise or whole-state perturbation. Marginal variance is one and
expected squared energy is unchanged across rho and allocations.

5184 certificates: 1728 rho0 certificates **reuse** study2 exposures exactly,
3456 new correlated exposures. No new fits. Rho0 predictions/probabilities
must match study2; do not count them as new independent evidence.
Actual paths only for random at first block of each cohort/c701, both graphs:
four clean plus 4 models ×2nonzero rho×3doses×2arms = **52 new paths**.
Other family/seed endpoints are certificates, not full new autonomous outputs.
Independently reconstruct all 5184 endpoints and fully replay all 52 paths.

Primary: random/brain1 allocation gain meets the same joint criterion at BOTH
nonzero rho in both cohorts. Pi/shuffled/partial outcomes are prespecified
secondary. Also report degradation versus rho0 separately for each arm;
retained allocation benefit does not establish absolute robustness.
Smoke: random first archived block/c701/both graphs, horizon8, 108
certificates/26 paths. Smoke rho0 equals the first eight study2 positions.

## Engineering, stopping and completion

Commit protocol/config/numerical implementation before outcomes. Two workers,
one numerical thread each. Per worker process-tree sampled RSS <=3GiB and
900seconds; per main stage <=7200seconds, verifier <=3600seconds. Preserve
partial artifacts on nonfinite/integrity/resource failures; cancel other jobs.
No overwrites, model/seed/dose replacement, best checkpoint selection or tuning.
Each stage pins its exact parent manifest at start, validates it and the preceding
stage's independent checks; the fixed source path is not chosen after outcomes.
Replays require exact symbols and numerical atol1e-12/rtol1e-10; metric reductions
use atol1e-11. Old result hashes and prior sealed reports must remain unchanged.

Complete with per-stage manifests/raw tables/checkpoints/plots, an 11-section
Korean synthesis, updated navigation and one next proposal. Measured physiology,
different neuron dynamics, whole-brain rewiring and additional task families
remain explicitly unexecuted. No success condition depends on those extensions.
