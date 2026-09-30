# Coordinate-SD noise: final bounded observation-calibration study

This protocol is committed before smoke/main outcomes. It follows the negative
[median-scale calibration study](graph-relative-noise-results.md), without changing
its criteria or artifacts. This study plus verification and synthesis closes the
currently authorized observation-noise extension; it does not close physiological,
whole-brain rewiring or new-dynamics research. No automatic chain of further noise
calibrations, strength searches, extra seeds or refits follows the outcome.

## Question, hypothesis and decision

Does directly matching preclip noise to each observation coordinate's training SD
further attenuate the degree-control minus intact autonomous-recall gap compared
with the previous own-median calibration?

Primary: random digits / degree control. Define G_c = control_c − intact_c for
calibration c, and **incremental attenuation = G_own − G_coordinate**. Unlike the
previous experiment, intact also changes between own and coordinate. Its change
must be subtracted; a decrease in the control alone is not this contrast.

H1 passes only if incremental attenuation averages at least 2 prefix symbols AND
.10 own-clean retention in BOTH existing cohorts, with strictly positive prefix
changes in at least 4/5 and 3/3 blocks respectively, and the same positive-count
requirements separately for retention. These thresholds are inherited practical
magnitudes, not biological standards. Failure means the conjunction fails;
undefined required retention means inconclusive. No p-value gate. Intact winning
is not required. Both cohorts are reused models, not new model confirmation.

Pi/degree and both role contrasts are descriptive secondary checks. Save all
three gaps and common−coordinate total attenuation plus common−own attenuation
(a fresh-noise replication of the previous contrast). None replaces the primary.
Absolute prefix and retention are distinct: retention has each graph's own clean
denominator and is capped at one. Clean0 remains undefined, including the known
pi/degree seed9143/circuit702 case; do not drop it or insert an epsilon.

## Fixed sources, controls and budgets

[Config](../configs/coordinate_sd_noise.json) pins structural parent/verification
manifests and the completed median-calibration parent/verification. Reuse all96
parent models: pi/random × intact/degree/role × circuits701/702 × eight paired
model/data blocks. Discovery9142–9146/data9242–9246 (n5); historical confirmation
440142–440144/data440242–440244 (n3). Do not pool cohorts. They share one connectome,
not independent animals; the32 rewired graph draws are reused across both tasks.
Degree/role sampler properties remain those audited in the structural study.

686 cells, threshold5,3309/3241 edges; identical inputs,48 observed MBONs,
482-parameter48→8tanh→10 head; K10/length200/prompt314/horizon197. Parent training:
2000Adam updates,LR.03,L2=1e-5,fixed32×4; incomingL1 gain.9,leak.6,mbon_after_kc.
No new graph generation, head fitting, recurrent learning or parameter selection.
Reproduce every clean baseline exactly and retain parent training loss/accuracy,
dataset identity, head/graph hashes and checkpoint references.

Let s_g be ddof0 SD of each coordinate across all199 training observations,
q_g=median(s_g), u_g=RMS(s_g), a_g=s_g/u_g. With relative strength r:

| Calibration | Noise amplitude per coordinate |
| --- | --- |
| common | r q_intact a_g |
| own | r q_g a_g |
| coordinate | r s_g |

Common matches expected raw energy across graphs. Own retains the graph-specific
q_g/u_g factor. Coordinate removes it: amplitude_i/s_i=r where s_i>0; zero-SD
coordinates receive zero noise and their count is reported. Median/RMS must remain
positive as in the source panel; invalid scales stop execution. Coordinate energy
is r² sum(s_i²), generally different across graphs. None of these conditions
matches clipping, joint covariance, head decision margins, realized noise energy
or physiological noise. This is preclip matching, not a promise of postclip SNR.

Fresh main noise498001–498003; smoke498901–498903; strengths .0003/.03/.3.
Canonical sorted-root-ID PCG64 SeedSequence([noise_seed,model_seed,circuit_seed,2]);
same draws paired across all calibrations, topologies, tasks and strengths.
Time-white independent-coordinate noise is added to48 observations then clipped
to[-1,1]. It is not written to the reservoir; generated symbols feed back.
The autonomous function receives prompt and noise, never the target sequence.

## Recording and analysis

Save all actual autonomous predictions/probabilities, position and region accuracy,
prefix/error position, own-clean retention, Llog2(10) prefix bits, raw/noisy MBON
states, active neuron count, norm, sparsity, effective rank, temporal cosine,
deviation from clean, injected/clipped energy and clipping. Prefix excludes prompt,
cap197; only pi calls this PMS. Prefix bits are not formal/Shannon memory capacity.
Teacher-forced first-error certificates are separate from post-error autonomy.
Save all certificate probabilities and compare through first error against every
actual path. Parent training diagnostics are retained; no new decay experiment.

Each model block averages18 circuit×noise×strength observations first, per
calibration/topology. Save all blocks and paired differences; report mean, median,
sample variance, paired dz, signs,10000-draw percentile95 bootstrap (seed498399).
NaN propagates through reductions. Dose curves, energy, clipping, q/u and
representation diagnostics are descriptive, not additional success searches.

96models ×27 noisy settings =2592 labeled noisy paths;96 clean repeats give2688
actual main paths. Common/own intact coincide (288 repeats), leaving2304 distinct
fresh noisy exposure settings. Independent replay covers ALL2688 paths plus96
teacher paths and64 control graph/task audits (32 distinct control graphs).
These counts are not independent statistical sample sizes.

## Smoke, stopping and verification

Smoke first: parent random seed9142/circuit701 across3topologies, horizon8,
all27 settings plus clean =84paths, with separate noise seeds. Verify independently
before main. Smoke is excluded from inference and cannot change scientific design.
One compute process/one numerical thread. Main≤3600s and sampled process RSS≤3GiB;
independent verification has the same limits. Checks occur between paths/cases,
not a real-time kill. Stop on budget/nonfinite/identity/verification failures,
preserve partial files, and report incomplete instead of selecting finished cells.
No outcome-dependent stopping or unregistered tuning. Engineering-only fixes must
be documented with their failed artifacts and preserve the scientific design.

Independent verification reconstructs recurrence, noise, SD/amplitudes, head
probabilities, all metrics, bootstrap and primary contrasts. Exact symbols/clean
parent replay; continuous atol1e-12/rtol1e-10, metric atol1e-11. Assert intact
common/own equality, graph invariants/normalization, frozen weights/heads and
canonical pairings. Pin source/config/protocol/git/environment, record runtime and
50ms sampled RSS, hash prior artifacts before/after. Use new output directories.

## Closing boundary

After smoke, full main, independent audit, plots and11-section results, publish a
scoped synthesis of feedback, sequence, structure, correlation and both calibration
studies. Preserve negative and undefined findings. This closes the specified
computational extension whether H1 passes or fails. New model/graph confirmation,
physiology, other dynamics and whole-brain rewiring remain explicitly unexecuted
extensions. A next highest-information experiment may be proposed once, not run
as an endless search. Claims remain computational representation/decoding under
synthetic observation noise; no animal memory or experience-driven internal learning.
