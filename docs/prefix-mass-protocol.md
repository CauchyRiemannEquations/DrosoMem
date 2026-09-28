# ACT II diagnostic: prefix loss mass at N512

Locked before treatment outcomes, 2026-09-28; base revision3553ddd.
Exploratory preregistration-style protocol, not external registration.
Question: does dilution of the first32 targets' normalized loss weight
contribute to N512 prefix-recall failure under this fixed decoding procedure?
This is one causal intervention in the computational training pipeline, not
a search for a better connectome, a new biological learning rule or capacity.

## Exact intervention

Reuse all20 N512 checkpoints from results/scaling_main/n512: model/dataset
blocks11142/11242 through11146/11246, strata701/702, legacy5/brain1.
Their baseline outcomes are already known; treatment outcomes are new.
Training state matrices, symbols, normalization, initialization algorithm/seed,
input/observed IDs, frozen network, readout and optimizer remain identical.
Refit from the original initialization, never from the trained baseline head.

N=512, K=10, prompt314,511 training pairs and509 generated outputs.
First32 generated targets are training indices[2,34); two prompt-internal
targets keep weight1. Remaining unweighted count U=511−32=479.
Baseline multiplier4 gives prefix mass128/607≈.210873.
The N200 pilot used prefix mass q=128/295≈.433898, so the SINGLE treatment
multiplier is qU/(32(1−q)) = **4×479/167 ≈11.4730538922**.
Use the float value from this exact formula, not a rounded decimal. Other
weights remain1 and the existing optimizer normalizes by total weight.
No alternative multiplier, weighted window or post-outcome adjustment.

Both conditions use the same511 rows, mean/std (floor1e-5),48→8 tanh→10 head
(482 parameters), SeedSequence([model_seed,9901,0]),2000 Adam updates, LR.03,
beta(.9,.999), epsilon1e-8, L2=1e-5, final head. Keep incoming-L1 gain.9,
leak.6, mbon_after_kc, KCEncoder fraction.1/amplitude.5; all graph weights frozen.
legacy5 has686 nodes; brain1 has138,639 nodes /15,091,983 edges. Exact input
IDs and48 observed MBON IDs/order match within each seed/stratum.

## Artifact reuse and baseline validation

Hash-check the complete source cohort and all20 source checkpoints. Verify
the earlier exact-replay records refer to those exact source run manifests.
Before treatment, newly replay and independently refit first block/stratum701
for both graphs: expected baseline prefixes12 /33, cohort means14.6 /9.5.
These are checks of existing data, not new independent baseline evidence.

The cached worker reads verified raw features and symbols, reinitializes and
refits the head, then builds the graph for target-free autonomous rollout.
Preserve source graph/feature/diagnostic hashes and explicitly mark copied
teacher-state and decay diagnostics. Independent verification regenerates
states/decay from the graph for every new head; it must not merely trust caches.
Compare resulting means/scales and features byte-for-byte to baseline.
Current source fingerprints must match the recorded numerical modules.

Smoke: cached features from results/scaling_smoke/n512, block11139/11239,
stratum701, both graphs,20 updates,2 treatment fits. Replay/refit both exactly;
exclude smoke scores from scientific inference. Main:20 treatment fits;
20 existing baselines, no unnecessary baseline cohort rerun.

## Metrics and analysis

Primary endpoint: paired change in exact_prefix_symbols (first error before
509-step horizon; prompt excluded). Average two strata within each of five
paired blocks separately per graph. A scoped loss-mass contribution is
supported if mean treatment−baseline prefix>=5 AND >=4/5 block deltas>0.
Failure means not established at this intervention/budget, not impossibility.
Report each graph; no favorable individual-seed selection or pooled rescue.

Prespecified tradeoff endpoint: teacher-forced accuracy on generated positions
33–509 (training indices[34,511),477 targets). Cost-free early improvement
requires the prefix criterion AND mean later-accuracy delta>=−.02 AND >=4/5
block deltas>=−.02. If prefix improves but this fails, report redistribution
with a late-decoding cost, not a general improvement. Autonomous late accuracy
is separately descriptive because earlier errors change subsequent inputs.

Save training histories, next-symbol and scored teacher-forced accuracy,
early1–32 and later33–509 teacher/autonomous accuracy, all generated symbols,
position accuracy/probabilities, confidence, first error, censoring, prefix
bits Llog2(10), and each updated weighted-majority/first-order control.
Training objective values across different weights are not directly comparable:
also report unweighted next-symbol cross entropy and the common BASELINE4x
weighted cross entropy for both saved heads, plus early/later cross entropy.
Neural training features, sparsity/cosines/rank/decay are held fixed and reused;
autonomous trajectories may diverge after a prediction error. Do not call
unchanged cached rank a newly improved representation.

Retain all40 condition rows, per-seed raw scores and paired block deltas,
mean/median/sample variance,95% percentile bootstrap intervals (10000 paired
block resamples, seed13399), d_z (undefined if delta SD=0), wins/ties/losses.
Model/data seeds are paired, not crossed; five blocks are not five animals.
Graph differences and difference-of-intervention-effects are descriptive,
not a new whole-brain-superiority success test. No p-value decision gate.

## Conditional independent confirmation

If either graph passes the main prefix criterion, run BOTH graphs and both
conditions on fresh blocks(13142,13242),(13143,13243),(13144,13244), strata701/702.
Generate new N512 random data using the same PCG64/SeedSequence2201 rule;
build baseline features once per graph/block/stratum and reuse them for treatment.
Twelve baseline plus12 treatment fits, exactly the same hyperparameters.
Only discovery-qualifying graphs can earn a confirmed claim: mean gain>=5 and
all3 block deltas>0. Cost-free claim also requires mean later delta>=−.02 and
all3 later deltas>=−.02. Report every confirmation row, even for nonqualifying
graphs. If neither discovery graph qualifies, do not run confirmation or tune.

Replay every new head exactly, regenerate its teacher-state/decay/probability
artifacts, and independently refit first block/stratum701 in each graph and
condition (2 main-treatment refits;4 confirmation refits if triggered).
No further seeds, offsets, lengths, alphabets, topology changes or plasticity.

## Budget, stopping and limitations

Each worker <=1800s /3GiB sampled process-tree RSS (.2s sampling), one numerical
thread. Preserve partial artifacts/logs on failure; never replace bad seeds.
Complete manifest only after all planned rows pass integrity checks. Save git
revision, source/script/protocol/config/dataset/graph hashes, source checkpoint
references, environment, runtime/RSS and all artifacts in fresh directories.
Cached treatment timing is not comparable to a baseline that regenerates states.

Even a confirmed gain would show an effect of external decoder loss weighting
under these conditions, not internal connectome learning or intrinsic capacity.
Larger prefix weight necessarily redistributes loss emphasis; a late cost is
scientifically informative. Preserve the prior anchored-window confirmation
failure: it expanded weighted windows in a200-symbol pi task and is a distinct
intervention. Do not rewrite it or promote a new default from this diagnostic.
