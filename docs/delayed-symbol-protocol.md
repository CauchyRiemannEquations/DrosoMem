# ACT II diagnostic — independently tested delayed-symbol decoding

This protocol must be committed before smoke or scientific outcomes. It follows
the confirmed N512 decoder tradeoff; it is not ACT III ablation or ACT IV learning.
ACT II alphabet-size experiments remain open. Do not change this protocol after
outcomes; retain failed runs and use fresh output directories.

## Question and hypotheses

Does the whole-brain threshold1 graph preserve more *linearly decodable past
input* at the same48 MBON observations than the legacy threshold5 partial graph?
H1: past inputs are decodable above frequency and temporally mismatched controls.
H2: brain1 improves the predefined delayed-accuracy aggregate over legacy5.
Failure of H2 is a valid result; neither hypothesis concerns biological memory,
internal synaptic learning, unseen pi prediction or formal total memory capacity.

## Conditions and fixed interface

Use the existing verified `whole_brain_memory.build_model` graph loader with
legacy5/brain1, strata701/702. Same source root IDs, KC-only digit patterns,
48 observed MBONs, incoming-L1 gain0.9, leak0.6, `mbon_after_kc`, input
fraction0.1 and amplitude0.5 as ACT I/II. Thresholds and graph sizes are logged.
Only graph condition differs within a seed/stratum pair. No graph or input tuning.

Independently seeded uniform iid decimal streams: 100 warmup symbols plus2,000
train /1,000 test symbols. State resets to zero separately for each stream;
there is no prompt. At t, collect state **after** input u[t], predict u[t-k].
Lags `[0,1,2,3,4,5,8,12,16,24,32]`. Warmup covers all lags; arrays are aligned
before slicing. Separate stream seeds prevent overlapping train/test inputs.
No test labels or test statistics enter fitting or standardization.

For each lag fit the existing affine RidgeDecoder: train-only mean/std,
std floor1e-5, unpenalized intercept, alpha1 on unnormalized summed squared
error. Ten one-hot outputs; 48x10 weights and10 intercepts =490 coefficients
per lag, identical between graphs. All lags are solved as independent outputs
in one matrix solve; outputs never become features for another lag.
This diagnostic head differs from the historical482-parameter nonlinear recall
head; no comparison between their task scores is implied. No optimizer or sweep.

Controls: theoretical chance10%, training-frequency majority predictor, and
an independently fitted affine head using training target rows circularly
shifted by1,000 (half the scientific train rows) while keeping features fixed.
Evaluate the shifted-label head on the original test targets. The circular
control preserves training class frequencies and destroys alignment at tested
short lags; it is a finite null diagnostic, not proof of graph-specific memory.
Smoke uses shift100 with200 training and100 test rows, warmup100.

## Seeds and budget

Main five paired blocks: input-mapping seeds14142–14146, train-stream seeds
15142–15146, test-stream seeds16142–16146, matched by final digits. Each block
has strata701/702 and both graphs:20 runs, 220 lag probes plus220 null probes.
Smoke: input14139/train15139/test16139, stratum701, both graphs:2 excluded runs.
No seed replacement or early stopping based on scores. Worker maximum1,800s
and3GiB sampled process-tree RSS every0.2s; one numerical thread. Stop cohort
on resource/error failure, preserve partial artifacts, report incomplete result.

Fresh confirmation only if H2 passes main: input18142–18144,
train19142–19144, test20142–20144, same two strata/both graphs,12 runs.
Do not trigger additional cohorts from isolated attractive lag/seed results.

## Metrics and predefined decisions

Save every lag's train/test accuracy, frequency accuracy, shifted-label test
accuracy, and `r2_vs_training_frequency = 1-SSE_model/SSE_training_frequency`.
Keep negative values. Ridge scores are not probabilities; do not call them
confidence. Save symbols, features, targets, scores, predictions, coefficients,
train-only scaling, graph/interface identity and source/config hashes.
Save active neuron counts, observed activity and cosine similarity, effective
feature rank/singular values, state decay and runtime/peak RSS/environment.

Primary past-input aggregate: mean test accuracy over lags1,2,3,4,5,8.
Lag0 is current-input access; lags12,16,24,32 are descriptive long-delay checks.
This arithmetic average is not a sum of memory capacities.
Average the two strata within each seed block before estimating uncertainty.

H1 per graph requires aggregate excess over BOTH frequency and shifted-label
controls >=0.05 in cohort mean and >=4/5 blocks, plus mean aggregate R2>0.
H2 requires mean paired brain1-minus-legacy5 aggregate accuracy >=0.03 and
positive differences in >=4/5 blocks. These are effect-size conventions, not
biological thresholds or p-value tests. H2 confirmation requires mean>=0.03
and all3 fresh block differences positive. H1 in confirmation is descriptive.
If H2 fails, no tuning or confirmation; inspect predeclared lag curves, rank,
activity and standardized score diagnostics before choosing one follow-up.

Report all per-run rows, block paired differences, means, medians, sample
variances, paired d_z (null for zero variance), bootstrap95% mean intervals
using10,000 paired-block resamples seed21399. Five/three blocks are small
computational samples of overlapping portions of one source brain.
Per-lag effects are descriptive; do not redefine the primary lag set afterward.

## Validation and stopping rule

Before main, verify old checkpoint integrity and reproduce one registered
N512 baseline in each graph (first block11142/11242,stratum701), including
features/head/predictions. Reuse old delay-alignment and known shift-register /
memoryless tests; add current-runner checkpoint/control isolation checks.
Smoke must cover injection, propagation, both streams, matched observation,
fitting, saving, metric reconstruction and complete independent repeat.
After main, independently regenerate every stream and trajectory, refit every
head and require exact equality of all saved numerical arrays. Recompute
metrics separately from saved predictions/scores; compare to the existing
`decode_delays` reference. No reinterpretation of failed exact checks as success.
Full scientific repeats are validation, not extra independent samples.

No autonomous rollout is run in this diagnostic; the prior verified pipeline
already covers it. All previous results and locked protocols remain unchanged.
Matched random topology, robustness, alphabet scaling and internal plasticity
remain separate studies. No new anatomical memory-circuit claim is supported.
