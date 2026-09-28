# ACT II: random-sequence length scaling

Locked before outcomes, 2026-09-28. Exploratory preregistration-style protocol,
not an external registration. Base revision: 0df1fc7. One question: how does
trained random-sequence autonomous recall change with sequence length under
the existing fixed decoder/update budget, and does brain1 alter that curve?

## Hypotheses and controls

H-length: normalized exact-prefix recall decreases as training length rises.
H-graph: brain1 improves exact-prefix recall over legacy5 across the length grid.
Reuse the ACT II frozen rate model and 48-MBON / 482-parameter external readout.
No internal synaptic learning. legacy5 is the original 686-node partial graph;
brain1 contains 138,639 nodes / 15,091,983 edges (threshold >=1). Two input-map
strata 701/702 are overlapping selections from ONE connectome, not animals.
Keep exact input IDs, amplitudes, observed IDs and order paired across graphs
and lengths within each seed/stratum. Validate source graph hashes.

## Dataset and paired budget

Lengths N = [32,64,128,256,512], K=10, supplied prompt [3,1,4], offset0.
For each dataset seed generate 512 symbols with NumPy PCG64,
SeedSequence([dataset_seed,2201]), uniform integers [0,10), and replace the
first three with the prompt. Each shorter dataset is the exact prefix of that
stream. Verify that the existing length-parameterized generator returns these
same prefixes. The suffix is iid conditional on the fixed prompt. Reuse of
prefixes pairs task difficulty; lengths are not independent replications.

Fresh model/dataset seed pairs: (11142,11242), (11143,11243), (11144,11244),
(11145,11245), (11146,11246). Both graphs and strata at all five lengths:
**100 fits**, no reuse of pilot outcomes as new data. Model and dataset seeds
are paired rather than fully crossed; their variance cannot be separated.
Each fit resets state and readout. No curriculum, warm start or transfer.
Run lengths in the listed order and all registered blocks, irrespective of score.

Smoke: both graphs, stratum701, block(11139,11239), lengths32 and512,
20 updates, four fits; excluded from all main estimates.
Before smoke, exactly replay and independently refit the pilot random family,
first block9142/9242 and stratum701, both graphs (expected33/32).

## Training and scoring rules

Train on N-1 teacher-forced pairs; autonomous horizon H=N-3 excludes prompt.
Teacher forcing supplies the true preceding symbol. Autonomous generation
receives only prompt, state transition, trained head and horizon, never targets.
Every fit uses 2000 full-batch Adam updates, LR.03, beta(.9,.999), epsilon1e-8,
L2 1e-5 on nonbias weights, final checkpoint. Dynamics: incoming-L1 gain.9,
leak.6, mbon_after_kc; KCEncoder fraction.1/amplitude.5. Train-only feature
normalization with std floor1e-5; 48->8 tanh->10 head, initialization namespace
SeedSequence([model_seed,9901,0]). Network weights remain frozen.

Weight the first min(32,H) generated targets by4, all other targets by1,
then normalize by total weight, exactly as in the pilot. Thus N32 has29
weighted targets, N>=64 has32. The two pre-evaluation teacher-forced targets
retain weight1. The weighted prefix mass is 4W/((N-1)+3W), W=min(32,N-3).
Record this mass and the weight sum. Increasing N changes this mass as well
as training examples per update; this is a FIXED-PROCEDURE scaling curve,
not isolated intrinsic storage capacity. Do not silently renormalize prefix
mass or increase updates to rescue longer sequences.

Primary curve: exact_prefix_symbols L and normalized_prefix L/H.
Report recalled-prefix bits L log2(10), full completion L=H, first error,
position/region accuracy, probabilities, teacher-forced and training accuracy,
loss and loss history. Full completion is right-censored at the horizon;
it does not identify maximum possible recall. Non-pi scores are not Pi Memory
Scores, Shannon information capacity or formal reservoir memory capacity.
Weighted-majority and weighted first-order predictors use the SAME training
pairs and weights; ties use smallest symbol, unseen rows use global counts.
Their autonomous prefixes are retained for each dataset/length.

## Analysis, success/failure and stopping

Preserve all 100 raw rows. Average the two strata within each of five paired
blocks before inference. For each length/graph report raw and block mean,
median and sample variance; paired graph differences, mean/median, sample
variance, standardized d_z (null when SD=0), wins/ties/losses, percentile95%
bootstrap intervals from10000 resamples with seed11399. Bootstrap blocks,
preserving all graph/length pairing, not neurons or individual strata.
No p-value gate. Tiny computational cohort; no population biological inference.

For each graph, H-length is supported in this cohort if the mean normalized
prefix drop from32 to512 is >=.25 and >=4/5 blocks drop (strictly positive).
Otherwise mark not established; report nonmonotonic intermediate values.
Operational collapse: at a sampled length mean normalized prefix <.5 AND
>=4/5 block means <.5. Report the first such length and EVERY qualifying
length. If already at32, label left-censored; if none, not observed through512.
If later lengths recover above this criterion, say so; do not infer monotonic
failure or interpolate a sharp physical capacity threshold.

H-graph is exploratory supported only if the per-block mean absolute-symbol
brain1-minus-legacy5 contrast over ALL five lengths averages>=5 with >=4/5
positive blocks AND at least TWO adjacent lengths each have mean gain>=5
and >=4/5 positive blocks. Report all per-length differences regardless.
A single favorable length is not sufficient. No condition selected for tuning.
If this H-graph rule passes, independently confirm the entire grid with fresh
blocks(12142,12242),(12143,12243),(12144,12244), 60 further fits, same rules.
Confirmation requires aggregate gain>=5 with all3 positive blocks and the
same adjacent-length rule with all3 positive at each length. Otherwise the
discovery remains unconfirmed. No confirmation if discovery fails.

Each subprocess <=1800s and <=3GiB sampled process-tree RSS at.2s. Complete
the fixed cohort unless a software, resource or artifact-integrity failure
blocks it; retain failed artifacts and report incompleteness, never replace
seeds or drop conditions. No extension to1024 in this study. Replay every
head exactly and independently refit both graphs / first stratum / first
block at each length (10 main refits, plus10 if confirmation triggered).

## State diagnostics and provenance

Keep per-step observed MBON features, active counts(abs state>1e-8), sparsity,
cosines, feature singular values/effective rank and32-step zero-input decay.
Compute standardized feature rank with training-only scaling. Prefix nesting
implies teacher-forced reservoir states for shared time steps should match
exactly across lengths before normalization; explicitly verify this invariant.
Different fitted scaling/readout parameters are expected across lengths.
These diagnostics describe representation; they do not identify causal circuits.

Save runtime, sampled peak RSS, hardware/packages/BLAS, graph size and source
identity, config/protocol/source/script hashes, git commit, checkpoint paths,
generated sequences and all per-run data. New directories only. Keep previous
results, failed attempts and protocols unchanged. No ablation, alphabet change,
new topology controls, plasticity or robustness sweeps in this study.
