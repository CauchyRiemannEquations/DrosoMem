# Current K4 structural control — preregistered 2026-09-28

## Question and scope

Does the particular real partial-connectome wiring improve held-out past-symbol
decoding over role-/degree-preserving rewiring, under the current rate dynamics?
This is a scoped ACT III comparison, following completed ACT I and ACT II work.
It is not autonomous recall, biological learning, or formal memory capacity.
Phase 3/3b already tested structural controls and did not establish real-wiring
superiority; this study changes the alphabet to K4 and uses current
`mbon_after_kc` dynamics. Independent streams remove the paired-sequence ancestry
overlap in the latest frozen cross-arm transfer study.

## Frozen design

Authoritative configuration: `configs/structural_k4.json`. Five paired main blocks
cross two fixed circuit samples (701,702), real versus role-shuffled: 20 runs.
Each block has separate mapping, training-stream, test-stream and rewiring seeds.
Each circuit/block has one rewired realization; neither circuit samples nor lags
are independent replicates. The two strata are averaged within block first.
Smoke is one separate block, both graphs, circuit701, 200 train/100 test rows;
it is engineering evidence only. No smoke metric enters the main analysis.

Use pinned 686-node threshold5 partial graphs, fixed node order and all48 MBON
observations. Each K4 symbol stimulates51 of512 KC at amplitude0.5, using the first
four patterns from the existing nested16-symbol encoder. Same identities,
patterns, streams and48 features within each graph pair. Input is never supplied
directly to the decoder. Incoming-L1 gain0.9, leak0.6, `mbon_after_kc` schedule.
Reset before each independent iid uniform K4 train/test stream. Generate via
NumPy default_rng(seed).integers(0,4,size), then cast uint8; save exact symbols.
Warmup100 plus2000 train or1000 test rows. Feature[t] is AFTER symbol[t].

Lags: [0,1,2,3,4,5,8,12,16,24,32]. Primary past aggregate: mean accuracy over
[1,2,3,4,5,8]. Lag0 measures current input, not past memory. Every lag uses
the same post-warmup rows. No next-symbol or autonomous-recall endpoint here.

Train-only feature standardization (std floor1e-5), affine ridge alpha1.0 with
unpenalized intercept, one-hot4 targets, summed squared-error objective. Closed
form, no optimizer sweep, epochs, early stopping or prefix weighting. Each head
has48*4+4=196 fitted coefficients;11 independent task heads per graph. Separately
refit every graph. Compare training-frequency majority and a second ridge fit
with target rows circularly shifted by1000 (smoke100). Scores are unbounded
regression outputs, not calibrated probabilities. Save training and test scores,
predictions, coefficients, labels, states and graph matrices.

## Structural intervention and audit

Reuse `role_shuffled` at5 accepted directed double-edge swaps per original edge,
maximum30 attempts per requested swap. Reject self-loops and duplicate edges.
Within source-role/target-role blocks, weights stay attached to the source.
Verify exact nodewise in/out degrees, role-block counts, outgoing signed weight
multisets BEFORE normalization, edge/node counts, and absence of self-loops.
Require all requested swaps; if a seed cannot complete, retain failure and stop
that study without replacement. Never choose graphs by decoding performance.
Record accepted/requested swaps, attempts, total and role-block edge overlap.
Singleton/constrained role blocks can retain many edges. This is a finite swap
chain, not a proved mixed or uniformly sampled graph ensemble.

Normalize each raw graph separately by identical incoming-L1 rule. This preserves
nonzero-row absolute input sum0.9 and signs, but does NOT preserve normalized
outgoing weight multisets or individual edge weights. Raw incoming strengths
can change. Thus the contrast includes redistribution of normalized strengths;
it cannot isolate topology from those weight changes. Record strength differences,
graph components, normalized matrix identity and input/observation identities.

## Hypotheses and immutable gates

H1 (past access), for each graph: primary test accuracy exceeds BOTH measured
controls by>=5 percentage points on average and in>=4/5 main blocks, with mean
R2 versus training-frequency vector>0. H2 (real-wiring benefit): paired real minus
rewired primary accuracy>=3pp on average and positive in>=4/5 main blocks, plus
real H1. H2 failing means no established real-wiring benefit under this design,
not equivalence or proof of no topology effect.

Only if main H2 passes, execute the three preregistered fresh confirmation blocks
with the same conditions and lengths (12 runs). Confirmation requires real H1
with all3 blocks meeting5pp controls, and H2>=3pp mean with all3 positive.
A confirmed advantage requires both cohorts. Always independently refit, replay
all trajectories and regenerate all rewired graphs; exact arrays/identities must
match. Independently verify ridge scores by augmented least-squares (1e-9
absolute/relative tolerance) and predictions exactly. Replay is reproducibility,
not another sample. No optional extra seeds or adjusted gates after outcomes.

## Analysis, diagnostics and stopping

Save all lag and circuit raw results. Show five blockwise paired values, mean,
median, sample variance,95% percentile bootstrap interval (10000 draws, seed38399)
and paired dz. Small-n intervals are descriptive; no p-value decision. Report
per-lag curves and both fixed circuit strata, without selecting best lags.
Do not label accuracy sums or R2 formal memory capacity.

For every graph save active-neuron counts, MBON state norm/std/sparsity,
consecutive-state cosine, centered singular values/effective rank and32 silent
decay steps. If H2 fails, analyze these representations without retuning or
adding experimental conditions. Diagnostics are associations, not mediation.

Before smoke, replay one existing independent-stream partial baseline from saved
symbols and exactly refit its saved ridge head. Check pinned data and historical
artifact checksums. Tests include lag alignment and shuffle preservation failures.
Record config/source/data hashes, commit, UTC timestamp, package/BLAS/hardware,
runtime and sampled peak RSS. Save every seed checkpoint and manifest.

Budget: single numerical thread, at most1800 seconds and3GiB sampled RSS per
cohort including its replay. Check between runs and abort on exceeded budget,
failed integrity, failed finite check or incomplete swaps; preserve partial files.
Existing successful cohorts are immutable. Smoke and main use new destinations.
Protocol/config are committed before any new structural outcomes are generated;
implementation is committed before smoke. No hyperparameter search.

## Claims and limitations

This evaluates representation plus external decoding in a computational model
using real Drosophila connectome structure. Recurrent weights remain frozen.
No claim that a fly remembers symbols, learns pi, or that this readout probe
proves autonomous memory, whole-brain benefit, physiological timing or internal
plasticity. Two fixed partial samples, five topology draws per sample, one graph
null, constrained mixing and normalization confounding limit generalization.
