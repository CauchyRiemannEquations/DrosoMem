# Frozen-state blocked decoding — ACT II preregistration

Lock and commit before new probe outcomes. Source cf8cba7: all40 discovery and
24 fresh-confirmation trajectories from the fixed-K intervention. Their recall
outcomes are known; the new probes are exploratory diagnostics on these saved
tasks. Keep the5-block main and3-block confirmation cohorts separate; no pooling,
selection or new neural simulation. Smoke trajectories are excluded.

## Alignment and question

Saved feature row t is the48-MBON state AFTER input symbols[t], t0..126.
Use the common t8..126 grid (119 rows) for all tasks. Targets are symbols[t-lag]
at lags0,1,2,3,4,5,8, and symbols[t+1] for next-symbol decoding. K4 throughout.
Lag0 is current access, past aggregate is the arithmetic mean of six positive
lags, and next is decoding of sequence continuation. Never combine them into
one memory-capacity score or autonomous recall metric.

## Fixed split and decoder

Three contiguous test folds split np.arange(8,127) with np.array_split(...,3):
t8..47,t48..87,t88..126. Train on the other grid rows only when distance from
every test row exceeds9. Thus full input/label windows t-8..t+1 do not overlap
between train and test. Reservoir state has no finite-support guarantee; purge9
does NOT eliminate all temporal dependence. Later training rows are allowed:
this is blocked within-trajectory decoding, not forward prediction or independent
stream generalization. The128-symbol trajectory is not reset or refit.

Use existing RidgeDecoder(alpha=1): train-only mean/std floor1e-5, unpenalized
intercept, unnormalized SSE with ridge penalty. Four one-hot outputs,48x4+4=196
coefficients per task. Solve tasks together only as independent output columns;
no label of another task enters features. No optimizer/search/nonlinear probe.
Training rows are uniformly weighted, distinct from the old recall head's
prefix weighting. All scaling and target frequencies use only train rows.

Controls: train-frequency majority and independently fitted ridge with training
target rows circularly shifted by floor(n_train/2). Test targets remain unshifted.
Chance25% is descriptive. No selection of shift/alpha based on outcomes. Shift
preserves marginal frequencies and is a diagnostic, not a perfect independent null.

## Metrics and gates

Save train/test indices, aligned labels, source feature hash, means/scales,
coefficients/intercepts, real/null scores and predictions, majority predictions,
per-fold train and test accuracy. Each test row occurs once. Aggregate held-out
accuracy by concatenating all119 predictions (not equal-weighting unequal folds).
Save R2=1-SSE_real/SSE_fold_training_frequency, keeping negatives; scores are not
probabilities. Summarize per lag and past/current/next separately.

For each graph/arm, above-control past access is supported in main if mean excess
over BOTH majority and shifted controls>=0.05, and both excesses>=0.05 in>=4/5
seed blocks, with mean past R2>0. Supported across cohorts only if the same
conditions hold in all3 confirmation blocks and positive mean R2. For each graph,
primary low-minus-high next accuracy passes main if mean>=0.05 and>=4/5 strictly
positive blocks; confirmation requires mean>=0.05 and all3 positive blocks.
These are diagnostic criteria, not biological significance. Report failures.

Average strata701/702 within seed before inference. Report all64 trajectories,
per-task512 rows, paired block differences, means/medians/sample variances,
paired mean/SD (undefined if zero SD), wins/ties/losses and percentile bootstrap95
with10,000 draws,RNG32399. Only5/3 seed blocks; no p-value-driven claims.
Past low-high changes and whole-partial differences are descriptive.

## Verification and budget

Reuse unchanged saved trajectories/checkpoints. Verify archived manifests and
reproduce one existing independently-tested delayed-symbol ridge baseline.
64x3x8=1536 real task heads plus1536 shifted heads, implemented as384 multi-output
solves. Exact coefficient replay and deterministic refit of all folds; independent
augmented least-squares reference scores must agree within atol1e-9/rtol1e-9,
and class predictions must match. Synthetic alignment, purged-window isolation,
train-only scaling, known delayed input and shuffled labels tests precede runs.
Budget15min/1GiB process RSS, one numerical thread; retain partial errors and stop
on any failed check. Store input/protocol/config/code hashes,commit,environment,
timestamp,runtime,peak RSS and all prior-result checksums. No new seed replacement.

## Interpretation and authorized continuation

Decodability with this affine head does not identify the internal storage
mechanism; failure does not prove absence of information. Different sequences
retain ordering confounds. Current task is representation/decoding, not internal
learning, autonomous recall, random-topology comparison or formal capacity.
After verification/reporting succeeds, the next single study is cross-arm
transfer: fit on one saved sequence and test the other with the same graph/input
map. Write its own protocol before outcomes. Continue regardless of whether the
scientific gates here pass; engineering failures must be resolved first.
