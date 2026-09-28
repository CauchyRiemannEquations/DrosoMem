# Fresh-seed confirmation of fixed moment-aligned transfer

Registered before generating fresh streams/graphs/states/scores. Follow-up to
moment-alignment-protocol, not a revision of either failed residual diagnostic.
Question: does the label-free alignment improvement replicate on new paired seeds?

## Cohort fixed in advance

Three new mapping seeds55142–55144; train56142–56144; test57142–57144;
rewire58142–58144; perturb59142–59144. Seed audit scans tracked config/result JSON
seed fields before registration, with no collisions. Reuse biological circuit
strata701/702; this is new pseudorandom realizations, NOT new biological brains.
Two conditions R=renormalized D(B)B and F=fixed_original D(A)B. Same raw shuffled
B, input mapping, neuron IDs, streams and observations within each pair.
12 fresh condition runs,12 directional transfers; no seed replacement or expansion.
Do not pool these3 blocks with previous5+3 observed blocks.

## Fixed procedure and baselines

Reuse normalization_control.execute without numerical edits. K4 iid symbols,
warmup100,train2000,test1000,48MBON,threshold5,686neurons,c7013309edges/c7023241edges.
Role/degree-preserving raw shuffle5swaps/edge, incoming-L1 gain.9,leak.6,
schedule mbon_after_kc,input_fraction.1,input_amplitude.5. Ridge alpha1 summed
loss with intercept, source standardization floor1e−5; no parameter tuning.
11heads for lags0,1,2,3,4,5,8,12,16,24,32; primary1,2,3,4,5,8.
196 supervised parameters per four-output head. Null heads use half-train cyclic
label shift. Train each condition's real/null heads once, then freeze. The own-
condition head is also the target-refit comparator for the opposite direction.
Thus12 new condition fits (132real+132null task heads), but ZERO additional
target-supervised fits within either transfer pipeline.

For both directions evaluate (1) original source-standardized unaligned transfer,
(2) target TRAIN-only per-feature mean/std alignment with unchanged source head,
(3) target-condition refit. Adapter formula is identical to moment_alignment:
z=(target_test−mean(target_train))/max(std(target_train,ddof0),1e−5).
Target labels, head parameters and test statistics do not enter fitting adapter.
Majority and identically aligned source null controls remain mandatory.
No autonomous rollout or Pi Memory Score; this measures driven past-symbol decoding.

Before any new-seed run, re-execute archived main c701/s34142 R/F baselines with
exact checkpoint/metric comparison and independent source refits. Re-execute old
smoke c701/s34001 R/F (train200/test100) and both aligned transfers; compare all
archived arrays exactly. Smoke remains excluded from fresh-cohort inference.

## Locked confirmation gates, per direction

Use the SAME three-block confirmation rule as the earlier alignment protocol:
H1 improvement: mean aligned−unaligned accuracy>=5pp and ALL3 block differences>0.
H2 access: mean accuracy exceeds majority AND aligned-null by>=5pp, ALL3 blocks
meet both margins, and mean R² versus source-training-frequency predictor>0.
H3 retention: mean aligned−target-refit>=−5pp, ALL3 blocks within tolerance.
Portable decoding requires H2+H3. Report H1,H2,H3 separately; H1 can succeed
while access/retention fail. No p-value gate or changing these criteria.
Primary lag mean→two-circuit mean→mapping block. n3, separate directions.
Report all raw runs/lag scores/predictions, mean,median,sample variance,
10000 paired-block bootstrap95(seed55399), paired dz for differences.
Intervals are descriptive with n3. Scores are unbounded regressions, not probabilities.

## Stopping, diagnostics and provenance

Execute all12 fresh conditions and both directions regardless outcome. Stop on
integrity/nonfinite failure,1800seconds or3GiB sampled process RSS; preserve partials,
report no completion, never substitute a seed. Prior results and source hashes stay
unchanged. Record git/config/code/environment, runtime, graph and dataset identity.
Preserve normalization_control neural/decay/perturbation diagnostics but do not use
them to tune or revise the primary decision. No new robustness sweep.
Exact full trajectory/head replay for every condition; exact saved transfer replay;
independent augmented solver checks; artifact verifier reconstructs symbol labels,
normalization, scores, paired metrics, bootstrap and gates. Baseline/smoke failures
block main execution. Save source and target provenance for each direction.

The fresh cohort tests replication of a computational-model decoding effect.
It does not establish actual-topology/whole-brain advantage, biological learning,
formal memory capacity, autonomous recall gains, or new independent biological data.
The prior failures of low-mode dominance and preferential orientation remain closed
within their scope. Do not re-run those hypotheses on this cohort to seek success.
