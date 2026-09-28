# Frozen cross-arm transfer — second authorized ACT II study

Commit before any transfer outcomes. The first blocked probe study (report056df14)
found past access in all four graph/arm conditions, but no confirmed low-minus-high
next-symbol gain. All verification passed. This follow-up tests whether the same
decoders remain useful on the opposite ordering, without refitting to that ordering.

## Fixed data and prediction

Reuse all192 fold checkpoints in results/frozen_state_probe and all64 immutable
source neural trajectories. Main5 blocks and confirmation3 blocks remain separate.
For each graph/seed/stratum and source arm, apply the source fold's fitted affine
head to opposite-arm feature rows at the SAME test indices. Directions low→high
and high→low. Preserve all coefficients, intercepts, means/scales, target class
frequencies, training indices and source training labels exactly. Do not train
on target-arm features or labels. Both arms share input mapping and48 observed
MBON IDs within a graph/seed/stratum; verify this before comparison.

The same frozen real/shifted heads are compared within-source and cross-arm;
training sizes70/61/71 and196 coefficients per task therefore match exactly.
The grid t8..126 and targets at lags0,1,2,3,4,5,8,next remain unchanged. Concatenate
40/40/39 held-out predictions,119 per direction. A shifted-label head remains
the original source-training shift, not a new shift of target labels.

## Primary criteria

H1, per graph/direction: transferred past aggregate (lags1/2/3/4/5/8) exceeds
BOTH source-training-frequency majority and source shifted-head controls by>=0.05
in cohort mean and>=4/5 main blocks, with positive mean R2 versus source frequency.
Support across cohorts requires all3 confirmation blocks to meet both excess
thresholds and mean R2>0. No new cohort or seed search is triggered.

H2, per graph: average both directions and strata within seed; delta is
transferred past accuracy minus WITHIN-SOURCE accuracy of the same frozen head.
Main retention criterion: mean delta>=−0.05 and>=4/5 blocks delta>=−0.05.
Confirmation: mean>=−0.05 and all3 blocks>=−0.05. This is an engineering tolerance
convention, NOT a formal statistical equivalence/noninferiority test. Report
every negative difference even if the tolerance passes. Individual directions,
current-symbol and next-symbol transfer are descriptive; do not select a lag.

## Artifacts and analysis

Save192 transferred fold archives, source head hashes, target trajectory hashes,
all scores/predictions/labels, per-fold accuracy,512 task rows,64 aggregate records,
paired seed tables, per-lag and past/current/next summaries. Real scores are not
probabilities. Preserve training accuracy from the original source fit; target
scores are only evaluated. Store per-task accuracy and count on positions where
the target symbol differs between the two sequences, plus identical-target
positions and source-target current-symbol equality fraction. These are
descriptive checks of the paired sequences' overlap, not new endpoints.

Bootstrap10,000 paired seed-block draws,RNG33399. Report means,medians,variances,
intervals,paired mean/SD,wins/ties/losses with only5/3 independent seed blocks.
No pooling of directions/strata as independent samples. No neural/autonomous
rollout or new learned target-arm decoder. Do not optimize transfer performance.

## Verification, stopping and limits

Verify every source manifest and require exact unchanged coefficient arrays.
Recompute scores from frozen coefficients, then refit on SOURCE rows only as a
verification (not an adaptation). Independent augmented least squares must agree
at1e-9 and yield identical class predictions. Test target-data isolation and
matched training budget on synthetic sequences.15min/1GiB,one numerical thread;
stop on a failed numerical/hash/resource check, preserving partial artifacts.
Record git/config/protocol/code hashes,environment,timestamp,runtime and sampled
peak RSS. Preserve all previous results and protocols.

The paired sequences share a multiset, initial shuffled ancestor and construction
procedure. Successful transfer does not prove arbitrary-sequence generalization,
formal memory capacity, biological circuits or internally learned connectivity.
It tests portable linear access in these frozen computational models using actual
Drosophila connectome structure. Negative next prediction cannot imply absent
past information. After reporting this second study, select one next experiment
but do not silently expand into an unbounded series or ACT IV plasticity.
