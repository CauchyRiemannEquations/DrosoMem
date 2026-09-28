# ACT III frozen-head normalization transfer — preregistered 2026-09-28

## Question and known evidence

Can a head fitted in one normalization condition decode past symbols in the
other condition without adapting its coefficients, intercept or standardization?
The prior normalization study (3ba4ee9) observed fixed-factor gains of1.245pp
main/0.894pp confirmation, failing its1pp confirmation criterion. That result
is unchanged. This follow-up evaluates decoder portability, not better gain
settings, new reservoir trajectories, autonomous recall or internal learning.

Existing within-condition results are known. Cross-condition predictions have
not yet been generated. The archived main and confirmation cohorts are both
reused in full, regardless of the new main outcome. They are not newly collected
confirmation data. Distinguish prior condition-specific refits from frozen
transfer; do not combine their scores or call transfer a new independent fit.

## Frozen design and sources

Config: `configs/normalization_transfer.json`. Use the committed
normalization_control_smoke/main/confirmation checkpoints. Only the two rewired
arms are transferred: R=renormalized (D(B)B), F=fixed_original (D(A)B).
Both directions R→F and F→R are mandatory. The real arm remains prior context.

Smoke: mapping34001,c701,200 train/100 test, two directions, excluded from
inference. Main: mappings34142–34146,c701/702,20 directions. Confirmation:
mappings41142–41144,c701/702,12 directions. Both cohorts have2000 training and
1000 held-out rows after100 warmup, K4 iid streams. Exact streams, seed identities,
raw graph, input mappings, observation IDs and target alignment must match within
each source/target pair. Train and test are independent within each checkpoint;
the two normalization conditions share those same train/test streams.

No graph simulation, sequence generation, new target fitting, alignment,
calibration, restandardization, bias correction, feature selection or tuning.
Apply SOURCE train-only mean/std (floor1e-5), source48×44 weights and44 biases,
equivalent to11 independent affine4-output heads with196 coefficients each.
Apply its frozen shifted-target null head too. Argmax selects the class; affine
scores are not probabilities. Only target held-out48-MBON states enter prediction.
Target labels are used for scoring, never for coefficients or transformations.

Lags0,1,2,3,4,5,8,12,16,24,32; primary is mean accuracy over1,2,3,4,5,8.
The input[t] has already entered feature[t]. Lag0 is current access, not past
memory. Every lag uses all1000 held-out rows (smoke100).

## Controls, metrics and immutable gates

Save per-lag transfer accuracy, training-frequency majority and transferred
shifted-null accuracy, both excesses, R2 versus source training-frequency vector
(keep negatives), test MSE, source-within accuracy, target-refit accuracy, and
both transfer-minus-source-within and transfer-minus-target-refit differences.
Save all source/target reference scores, transfer/null scores, predictions,
target labels, frozen head/scaling and source artifact hashes.

H1 past access, separately for EACH direction and cohort: mean primary accuracy
exceeds BOTH measured controls by>=5pp, and>=4/5 main blocks (all3 confirmation)
each meet both5pp margins; mean R2 versus frequency must be>0.

H2 retention, separately for EACH direction and cohort: primary transfer minus
TARGET's existing condition-specific refit accuracy has mean>=−5pp and>=4/5
main blocks (all3 confirmation) are>=−5pp, AND H1 passes. The5pp tolerance is
descriptive, not a formal equivalence/noninferiority test. Source-within losses
are secondary and cannot replace the target-refit comparison after outcomes.

Supported access or retention requires the corresponding criterion in BOTH
archived cohorts. Overall portable decoding requires BOTH directions to pass
retention in both cohorts. Do not average directions to hide a failing direction,
pool cohorts, select successful lags, add seeds, or change thresholds.
Failure means frozen portability is not established by this test; it does not
mean past information is absent (the condition-specific refits are controls).

## Diagnostics and statistical unit

Two circuit strata are averaged within each mapping seed before statistics.
Directions and lags are repeated conditions, not new samples. Report all raw
stratum/lag values and seed-block values, mean, median, sample variance,
95% percentile bootstrap (10000 draws, seed48399), and paired dz for differences
only. With n5/3, intervals are descriptive; no p-value gate.

Descriptive train-feature diagnostics: target mean displacement in source std
units, target/source std ratio (using the1e-5 floor), same-time state cosine,
and centered paired-feature correlation. Do not apply these statistics to the
predictions; no target-distribution alignment is permitted in this study.
These may reveal distribution changes but cannot alone establish causal mediation.
No new representation-capacity or biological inference follows.

## Reproduction, independence checks and stopping

Before transfer, validate source/target manifests and reproduce each source's
within-condition scores exactly from saved coefficients. Source-only ridge
verification refits must reproduce parameters, and independent augmented least
squares must reproduce transferred scores to1e-9 absolute/relative tolerance
with identical predicted classes. These are verification-only source fits;
there are ZERO new target-trained heads.

Tests: identity transfer reproduces within scores; modifying target labels
cannot change predictions; unrelated target-head coefficients are inaccessible
to the prediction function; all source head/stat arrays remain unchanged.
Pairing checks include raw graph hash, stream/label identity, node IDs, input
patterns, feature shapes and exact source/target time alignment.
Reapply each saved head and exactly replay every transferred array. Verify all
saved metrics, paired aggregation and gates independently from saved arrays.

Commit this protocol/config before new predictions and implementation before
smoke. Save a separate smoke subdirectory, all32 main/confirmation directions,
manifests, config/source hashes, Git commit, timestamps, environment, runtime
and sampled peak process RSS. Preserve historical results and protocol/source
bytes. Use a fresh destination and keep partial outputs on failure.
One numerical thread; whole study including verification<=1800 seconds and3GiB
sampled RSS, checked between directions. Stop on budget/integrity/nonfinite
failure, without replacing seeds or outcomes. Complete both registered cohorts
otherwise. No follow-up rescue condition in this run.

## Claims boundary

This is a computational model using actual Drosophila connectome structure.
Frozen recurrent connectivity and trained external heads concern representation
and decoding, not experience-dependent recurrent learning. Transfer is between
normalizations of the SAME raw graph on the SAME symbol streams, not between
arbitrary graphs or alphabets. It does not measure autonomous sequence recall,
formal reservoir/Shannon capacity, biological memory or whole-brain superiority.
