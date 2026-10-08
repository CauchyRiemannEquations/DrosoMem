# M2: joint current/history decoding tradeoff — prospective protocol

Date: 2026-10-08. Baseline `e52d1ff959d4426ce61315ad17788aa7c66fcb88`.
[Fixed config](../configs/temporal_tradeoff.json). Register and commit before any
M2 neural outcome. New namespace `results/tdc_tradeoff_v1/`; reject existing run
paths and preserve every failed/interrupted attempt.

## Question and relation to earlier work

With the established model and readout held fixed, does enabling direct carry
without previous-state synaptic drive improve short-lag historical decoding
while reducing current-symbol decoding relative to an instantaneous reference?

This question, its directional hypothesis and lag1–5 focus are explicitly
informed by M1 observations. It is a prospective test on fresh fixed M2 streams
and mappings, not a blind replication. M1 remains **assay-invalid** with its90%
all-arm eligibility requirement and undefined material-synaptic-increment
endpoint. No M1 threshold, data, decoder or interpretation is changed.
M2 does not confirm M1's original full-versus-carry question.

Here a tradeoff means opposite changes in two separately fitted linear decoding
tasks. It does not mean a conserved information budget, mutual inability to
decode both labels, formal memory capacity, optimal allocation or biological
resource competition. No joint multitask decoder is introduced.

## Frozen dynamics, graphs, input and readout

Reuse the validated M1 four arms without changing their implementation:
full, carry_only, synaptic_only, instantaneous. Drive b=.6, carry a=.4 where
enabled; incoming-L1 gain.9, encoder amplitude.5/fraction.1, `mbon_after_kc`.
Keep the identical normalized W and current KC-to-MBON block in every arm.
Full uses the original `TimedReservoir`; the other arms use the frozen M1 factor
update. Turning carry off does not change the input/drive multiplier.
Previous-state synaptic transmission includes delayed feedforward paths as
well as anatomical feedback cycles, and is not a feedback-cycle isolation.

Primary: source-derived legacy5 circuit701 (686 neurons/3,309 edges), all six
fresh blocks1200001–1200006. Secondary: brain5 (138,639 neurons/2,700,513 edges),
first three blocks in numeric order. Whole scope is based on measured M1 costs,
not new outcomes. No null ensemble, edge sweep, carry/gain scan or P3.

Fixed mapping seeds1210001–1210006, independent train seeds1220001–1220006,
independent test seeds1230001–1230006. These do not overlap M1 namespaces.
Uniform iid K10; reset train/test states separately; warmup200; train4,000 /
test2,000 scored rows per lag. All arms/levels in a block share input source IDs,
48 observed MBON root IDs, stream bytes and labels. Targets never affect states.

Decode `state(t) -> symbol(t-lag)` after incorporating input t, lag0–20.
Labels stay within each independent split after warmup. Lag0 is current access;
lag1–20 is the full descriptive TDC. Fixed affine alpha1 ridge, train-only means
and SD floor1e-5, 490 parameters per lag; exactly the established decoder.
Chance10%, train-only frequency and current-symbol-only lookups are reported.
No sequence prediction, autonomous recall or post-test refitting/tuning.

## Primary endpoint: both dimensions together

For arm A and block s let C(A,s) be lag0 test accuracy, H(A,s) the unweighted
mean `(accuracy-.1)/.9` over lag1–5, without clipping. Define

`dH(s)=H(carry_only,s)-H(instantaneous,s)`;

`dC(s)=C(carry_only,s)-C(instantaneous,s)`.

Primary PASS, on a valid assay, requires **all**:

1. Mean dH over all six primary blocks >=.03 (2.7pp raw mean historical gain).
2. Mean dC over those blocks <=-.05 (at least5pp raw current-accuracy loss).
3. Every one of the six paired blocks has dH>0 **and** dC<0.

Use integer-correct-count rational arithmetic; exact equality at the mean
thresholds passes, zero paired direction does not. These modest material scales
are chosen before new outcomes using M1 observations and the prior .03 adjusted
effect scale. They are not claimed universal scientific significance thresholds.
Report each component, both exact fractions and all paired values even if the
conjunction fails. Failure of either mean or any paired direction means FAIL.
No more seeds, changed lag weights, different decoder or lowered threshold.

Secondary brain5 uses the same joint rule on its first three fixed blocks,
reported separately and unable to rescue primary. Full and synaptic_only's
two-axis values, all pairwise descriptive contrasts and mean lag1–20 scores
are descriptive only; do not promote them into substitute endpoints.

## Assay validity and reference checks

Technical validation requires source/graph/input/observation pairing, split-safe
labels, exact predictions, independent trajectories/refits/metrics and rational
criterion replay. Analytical current-symbol prototypes must match each arm's
zero-state response. Instantaneous must match its current-symbol prototype at
every coordinate and time regardless of history; this certifies no history.

The instantaneous reference must have lag0 accuracy >=.99 in every relevant
cohort block. Its entire state is a fixed current-symbol mapping, so require
near-exact access through the frozen readout. Carry_only lag0 accuracy is an
**outcome**, not an eligibility requirement in this distinct question. Full and
synaptic_only lag0 are also reported without gating the primary contrast.
This is not a relaxation of M1's all-arm90% rule; that study remains closed.

If reference validity fails, outcome is `assay-invalid` and scientific endpoint
undefined. If technical verification fails, keep `validation-failed` status and
do not claim scientific confirmation. The original result remains immutable;
independent verification certifies it separately. Finite-sample historical
accuracy of the no-history reference is descriptive, with no empirical chance
gate: exact history independence is the reference certificate.

## Prespecified descriptive diagnostics and reporting

For every arm/block, save current-symbol confusion counts and lag0 accuracy
separately for trials with `symbol(t)==symbol(t-1)` (repeat) and trials where
they differ (switch). Among lag0 errors, count predictions equal to the actual
previous symbol. If there are no errors, report that conditional fraction as
null, not zero. These probes use existing held-out predictions and require no
new fit. They are descriptive associations, not unique mechanisms or gates.

Save raw correct counts, labels, scores, predictions, fitted coefficients and
training moments; both complete streams/features/final states; full-state hashes;
source/model/data/graph/input hashes and manifests. Report mean/median/sample SD
and paired10,000-draw descriptive bootstrap2.5–97.5% intervals, seed1240001.
Resample entire blocks together across arms, lags and both dimensions. Do not
interpret computational blocks as animal replicates. Curves are not forced
monotone and are not formal Shannon/reservoir capacity measurements.

Main36 cases,72 streams,756 lag heads; smoke first block/both graphs/all arms,
train300/test200, warmup200, eight cases/16 streams/168 heads, no scientific
endpoint. Smoke checks apply and must independently pass before main.
Four workers, one numerical thread each, max7,200 seconds/4GiB aggregate tree
RSS per stage; identical M1 cost-based budget. Record parent wall time/lifetime
working set and50ms sampled process-tree RSS, which can count shared pages twice.
Use coordinate-first indexing to avoid the archived M1 verification allocation.
Retain exception/interruption records; retries keep identical scientific settings
and use fresh paths. Arrays atol/rtol1e-9, scalar comparisons1e-12, exact
predictions/count gates; no post-outcome tolerance relaxation.

## Execution and closure

Audit/seal all42,717 baseline result blobs and historical scientific source,
data/config/protocols; commit this protocol/config before new neural outputs.
Implement a new wrapper around the frozen experiment and a separate verifier
around frozen independently implemented graph/trajectory/SVD-fit helpers.
Main summary/diagnostic/criterion code is independent of verifier summary code.
Meaningful safeguard tests, smoke, independent smoke validation, main,
independent main validation, report/synthesis, overview/roadmap updates, final
artifact/recorded-source/history audit, commit and push verified work in order.

Any PASS supports only the specified opposite changes in readout access under
these fixed computational interventions. It establishes no storage site,
feedback-cycle role, real-wiring/whole-brain superiority, physiological dopamine,
biological learning, simultaneous decoding limit or formal capacity. Every prior
negative and M1's assay-invalid result remains unchanged. No further study is
automatically part of this finite protocol.
