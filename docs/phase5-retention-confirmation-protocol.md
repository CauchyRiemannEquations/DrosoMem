# Locked anchored-weighting confirmation — 2026-09-27

Editorial update: research terminology only; the registered numerical design is
unchanged. Original protocol bytes are preserved in Git history; see
[historical reproduction](historical-reproduction.md).

Base: 96f0ccdb1a5c39722bd5a06833d51cbf741d8151. Written and committed before outcomes.
This completes the open confirmation step of the Phase 5 readout investigation;
it does not complete Phase 5B, spiking dynamics or Phase 6.

## Fixed design

Keep the discovery algorithm and every training hyperparameter unchanged.
Compare fixed first-32 weighting, ordinary 32→64→128 weighting, and preservation
of the first-32 normalized loss share 128/295. Each gets 6,000 Adam updates with
stage heads at 2,000/4,000/6,000, continuous optimizer state, 482 trainable readout
parameters, learning rate .03 and L2=1e-5. Final 6,000-update heads are primary;
no best-stage, best-seed or best-segment selection.

Use the existing s701/s702 circuits, real/role-degree-shuffled graphs, incoming-L1,
mbon_after_kc, new model seeds **4142/4143/4144**, and initializations 0/1/2.
Train each 200-digit segment at offsets **0/1000/2000** separately. The prompt is
the first three digits of that segment and recall horizon is 197. The nonzero
offsets were used in earlier weighting studies; they are new to this anchored
intervention, not untouched test data. New seeds are not new biological animals.

36 state conditions × three treatments × three initializations = **324 fits**,
**972 stage-head evaluations**. Match graph, input encoding, initial head and
normalization statistics within each comparison. Use one numerical thread.

## Decisions fixed before results

Apply the original discovery criteria separately at every offset:

1. Anchored mean final free recall exceeds both controls, and at least 4/6
   initialization-averaged real-graph conditions beat each control.
2. Anchored loses no more initially complete 32-digit models than fixed weighting,
   and completes at least as many first-32 prefixes at the final stage.
3. Stronger joint success additionally requires later-165 teacher-forced accuracy
   at least that of fixed weighting.

Cross-segment primary confirmation requires criteria 1 and 2 at **all three**
offsets. Joint confirmation requires all three criteria at all offsets. A failed
offset cannot be hidden by pooled means or another successful offset. Report
each criterion, both controls, all graph types, stage-wise prefix retention,
32/64/128/197 completions, position bands, overall accuracy and cross entropy.
These are descriptive prespecified decisions, not significance tests.

A primary pass supports the recall/retention effect within these small circuits;
report any later-accuracy cost. A failure leaves fixed-32 as the validated default.
Do not change previously archived artifacts in this research run regardless of outcome.

## Verification

Use the existing atomic per-condition checkpoint engine and strict context checks.
Stop after the first actual condition, then resume in another process. Replay all
972 stage heads and full autoregressive strings, verify frozen connectivity and
all metrics. Independently refit all three arms × three initializations on the
first real circuit/first seed at every offset: **27 fits / 81 stage heads** and
histories must match exactly. Compare pi generators through digit 2200. Add tests
for all-offset decisions, fresh-seed validation and cross-offset resume/replay.
Run the full test suite before the main study. Keep protocol/source unchanged
during the study; retain unsuccessful outcomes.
