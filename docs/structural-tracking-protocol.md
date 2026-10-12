# M9: graph-specific fitted then frozen historical-value tracking — protocol

Date: 2026-10-12. Commit before new M9 head/trajectory/decoding outcomes.
[Prior audit](structural-tracking-prior-audit.md),
[fixed config](../configs/structural_tracking.json).
Baseline `b2ef8412832ce64e208287fe063abe8068b5df5c`;
namespace `results/structural_tracking_v1/`. No old result is overwritten.

## Question and exact primary criterion

Does source-derived delayed wiring yield materially better **unconditioned
paired past-symbol tracking** than the entire M8 matched rewiring bank after
equal-budget graph-specific readout training? The engineered delay is a
separate reference, not an anatomically matched null or the primary comparator.

For each graph/block, joint=number of64 anchors with BOTH original prediction
equal s AND replacement prediction equal r=(s+1)%10. All64 count, including
original errors. Graph score is the equal-block mean of joint/64; partial six
blocks, whole firstthree. No autonomous recall, current-only target, conditioning
on clean success or fitted test-time adaptation enters this metric.

First require ALL real AND null lag0 test-current cells in that graph level
to have exact `10*correct>=9*test_samples`. If any fail, that level's comparative
endpoint is **assay-invalid** (pass=null); no bad graph/block is discarded.
Report every raw score anyway. If valid, advantage PASS requires real score
STRICTLY ABOVE EVERY fixed null graph score AND real−null mean>=.03 RAW joint
fraction (3pp). Otherwise FAIL. Use integer-count Fractions for graph means,
ordering and3/100 comparison; score tolerance never enlarges this criterion.
Primary legacy5, secondary brain5 cannot rescue a failed/invalid primary.

Report the fixed90%-in-every-block tracking diagnostic for real and each null
separately; it does not replace comparative advantage. For the delay reference,
current>=90% in every block is its own validity rule and joint>=90% in every
block its separate benchmark criterion. It cannot rescue or invalidate the
real/null primary. Real−delay differences are descriptive because dynamics/
anatomical matching differ. Near-perfect/equal scores are a reportable ceiling;
no lag/noise/alphabet/head/threshold sweep follows a FAIL or assay-invalid result.

## Frozen graph bank, model and exactly learned parameters

Reuse ALL M8 members: partial1840001–1840010; whole1841001–1841003. No graph
generation/search or extra nulls. Parent main manifest SHA256
`528d420d16b55ccc99d8c180fdd07886cab1235d14b6ce1043bb381f2a0780eb`,
validation manifest `abfd3bfeb500b9a643afd268042c1551023af6bc6eb1507cbfad2a57a99ec272`.
Authenticate parent main/validation/config/source/graph audit/raw matrices and
all source/model/input/prototype/layout hashes before using them. Degrees,
role counts, both raw weight multisets and current raw/normalized coefficients
match; normalized outgoing multisets/delayed paths are not constrained.
This finite restricted bank is not uniformly sampled graph-space or animals.

Real/null: source circuit701, incoming-L1 gain.9, zero DIRECT carry,
synaptic history on, drive.6, mbon_after_kc. Same512 source KC pool/51 active
KC per K10 symbol at amplitude.5 and48 MBON observations. Normalize each raw
graph by the same row rule; exact M8 factors/protected coefficients authenticate.
Reuse validated FactorReservoir equations, never full carry-on baseline.

Delay: exact M8 padded686/138639-state layout,608 active coordinates and unit
shift D; source instantaneous code templates split by ascending root IDs into
current24/past24. Each step supplies512 instantaneous KC values and current24/
bank0 codes; prior bank0→bank1→bank2 gives lag2. Native observation order is
restored. Inheritance of the source front end and different roles/weights/
dynamics are explicit limitations. For byte-stable reproduction, M8 float64
templates are FIXED model data: independently recompute their raw-source
formula within1e-9, authenticate parent bytes, then use exact archived templates
for official delay state replay. Do not use new main state arrays as inputs.

Learn ONLY external affine alpha1 ridge heads from4000 scored training states:
train-only48-coordinate mean/SD floor1e-5. Simultaneously fit two separable
ten-class targets ordered lag0 then lag2. Each lag has48*10+10=490 scalar
coefficients/intercepts; case total980, identical in ALL conditions. Scaler
training is matched and never uses test/replacement/reference states. Freeze
scaler/heads before held-out test. No recurrent or encoder/template learning.
Main87 cases contain174 lag heads (85260 nominal fitted affine scalars).
Independent SVD refits check those learned heads, not biological replication
or a second newly selected training bank. M6/M7 OLD heads remain untouched.

## New independent data and actual input intervention

Blocks1900001–1900006; reuse M8 mapping1810001–1810006. Fresh train1910001–1910006,
test1920001–1920006, uniform iid K10 PCG64; independent train/test reset zero,
warmup200 then train4000/test2000. Within every level/graph the same block
streams/map/source input IDs are paired. Whole firstthree pair with partial,
not three extra independent input replicates. Test labels stay in their own
stream: after incorporating t, read state(t)→symbol(t) or symbol(t−2).

64 anchors `warmup+floor(linspace(0,test_samples-1-2,64))`. Copy the uninterrupted
test state at a−1; replay a,a+1,a+2 with original s and actual r=(s+1)%10 at a
only. Prefix/future/current symbol at a+2 are identical. Evaluate lag2 frozen
heads on both worlds. Record full pre/window/final/baseline digests and compact
48-coordinate windows. Original replay must match the uninterrupted trace
exactly. Every graph/block has the same targets/times, with no dropped anchor.

Instantaneous references disable synaptic history/direct carry (real/null)
or reset prior delay banks each step (delay), retaining current input/front end
and the SAME case lag2 head/scaler. Final full states/scores/predictions must
be equal across their worlds. Their paired joint0 and unchanged frequency/
current-only joint0 follow mathematically because s!=r; they are not a10%
joint-chance estimate. Uniform10% is only a single-arm guess. The reference
head is off distribution; its marginal accuracy has no validity gate.

Training-only class-frequency and current-symbol lookup controls accompany
lag0/lag2 full-stream scores; do not force a meaningless iid Markov baseline.
Report paired original/replacement/joint values, prediction transitions and
affine contrast-margin projection, plus stream accuracies/baselines. No new
noise/clipping/feedback, pi sequence prediction or biological plasticity.

## Summary, validation and resource rules

Report each graph score and paired block values; null mean/median/sample SD/
min/max; empirical percentile100*(below+.5*tied)/N; real−null mean and
standardized effect (undefined if SD0); real−delay block differences. Describe
block mean/median/SD and10000-draw bootstrap seed1930001 for paired differences.
These are finite computational descriptions, not biological population tests,
equivalence proof, formal capacity or all-wiring conclusions.

Main72 partial cases (real+10null+delay ×6),15 whole (real+3null+delay ×3),87 total.
174 train/test full-state traces,5568 unconditioned probe pairs,22272 four-world
window/final digest entries. Source-neural cases78 (156 train/test trajectories),
engineered-delay cases9 (18 trajectories). Four workers/one numeric thread;
7200s/4GiB aggregate process-tree envelope per main/verification stage, based
on measured P2/M6/M8 costs. Save per-case runtime/OS peaks and50ms aggregate
RSS (shared pages can be counted repeatedly). Do not adapt budgets/conditions
to new performance. A resource/interruption/validation failure preserves its
files and is incomplete; same-settings technical retries use new directories.

Order: repo/parent audit and baseline seal; protocol commit; implementation
freeze; synthetic safeguards; smoke with first block and real/first-null/delay
at each level, train300/test200; independent smoke; fixed87-case main;
independent main; report/synthesis and four overview updates; final audit/push.
Smoke endpoint is null (stage label smoke), excluded from all main statistics.
All stages use fresh exist_ok=False directories and stable clean HEAD with
Git-authenticated source/config/protocol/data/parent hashes rechecked at end.

Independent verifier never imports main M9 reservoir/step/trajectory/fitter/
scorer/labels/control/summary algorithms. Reuse only previously independent
raw loaders/normalization-source checks and generic IO/resource/hash helpers.
Reconstruct graphs/mapping/delay layout, all174 full-state traces and5568
prestates/22272 windows/endpoints, ridge/scaler via augmented SVD, labels,
predictions, controls, paired metrics, every summary and exact validity/advantage
decision. Require exact state hashes/integer predictions; numeric coefficient/
score checks atol/rtol1e-9 do not affect scientific integer thresholds.

P0/P1 past-input evidence, P2/M7 FAIL, M1 invalid, M4 INFEASIBLE and all older
failed runs/provenance are preserved. Both comparative directions and invalid
or ceiling outcomes are valuable. No storage-site/physiological learning/real
fly memory/whole-brain superiority/capacity claim; no P3 or rule-transfer work.
