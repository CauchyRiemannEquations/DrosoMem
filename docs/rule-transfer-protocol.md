# M11: matched fitted then frozen rule-coefficient continuation

Date: 2026-10-13. This protocol/config is committed before new neural task
trajectories, fitted heads or performance outcomes. Baseline main
`5467811412cb36a0a56f6137e98f93b9e31f8206`.
[Config](../configs/rule_transfer.json), [prior audit](rule-transfer-prior-audit-m11.md),
[M10 task](rule-transfer-design-results.md).

## One question and finite comparison

Under equal observation/affine-head/training budgets, does the original fixed
source-derived circuit show a material held-out affine-coefficient continuation
advantage over a simple three-slot delay? The secondary question is whether
any such advantage exceeds every member of the full ten-graph matched M8 bank.
Neither endpoint is internal recurrent learning or biological mathematics.

Reuse every authenticated M10 train/instance/parameter array: 275/275/440
episodes, mod 11, prefix 3, horizon 8. Multipliers 2..6 train, 7..10 held out.
Parameter-test rules with multiplier 10 have period 2 and can be copied by a
fixed lag-2 predictor. Therefore the primary uses ONLY a=7,8,9 (330 episodes),
with their individual scores required; all a=10 and full 440 scores are also
reported. This exclusion is prospective, based on M10's known algebraic limit.
No examples are filtered by the fitted model's success.

All 12 models: original partial circuit s701, all ten M8 partial rewires
g1840001..g1840010, and an engineered 3-slot one-hot shift. No new graphs,
whole-brain extension, lag/noise search, nonlinear head or P3 is included.
Use fresh K11 input maps 2710001..2710006; first 3 are discovery, last 3
confirmation. Every model runs EVERY affine/iid/shuffled M10 family for each
map: 216 heads total. Negative families and seen-rule/training scores are
diagnostic, never alternate successful primary families. All task symbols,
formula-oracle/table performance and M8 bank properties are previously known;
these new circuit/interface fits and trajectories have not been measured.

## Fixed interface, dynamics and learned budget

Preserve partial raw signed source weights, root-ID ordering, roles and all
48 observed MBONs. Select the original 512 KC pool in neuron-array order.
PCG64 seeded with each input map selects 51 distinct pool entries for each
integer symbol 0..10, sequentially, with amplitude .5. Exact patterns/IDs are
shared across real and every rewire. No rule metadata, target, equation parameter
or test-specific information is an input. Input maps and weights never learn.

Incoming-L1 gain .9, drive .6, ZERO direct carry, one update per neuron, then
MBON recomputation using the same step's KC and previous other presynaptic
states. Reset to 686 zeros for EVERY episode, train or test; no across-episode
state. The M8 bank preserves current KC→MBON coefficients, degrees/roles and
incoming/outgoing signed raw weight multisets. Its conditional limited mixing
and shared structures remain limitations, not biological null-population sampling.

The shift has the SAME 686 scalar state slots (33 active, rest idle), observes
its first 48 slots and shifts three 11-way one-hot symbol slots in order current,
one back, two back. Current one-hot amplitude 1; no anatomical/source front end,
feedback nonlinearity or training of its transition. This is an engineered
reference with matched scalar/observation/head budgets, different representation
and dynamics. It is canonical and independent of the input-map seed; its six
fits are exact-repeat paired references, not six independently drawn delay models.

One 48→11 affine ridge head per model/family/map: 528 coefficients +11 intercepts
=539 nominal learned scalars, plus 48 train-derived means and 48 SD values.
Alpha 1, train-only SD floor 1e-5, all 2200 unweighted training transitions,
uniform one-hot labels. Nothing else is learned. Fixed learning order is the
M10 episode order and eight next-symbol rows per episode. Once fitted, mean,
scale, coefficients/intercepts become read-only before ALL evaluations.
Scores within absolute 1e-10 of the maximum tie to the lowest integer, declared
before outcomes to stabilize rank-deficient/symmetric numerical comparisons.

## Three evaluation modes and explicit references

Teacher trajectories feed only the ten actual input symbols x0..x9. Fit/evaluate
after inputs x2..x9 against targets x3..x10. First-next needs only the initial
three symbols. Teacher-forced scoring exposes actual past inputs, never future
targets; it does not establish target-free completion.

Autonomous rollout's function accepts model, frozen head, PREFIX and horizon,
and has no target argument or metadata. Reset, feed 3 prefix symbols, then
generate 8 outputs using only the same head and feed each generated output back.
Record 11 full-state updates including final generated-symbol feedback. Score
after generation in a separate function: exact prefix before first error and
all-8 completion. Never replace an incorrect generated symbol with the target.
Evaluate all splits: training is trained-instance recall, instance test is new
starts under seen rules (shared cycles), coefficient test is held-out parameters.

Report EACH of 7 M10 controls as authenticated prior reference, without new
fitting or treating them as matched neural heads. Additionally execute zero-fit
fixed lag-2/lag-3 copies on all families/splits in teacher and autonomous modes;
these know only the available 3-symbol history. The known-form analytic oracle
is privileged with the formula/modulus and is an assay reference, not a learned
or equal-knowledge comparator. Coefficient-test lag-2's known period-2 shortcut
and its a=7..9 score must be separately visible.

## Registered primary and secondary decisions

Technical/task validity requires authenticated parent READY, exact split/labels/
input sharing/reset/no-target-flow, finite states/heads/scores, complete registered
case set and independent exact trajectory/prediction/count replay. If any fail,
the study is incomplete/invalid; preserve the failed attempt and raw outcomes.
Low training or test accuracy is a scientific failure of this learner budget,
not a reason to replace the assay or tune the head after seeing outcomes.

For EACH cohort separately, compute exact rational averages over its 3 input
maps on affine data. Registered primary PASS requires ALL:

- Original train teacher accuracy >=1/2.
- Original first-next accuracy over a=7..9 >=1/4.
- Original autonomous full-completion rate over a=7..9 >=1/10.
- Original full-completion rate >=1/20 separately for EACH multiplier 7,8,9.
- Original minus matched shift full-completion rate over a=7..9 >=1/20 (5pp).

The overall primary passes only if BOTH discovery and confirmation pass all
gates. No best input-map/family/metric selection. The secondary fine-wiring
endpoint requires the primary to pass AND in EACH cohort the real full-completion
rate (a=7..9) is strictly above EVERY one of the ten null means and real minus
their equal-graph mean >=1/20. Ties fail the strict ordering. Report secondary
raw effects even when primary fails; it cannot rescue that failure.

Archive all denominators and exact Fractions; do not decide from rounded plots.
Report per rule (a,b), multiplier, split, family, map and model, plus paired
real−shift and real−null differences, finite mean/median/SD and undefined
standardized effects for zero variance. No p-values or biological confidence
intervals from repeated cycles/maps. Shift repeats and shared exhaustively
enumerated task sets are not independent data/biological cohorts.

## Finite execution, independence and preservation

Before outcomes: source/task/bank authentication and baseline immutable-file
audit, synthetic safeguards, then a finite smoke (first 11 episodes per split,
affine only, first input seed, real/first null/shift). Smoke fits 88 transitions
and has no scientific PASS/FAIL gate. Smoke is excluded from main and is not an
independent confirmation cohort. Do not choose changes using smoke performance.

Main: all 216 full-budget fits; independent replay/refits of ALL 216, with
synthetic and smoke verification first. The independent implementation must not
import the M11 model/trajectory/fitter/scorer/summary. Independently reconstruct
source roles/weights and all partial nulls from pinned M8 swap seeds, K11 input
maps, M10 datasets/labels, episode-reset manual dynamics, and augmented-SVD ridge.
Full state-update byte digests must match; observations are independently rebuilt.
Mean/scales/coefficients/scores within fixed 1e-9 absolute/relative checks,
EVERY teacher/autonomous prediction and all exact metrics/gates must match.
Also re-evaluate predictions with authenticated producer head coefficients to
separate trajectory correctness from independent SVD fitting tolerances.

Use compact observation hashes, full-state trajectory hashes per episode,
all predictions/scores and head arrays; do not archive redundant full neural
trajectories. Four workers, one numeric thread each, 7200 seconds /4 GiB sampled
aggregate process-tree RSS per main or verification stage; maximum 1 GiB output
artifacts per stage. Record per-process OS peak, 50ms tree samples, environment,
source/config/data hashes, every failure and complete checksum coverage. These
are computational limits, not physiology. All output directories must be fresh.
No budget expansion or changed statistical criteria after observed failure.

Only `results/rule_transfer_v1/` holds new evidence. Every old source, result,
manifest, protocol, data file, test and negative remains unchanged. Four current
overview files may receive dated handoffs. Historical replay remains pinned to
recorded source revisions. After independent verification, report acquisition,
held-out predictions, autonomous completion and limitations separately; do not
equate implementation completion with hypothesis success. Commit/push verified
work to main as the user requested, preserving branch protection.
