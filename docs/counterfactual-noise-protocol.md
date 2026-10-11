# M7: frozen-readout past-symbol tracking under artificial observation noise

Date: 2026-10-12. Commit before fresh trajectories/noisy scores.
[Prior scope audit](counterfactual-noise-prior-audit.md),
[config](../configs/counterfactual_noise.json).
Baseline main `191e8edbcb7b90e066c85bfead24f00d02d04c6e`;
new namespace `results/counterfactual_noise_v1/`. No old artifact is overwritten.

## Question and fixed primary rule

Does M6-style lag2 tracking survive ONE fixed nonzero artificial observation
noise dose when the original graph, input encoding, dynamics, M5 readout and
scaler stay frozen? Primary is unconditioned paired transition accuracy:
original noisy prediction==s AND replacement noisy prediction==(s+1)%10.
Require exact `10*joint_correct>=9*64` in EVERY six partial blocks (minimum58).
No numerical tolerance enlarges this integer threshold. Whole firstthree is
separate paired secondary. No performance eligibility gate, successful-clean
filter, dropped anchor, new dose/seed, clipping, adaptation or retraining follows
a FAIL. Smoke's scientific endpoint is null; stage label `smoke` is allowed.

## Model, parent and new inputs

Original source-derived legacy5/brain5 circuit701, incoming-L1 gain.9,
drive.6, carry0, mbon_after_kc, previous-state drive on. Same512 KC input pool,
51 at amplitude.5 per K10 symbol,48 MBON observations. Reuse exact M5
mean/scale, coefficients[:,20:30]/intercept[20:30] (490 affine parameters);
lag0 head is clean sanity only. Parent source `c541993484f6778a031d71ad4eb1798c748f6ce7`,
protocol `37212abeb6cfc1a4907e08348ff41030f3edfb8b`, main manifest
`98eb3b4c3c49934365caa30608fd50b677dd883b7e3ed722271db4aceeaf73af`.
All parent/model/encoder/coordinate/head/control hashes authenticate.
No newly learned parameters. Independent OLD-training SVD refits authenticate
the parent only; official scoring always uses archived bytes.

Blocks1700001–1700006 retain parent1500001–1500006 and mappings1510001–1510006.
Fresh uniform iid streams1710001–1710006, new noise seeds1720001–1720006.
Reset zero, warmup200, test2000;64 anchors
`warmup+floor(linspace(0,test_samples-1-2,64))`.
Copy exact state at a−1, directly change only symbol at a from s to(s+1)%10,
then replay a,a+1,a+2 with both future symbols held fixed. Evaluate a+2.
Instantaneous reference disables prior drive and keeps carry0/current KC→MBON.
All64 anchors count. Original/replacement windows match M6 semantics; this is
an actual categorical input intervention, not a continuous derivative.

## Precisely specified observation-noise allocation

Fix standardized sigma=.10 BEFORE outcomes. From PCG64(noise_seed) generate
64×48 independent standard normals in ascending numeric observed root-ID order;
restore to each archived head's coordinate order. Original, replacement,
instantaneous-original and instantaneous-replacement at one anchor share the
EXACT same Z vector. The firstthree partial/whole counterparts share standardized
draws by root ID; their raw amplitudes/head parameters are not assumed equal.
One draw per anchor; no additional favorable realizations or dose curve.

`epsilon = .10 * frozen_M5_scale * Z`, `noisy_observation = clean_endpoint + epsilon`.
The scale includes M5's existing1e-5 floor. Thus noise has SD.10 in standardized
decoder space, including coordinates with floored scale. It is NOT exact
per-coordinate physical training SD when that floor binds, not matched raw
energy between graphs, and not physiological noise. Do not estimate scale
from new clean/counterfactual/test states. No clipping and no write-back to
reservoir states or future inputs. Only64 lag2 endpoint observations are noised;
full-stream noisy accuracy is not an additional experiment.

Apply exact frozen scaler/head to noisy endpoints. For every world also
independently check affine identity `noisy_scores = clean_scores + .10*Z@head`.
Within each pair the intervention score difference and class-margin change
must equal their clean equivalents within the predeclared1e-10 score tolerance;
shared additive noise cancels these differences. This does not guarantee both
argmax predictions remain correct: shared noise can move the pair across other
class boundaries. Lowest-index argmax ties are preserved.

Old training-derived frequency/current-symbol lookup never changes across
worlds and has exact joint0 (s!=replacement). Instantaneous final states and
shared noisy observations/scores/predictions must remain equal and joint0.
These are mathematical paired controls, not a10% joint-chance estimate.
Single-arm uniform chance.1 is reported separately. The instantaneous frozen
head is off-distribution; no marginal accuracy gate is imposed.

## Descriptive clean reference and output records

Generate clean tracking on the SAME fresh streams alongside noisy tracking,
but do not gate/filter on clean correctness. Sealed M6 clean results remain a
separate historical reference; no rescore or reopened claim. Report clean/noisy
original accuracy, replacement accuracy, JOINT accuracy, prediction changes,
clean-minus-noisy joint difference, every64 raw target/prediction/score/noise
vector, marginal errors and shared-noise label transitions. Record injected
raw/standardized energy, amplitudes and floored-scale count. Margins/energy
are descriptive, no alternative success metric. All main states remain clean.
Compute block mean/median/sample SD and10,000-draw descriptive bootstrap
seed1730001; no animal inference or pooling of three paired counterparts.

## Validation and reproducibility

Baseline audit/identity seal; protocol commit; frozen implementation/guards;
smoke and independent smoke; fixed main and independent main; results and
final artifact audit; four overview updates in that order. No source edits
during captures/runs. Use stable clean HEAD, authenticate captured source
hashes at recorded Git revisions, recheck bytes before sealing. Fresh
exist_ok=False dirs; preserve every interruption, source/validation failure
and same-settings repair. Scientific failures are final within this scope.

Main may reuse frozen M6 dynamics/clean replay. Independent verifier reuses
ONLY the older independent raw-source/manual-step/OLD-head SVD authentication
helpers, never main dynamics/scoring/noise/endpoint algorithms. It independently
rebuilds graphs, original streams, full trajectory/pre-state/four-world digests,
frozen heads, canonical Gaussian draws, noisy observations, score arrays,
integer predictions/control counts, primary and all summaries. Check exact
digests/draws/predictions and same-source/head bytes; numeric score checks
atol/rtol1e-10 do not alter the scientific gate.

Synthetic safeguards cover canonical-ID draw restoration/pairing, existing
scale floor semantics, no mutation/no clipping/feedback, shared-noise controls
exact joint0, affine score-difference identity, old frozen parameters,
unconditioned57/64 FAIL versus58/64 PASS/all-six conjunction, smoke-null,
clean-success exclusion rejection and independent noise reconstruction.
No source/noisy outcome is inspected to select test tolerances or dose.

Four workers, one numeric thread,7200s/4GiB sampled aggregate process-tree
budget per stage. Record individual RSS/OS peaks and50ms aggregate sampling
(shared pages may be counted repeatedly). No adaptive resource/seed/draw
increase. Store compact ROI windows/noise/scores plus full state digests;
no duplicate M5 full finite-difference banks or dense Jacobians.
Save config, protocol/seeds/source/environment/runtime/memory, parent/model/
data/graph/head hashes, manifests, raw metrics, summaries and independent checks.

Prior P2 material FAIL, M1 assay-invalid, M4 strict INFEASIBLE, noise-extension
negative gates and technical failures remain unchanged. A PASS supports this
one synthetic decoder-space condition; a FAIL does not mean internal stored
information was erased, because observation alone is corrupted. Neither
establishes biological robustness/learning/storage, unique cycle effects,
real-wiring/whole-brain superiority, living-fly pi memory or formal capacity.
P3 physiology/spiking/plasticity, later structural/rule-transfer/release projects
are outside this bounded M7 execution.
