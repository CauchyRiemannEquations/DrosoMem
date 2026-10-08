# M6: counterfactual past-symbol tracking by frozen readouts

Date: 2026-10-08. Prospective protocol to be committed before new held-out
trajectories or counterfactual decoder outcomes.
[Config](../configs/counterfactual_tracking.json).
Baseline main `43d6b068af853dd317241fc611f0df78b7110fca`;
new namespace `results/counterfactual_tracking_v1/`. Old results are immutable.

## Question and primary endpoint

If one actual past input symbol is changed while the entire prefix and future
input stream remain identical, does an already-trained lag2 decoder follow
that replacement on new independent held-out input?

M5 quantified local sensitivity and finite state differences, not registered
counterfactual decoding. M6 freezes its model, encoder, decoder/scaler and all
training-derived baseline predictors. No new learned parameters, readout fit,
normalization, mapping, dynamics or graph search is permitted in the experiment.
Any independent refit in verification authenticates OLD M5 training only.

At each of64 fixed anchors a, evaluate state after symbol a+2. Original target
is s=stream[a]; replacement target is r=(s+1)%10. For ALL anchors define
`joint_correct = count(pred_original==s AND pred_replacement==r)`.
Primary PASS requires `10*joint_correct >= 9*64` in every one of six primary
blocks (at least58/64). Equality at the exact9/10 threshold passes. There is
one fixed lag and one unconditioned paired-transition endpoint. Do not retain
only anchors whose original prediction succeeded, change the denominator,
select a different alternative symbol, add seeds or lower the threshold.

There is **no performance eligibility gate**. Poor unperturbed accuracy or
frozen-head transfer counts toward scientific FAIL. A joint success inherently
requires both original and replacement accuracy at the same anchor. This
decision avoids converting a functional tracking failure into assay invalidity.
Source/semantic/replay violations are technical validation failures, archived
separately. Non-distinct encoder patterns would invalidate the replacement
assay without resampling. Whole firstthree is a separate paired secondary
analysis, not three extra independent input replicates. Smoke scientific
endpoints are null (stage label `smoke` allowed).

## Frozen source model, encoder and readout

Reuse original source-derived circuit701 legacy5 and brain5, signed incoming-L1
gain0.9, drive0.6, direct carry0, previous-state synaptic history on,
mbon_after_kc. These are the M5/M3 intact_synaptic computational conditions,
not the carry-on v1/full model. Same512 KC input pool,51 selected KC per symbol
at amplitude0.5, same48 observed MBON IDs. No recurrent weights change.

Import each block's authenticated M5 parent from
`results/functional_sensitivity_v1/main/{level}_s150000i/`.
Parent main source `c541993484f6778a031d71ad4eb1798c748f6ce7`, parent protocol
`37212abeb6cfc1a4907e08348ff41030f3edfb8b`; parent main manifest SHA256
`98eb3b4c3c49934365caa30608fd50b677dd883b7e3ed722271db4aceeaf73af`.
Read only required arrays, not parent replacement outcomes. Frozen lag2 is
`coefficients[:,20:30]`, `intercept[20:30]`; original mean/scale each48.
The490 affine parameters were trained on4000 old M5 iid samples with alpha1,
train-only mean/SD floor1e-5. Scoring is
`((observed_state-mean)/scale) @ coefficients[:,20:30] + intercept[20:30]`.
Argmax chooses the lowest class index on a tie. No centering/refitting on new
original or counterfactual test data. Lag0's existing columns0:10 are a separate
unperturbed sanity output, not another scientific pass criterion.

Authenticate exact original graphs, input patterns and coordinate IDs/roles
against the frozen parent records. Distinct K10 patterns must remain exact;
partial and whole source IDs/mappings are paired while their normalized weights,
response amplitudes and heads are not asserted equal.

## New held-out inputs and intervention

New blocks1600001–1600006 inherit mappings1510001–1510006 and M5 parent
blocks1500001–1500006. New uniform iid K10 streams1610001–1610006 are independent
of both old M5 train152... and test153... streams. Whole uses the firstthree
paired new streams. Reset each baseline trajectory to zero; warmup200,
test2000 scored rows. No new training stream or autonomous feedback.

Fix64 anchors `warmup+floor(linspace(0,test_samples-1-2,64))`.
Each anchor's original and replaced three-symbol windows start from the SAME
unperturbed full state at a−1. Replay symbols at a,a+1,a+2; in the replaced
window directly substitute actual symbol r at a only. Prefix, both future
symbols, model and head are identical. This is a valid categorical intervention,
not local interpolation, gradients, an encoder-wide update or feedback recall.
Save original/current/replacement targets explicitly with evaluation time a+2.

Process probes during baseline replay or use authenticated pre-state digests
to avoid giant histories. Save observed windows/scores/predictions and full
pre-state/window/final-state digests rather than64×N raw snapshots or full
Jacobians. Independent replay reconstructs pre-states from the entire fresh
stream and verifies every digest. No source state information is replaced by
label lookup or teacher reference during scoring.

## Paired controls and descriptive outputs

Use the OLD M5 training-derived `majority[2]` and `current_tables[2]` (with old
majority fallback for empty rows). No new label fitting. Current symbol at a+2
is unchanged by past replacement; both predictor outputs must be identical
within every pair. Since s!=r, unchanged predictions have exact joint count0.
This is a mathematical paired-control result, not a biological population
inference or a made-up10% joint-chance threshold. Single-arm uniform chance is
1/K=.1; report it separately. Independent random guesses in the two worlds
are not substituted for the paired unchanged controls.

Instantaneous reference disables prior-state drive, retains carry0 and current
KC→MBON access, and uses the SAME frozen lag2 head/scaler. Its final full states,
scores and predictions must be exactly identical for both windows and its joint
count exactly0. That head is off its training distribution in this reference;
do not require a particular marginal accuracy or call the reference head a
matched physiological decoder. This is a semantic negative control.

Record every anchor, original/replacement marginal accuracy, unconditional
joint count, predicted-label changes/unchanged predictions, true/predicted
classes, class-frequency/current-only/reference controls, score vectors,
original/replacement class margins and their paired changes. Verify margin
change equals the frozen linear head's projection of observed-state difference.
Margins/norms are descriptive, no extra gate. Report full new-stream lag0 and
lag2 accuracy and old controls as sanity/descriptive transfer context.

Summarize fixed block paired values, mean/median/sample SD and descriptive
10,000-draw block bootstrap seed1620001. The64 probes share each stream/model;
they are not64 independent models or animals. No all-lag curve or general
memory capacity is inferred from the registered lag2 study.

## Verification, safeguards and resources

Sequence: baseline repository/provenance audit, protocol/config commit,
immutable baseline seal, implementation/source freeze, synthetic safeguards,
smoke, independent smoke, fixed main, independent main, synthesis/final audit
and four overview updates. Source remains frozen during all captures/runs.
Capture stable clean HEAD and authenticate every tracked SHA at that Git
revision, and recheck bytes/HEAD before stage sealing. Fresh exist_ok=False
directories; all interruptions/validation failures remain sealed and registered.
Same-settings technical repairs never change lag/threshold/probes/seeds/head.

Synthetic safeguards cover only-one-window-symbol replacement and a+2 labels,
same prefix/future, old lag-major head extraction, frozen scaler/no new fit,
lowest-index tie policy, joint-versus-marginal credit without filtering,
exact57/64 FAIL versus58/64 PASS/all-six conjunction, low original accuracy
counting toward FAIL, unchanged controls joint0, instantaneous zero-history,
source hashes/metadata shapes and smoke scientific-null. No extra performance
gate is added after seeing a failure.

The independent verifier rebuilds raw source graphs/normalization/roles,
encoder mappings, fresh full-state trajectories/pre-states, direct symbol
replacement/reference windows, fixed scores/predictions, all targets and
controls, integer primary gate and summaries. It imports no main reservoir,
runner, scoring or endpoint helpers. It also reconstructs the OLD M5 lag2
ridge training from authenticated old features/labels (independent SVD) and
checks the archived head, then evaluates the exact frozen archived parameters.
This is old-head authentication, not new model training.

Four workers, one numerical thread each; each stage7200 seconds and4GiB
sampled aggregate process-tree RSS maximum. Save individual process OS peak and
50ms aggregate measurements (shared pages may be counted repeatedly). No
adaptive resource/stream/probe count. New archives should stay modest; no
duplicate M5 full finite-difference vectors or dense full-state probe banks.

Each stage includes protocol/config/seeds/environment/source commit, pinned
parent/model/graph/data/head/scaler hashes, runtime/memory, raw trials,
summaries, manifests and independent verifier records. Final audit protects
all44,571 baseline result identities and old source/protocol/config/data/tests,
including sparse paths and M4's provenance-invalid guard snapshot. No fresh
replay of sparse old results is claimed. P2 failed material gates, M1
assay-invalid, M4 strict INFEASIBLE and all historical negatives remain.

Even a PASS supports only this fixed computational input→state→frozen-readout
tracking response under stipulated conditions. It does not establish storage
location, unique cycle contribution, real-wiring or whole-brain superiority,
biological learning, living-fly pi memory or formal Shannon/reservoir capacity.
P3 physiological/spiking/plasticity work remains excluded.
