# ACT III-A bounded neuron-population closeout protocol

Status: locked before any new trajectories or outcomes. Historical source: commit
a6bcbea, KC confirmation top manifest pinned in `configs/neuron_panel.json`.
No tuning, optional extra seeds, outcome-dependent subgroup selection or changed
criteria. Negative studies count as completed studies. See annotation/seed audits.

## Scope and stopping rule

Complete the present two 686-neuron MB rate-model ACT III-A study, not every
possible whole-brain neuronal localization. Previously executed KCgamma studies
remain unchanged. Three work packages: input-only control; a population panel;
independent confirmation of every panel condition and a sensitivity map/report.
ACT III-B/C/D and IV are not part of this run. Downstream populations are absent
from these graphs; whole-brain localization remains explicitly untested. This
scope limitation is not a successful result. No autonomous recall claim follows
from an independent-stream delayed-symbol decoding endpoint.

Stop after all prescribed smoke, discovery and confirmation cases and validation.
Resource cap: 3600 seconds and sampled peak RSS 3 GiB per command. On scientific
negative findings continue unchanged. On integrity failure stop, preserve failed
output and fix implementation in a new commit/directory; never replace seeds.

## Shared fixed design

Exact machine-readable config is `configs/neuron_panel.json`. Actual threshold5
circuits c701/c702: 512 KC,48 MBON,125 DAN,1 APL,686 neurons. Same biological
graphs in confirmation; computational seeds are new, not biological replication.
K=4 pseudorandom integer input; 100 wash-in then 2000 train/1000 independent test
symbols; smoke200/100. Lags0,1,2,3,4,5,8,12,16,24,32; primary past lags1,2,3,4,5,8.
Incoming-L1 normalization before interventions, gain0.9, leak0.6,
mbon_after_kc schedule, input fraction0.1/amplitude0.5. No renormalization after
intervention. Keep all48 MBON slots, including zero-valued lesioned slots.
Ridge alpha1, train-only mean/std floor1e-5, 196 coefficients per 4-output lag
head. Cyclic half-train null and training-frequency predictors. Scores are
unbounded affine outputs, not probabilities. R2 may be negative and is not clipped.
Frozen means SOURCE moments/real+null weights/bias unchanged; refit uses each
target's training states and labels. Both always reported separately.

## Package 1: direct input versus full neuron lesion

Use EXACT stored masks, streams, mapping and source heads from kc_confirmation
smoke/main, gamma and matched0..2: 4 smoke+24 main NEW conditions. Silence direct
input on the mask but retain every recurrent weight. Lesioned neurons here may
be driven recurrently and must not be forcibly clamped. Compare frozen/refit
with archived full lesions using identical source heads. Primary H-input:
input-only accuracy minus full-lesion accuracy >=5 percentage points on average
and >0 in all3 blocks, separately frozen/refit; gamma primary, matched descriptive.
Also report state differences and whether graph reachability or zero raw incoming
degree forces an exact null. Failure to pass is not equivalence. This estimates
the additional edge-removal effect conditional on direct-input removal, not a
standalone topology effect. Existing seeds reused intentionally for causal pairing.

## Package 2: fixed population panel

Five target groups: KCab (KC cell_type starts KCab), KCapbp (starts KCapbp),
DAN role, APL role, and high-degree hubs: top ceil(0.05*N_nonMBON)=32 non-MBON
neurons by raw in+out edge degree, ties by numeric root ID. These masks use graph
annotations only, never outcomes. The prior gamma group completes KC coverage.
For each target draw three controls without replacement within each draw;
across draws overlap is allowed. KC subtype controls use all KC and exactly match
joint raw in-degree/out-degree/four-symbol input bitmask counts. DAN/APL/hub
controls use all non-MBON neurons and exactly match input-bitmask and count.
Targets remain eligible in pools: report target overlap, cannot call controls
disjoint. Infeasible matching is an integrity failure, not permission to relax.
For DAN/APL/hubs degree and strength are NOT matched; report their imbalance and
interpret as count/input-matched sensitivity only. Hubs intentionally contrast
high degree with ordinary cells. No cell-type-exclusive causal claim is warranted
from unbalanced graph properties. All controls remove zero observed neurons.

Neuron lesion zeros both recurrent rows/columns and direct input and leaves state
zero from zero initialization; surviving normalized weights remain unchanged.
Two positive engineering controls: remove all KC (all direct input removed),
remove all MBON (all observed features zero). These are NOT specificity tests;
loss of an observation channel is not evidence of an internal memory mechanism.
Intact+5*(target+3controls)+2positives=23 conditions per block/circuit.
23 smoke,138 discovery,138 confirmation=299 new fits/full replays; all286
non-intact conditions also frozen-source evaluation. Do ALL confirmation arms
regardless of discovery outcome. No pooling cohorts for a pass.

## Hypotheses and inference

For each target and decoding mode, impairment = intact-target and excess
impairment = mean(3 matched controls)-target, averaged over primary lags and
then the two circuits within each computational seed block. A material
population sensitivity requires BOTH mean impairment >=0.05 and mean excess
impairment >=0.05, with both differences >0 in all3 blocks, AND intact access:
frequency/null margins>=0.05 in every block and positive mean R2. Confirmation
requires that same gate in BOTH independent cohorts for the SAME mode/group.
Frozen and refit gates are distinct. A frozen-only effect is decoder dependence,
not destroyed information. A confirmed refit effect means reduced accessible
past-symbol information under this decoder/model, not biological memory necessity.
Five comparisons are exploratory, no p-value or family-wise significance claim.
Publish all cases, not just those passing. Failed thresholds do not show equivalence.

Store per-lag/per-seed accuracy, training loss/accuracy, frequency/null margins,
R2, frozen drift, raw states, masks/IDs, removed edges/strength, active neurons,
MBON norms/sparsity, temporal cosine, centered effective rank and decay32.
Zero-state decay ratio is null with an explicit zero-baseline flag, not NaN/0/1.
Every condition exact trajectory replay plus independent augmented least-squares
refit; frozen SOURCE least-squares verified on target states. Keep checkpoints,
git/config/data/source hashes, environment/runtime/sampled memory and old-output
hash audit. Independently regenerate masks and intervention weights in verifier.
Statistics: per-seed values, mean/median/sample variance, paired differences,
10000 bootstrap resamples of THREE seed blocks, seed94399, paired dz when defined.
Masks/circuits/lags are not independent replicates. Intervals with n=3 are limited.

## Completion deliverables

Protocol/config/seed audit committed before outcomes; implementation committed
before execution. Raw tables, manifests/checkpoints, independent verification,
sensitivity map, negative findings and scope/claim report. ACT III-A current-model
panel closes when these are complete even if no population passes. Broader
whole-brain/downstream and autonomous-recall localization remain open extensions.
Choose ONE next information-rich experiment after interpreting the results.
