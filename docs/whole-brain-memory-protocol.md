# ACT I: matched whole-brain sequential recall protocol

Locked before smoke/main outcomes on 2026-09-28. Exploratory preregistration-style
protocol, not an externally registered confirmatory trial. The preceding repository
audit and historical baseline replay are not new study outcomes.

## Question and hypotheses
Does expanding the actual Drosophila connectome improve autonomous recall under a
fixed input and observation budget? H-scale: brain5 improves over legacy5.
H-completeness: brain1 improves over brain5. Null/negative outcomes are equally
reportable. This is a computational model using actual fly connectome structure.

## Baseline and model choice
The established fixed-32 baseline freezes recurrent weights and trains an external
48→8 tanh→10 softmax head (482 parameters). Use its rate dynamics, incoming-L1
normalization (gain .9), leak .6 and mbon_after_kc schedule at every scale.
Do not compare the historical rate score to whole-brain LIF as if graph size
were the only change. Phase 6 LIF feasibility remains a separate result.

All neurons update once per symbol; all annotated MBONs read newly updated KC
states and old states of other populations. Only the exact original 48 left
MBON root IDs, in original order, are observed. Enlarged graphs introduce neither
additional observed features nor direct stimulation of added neurons.

## Conditions and matched controls
- legacy5: original bundled 686-neuron circuit, threshold >=5.
- left5: all 2,795 left KC/MBON/DAN/APL neurons, threshold >=5.
- left1: same left node universe, threshold >=1.
- brain5: all 138,639 source nodes, threshold >=5.
- brain1: same complete node universe, threshold >=1.

Use both original sample circuits (701/702). They share MBONs and come from one
fly; these are two input-map strata, not independent animals.
For each seed, construct the original KCEncoder in the original circuit order,
then map exact input root IDs and amplitudes into each graph. Never encode by
sampling from the expanded KC population. Hash and save every mapping.

Normalize each graph separately by the same incoming-L1 rule. This necessarily
rescales retained edges when new incoming edges appear. Therefore effects concern
graph preprocessing plus the fixed normalization rule, not a pure biological
size effect. left5−legacy5 and brain5−left5 diagnose expansion at threshold 5;
left1−left5 and brain1−brain5 diagnose weak-edge restoration at fixed node coverage;
brain1−left1 diagnoses expansion at threshold 1. These contrasts do not uniquely
identify feedback pathways or topology causation. No additional tuning.

## Dataset, training and evaluation
Pi is an integer-symbol dataset with K=10. Index 0 is leading 3. Use digits [0,200),
independently checked by mpmath and Decimal. Train 199 next-symbol pairs;
prompt is 314, autonomous horizon is 197, all within the trained sequence.
State starts at zero for training and each recall. Train-only means and
std floors 1e-5; full-batch Adam (.9,.999,epsilon=1e-8), LR .03, L2=1e-5,
2,000 updates. Prefix weights 4 on the first 32 generated targets (training
indices 2..33), 1 elsewhere, normalized by their sum.
Readout initialization uses the existing SeedSequence([seed,9901,0]).
Use final checkpoint only; no best-epoch or best-seed selection.

Teacher forcing supplies true symbols. Autonomous generation receives only the
prompt, frozen state update, trained head and requested horizon; targets are
absent from generation. Continue generation through errors, never stop simulation
at the first error. Save all symbols and 10-way probabilities.

## Seeds and finite budget
Main seeds: 7142,7143,7144,7145,7146, paired across all five levels and two strata:
50 fits. One initialization per pair; no sweep. Smoke: seed 7139, circuit 701,
all five levels, length 40/horizon 37, 20 updates, same prefix rule. Smoke
results are excluded from main estimates and cannot trigger tuning.
Each isolated worker: <=1,800 seconds and <=3 GiB sampled process-tree RSS.
Sampling every .2 seconds. Stop and preserve failed worker logs; do not replace
failed seeds. No complete result manifest unless every prespecified run finishes.
If the budget is exceeded, report incomplete feasibility instead of reduced
post-hoc success criteria. Graph preparation is separately bounded by Phase 6.

## Metrics and immutable records
Primary Pi Memory Score = exact correct generated prefix length, excluding prompt,
capped at 197. Save first-error generated position (1-based; null if censored),
absolute sequence index (0-based), full generated sequence, position accuracy,
regions 1–32,33–64,65–128,129–197, and confidence/probability arrays.
Also save recalled-prefix bits = L log2(K); this is not Shannon capacity or formal
reservoir memory capacity.

Save training objective/cross entropy curves, next-digit accuracy and
teacher-forced scored-region accuracy separately from autonomous metrics.
Online state diagnostics: active |state|>1e-8 counts and fraction/sparsity,
48-MBON trajectories, consecutive observed-state cosine similarity,
centered observed-state singular values and entropy effective rank. Save zero-input
32-step state-norm decay after training; report relative full/observed norms.
This probe is not autonomous recall and does not change fitted data.
Never allocate a time-by-all-neurons or dense N-by-N matrix.

Every run stores commit, source hashes, protocol/config hashes, timestamp, all
seeds, platform/CPU/RAM/BLAS/package versions, dataset bytes/hash, source and graph
identities, input/observation root IDs, runtime, sampled peak RSS, edge counts,
threshold, head, raw features, predictions, history and file hashes.
New directories only; archived artifacts remain untouched.

## Decision and analysis fixed before outcomes
Report all 50 raw scores. Average the two input-map strata within each paired
seed for uncertainty estimates (five independent computational-seed blocks).
Report per-condition mean, median, sample variance, paired deltas, win/tie/loss,
standardized paired mean difference d_z (undefined when SD zero), and percentile
95% paired bootstrap intervals from 10,000 resamples using seed 7199.
Small seed count: descriptive intervals, no p-value gate and no biological
population inference.

A condition improves over legacy5 in main only if paired mean improvement >=5
digits and >=4 of 5 seed-block deltas are strictly positive. Report separately
brain5 and brain1, even if neither passes. Weak-edge evidence requires mean
brain1−brain5 >=5 and >=4/5 positive seed blocks, separately labeled.
Failure to pass is lack of established improvement under this design, not proof
of equivalence or impossibility. Report all other contrasts as diagnostic.

If either whole-brain condition passes, confirmation uses fresh seeds
8142,8143,8144, both strata, offsets 0 and 1000, all five conditions (60 fits).
Each offset must have >=5 mean improvement and all three block deltas >0 for
the originally qualifying comparison. Keep design/optimizer fixed. Independently
refit every qualifying whole-brain and legacy head for the first seed/stratum
at both offsets. Replay every saved head exactly. Main positive followed by
failed confirmation remains unconfirmed.
If neither passes, run no hyperparameter search or extra memory benchmark:
analyze rank, amplitude, activity and decay diagnostics already collected.

## Limits and stopping boundary
Only ACT I is executed here. ACT II starts after this research paragraph is
reviewed; the latest user instruction prioritizes closing ACT I first.
Representation, decoding and internal learning are distinct: frozen connectivity
plus trained readout concerns representation/decoding. It cannot establish
internal synaptic learning, biological pi memorization, unseen pi prediction,
physiological realism, arbitrary-sequence memory or real-topology superiority
over matched shuffled graphs. Additional structural controls require a new
protocol. Robustness belongs to ACT V and is not pooled into this main result.
