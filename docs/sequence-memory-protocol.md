# ACT II pilot: four sequence families

Locked before any new outcome, 2026-09-28. Exploratory preregistration-style
protocol; no external registration. Base evidence: ACT I revision 0a19126.
One question: does the fixed-observation model support trained-prefix recall
beyond pi, and does brain1 outperform legacy5 across these sequence families?

## Hypotheses and baseline

H-generalization: appreciable exact-prefix recall of independently seeded random
and shuffled-pi sequences occurs in at least one network, above simple
statistical-prediction controls. H-scale: brain1 improves over legacy5.
Periodic performance is a predictability sanity check, not independent proof
of long-term arbitrary information storage. Pi is not assumed mathematically random.

Keep ACT I's rate dynamics and fixed-32 decoder; recurrent connections frozen.
legacy5 uses both original 686-node input-map strata (701,702); brain1 uses all
138,639 source nodes and 15,091,983 edges. Same exact stimulated root IDs and
amplitudes, 48 observed MBON IDs/order and 482 readout parameters across graph
levels and sequence families for a given model seed/stratum. No extra input
or observation for whole brain. Reuse cached graphs only after hash validation.

## Four datasets, fully specified before generation

All contain 200 integer symbols, K=10, common supplied prefix [3,1,4].
Training covers 199 next-symbol pairs; autonomous horizon197 excludes prompt.
Each stream is reset and separately trained; no transfer learning.

- pi: indices [0,200), leading 3 included; mpmath/Decimal agreement required.
- random: PCG64 via SeedSequence([dataset_seed,2201]); draw 200 uniform integers
  in [0,10), replace first three with [3,1,4]. Thus the suffix is iid uniform,
  conditional on a fixed prompt; do not claim the entire stream is iid.
- shuffled_pi: retain pi's first three symbols and permute its remaining197
  symbols using SeedSequence([dataset_seed,2202]). Preserve the complete
  200-symbol multiset and the prompt exactly, while altering suffix order.
- periodic: repeat the fixed period10 motif [3,1,4,0,2,5,6,7,8,9] to length200.
  Each symbol has one deterministic successor; a first-order predictor should
  solve it. This is deliberately easy and has low generative complexity.

Save exact arrays, generator specification, dataset seed and SHA256. Pi and
periodic are identical across dataset seeds and must not be counted as five
independent sequences. Random/shuffle each have five different realizations.

## Fixed budget and paired design

Model/dataset seed pairs: (9142,9242), (9143,9243), (9144,9244),
(9145,9245), (9146,9246). Each block crosses all four families, both graphs,
and both input-map strata: **80 fits** total. Model and dataset seeds are
paired rather than fully crossed; variance combines those sources and they
cannot be separately estimated. Statistical unit is the paired block, not
individual stratum or animal. One connectome; no biological replication.

Smoke: block(9139,9239), stratum701, both graphs, all families; length40,
horizon37, 20 updates, eight fits. Smoke is excluded from all main estimates.
Per isolated worker: <=1800 seconds and <=3GiB process-tree RSS sampled every
.2 seconds. Preserve failed logs/partial artifacts, do not replace failed seeds.
No complete manifest unless all planned conditions finish. No hyperparameter
sweeps or result-based checkpoint selection.

## Dynamics and learning held unchanged

incoming-L1 gain .9, leak .6, mbon_after_kc schedule. Original KCEncoder
fraction .1/amplitude .5; 51 stimulated KCs per symbol in each original map.
48→8 tanh→10 readout, SeedSequence([model_seed,9901,0]), train-only feature means/
std floor1e-5. Full-batch Adam beta(.9,.999), epsilon1e-8, LR .03,
L2=1e-5 on nonbias weights, 2000 updates, final head only.
First32 generated targets get weight4, others1, normalized by total weight.
Teacher forcing supplies true input symbols; autonomous generator receives only
prompt, state update, head and horizon, never reference targets.

## Prediction controls

For each exact dataset fit two deterministic non-reservoir controls to the same
199 weighted target pairs: weighted majority-symbol output and a weighted
first-order transition table. The latter observes only the current symbol.
Argmax ties resolve to smallest symbol; an unseen transition row uses weighted
global target counts. Greedy autonomous generation starts from the same prompt,
continues197 steps; save arrays, prefix score and teacher-forced accuracy.
These controls separate simple statistical predictability from stateful decoding;
they do not prove connectome topology superiority or replace matched graph controls.

## Metrics and raw evidence

Primary generic exact_prefix_symbols (first error, horizon-censored), and
autonomous_prefix_bits = L log2(K), also called recalled-prefix bits.
Only pi rows use the name Pi Memory Score; non-pi rows never inherit that label.
Do not equate this score with Shannon capacity or formal reservoir memory capacity.

Retain per-run generated symbols, first error (1-based generated position and
0-based full-sequence index), position and region accuracies (1–32,33–64,65–128,
129–197), 10-way probabilities, training loss/accuracy curves, teacher-forced
accuracy and independent control scores. Neural evidence: 48-MBON training/
recall features, active counts(|state|>1e-8), sparsity, consecutive-state cosine,
centered effective rank/singular values, 32-step zero-input decay.
Record runtime, sampled peak RSS, node/edge counts, threshold, git commit,
source/protocol/config/data hashes, environment lock/CPU/RAM/BLAS and artifacts.
Use fresh directories; never overwrite ACT I or earlier result files.

## Analysis and decisions

Report all80 raw rows, per-family/network mean, median, sample variance,
exact-prefix completion fractions (>=16,>=32 and full horizon), and bits.
Average two strata within each block. Paired graph differences and paired family
differences versus pi retain all five blocks. Percentile95% bootstrap intervals:
10000 resamples, seed9299; standardized paired d_z undefined if SD zero.
Intervals describe this small computational cohort; no p-value success gate.
Do not treat family instances/model initializations as independent animals.

Limited beyond-pi evidence for a network requires, for BOTH random and shuffled_pi:
mean prefix>=16, at least4/5 block means>=16, mean advantage over the stronger
of the two simple predictor prefix scores>=5, and at least4/5 such advantages>0.
Failure means not established under this budget, not impossibility.
Passing supports trained-instance recall across these sampled families, not
unseen-sequence prediction or arbitrary-sequence universality.

Graph superiority per nonperiodic family requires brain1−legacy5 mean>=5 and
at least4/5 positive block deltas. General non-pi graph superiority requires
both random and shuffled_pi to pass; pi alone cannot establish it. Periodic is
excluded from superiority decisions because of predictable ceiling behavior.

If any nonperiodic graph comparison passes, independently confirm ONLY those
families using blocks(10142,10242),(10143,10243),(10144,10244), both strata and
graphs, same settings. For pi use offset1000 and its own first-three prompt;
for random/shuffle use new dataset seeds, same generating rule. Each qualifying
comparison must gain>=5 with all3 positive block deltas; failed confirmation
remains unconfirmed. No new confirmation if none qualifies.
Limited family-generalization findings remain exploratory across five sampled
sequences; do not relabel the above conditional test as confirmation of all
possible sequence memory.

Replay all saved heads/features/probabilities exactly and independently refit
both graphs/all families for the first block/first stratum (eight heads).
Verify periodic transition-control ceiling, shuffled multiset and common prompt.
Negative outcomes lead to existing representation diagnostics, not tuning.
Length/alphabet scaling, neuron/edge ablation and plasticity are not run here.

## Claims and provenance boundaries

Representation, decoding and recurrent learning remain distinct. A frozen
connectome plus trained head addresses the first two. Both graph levels share
incoming normalization assumptions and sparse artificial stimulation.
Keep old source/protocol files and archived artifacts unchanged. Because historical
verifiers fingerprint the entire source tree, adding new modules changes current
context; strict ACT I replay uses revision0a19126, not rewritten old manifests.
