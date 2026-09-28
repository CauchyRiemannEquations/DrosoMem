# ACT III normalization control — preregistered 2026-09-28

## Question and prior knowledge

On the SAME raw role-/degree-rewired graph, does recalculating incoming
normalization materially change independent-stream past-symbol decoding?
The preceding K4 study (result6b1d201) found real76.965% versus rewired78.603%,
with no real-wiring advantage. This informed follow-up is exploratory, with
new outcomes and gates frozen here before execution. It is not a retrospective
change to the prior study or evidence of autonomous recall/internal learning.

## Intervention and fixed comparison

Let A be the original signed W[post,pre] partial graph and B its audited
role-block shuffled raw graph. Define D(X)[i]=0.9/sum_j(abs(X[i,j])), or0 for
zero-input rows. Compare THREE arms:

- real: D(A) A (reference only);
- renormalized: D(B) B (previous structural-control construction);
- fixed_original: D(A) B (only the normalization factors change versus above).

Do not rescale again, tune gain, or repair a poor score. Both B arms have exactly
the same raw edges/weights. A/B nodewise degrees and role-block counts match;
zero-input rows therefore match, so frozen zero factors cannot delete B edges.
Source signs and raw outgoing weight multisets are retained before normalization.
Neither B construction guarantees normalized outgoing weight preservation.
This intervention isolates the effect of choosing D(B) versus D(A), conditional
on B. It does not isolate a universal topology effect or biological homeostasis.

## Cohorts and constants

Authoritative config: `configs/normalization_control.json`.
Main: five paired blocks, circuits701/702, three arms =30 runs. Reuse the10 raw
B graphs and mapping seeds34142–34146 from structural_k4_main, verifying exact
raw-matrix identity against the saved checkpoints. New train39142–39146 and
test40142–40146 streams; reuse is disclosed and no graph is selected by score.
Circuit samples and lag targets are not independent replicates: average both
fixed strata within each of five blocks before inference.

Smoke: circuit701, mapping34001/rewire37001 from the prior smoke, new train39001,
test40001;200 train/100 test rows plus100 warmup, three arms. Engineering only.

Both streams are independent iid uniform integer K4, generated using NumPy
default_rng(seed).integers(0,4,size), then uint8. Main train2000/test1000 rows,
each with100 warmup, each reset to zero separately. First4 patterns of the
existing16-symbol bank; each symbol stimulates51 of512 KC with amplitude0.5.
Same IDs, input patterns, streams,48 MBON observations and readout budget within
every triple. Rate model gain0.9/leak0.6/mbon_after_kc, no recurrent learning.

Feature[t] follows input[t]. Targets at lags0,1,2,3,4,5,8,12,16,24,32; primary
accuracy averages lags1,2,3,4,5,8. Lag0 is current access. Each graph/arm refits
affine ridge alpha1 on train-only standardized features (std floor1e-5),
unpenalized intercept,4 one-hot outputs,196 learned coefficients per lag.
Eleven heads and11 shifted-target null heads; shift1000 rows (smoke100).
No epochs, Adam, prefix weighting, hyperparameter search or autonomous rollout.
Unbounded regression scores are not probabilities. Save all labels/predictions,
train/test accuracy and MSE, frequency and shifted-null accuracy, and R2 versus
training-frequency scores, including negatives.

## Primary gates and conditional confirmation

Primary paired contrast: delta = renormalized minus fixed_original past accuracy.
Two-sided material-effect gate: abs(mean delta)>=1 percentage point and at least
4/5 blocks have the same strict sign as its mean. Additionally both B arms must
pass past-access H1: mean excess over BOTH measured controls>=5pp,>=4/5 blocks
each exceeding both by5pp, and mean R2 versus frequency>0.
Failing means no established >=1pp effect by this design, NOT equivalence.

Only if main passes, execute three preregistered confirmation blocks on NEW
mapping/train/test/rewire seeds41142–41144/42142–42144/43142–43144/44142–44144,
both circuits and all three arms =18 runs. Same lengths/hyperparameters/gates,
but all3 blocks must pass H1 and all3 deltas must have the MAIN direction, with
mean in that direction>=1pp. No pooling cohorts to rescue failure. Positive or
negative normalization contributions are treated symmetrically. Real-minus-B
contrasts are descriptive; they cannot trigger different extra cohorts.

No confirmation if the main gate fails. In either case analyze saved state
diagnostics without adjusting parameters. The prior result is not combined with
these streams to enlarge n. All seeds, failures and condition metrics are kept.

## Stability/representation diagnostics fixed before outcomes

Save normalized row L1 sums and row factors. With mbON-after-KC schedule, a
sufficient infinity-norm Lipschitz bound uses L_i=(1-leak)+leak*rowL1_i for
non-MBON neurons. For MBON i use (1-leak)+leak*(sum_nonKC|W_ij|+
sum_KC|W_ij|*L_j); take the maximum over rows. Bound<1 certifies contraction
for shared inputs; a bound>=1 means this sufficient test is inconclusive,
not that the dynamics are unstable. Report this distinction explicitly.

Check every integrated state is finite and abs(state)<=1+1e-12 (zero/within-box
initial states and leaky tanh). Abort a cohort on violation, preserve partial
artifacts, no replacement/tuning. Record max absolute state and rowL1>1 counts.
Physical/biological stability is not assessed.

For each run compare two initial states (zero and uniform[-1e-6,1e-6], fixed
perturbation seed in config), driven by the SAME first256 training symbols.
Save257 points of full-state infinity distance and MBON L2 distance plus initial
perturbation. Empirical forgetting flag: final full distance / initial <=1e-3.
This flag does not filter or suppress accuracy results and is not proof of an
echo-state property. It measures one small perturbation under one input.
Also save32 silent-decay steps after training, active-neuron counts, MBON norm,
sparsity, consecutive-state cosine, centered singular values/effective rank.
Diagnostics describe associations, not causal mediation or formal capacity.

## Audit, baseline, verification and resource stopping

Before smoke, reproduce both real and shuffled c701/s34142 from the previous
K4 study: every shared checkpoint array, graph weight identity and all metrics.
Regenerate every B from its pinned swap seed (5 accepted swaps per edge,
30-attempt cap), verify structural invariants and archived graph identity where
available. Stop on incomplete swaps; do not select another graph.

Every run regenerates graph, replays trajectories and perturbation probes,
and independently refits; all saved arrays must match exactly under one BLAS
thread. Independent augmented least-squares scores agree to1e-9 and predicted
classes exactly. Separate saved-artifact verification checks label alignment,
matrices/factors, paired identities, raw metrics, aggregation and gates.
Record code/protocol/config/data hashes, Git commit, UTC time, environment,
runtime, sampled peak RSS, source artifact hashes and checkpoint paths.
Preserve all historical result files and numerical/protocol source hashes.

One numerical thread; each cohort (including replay) <=1800 seconds and3GiB
sampled process RSS, checked between runs. Abort on resource/integrity/nonfinite
failure, preserving partial outputs. No reuse of output directories. Commit
protocol/config before new outputs; commit implementation before smoke.
Report every seed and stratum, paired mean/median/sample variance/95% percentile
bootstrap (10000 draws, seed47399), paired dz, full lag curves. No p-value gate.
Raw cell mean/SD is NOT labeled as a paired effect.

## Limits

This is representation plus external decoding in a computational model using
actual Drosophila connectome structure. No pi prediction, autonomous recall,
whole-brain advantage, learned internal connectivity, identified biological
memory circuit or Shannon/formal reservoir memory capacity claim follows.
Five main blocks and two fixed circuit samples limit population inference;
finite role-constrained rewiring is not a uniformly mixed random ensemble.
