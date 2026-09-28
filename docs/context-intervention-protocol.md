# Fixed-K context-conflict intervention — preregistered protocol

ACT II task/decoding experiment following the context diagnostic at82aaf19.
The previous alphabet and n-gram outcomes are known. Commit this protocol and
all cohort configs before generating the new sequences or fitting any new head.
No assertion that sequence conflict is the only property changed by reordering.

## Question and construction

At K4 and N128, does training on lower-conflict three-symbol contexts improve
autonomous trained-sequence recall with input/output capacity held fixed?
Prompt[0,1,0]; each complete sequence has exactly32 occurrences of each symbol.
Tail multiset:30 zeros,31 ones,32 twos,32 threes. For each data seed, use
NumPy default_rng(seed) to permute that tail once, then draw2000 pairs of indices
uniformly with rng.integers(3,128,size=(2000,2)). Reuse the identical starting
sequence and proposal list in both arms. For each proposed swap, accept only
strict decreases (low arm) or strict increases (high arm) in the integer error
`125 - sum_context max_symbol count(context,symbol)` over target positions3..127
and three preceding symbols. Reject ties, identical positions and equal-symbol
swaps; never alter the first three symbols. Use every proposal; no early stopping,
restarts or searches over seeds. Save initial sequence, proposal indices, all
2001 objective values and acceptance indicators for both arms.

Construction is a manipulation of a declared DATA property, not selection by
neural performance. Before any cohort's neural fitting, require high-minus-low
error floor>=0.10 for every pair, prompt/multiset preservation and deterministic
reconstruction. On failure retain all sequences/logs and stop that entire cohort;
do not replace a failed pair or relax the cutoff. Record short-context orders
1,2,3,4,5,8, early/late ambiguity, regional symbol frequencies, transition counts
and repeated-neighbor fraction. Other ordering properties may change too.

## Cohorts and matched neural conditions

Main: model27142–27146/data28142–28146 (paired), strata701/702, legacy5/brain1,
low/high arms:5x2x2x2=40 fits. Smoke: model27139/data28139, stratum701, both graphs
and arms,20 updates (4 fits), excluded from inference. Confirmation, only if any
main graph passes the neural criterion below: model29142–29144/data30142–30144,
both strata/graphs/arms,24 fits. Only a main-qualified graph can earn a confirmed
claim. Never pool smoke, discovery and confirmation to pass a criterion.

Use the previous K4 adapter and unchanged numerical core:512 eligible KCs,
51 stimulated KCs per symbol, amplitude0.5; identical four symbol patterns and
same48 observed MBON IDs within each block/stratum across arms and graphs.
Readout48→8 tanh→4 (428 parameters), external learning only, initialization0
with the historical nonlinear_seed mapping. Incoming-L1 gain0.9, leak0.6,
mbon_after_kc; fixed recurrent weights.127 teacher-forced transitions from reset;
train-only feature scaling with floor1e-5; Adam2000 updates,lr0.03,L2=1e-5;
target positions3..34 weight4, others1. Autonomous rollout from reset after010,
125 steps, feeding back only generated symbols. Each arm has a fresh state/head;
same initial head randomness across arms. No hyperparameter tuning.

## Controls and measurements

For every unique sequence fit all six prior fixed context controls, including
their unchanged weighting, backoff and tie rules. Primary descriptive control=m3;
report every order/storage size and verify with the independent scan oracle.
No matched-parameter or connectome-topology claim for these count tables.

Save every head/checkpoint/config/manifest and raw per-seed results. Training:
loss and teacher-forced next-symbol accuracy. Autonomous: exact-prefix symbols,
first error, generated integer sequence, probabilities, position and region
accuracy, prefix bits. Neural: active neurons, observed MBON trajectories,
sparsity, cosine similarity, effective rank and32-step state decay. Engineering:
runtime, sampled peak memory, graph IDs/sizes/threshold, environment, commit,
source/config/data hashes. Reuse the alphabet runner's scoring unchanged.

## Primary analysis and gates

Within each model seed and graph, average the two input strata, then compute
low-minus-high exact-prefix symbols. Each graph passes discovery if mean>=10
and strictly positive in>=4/5 blocks. An eligible graph is confirmed only if
fresh confirmation mean>=10 and all3 blocks are strictly positive. Report
negative results and inconclusive confirmation without changing these rules.
Whole-minus-partial differences within each arm and graph-by-arm interactions
are descriptive; no new whole-brain superiority claim is registered here.

Report raw rows, paired differences, mean, median, sample variance, bootstrap95%
percentile intervals (10,000 block draws,RNG31399) and paired mean/SD (null if
zero SD). Only5 or3 data/model seed blocks; no p-value-based declarations.
Report context3 control changes, prefix-region conflict and teacher/autonomous
tradeoffs. No post hoc best-order or best-seed selection.

## Verification, budget and stopping

Replay/refit one archived K4 partial baseline before new runs. Verify all smoke
heads and refit all four. Replay all main/confirmation heads exactly, rebuilding
graph/trajectories; independently refit both arms and graphs of the first block,
stratum701 (4 per cohort). Grouped graph reuse is allowed, state/head reuse is
not. Max1800s/3GiB per graph/seed/stratum worker, measured every0.2s; one numerical
thread. Max40 main+24 conditional confirmation fits, not counting4 smoke.
Preserve partial failures and stop on numerical/hash/resource errors. No expanding
the budget to rescue a failed hypothesis. Baseline and graph cache already exist;
avoid rerunning older sweeps. Preserve all old results/protocols/source bytes.

## Interpretation and next-study rule

Success supports sensitivity to this fixed-K ordering intervention, not isolation
of context conflict as a unique causal variable. Unigram frequency, prompt,
input identities and head capacity are matched; higher-order frequencies,
predictability and conflict positions may differ. This is trained-instance
recall, not unseen random prediction, formal memory capacity or biological
synaptic learning. Whole versus partial still conflates graph scale/weak edges.
After a confirmed effect, prioritize one region-matched conflict intervention
to separate early-prefix weighting from global conflict; after a negative or
unconfirmed effect, prioritize one frozen-state representation/decoding diagnostic
on these saved pairs. No ACT III or plasticity expansion in this experiment.
