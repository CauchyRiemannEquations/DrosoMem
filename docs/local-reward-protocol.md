# ACT IV — Scalar reward with local rate eligibility: locked protocol

One question: can scalar correctness reward, without an activity-target vector or
trained-head gradient, teach useful lag2 codes to the current partial rate model?
This follows the completed bounded IV-A comparison, not a restart of ACT I.
Source/protocol/config must be committed before smoke or main outcomes. One
learning rate, no tuning after failure. Confirmation runs regardless of discovery.

## Scientific scope and source review

Three-factor plasticity combines local pre/post activity traces with a scalar
reinforcement signal. [Legenstein, Pecevski & Maass (2008)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2543108/)
analyze reward-modulated STDP in spiking models. This study borrows that general
organization only: its rate covariance rule is NOT their STDP equations, a
physiological dopamine simulation, or an unbiased policy gradient. All constants
below are engineering choices. Existing Phase5A RewardPlasticity requires a trained
classifier gradient; it cannot answer the present local-only question. Existing
Phase5B failures and IV-A artificial-teacher outcomes remain unchanged.

## Fixed interface and budget

Only real graphs c701/c702: 686 cells, threshold5, 3309/3241 directed edges.
Existing input mapping (fraction .1, amplitude .5), incoming-L1 gain .9,
synchronous tanh update with leak .6, one microstep. Same 48 observed MBONs,
K4 symbols, independently generated iid train/test streams, warmup100,
train2000/test1000, target symbol2 steps in the past. Ten epochs, 20000 reward
opportunities, no early stopping or best-checkpoint selection. Fixed final checkpoint.
Input, codebook, symbol streams and noise draws paired across all three arms.
Readout: fixed nearest-code squared distance, four disjoint groups of12 MBONs at
activation .25, zero elsewhere; seeded before outcomes, zero fitted parameters.
Greedy argmin, lowest index on ties. Labels enter only the environment's binary
correctness calculation after action selection, never the synaptic learner.

## One rule, explicitly specified

At each epoch reset state, postsynaptic mean and eligibility to zero, reward
baseline to .25; retain learned weights. Drive = W previous_state + input.
Add independent Gaussian noise SD .02 to the48 MBON drives during ALL training
steps (including warmup, all arms). Activation a=tanh(drive); state=.4 previous+.6 a.
For each existing KC i -> MBON j edge:

    local_ji = previous_state_i * (a_j - old_post_mean_j)
    eligibility_ji = .8 * old_eligibility_ji + .2 * local_ji
    post_mean_j += .05 * (a_j - post_mean_j)
    advantage = scalar_reward - old_reward_baseline
    delta_ji = advantage * eligibility_ji / (1 + sum_KC_inputs previous_state_i^2)

Warmup updates activity means but clears eligibility after each step and neither
updates weights nor reward baseline. After each training action, replace plastic
magnitude by max(1e-4 original_magnitude, current_magnitude + .05 sign * delta).
Rescale each postsynaptic plastic row back to its original absolute mass; retain
original edge signs and all nonplastic values. The floor is applied BEFORE row
rescaling, not an absolute lower-bound guarantee afterwards. Then baseline +=
.05 advantage. This row normalization is a postsynaptic homeostatic constraint,
not purely synapse-autonomous. No tanh derivative, code error, downstream weights
or target class enters eligibility/update. Exploration is local activity noise;
actions themselves are deterministic. Every noise stream is seeded and retained
through manifest identity for exact regeneration. No physiological calibration.

Arms: contingent binary reward(action equals lag2 label); frozen learning-rate0
with identical forward/noise schedule; yoked learns from the contingent arm's
reward sequence circularly shifted1000 positions WITHIN each epoch. Yoked starts
from original weights and the same noise stream, matches reward counts per epoch
and ignores its own correctness for weight updates. Store actual and applied
rewards separately. This destroys time-aligned reward contingency approximately,
not all possible reward/input correlations; it is not a reward-free control.

## Seeds, evaluation and statistics

Smoke241001/c701,200 train/100 test,one epoch; excluded from main statistics.
Discovery241142–241144; confirmation251142–251144. Train seed=input+1000,
test+2000, code+3000, noise+4000. Bootstrap256399,10000 draws. Config enumerates
all values; prior config numeric-token collisions audited before execution.
3 smoke +18 discovery +18 confirmation=39 training cases; no structural sweeps.
Freeze weights, remove exploration, reset states and recollect train/test features.
Fixed-decoder primary on test lag2. Secondary affine ridge alpha1 (196 parameters
per lag), training-only standardization, same existing shifted-label null, at
lags0,1,2,3,4,5,8,12,16,24,32. No test-set fitting. All scores and symbols saved.
This is continuous-input delayed decoding, NOT autonomous recall, Pi Memory Score
or formal memory capacity. Those recall metrics are not applicable to this task.

Average the two circuits within each paired input-seed block; n=3 per cohort.
Report each block and each circuit, mean/median/sample variance, bootstrap95,
paired differences and paired dz; small n makes intervals descriptive. Never pool
cohorts, select seeds, or use p-values to change interpretation.

Primary reward-coding success requires BOTH cohorts independently:
1. contingent fixed accuracy >= training-frequency baseline +5pp in every block;
2. contingent minus frozen AND contingent minus yoked mean >=5pp, each contrast
   positive in all three blocks.
Failure of any gate means no confirmed reward-coding improvement for this rule.
Secondary representation improvement uses the same paired5pp gates for ridge,
plus frequency/null excess>=5pp every block and mean R²>0. Frozen representation
access separately uses that latter access criterion. Ridge success cannot rescue
fixed-decoder failure. Lag2 choice is informed by historical data and now fixed.

## Completion and failure handling

Reproduce the prior real/aligned c701 s221142 positive case by independent internal
training replay and final trajectory; preserve its checkpoint/hash. Save per-case
initial/final/raw sparse graphs, symbols, input/code mapping, actions, rewards,
advantage/trace histories, final features, scores, diagnostics (rank, activity,
sparsity, norms/cosines), manifests with Git/config/source hashes and environment.
Root runtime/RSS limit7200s/3GiB; deterministic finite workload, no extra trials.
NaNs, invariant failure, resource limits or replay disagreement stop the study,
preserve failure artifacts and investigate implementation before interpreting.
Independent verifier reconstructs every local update and compares final weights,
histories and traces exactly; independently regenerates noise/inputs, verifies
yoking, trajectories, decoder fits/scores, summaries, criteria and old hashes.
Smoke validates engineering only. If primary is negative, report rank/weight/trace
diagnostics from the locked run; no additional learning rates. Any subsequent
mechanistic experiment needs its own question/protocol before outcomes.
