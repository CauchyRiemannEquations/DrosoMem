# ACT II-A — fixed-length symbol-alphabet comparison

Commit before smoke or scientific outcomes. Follow the matched delayed-symbol
diagnostic; keep ACT III ablations and ACT IV internal learning separate.

## Questions and hypotheses

At fixed length128, how does trained-sequence autonomous recall change for
K=2,4,10,16? Does whole-brain connectivity improve recall within each K when
input/observation and decoder budgets are matched? H1 is scoped trained-prefix
recall beyond weighted simple predictors. H2 is whole-brain superiority within
K. A descriptive endpoint contrast tests sensitivity to alphabet load.
These are computational claims, not living-fly memory or internal learning.

## Locked design

Graphs legacy5 / brain1; source strata701/702. Same original512 eligible KC
root IDs and48 MBON root IDs within each pair. Use the existing immutable graph
loader/dynamics: incoming-L1 gain0.9, leak0.6, mbon_after_kc, input amplitude0.5,
fraction0.1. Legacy graphs have686 neurons with3,309/3,241 edges, threshold5;
brain1 has138,639 neurons/15,091,983 edges, threshold1. The two conditions do
not independently identify graph expansion versus weak-edge effects.

Generate a bank of16 symbol patterns using the historical DigitEncoder rule:
default_rng(model_seed), sequential choice of51 distinct original KCs per
symbol. Use its first K patterns. Shared symbol identities retain exactly the
same stimulation across K and graphs; eligible KC pool is fixed. The union
of actually stimulated KCs can grow with K and is reported, not falsely held
constant. The first10 patterns must match historical encoding byte for byte.

Use existing SequenceDataset('random') with alphabet K, N128, offset0 and
common valid prompt [0,1,0]. Its namespace2201 seeded generator produces iid
suffixes; prefix replacement is explicit. Same dataset seed across K, but
integer draws depend on K, so these are paired/correlated task conditions,
not four independent datasets. Graphs and strata within K receive exactly the
same symbols. K10 with prompt010 is a new anchor, not a reused314 baseline.

Observe48 MBONs only. External readout48→8 tanh→K, no recurrent readout state,
no sequence-position features. Parameters392+9K =410/428/482/536. Architecture
and optimizer are matched within K; parameter count is not constant across K.
Original nonlinear seed namespace9901, initialization0; w1 is identical across
K. K10 initialization and full fitting must match the old head exactly.

Teacher-force127 transitions from reset state. Train-only mean/std, floor1e-5.
Adam2,000 updates, LR0.03, L2=1e-5; fixed first32 generated targets have weight4,
all other transitions weight1. Weighted indices[2,34), total weight223, early
mass128/223, unchanged across K. No new weight, architecture or optimizer sweep.
Reset, supply prompt010, then generate125 symbols using predictions only.
Keep teacher-forced and autonomous accuracies separate. Save integer symbol
arrays (including values10–15); do not concatenate them into ambiguous digits.

## Samples, resources and stopping

Main: model21142–21146 /dataset22142–22146, paired in order, two strata and two
graphs at all four K:80 fits. Compute graphs in20 bounded groups, each group
reusing one verified frozen graph across K2/4/10/16 while resetting state and
initializing a fresh head for every K. Record group graph-build/runtime/RSS and
per-fit runtime; do not sum repeated group resources as independent costs.
One numerical thread. Each group limit1,800s/3GiB, RSS sampled every0.2s.
Stop on engineering failure, preserve partial artifacts, never replace a seed.

Smoke: model21139/dataset22139, stratum701, both graphs/all four K,20 updates;
8 fits, excluded from every scientific estimate. Require complete execution,
saved metrics/checkpoints, exact rollout replay and all8 independent refits.

H2-positive discovery triggers one fresh confirmation at ALL four K:
model23142–23144 /dataset24142–24144, both strata/graphs,48 fits. Only K that
qualified in main can earn confirmed superiority; other confirmation cells
remain descriptive. No confirmation from an isolated favorable seed or lag.
If H2 fails everywhere, stop at representation/decoding diagnostics, no tuning.

## Metrics and analysis, fixed before outcomes

Primary within-K measure: exact autonomous prefix symbols (prompt excluded).
Report L log2(K) as recalled-prefix bits, NOT Shannon information capacity or
formal reservoir memory capacity. Also report fraction of125, full completion,
first error, generated symbols, position/region accuracy, confidence, teacher
accuracy, objective/CE and training loss history. Chance expectation is1/K but
is not itself an autonomous-memory baseline.

Fit weighted majority and first-order Markov controls from the SAME training
symbols/weights. Their autonomous generation also receives only the prompt.
Per run use the larger of the two control prefix scores as conservative control.
H1 per graph/K requires mean prefix>=8 symbols, mean excess over that control
>=5, and positive excess in>=4/5 paired blocks. This is a scoped operational
criterion for trained-instance recall, not universal alphabet generalization.

H2 per K requires mean paired brain1-minus-legacy5 prefix>=5 symbols and
positive differences in>=4/5 blocks. Confirmation requires mean>=5 and all3
fresh differences positive. No new success criterion after observing results.
Average two strata per seed block before inference. Report every raw run,
paired differences, mean, median, sample variance, 95% bootstrap intervals with
10,000 paired-block resamples seed25399, and d_z on differences (null when
delta SD=0). Four K comparisons are exploratory; do not overinterpret p-values.

Descriptive alphabet-load endpoint: per graph K16-minus-K2 exact-prefix
fraction<=−0.10 in mean and negative in>=4/5 blocks. Report all intermediate K
and all bit scores even if the endpoint criterion fails. Do not turn this curve
into intrinsic capacity: fixed symbol length changes nominal information load,
input-pattern union and output-layer size with K.

Neural diagnostics: active neurons, MBON activity, sparsity, adjacent-state
cosine, singular values/effective rank, decay after silent input. Save graph
identity, edge threshold, source hashes, dataset identity, environment, Git
commit, scripts, configuration hash and complete per-seed raw artifacts.

## Validation and provenance

Before main reproduce the existing N128 first-block11142/11242/stratum701
baseline in both graphs, including independent exact refit. Then use the new
K10 adapter on those same314-prompt data to require exact patterns, features,
head parameters, probabilities and rollout equivalence. This compatibility
check is separate from the new010-prompt study and excluded from estimates.

Tests must cover nested/equal-cardinality symbol coding, support of symbol15,
K10 legacy parity, actual K2/4/16 training gradients, out-of-range labels and
symbol bounds, parameter counts and prompt validity. Retain existing tests.
After main independently regenerate every graph/feature trajectory and replay
all80 heads. Independently refit first block/stratum701 at every K and graph
(8 fits). Confirmation repeats48 heads with8 corresponding refits. Compare
saved arrays, generated symbols/probabilities, controls and losses exactly;
single-thread analysis is mandatory. No relaxed tolerance to hide mismatches.

Keep all prior result files/protocols/numerical modules unchanged. New adapters
reuse old implementations, with script hashes recorded as additional provenance.
End at the finite cohort and its registered confirmation gate; choose one next
experiment after interpreting positives and negatives together.
