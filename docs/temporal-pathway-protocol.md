# DAN→MBON temporal-pathway intervention: prospective protocol

Registered before new smoke/main outcomes. This study asks whether the
held-out delayed-symbol decoding loss from DAN→MBON removal is dominated by
the **current update into the observed MBONs** or by a changed recurrent
history. It follows the earlier pathway and observation-location results;
their results and artifacts are not changed.

## One question, four crossed states

Use the same input stream and fixed source-derived partial graph in four
counterfactual arms at each time t:

| Arm | State before input t | Weights for update t |
| --- | --- | --- |
| intact | intact history | intact |
| current_only | intact history | DAN→MBON cut |
| history_only | continuously cut history | intact |
| persistent | continuously cut history | DAN→MBON cut |

The current-only and history-only states are **one-step probes**, not
independently propagating networks. Intact/persistent histories each receive
the same externally specified K4 symbols. The cut removes all and only
DAN→MBON normalized entries, without renormalizing other weights. The input
encoder, MBON set, KC comparison set and 48-dimensional ridge budget are
identical across arms. No recurrent weights learn.

Train a separate ridge decoder on each arm's **training stream** and test it
on an independent held-out stream, with lags 1,2,3,4,5,8 primary. This refit
tests information accessible at the same MBON site and decoder capacity, not
just transfer of an intact decoder. A frozen intact ridge applied to each
arm is secondary, as is the selected 48 unstimulated KCs. Selection is by
root-ID order and a fixed seed before seeing activity or accuracy.

## Decision

For each seed block, average six primary lags within each circuit, then two
circuits. Define `loss_arm = intact MBON refit accuracy − arm MBON refit
accuracy` on held-out positions. The **current-path-dominance** criterion
passes only if **both** new cohorts (three seed blocks each) satisfy all:

1. Mean persistent loss at least 5 percentage points and positive in all
   three blocks.
2. Mean current-only loss at least 5 points and positive in all blocks.
3. Mean current-only loss at least 75% of mean persistent loss.
4. Mean history-only loss at most 2 points.

The first two thresholds preserve the earlier practical 5-point magnitude;
75% and 2 points operationalize dominance. The test is a fixed conjunction,
not an equivalence test or a statistical proof of anatomical storage.
Report all signed block differences, means, medians, sample variances, paired
dz and 10,000 seed-block bootstrap percentile intervals. Keep cohorts
separate. Do not change lag weights, controls or thresholds after outcomes.

## Fixed design and scope

The two 686-neuron threshold-5 left MB circuits (701/702), K4 iid input,
warmup 100, train 2,000, test 1,000, gain .9, leak .6, incoming-L1
normalization, `mbon_after_kc` schedule, 48 observed MBONs and alpha-1 ridge
are inherited. Lags 0,1,2,3,4,5,8,12,16,24,32 are saved; the six primary
lags above alone decide. Independent streams and train-only standardization
prevent test labels from entering fitting. Discovery model seeds 720001–720003
and confirmation 721001–721003, with disjoint training, test and KC-selection
seeds in the config. Six blocks × two circuits = 12 paired state panels and
12 × four arms × two sites = 96 refit decoders. Frozen readout and neural
state differences are descriptive secondary outputs.

One excluded smoke panel uses smaller train/test streams. The main and
independent replay budgets are each 3,600 s
and 3 GiB sampled RSS, one process and one numerical thread. Stop on budget,
nonfinite, identity, checksum or replay failure and preserve partial output.
No favorable seed, lag, KC set or post-hoc arm selection.

## Autonomous secondary check

Separately reuse the 16 fresh **random-digit intact** readouts from the
completed coordinate-SD study (eight model/data blocks × two circuits),
without refitting. For each, run target-free autonomous recall with the same
four crossed-state rules, clean observations and horizon 197. Each arm feeds
back its **own generated symbol** and is scored at its first error. This
secondary panel asks whether the temporal pattern is visible in closed-loop
recall; it cannot replace the delayed-probe primary decision. Parent head and
data hashes, generated digits, probabilities and state diagnostics are saved.

## Verification and interpretation

Save source/config/code hashes, edge identities, exact train/test symbols,
the four site trajectories, fit outputs and raw per-lag metrics. Independently
rebuild the cut, replay all four state trajectories using the existing
`TimedReservoir.step` implementation, refit each ridge from saved training
features, reconstruct frozen predictions and all summary contrasts, and
replay every autonomous path. Keep the old research outputs and manifests
unchanged. These are computational counterfactuals on one connectome release,
not living-fly memory, physiological dopamine action, a unique storage site,
or formal memory capacity. A positive result supports current MBON readout
path dominance **within this fixed model and task**; it does not prove that
no state elsewhere carries the information.
