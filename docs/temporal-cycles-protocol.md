# M3: cycle-free delayed feedforward sufficiency — prospective protocol

Date: 2026-10-08. Baseline `eb571bd83107e2b108bf270003a918f8eb21e797`.
[Fixed config](../configs/temporal_cycles.json), namespace `results/tdc_cycles_v1/`.
Commit before any M3 neural trajectory/decoding outcome. Structural graph checks
may precede registration; they never fit readouts or evaluate neural performance.

## Question and interpretation fixed before outcomes

Can a source-derived **acyclic** graph with **no direct state carry** retain
decodable iid past-input information through delayed feedforward transmission?
The next-work cycle question is narrowed to sufficiency, because removing cycles
also removes degrees, weights and paths. An intact−DAG performance difference
does not uniquely quantify a cycle effect. Positive demonstrates sufficiency
under the registered retained graph/model/readout; negative does not establish
cycle necessity. No new hypothesis rescues M1 or the earlier wiring gates.

M1 remains assay-invalid, M2's specific joint decoding result remains closed,
and all earlier positive/negative evidence is immutable. This is a prospective
question informed by those studies, using fresh fixed mappings/streams.
No physiological/spiking/plasticity P3, neuron copies, unrolling or null-graph
superiority analysis is included.

## Graph intervention and mathematical controls

Keep the source node coordinates and fixed48 observed MBONs. Order all KC
neurons first, other roles next, MBONs last; within each tier use ascending
numeric source root ID. Retain only edges `rank(pre)<rank(post)`. Every retained
edge strictly increases rank, proving no directed cycles or self-edges.
This artificial ordering is not a biological hierarchy and is not optimized.

Start with the original gain.9 incoming-L1 normalized W, mask it, and do not
renormalize afterward. Preserve the entire same-step KC-to-MBON block, including
its exact coefficients. Input encoder/amplitude.5/fraction.1 and source IDs
are unchanged. The topology-only audit gives legacy5:3309→2500 edges, protected
1474; brain5:2700513→1353524 edges, protected21438. Per-neuron degree/weight mass,
role-block flows, SCCs and rank proofs are saved, including the removal confound.
Cross-SCC-only was rejected before neural outcomes because it destroys every
protected current input path; restoring that block still collapses delayed paths.

| Arm | Graph | Direct carry | Prior-state synaptic drive |
| --- | --- | --- | --- |
| intact_synaptic | Original source-derived W | 0 | on |
| dag_synaptic | Fixed acyclic subset of original W | 0 | on |
| instantaneous | Original W; protected current block only active | 0 | off |

Reuse the validated M1 factor update, with drive `b=.6` in every arm. Carry is
explicitly zero; do not change leak to1. With selected W and KC-only stimulus u,
`q_t=.6*tanh(W*x_(t-1)+u_t)` for synaptic arms; replace MBON coordinates with
`.6*tanh(W[MBON,:]*v_t+u_t[MBON])`, where v uses updated KC and previous other
coordinates. Instantaneous uses zero prior-state drive but the same current
KC-to-MBON block. All arms have identical zero-state current responses.
Intact_synaptic is M1's zero-carry synaptic arm, not its full carry-on model.

Under this schedule, KC→MBON edges have zero symbol-step delay and other edges
delay1. DAG longest weighted path from ANY initial coordinate is H=11 partial /
153 whole; topological recurrence certifies finite dependency. These are upper
bounds, not measured biological forgetting times or decoding capacities.
The bound from all KCs may be smaller and is not the actual selected-input bound.

For each DAG block, start two seeded Gaussian initial vectors and independently
generated64-symbol prefixes, followed by the same seeded suffix of length H+1
(12/154). Save both inputs/initial/prefix-final/end states and complete trajectory
hashes. After that suffix **all state coordinates must agree exactly**. Distinct
prefix-final states document that the probe exercises earlier-history differences.
Topology proves the general bound; this finite probe checks its implementation.
Instantaneous has its independent current-symbol prototype/no-history certificate
at every coordinate/time. Neither certificate relies on empirical chance scores.

## Fixed data, decoder and prospective primary

Primary legacy5 circuit701,686 neurons: six blocks1300001–1300006.
Secondary brain5,138639 neurons: first three paired blocks. Budget derives from
measured M2 costs; no result-dependent graph or seed additions.
Mapping1310001–1310006; independent train1320001–1320006 and test1330001–1330006.
Certificate initial-A1350001+i /initial-B1350101+i, prefix-A1360001+i /
prefix-B1360101+i, suffix1370001+i, i=0..5, fixed in the config.

Uniform iid K10, independent train/test state resets, warmup200 (greater than
both conservative DAG bounds), train4000/test2000 scored rows per lag0–20.
After incorporating input t, fit `state(t)->symbol(t-lag)`; labels remain in
their independent split. All arms/levels pair streams/source input IDs/observed
root IDs. Fixed alpha1 affine ridge, train-only mean/SD floor1e-5,490 coefficients
and intercepts per lag. Only external readouts learn. Chance10%, train-only
frequency and current-symbol-only lookup baselines accompany every lag.

At **lag2**, define each DAG block's excess as accuracy minus
`max(.10, frequency accuracy, current-symbol-only accuracy)`.
Primary PASS requires excess **>=.10 in every one of the six partial blocks**.
This reproduces the previous short-lag10pp material scale, selected before any
new neural result. Use integer-count Fractions, exact equality passes. Failure
in any block means FAIL; no alternate lag, lower threshold or added seed.
Whole firstthree uses the same rule as a separate secondary endpoint and cannot
rescue primary. All other lag results and intact−DAG differences are descriptive.

Assay eligibility requires exact graph/rank/protected-input and zero-state
certificates, independent trajectory/label/readout/prediction/criterion validation,
the finite dependency probe, and instantaneous lag0 accuracy>=.99 in every
relevant cohort block. DAG/intact current accuracies are outcomes, not M1's
all-arm90% eligibility requirement. If reference validity fails, scientific
endpoint is undefined/assay-invalid; if technical replay fails, preserve
validation-failed evidence and do not confirm any endpoint. M1 remains unchanged.

## Reporting, resources and execution order

Save all raw lag counts/predictions/scores/labels/features, train moments and
coefficients, full final-state vectors/trajectory hashes; original and masked
graph hashes plus rank/per-neuron audits; config/protocol/seeds/environment/source
commit/model/data hashes and manifests. Report all paired values, mean/median/
sample SD, chance-adjusted accuracy and strongest-baseline excess, and10,000-draw
paired bootstrap2.5–97.5% intervals, seed1340001. Blocks are computational inputs,
not animal replicates. All lag0–20 curves remain unclipped and unconstrained;
do not call TDC formal Shannon/reservoir memory capacity.

Main:27 arm/graph/block cases,54 streams,567 lag heads and9 DAG certificate
pairs. Smoke: first block/both graphs/three arms, train300/test200, warmup200;
6 cases/12 streams/126 heads,2 certificate pairs, no scientific gate.
Four workers, one numerical thread each, max7200 seconds/4GiB aggregate
process-tree RSS per stage, with parent lifetime RSS/OS peak and50ms tree samples.
Shared pages may be counted twice; report limitations of memory measurement.
Coordinate-first prototype indexing preserves the fixed budget.

Audit/seal43,495 prior result-file Git identities and old scientific code/data/
config/protocol/tests; register protocol/config and structure-only audit first.
Implement new runner and independent graph-mask/dynamics/fit/criterion verifier;
numerical safeguard tests; smoke; independent smoke validation; main;
independent main validation; results/synthesis and overview/roadmap/handoff;
final artifact/source/history audit; commit and push verified work.
Never overwrite existing paths. Keep failed/interrupted attempts and original
source; technical retries use identical scientific settings and fresh paths.
Arrays atol/rtol1e-9, scalars1e-12, exact predictions/count gates and certificates;
no post-outcome tolerance change or success-seeking tuning.

Any result closes this finite question. It establishes no biological storage,
physiological learning, cycle necessity, unique cycle contribution, real/whole
wiring superiority or formal capacity. Further matched cycle-attribution work
would require a separate prospective design and validity argument.
