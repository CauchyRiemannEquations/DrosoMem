# M4: feasibility of an exact cycle-attribution control

Date: 2026-10-08. Prospective **methodological/structural study**, to be committed
before generating new structural results. Baseline main `5afe8f28cc1d820973b9156c20501a0d31c88a26`.
[Configuration](../configs/cycle_attribution_feasibility.json).
New namespace: `results/cycle_attribution_v1/`. Historical results remain immutable.

## Question and scope

Can a same-neuron acyclic control preserve the source graph's labeled degrees,
role-block edge totals, normalized weight constraints, current KC-to-MBON block
and complete delayed input-to-observation dependency-walk support?

M3 establishes delayed feedforward sufficiency, not a unique cycle contribution.
Its mask deletes degrees, weight mass and paths. This study checks the strict
ideal behind the proposed attribution comparison before spending new neural
compute. The all-lag requirement is deliberately an **exact ideal**, not an
assumption that every scientifically useful restricted comparison must satisfy
it. Finite-lag or approximate controls can pose a distinct future question.

No neural trajectories, iid decoding outcomes, ridge fits, autonomous recalls,
graph ensemble or physiological/plasticity P3 are executed here. If feasibility
is rejected, close this study rather than inventing a weaker control after
results. If no obstruction is found, record UNRESOLVED; absence of a certificate
is not a feasible construction. Any later performance study needs its own
protocol, controls, fresh streams, decoder budget and material endpoint.

## Frozen sources and mappings

Reuse circuit701 partial legacy5 and pinned whole brain5, with source IDs/roles
unchanged. Original signed matrices are incoming-L1 normalized to gain0.9.
Use the exact sealed M3 rank DAG as a **reference for checking certificates and
describing mismatches**, not as an allegedly degree-matched new control.
No normalization after masking. Same48 observed MBON IDs, no neuron copies.
Schedule mbon_after_kc, drive0.6, direct carry0: KC-to-MBON reads updated current
KC (edge delay0); every other edge reads previous state (edge delay1).

Six fresh mapping blocks1400001–1400006, mapping seeds1410001–1410006, are fixed
now. Each symbol independently chooses51 of the same512 KC source neurons
without replacement, using the existing encoder algorithm; amplitude0.5, K10.
Whole secondary uses the firstthree mappings, paired by root ID. No train/test
symbol stream is generated. These are mapping probes, not model performance
replicates or animals. Smoke uses firstone mapping in each level and has no
scientific endpoint. Main uses exactly six/three; no added seeds after results.

## Predefined impossibility certificates

Authenticate all source hashes before construction. Matrices are simple
directed graphs without self-edges, represented as W[target,source]. Degrees
count nonzero edges; signed weights cannot cancel graph-support edges.

For fixed labeled in/out degrees and role-block totals, report these sufficient
obstructions separately:

1. A nonempty DAG, ignoring isolated vertices, must have a nonisolated source
   with in-degree0 and a sink with out-degree0. Missing either forbids a DAG.
2. With M nonisolated vertices, any vertex's in+out degree greater than M−1
   forces a reciprocal neighbor; E>M(M−1)/2 likewise forbids a DAG.
3. A role with n vertices cannot contain more than n(n−1)/2 directed edges
   in a DAG. For distinct roles r,s, E(r→s)+E(s→r)>n_r*n_s forces a reciprocal
   pair, because an unordered cross-role pair can support at most one edge
   without a 2-cycle. These tests use fixed role totals, not actual reciprocal
   edges that rewiring might remove.

Any certificate disproves a DAG under those exact count constraints even if
weights and current access were relaxed. Non-triggering certificates leave
degree/role feasibility UNRESOLVED. Do not claim universal degree-matching
impossibility: include a synthetic DAG/cyclic pair with equal labeled degrees,
incoming/outgoing strengths, presynaptic weight multisets, role totals and
current block to verify this distinction. A seven-node fixture with edge
weights0.3 and all active incoming sums0.9 can satisfy these constraints while
its delayed walk profiles differ; it is not a source-derived connectome control.

Independently, for each of10 symbol input sets in each mapping, find a **directed
positive-delay cycle reachable from a stimulated KC and able to reach an
observed MBON**. A cycle outside this cone does not certify this obstruction.
Selection is deterministic as specified in config. Archive source-ID paths
input→anchor, anchor→...→anchor (cycle), and anchor→observed neuron; every edge
and its scheduled delay must be independently checked.

Let A and B be the two access-path delays, P the cycle delay, and H the reference
DAG's longest scheduled dependency bound starting at any coordinate. Require
P>=1, and choose k=max(1,floor((H−A−B)/P)+1). The original graph has an explicit
walk with delay A+kP+B>H; the reference DAG has none. More generally, repetition
of the relevant cycle produces unbounded delays, while **any finite DAG on the
same N neurons** has a finite longest path (at most N−1 delayed edges). This
certifies that no finite DAG can preserve the complete all-lag structural walk
support, regardless of its rank or its degree/weight matching.

There cannot be a cycle consisting only of delay0 edges: such edges go KC→MBON,
and a subsequent edge from MBON has delay1. A DAG under this schedule has finite
input-history dependence when direct carry is zero. A graph-support walk is
not proof of nonzero nonlinear functional influence, signed transfer gain,
readout accuracy, useful long-term memory or formal capacity. Signed paths can
cancel and tanh/decoders can suppress accessible paths.

## Prospective endpoint and interpretation

For each level, exact-control feasibility is
**INFEASIBLE** if any independently validated degree/role obstruction or any
input-to-observed positive-delay-cycle witness is found. The summary must
separately label `degree_role_status` and `all_lag_path_status`, and list all
mapping/symbol cells including cells without a witness. If no obstruction,
the result is UNRESOLVED, never automatically FEASIBLE. Technical/source or
certificate failures make the study validation-invalid pending an archived,
same-settings technical repair. Smoke endpoint is null.

Partial is primary; whole firstthree is secondary. Report the scope of
certificates (e.g. how many of60/30 symbol cells are obstructed). No threshold,
lag, seed, weight or rank sweep. No inference about whether cycles improve
decoding follows from INFEASIBLE. Validated obstruction is a completed negative
feasibility result, not a successful memory-performance hypothesis.

## Descriptive structural panel

For original versus sealed M3 DAG, archive exact per-neuron in/out degrees,
incoming/outgoing absolute weight sums, role-block counts, source matrix/ID/role
hashes and the protected current-block equality. Report degree/weight
mismatches rather than treating them as matching. Check zero-state observed
responses analytically for all10 symbols at each fresh mapping; equal
zero-state current responses do **not** prove equal current decoding under
history. No current-symbol prediction or neural trajectory is computed.

For lags0..20, record boolean existence of scheduled dependency walks from
each symbol's stimulated neurons to each observed coordinate, separately for
original/reference DAG. Apply delay0 closure to current KC→MBON paths, then
one delayed edge and closure for successive lags. This is a structural
reachability panel, not a new Temporal Decodability Curve. No new performance
gate is defined from these descriptive values.

## Implementation, independent verification and stopping

After this protocol/config commit, implement runner, guards and a separate
verifier. Main may reuse source-derived loaders and M3 graph helper; verifier
reconstructs partial graph from pinned CSV and whole graph/roles from pinned
parquet/annotations, independently masks rank rows, verifies graph/arrays and
per-neuron statistics, checks every witness edge/delay and theorem arithmetic,
recalculates each mapping and finite-lag support panel without importing the
runner or its certificate/summary algorithms. No rerun of historical neural
studies is claimed. Preserve all historical result Git identities including
sparse paths, and verify materialized prior manifests/source hashes.

Synthetic safeguards cover relevant versus disconnected/unobservable cycles,
delay0 schedule, DAG history bound, degree/role obstructions versus the matched
feasible counterexample, invalid witness edges/endpoints/arithmetic, no inference
from absent obstruction, smoke-null and strict stopping/summary semantics.
Run smoke, independent smoke verification, then fixed main and independent
main verification. Archive config/source/environment, source commit/protocol
commit, mapping seeds, graph/data/array hashes, raw structural metrics,
witnesses, resources, manifests and checks for every stage. Use fresh directories
with exist_ok=False; every interruption/failure is retained, never overwritten.

Each stage uses one numerical thread,7200 seconds and4GiB maximum process RSS
including OS-reported peak where available;50ms sampled RSS is an estimate.
No adaptive increase of budget or certificate set. Finish with independent
artifact/provenance audit, synthesis and updates to the four current overview
documents. Historical protocols, conclusions, raw results and negative outcomes
remain unchanged, including M1 assay-invalid, P2 failed material gates and M3's
limited sufficiency conclusion.
