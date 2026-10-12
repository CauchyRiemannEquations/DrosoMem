# M8: structural-specificity control construction — prospective protocol

Date: 2026-10-12. Commit before new graph/prototype outcomes.
[Prior audit](structural-specificity-design-prior-audit.md),
[fixed config](../configs/structural_specificity_design.json).
Baseline main `4185b214ae831499b556764fa288295b1dbd917b`;
new namespace `results/structural_specificity_design_v1/`.

## Question and finite endpoint

Can we construct nontrivially different recurrent controls preserving both
raw weight multisets, degrees, roles and the exact normalized current KC→MBON
block, plus an explicitly engineered current/lag2 delay reference within the
same scalar state-count/48-observation/490-affine-head budget?

This is a design preflight, not a real-wiring advantage, memory performance,
noise robustness or cycle-attribution experiment. No new source neural
trajectories, iid train/test streams or fitted readouts. Old P2/M7 FAIL,
M1 assay-invalid and M4 INFEASIBLE remain unchanged; P3 is excluded.

Primary READY requires ALL ten partial graph constructions to preserve every
specified invariant, complete the exact swap budget, retain at most90% of
original delayed edge locations, and have distinct raw graph hashes; ALL six
fixed input maps must pass the numerical delay-reference rank/budget checks.
Whole's three graphs/firstthree mappings have a separate secondary READY rule.
Otherwise NOT_READY: this finite construction/certificate is insufficient,
not a universal impossibility theorem. Smoke has no scientific READY endpoint.
No extra seed, looser overlap, different coordinate split, proposal budget,
rank threshold or hidden neural test is added after an unfavorable outcome.

## Locked equal-weight delayed-edge rewiring

Source circuit701 legacy5/brain5, original signed raw weights, source node IDs
and roles. Matrices are W[target,source]. All KC→MBON edges are protected
(zero-delay under mbon_after_kc); every other edge is eligible delayed input.
From original CSR order, group eligible edges by
`(presynaptic role,postsynaptic role,exact signed raw weight)`.
Keep proposal groups having at least two distinct sources AND targets.
The sorted original edge indices in these groups are the proposal pool.
This deterministic filter is fixed before any graph outcome; full eligible
edge count, not pool size, defines the requested swap budget.

For each fixed seed start independently at the original graph. PCG64 default
generator: choose a pool index uniformly, then a uniform index in its original
group. Reject equal sources/targets, newly created self edges or existing
endpoint pairs. Otherwise exchange postsynaptic endpoints, leaving weights
attached to the unchanged sources. Proposal grouping stays fixed; endpoint
multisets within groups stay fixed. Request5 accepted swaps per delayed source
edge partial,1 per delayed edge whole; at most200 proposals per requested swap.
An empty pool or exhausted finite budget is NOT_READY, with partial graph/log
retained. A malformed invariant or execution/resource failure is technical
validation failure, never scientific READY.

Seeds1840001–1840010 partial,1841001–1841003 whole. Counts are final, based on
prior P2/4B measured runtime/memory. One graph worker/one numeric thread;
7200s/4GiB per construction/verification stage. No uniform/mixed graph-space
sampling claim; report protected/eligible/proposable counts and all overlaps.

Independently verify exact node in/out degrees, per-node in/out signed RAW
weight multisets, directed role blocks, source IDs/roles, no duplicate/self/zero
edges, exact protected raw block and original/null incoming absolute strengths.
Incoming-L1 gain0.9 factors must match exactly; normalized protected coefficients
must match exactly and nonempty normalized rows retain gain0.9 within1e-14.
Normalized outgoing weight multisets are NOT constrained. Delayed path/cycle
organization can change; neither acyclicity nor all-lag support equality is
required. This tests a stricter conditional ensemble than P2, not a pure cycle
control or all connectome structure.

Define delayed-edge overlap as
`(common endpoint pairs - protected edge count)/(source edges - protected edges)`.
Require exact integer `10*common_delayed<=9*delayed_edges` for every graph.
Weights can move between existing locations without changing this topological
overlap; report raw/normalized matrix hashes separately. Accepted swap count
is an engineering budget, not a guarantee of ensemble independence/mixing.

## Engineered minimal-history reference

Input maps1810001–1810006; whole firstthree, paired on the same512 source KC
pool (51 per K10 symbol at amplitude0.5). Same48 source MBON observation labels.
Derive ten unperturbed zero-history current codes analytically:
`P(s)=0.6*tanh(W[observed,KC] @ (0.6*tanh(u_s[KC])))`.
No source-state stepping or history trajectory is used. This response is
identical for every valid null because protected normalized coefficients match.

Sort observed root IDs numerically. Reserve first24 code coordinates for
current input and last24 for lag2; fix the split before new code/rank outcomes.
Reference state has512 instantaneous KC activation slots,24 current slots,
and THREE banks of24 past-code slots: bank0 receives past-channel code at t,
bank1 copies previous bank0, bank2 copies previous bank1. Total608 active slots,
pad idle coordinates to original N (686/138639). Observe current24+bank2's24,
restored to the original48 source-observation coordinate order.

The fixed transition has48 unit shift edges, no cycles, D² with24 edges and
D³=0. Thus, after initial transients, observations contain current code at t
and past code at t−2. Current access is explicitly reserved, not sacrificed
to an all-lag2 observation. Padding does not make its coordinate identities,
roles, degrees, weights, gain or dynamics equivalent to source neurons.
This is a different engineered computational system, not a rewired biological
graph or a same-node DAG preserving source all-lag dependencies. Input static
front end includes the fixed source KC→MBON map; it is not learned or free
extra memory, and512 stimulus slots are counted in608.

For EACH current24 and past24 code table, concatenate a constant1 column;
require numerical row rank10, counting singular values strictly greater than
`1e-12*largest singular value`. This is a sufficient numerical affine-code
certificate at the computed float64 prototypes, not a measured ridge accuracy.
Independently recompute rank with a separate SVD driver. Full row rank permits
an affine ten-output map using at most the fixed490-parameter budget; no map
is fitted, no target sequence is used and finite-sample alpha1 training success
is NOT inferred. Failed sufficient rank does not prove nonseparability.
Record singular spectra/ratios, exact layout, padded N, input/observation IDs,
prototypes, normalization, D hashes and nilpotency checks.

## Order, provenance and independent validation

Audit current repo/history; commit protocol/config; implement/commit stable
source; seal baseline identity; synthetic safeguards; one-primary-graph/one-map
smoke and independent smoke; all13 constructions/9 maps; independent main;
results/synthesis, four overview updates, final artifact audit and push.
Smoke repeats first fixed graph with the SAME complete swap budget; it is
not another ensemble realization and has null scientific endpoint.

Independent verifier never imports the main graph generator, normalizer,
prototype/rank/layout or summary algorithms. It reuses only old independent
raw-file loaders/whole-source checks plus generic IO/resource/hash helpers;
regenerates each RNG proposal/acceptance sequence, raw/normalized matrices,
both multisets/degrees/role blocks/protected coefficients, root-based input
maps and analytic prototypes, independent SVD ranks, delay layout/nilpotency
and exact integer READY/NOT_READY conjunction. Float comparisons1e-10 do not
loosen structural overlap or rank decision thresholds.

Fresh exist_ok=False output paths; stable clean HEAD at capture/end; all
tracked source hashes authenticated at recorded Git revisions, pinned ignored
input hashes and protected baseline Git identities. Archive config/protocol/
seeds/environment/runtime/per-process OS peaks/50ms aggregate RSS, graph/data/
model hashes, raw graphs/audits/spectra/certificates/summaries/manifests/checks.
Retain failures and technical retries separately, never overwrite old results.
READY authorizes designing a later performance protocol only; no new
historical decoding superiority or biological claim follows this preflight.
