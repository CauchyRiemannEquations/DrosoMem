# P2: real wiring versus matched null ensembles — prospective protocol

Registered **after P1 is fully executed, independently validated and closed**
(`08af2a6`), before any new null graph or neural outcome. The P1 real TDC is
known. This is a prospective null comparison, not an outcome-blind registration
of the entire real/null pair. [Config](../configs/temporal_null_ensemble.json)
and the unchanged [P1 config](../configs/temporal_memory_curve.json) fix the design.

## Question and fixed resource choice

Does real Drosophila wiring preserve historical input information better than
a finite distribution of structurally matched rewired controls, in this rate
model and fixed observation/decoder interface?

Primary: one `legacy5` real graph and **20** independent null graphs, graph seeds
920001–920020. Use all six P1 input/data blocks. Secondary: one `brain5` real
graph and **10** independent null graphs, seeds930001–930010. Use the first three
P1 blocks910001–910003 by numeric order, not by score. Both analyses use circuit701
root mappings. Counts are final regardless of performance; no extra seeds/nulls.

Resource basis: P1 main 905.89s for12 cases, whole-brain case timings
116.69–209.45s and peak process working set285.1MiB; independent replay494.36s.
Earlier 4B whole-brain generation took87.70s and peak sampled RSS1.19GB at one
accepted swap per edge. Two graph workers and four trajectory workers, each
one numerical thread, fit a conservative4GiB process-tree envelope. The larger
partial ensemble is inexpensive; three whole-brain input blocks and ten nulls
bound the secondary cost while addressing the earlier one-null limitation.
These are engineering choices based on time/memory, not P1 accuracy.

## Null construction and audits

Reuse the verified `role_shuffled` double-edge-swap generator. Each null starts
from the original raw graph independently. Request five accepted swaps per
source edge for partial graphs, one per edge for whole brain as in4B. Require
the entire requested acceptance count; no new seed if a graph fails. Preserve:

- every neuron's in-degree and out-degree;
- exact directed pre-role/post-role block edge counts;
- each presynaptic neuron's signed raw outgoing-weight multiset, hence global weights;
- source node/root IDs and roles, no self edge, duplicate pair or zero edge.

Record attempts, accepted swaps, global and per-block overlap, raw/normalized
hashes and incoming-strength changes. Independently reconstruct source graphs
from pinned data, regenerate each null with its fixed RNG sequence, and separately
audit every invariant. No neural outcomes are used in graph generation/auditing.
The finite swap process is not claimed to uniformly sample or fully mix the
entire degree/role-constrained graph space. Highly constrained blocks can retain
source edges; audit that overlap rather than hiding it.

Normalize original and each null independently by incoming-L1 with gain.9.
Raw presynaptic weight multisets stay fixed; normalized outgoing multisets need
not. Thus this is wiring plus the common row-normalization rule, not a pure
individual-synapse-identity intervention.

## Paired TDC and scalar endpoint

Apply the unchanged P1 TDC: iid K10, warmup200, train4,000/test2,000, lag0 sanity
and lag1–20 history; same48 observed MBON root IDs, input seed/mapping, reset
state, leak.6, `mbon_after_kc`, train-only standardization and alpha-1 ridge,
490 affine decoder parameters per lag. No autonomous feedback/internal learning.
Original graph cases are rerun; require their arrays to match the sealed P1
parents exactly. All nulls use the same paired streams and observation interface.

For graph g, the scalar is **the unweighted arithmetic mean of chance-adjusted
accuracy `(accuracy−.1)/.9` over lag1–20 and the fixed input blocks**. No clipping;
every lag/block has equal weight. This is a discrete mean, not an interpolated
AUC or formal memory capacity. Lag0 and the data-dependent baseline maximum
do not enter the scalar; frequency/current-only controls remain in raw reports.

Primary real-wiring advantage PASS iff, for `legacy5`, the real score is
**strictly greater than every one of20 null scores** and real minus null mean
is **≥.03** in chance-adjusted score units (2.7pp mean raw accuracy). Otherwise
FAIL. Equality at .03 passes; ties with a null fail. Compute the gate from exact
integer correct counts/rational arithmetic to avoid rounding ambiguity.
The historical3pp-scale material-effect standard motivates a fixed magnitude
rule; P1's positive access gate does not imply wiring superiority. Whole-brain
same-rule status is secondary and cannot rescue a failed primary endpoint.

Report every graph score, null mean/median/sample SD/min/max, empirical percentile
`100*(number null scores below real + .5*number tied)/N`, and standardized effect
`(real−null mean)/null sample SD` (undefined if SD0). Report paired per-block
scores, each null's differences and every lag's real−null-mean difference.
Ranks/effects are descriptive in this finite ensemble conditional on source,
tasks, mappings and dynamics; no biological-population p-value or equivalence claim.

## Execution, validation and failure preservation

1. Commit protocol/config; implement/commit code before graph generation.
2. Generate all30 nulls and independently regenerate/audit them before smoke.
3. Smoke uses first input block, real and first null of each graph level,
   train300/test200, same warmup/model/lag set. Exclude from main statistics.
   Independently validate smoke before main.
4. Main:126 partial cases and33 whole-brain cases (real included),159 total,
   3,339 lag heads and318 separate train/test streams. Require every case.
5. Independent manual dynamics, label/control reconstruction, augmented least-
   squares refits, exact predictions and rational scalar/gate recalculation.
   Structural verifier has its own source loader and swap replay implementation.

Each generation/run/verification stage has a fixed two-hour/4GiB process-tree
budget. Save per-job runtime, sampled RSS, OS process lifetime peak working set,
50ms sampled aggregate tree RSS, environment, source/protocol/config/input/model/
graph/array hashes, structural audits, raw results and manifests. Worker process
parallelism changes scheduling only; numerical threads remain1. A technical
failure or interruption retains attempt/failure records and partial artifacts,
never counts as a complete experiment or scientific PASS. Same-config retries
use new paths; prior outputs remain immutable. No tolerance, seed, metric,
architecture or null-count changes after null outcomes. No P3 work.

Both outcomes are reportable: a qualifying advantage is limited evidence about
these structures in this computational task; a score within the null ensemble
supports the possibility that generic matched recurrence/roles/weights explain
much of the accessible history. Neither locates biological storage or establishes
physiological learning. Earlier v1 negative results remain unchanged.
