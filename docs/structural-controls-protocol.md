# ACT III-C: which structural properties support delayed-symbol decoding?

Prospective bounded closeout of structural controls in the current two partial
models. No new task outcomes inspected before this protocol/code commit.
III-B found a raw-bin DAN excess near7pp, but normalized-strength controls
failed its5pp criterion. We therefore preserve incoming weights, rather than
selecting a favorable new dynamics setting. Read the III-B report for overlap
and different-cohort limitations. This experiment is about representation and
external decoding, not internal synaptic learning or autonomous recall.

## Hypothesis and primary comparison

Primary: intact wiring exceeds role/signed-degree/incoming-weight-preserving
rewiring by mean >=0.05 test accuracy, with positive paired differences in all
three mapping/sequence blocks, in BOTH discovery and fresh confirmation.
Intact must have frequency and cyclic-null excess >=0.05 in each block and
mean R2 versus frequency >0. Primary mode is refit. Frozen is a separate
secondary sensitivity question. Other three controls are secondary comparisons;
no promoting them to primary after outcomes. The same gates are reported for
secondary modes/families, with no p-value selection. Failure is failure to meet
this material-effect criterion, not equivalence or absence of smaller effects.
Both cohorts run regardless of discovery; do not pool them for a pass.

## Four controls and preserved properties

All controls retain all686 neuron IDs/role annotations and3309/3241 edges,
threshold5 source weights, with no self loops or duplicate directed edges.
Three independent draws per family, paired between c701/c702 using identical
RNG seeds (different source graphs). Raw signed weight multiset always retained.

| Family | Preserved | Deliberately not preserved |
|---|---|---|
| random | N,E,global signed weights | Per-neuron degrees/strengths, role-block wiring, endpoints |
| degree | Signed in/out degree at every neuron; each post-neuron incoming signed weight multiset | Role mixing, endpoints, outgoing weight multiset/strength |
| role (PRIMARY) | All degree properties plus per-neuron signed degrees to/from every role | Exact partners/motifs, outgoing weight multiset/strength |
| weight | Exact adjacency and sign at each edge; each post-neuron incoming signed weight multiset | Assignment of magnitudes to presynaptic partners, outgoing strengths |

Random: sample E distinct non-self directed positions uniformly, then permute
raw signed weights globally. This is a deliberately weaker matched control,
not a biologically plausible model or a pure topology-isolation test.
Degree/role: double-edge swaps of presynaptic endpoints; weights stay with each
postsynaptic edge slot. Select first edge uniformly, second from same-sign pool;
role also requires same pre-role AND post-role. Reject shared endpoints,
self loops and duplicates. Stop at10*E accepted swaps, fail at200*E proposals;
never relax a constraint or pick a draw by outcome. This finite sampler is NOT
claimed to be a uniform draw from all matched graphs or a proven mixed chain.
Graph-only pilot with seed169001 achieves requested swaps, with roughly43%
degree and50% role edge overlap. Report overlap and preserved invariants for
EVERY draw; do not claim independent/disjoint topologies. Weight permutations
are independently within each postsynaptic neuron/sign group.
All matrices get incoming-L1 gain0.9 normalization. For degree/role/weight,
per-post incoming raw multiset preservation gives IDENTICAL normalization factors
and incoming normalized weight multisets. Random does not preserve these.
Per-neuron outgoing strengths are not matched, including in the primary control.

## Fixed task and budgets

K4 independent uniform random train/test streams; warmup100,train2000,test1000;
lags0,1,2,3,4,5,8,12,16,24,32;primary1,2,3,4,5,8. Small smoke200/100.
No Pi Memory Score endpoint because this is delayed input-symbol decoding,
not autonomous rollout. Same biological graphs as III-B; fresh computational
cohorts are not independent animal samples. Alphabet/length scaling stays archived.
Same paired input maps (fraction0.1,amplitude0.5),48 MBON observations,leak0.6,
mbon_after_kc schedule,train-only feature moments,std floor1e-5,ridgealpha1,
196 parameters per lag. Frozen uses intact source moments/coefficients/bias;
refit trains a separate head; internal connectivity never learns.
13 arms (intact+4*3), smoke13,discovery78,confirmation78:169 fits/full replays,
156 frozen evaluations. Discovery161142–161144,confirmation171142–171144;
full dataset/input/graph seeds and bootstrap174399 fixed in config and seed audit.
Three graph draws averaged inside each block, then circuits/lags averaged equally.
Statistical n=3 per cohort, NOT edges, graphs, timepoints, lags or masks.
Save per-seed values,paired deltas,mean,median,sample variance,10,000 block
bootstrap95 intervals,paired dz where defined. Small-n intervals descriptive.
Save both negative and positive effects; fixed reporting order random/degree/role/weight.

## Execution, validation, failure and stop

Commit protocol/config/graph-only audit before task outcomes, then committed
implementation/tests before executing smoke -> discovery -> confirmation.
Replay archived intact baseline. Save all raw weights, features, symbols, labels,
heads,train/test scores,neural activity/rank/decay,graph overlaps,source hashes,
Git commit,environment,run time and sampled peak RSS in sealed manifests.
Verify exact full reruns and independent augmented least-squares fits. Independent
validator audits preserved graph properties, regenerates deterministic graphs,
rebuilds dynamics via a separate formula and recomputes metrics/statistics/gates.
Graph generator replay is not an independently written random sampler; property
checks and trajectory/statistics checks are separate. Keep this distinction.
Hard resource limits3600s and3GiB per command. Any integrity failure stops work;
preserve the failed directory and error before repair/fresh run. Smoke is excluded
from inference. Negative hypothesis outcomes do not trigger tuning or new seeds.
Stop after169 valid conditions, independent checks, reports and preserved artifacts.
This closes III-C only within current partial-model/delayed-decoding scope.
Whole-brain, physiological validation, autonomous recall and uniform graph-ensemble
sampling remain untested. Do not start III-D or ACT IV in this package.
