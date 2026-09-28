# KCg annotation-group ablation with exact matched KC controls

Registered before lesion trajectories or performance. ACT III-A, refit-only.
Annotation audit: cell_type prefix `KCg` within role KC selects254/512 KC in
circuit701 and226/512 in702. Names include KCg-m,KCg-d and (701)KCg-s2/s3.
Use this operational definition; no claim that the annotation is a memory circuit.

Question: is loss of driven past-symbol decoding larger after removing the KCg
group than after removing an equal-count, degree/input-exposure-matched KC set?

## Design fixed before outcomes

Real source-derived partial graph only:686neurons,threshold5,c7013309edges/
c7023241edges. No graph shuffle in this study. Conditions: intact, all KCg,
three matched random KC sets. Masks are drawn uniformly without replacement
within EXACT strata (raw incoming degree,raw outgoing degree,four-bit input
membership). For each stratum draw the same number as KCg from ALL KC, including
KCg. This exactly preserves the joint degree/input histogram and counts per
symbol; it does not preserve synaptic strength, removed incident edge count,
spatial position or other annotation. Preserve those imbalances in diagnostics.
Expected target/control overlap in graph-only audit is63–67%; overlapping sets
reduce the anatomical contrast. Do not exclude overlap, optimize masks after
results, or call this a disjoint gamma-versus-nongamma comparison.

RNG loops strata in sorted numeric tuple order; candidates in original neuron-ID
order; choice without replacement. Main mapping61142–61144,train62142–62144,
test63142–63144. Match seed64142+3*block_index+control_index (64142–64150).
Circuit strata701/702 share these seed numbers but masks depend on their graph.
Smoke61001/train62001/test63001,control64001–64003,c701 only,train200/test100.
Main train2000/test1000,100warmup,independent iid K4 streams. 30main+5smoke runs.
No fresh biological specimens. Three mapping blocks are primary sampling units.

Use original incoming-L1 normalization gain.9 BEFORE lesion. Disable masked
neurons by zeroing their incoming/outgoing normalized weights and external input;
start zero state and verify masked states stay zero. Keep surviving weights
exactly unchanged. Do NOT renormalize or remove observation slots. All48MBON
observations and readout size stay fixed. leak.6,mbon_after_kc,input fraction.1,
amplitude.5,all other dynamics from structural_k4. Save intended and effective
input patterns separately; matching preserves removed input count per symbol.

Fit each condition's real and half-train shifted-label null ridge heads anew:
alpha1,source train mean/std floor1e−5,unpenalized intercept,196parameters per
4-output lag head. No frozen-head lesion score in this experiment. No internal
learning. Lags0,1,2,3,4,5,8,12,16,24,32; primary1,2,3,4,5,8.

## Decisions and reporting

Primary paired extra impairment = mean accuracy of three matched controls minus
KCg-lesion accuracy. Average primary lags→controls→circuits per mapping block.
Support disproportionate KCg impairment only if mean extra impairment>=5pp and
all3 blocks>0, plus intact baseline meets access: mean and all3 block margins
over majority AND shifted-null>=5pp and mean R²>0. Report intact−KCg and
intact−matched deficits separately; a lesion deficit alone is not specificity.
Report every control draw,seed,lag,mean,median,sample variance,bootstrap95
(10000 block draws,seed64399),paired dz for differences. No p-value gate.
Never count control masks,circuits or time rows as independent samples.

Keep raw masks/root IDs,annotation/source hashes,stratum audit,overlap,removed
edges/strength,features,labels,real/null weights/scores,predictions,training and
test metrics,neural rank/sparsity/decay,environment/runtime/sampled RSS.
Graph-only matching audits may be viewed before performance; no adapting matches.

First reproduce archived structural_k4 real c701/s34142 complete checkpoint and
metrics. Then smoke5,then all30main regardless result. Exact full replays,
source-only independent augmented least-squares, independent saved-graph/mask/
stream/metric/aggregation verifier. Stop on integrity/nonfinite failure,1800s
or3GiB sampled RSS, retain partial artifacts and never replace seeds. Old files
and numerical modules remain unchanged. Protocol and implementation commit first.

Limits: source-derived partial rate model, refit decodability on independent
driven streams; not autonomous memory capacity,whole-brain benefit,actual fly
learning or causal biological memory circuit. A negative result is inconclusive
about all KCg functions, especially with partial sampling and overlapping controls.
