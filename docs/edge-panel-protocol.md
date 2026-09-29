# ACT III-B: edge ablation preregistration

Locked before new state trajectories/outcomes. Baseline revision fc20e23;
machine-readable config `configs/edge_panel.json`, seed/graph audits alongside.
Question: which edge sets affect accessible past-symbol information beyond
matched removal count, and does DAN-associated impairment survive weight-bin
matching? Frozen and refit decoding are different questions. No internal learning.

## Scope and execution order

Complete a bounded edge-ablation panel on the existing c701/c702 686-neuron
threshold5 MB rate models: DAN-associated; weakest; strongest; high betweenness;
within-role; between-role; random controls. These cover the ACT III-B edge
categories in the current model. Roles KC/MBON/DAN/APL are operational partitions,
not inferred anatomical modules. Whole-brain ablation, detected community modules,
autonomous recall and exhaustive dose-response curves remain open extensions.
Do not move to III-C/D/IV in this study.

Commit protocol/config first; implementation next; smoke17 conditions, discovery
102, then confirmation102 regardless of discovery outcomes. Independent
verification, figures, all-seed report and scope closeout complete the stage.
No tuning, seed substitution, threshold changes or optional extra conditions.
Time limit3600s and sampled RSS3GiB per runner/verifier. Integrity failures stop;
preserve failed artifacts and repair in a new revision/output path. Negative
scientific results do not stop the registered sequence.

## Fixed task and model

Keep all686 neurons, unchanged direct input patterns and48 MBON observation
slots. Normalize incoming L1 on intact graph once (gain0.9), remove selected
normalized entries, never renormalize. leak0.6,mbon_after_kc schedule. K4 seeded
random inputs;100 warmup then2000 train/1000 independent test (smoke200/100).
Input fraction0.1/amplitude0.5. Ridge alpha1; train-only moments/std floor1e-5;
196 coefficients per lag. Frozen uses intact source moments/real and null heads
unchanged. Refit uses lesion training states. No target labels enter frozen fit.
Null is half-train cyclic labels; frequency baseline from training labels.
Lags0,1,2,3,4,5,8,12,16,24,32; primary average1,2,3,4,5,8. This is delayed-symbol
decoding, not autonomous recall or formal memory capacity. Scores not probabilities.

## Edge definition, matching and graph-only choices

One edge is one nonzero aggregated directed neuron-pair entry, row=postsynaptic,
column=presynaptic. Canonical enumeration is sorted CSR row then column.
DAN target removes every entry incident to a DAN role neuron:552 edges c701,
524 c702. Three controls sample the same counts in EACH raw sign and absolute
weight bin: [5,10),[10,20),[20,50),[50,100),[100,infinity). No replacement within
a mask. The pool includes all edges, including DAN target edges; draws may overlap.
Disjoint matching is infeasible in several audited strata, so report overlap and
do not call controls non-DAN. Exact strength or normalized-weight matching is
NOT claimed; report raw/normalized removed L1 and endpoint role counts.

Other families remove floor(0.05*E):165/162 edges. This count fits the sparse
within-role pool217/220. Weak/strong rank absolute RAW weights ascending/descending.
High betweenness uses exact unweighted directed shortest-path edge betweenness
on binary intact connectivity, ignoring sign/weight as path costs. For all ranked
ties sort numeric (presynaptic root ID,postsynaptic root ID). No outcomes used.
Three uniform random masks sample all edges. Three within-role masks sample
edges with equal endpoint roles; three between-role masks sample unequal roles.
All use no replacement within a mask, with cross-mask overlap allowed.
Uniform controls match COUNT ONLY; sign/weight/direction changes are reported,
not interpreted as isolated topological effects. No graph rewiring occurs.

17 conditions: intact,DAN,3 signed-bin DAN controls,weak,strong,betweenness,
3 uniform,3 within-role,3 between-role. 13 circuit/block combinations including
smoke =>221 fits/replays and208 frozen evaluations. Every main condition uses
both circuits and every seed; within/between/control masks averaged within block.
Discovery seeds121142–121144,train122142–122144,test123142–123144;
confirmation131142–131144,132142–132144,133142–133144. Full RNG seeds in config.
Same biological source graphs, fresh computational cohorts, no biological n=6 claim.

## Hypotheses and decisions

Primary H-DAN-refit: intact minus DAN and mean signed-bin-controls minus DAN
must EACH have mean >=5pp and be >0 in all3 seed blocks, with intact frequency/
cyclic-null margins >=5pp in every block and positive mean R2. Require this same
refit gate in both discovery AND confirmation. No pooling to rescue a failure.
Secondary DAN-frozen and each other family/mode use the same gate, comparing
target/mask-family average against mean uniform controls. Frozen/refit are never
mixed across cohorts. Predeclare within-minus-between accuracy as a descriptive
paired contrast, without inventing a post-hoc successful direction. No formal
multiple-comparison significance claims; this is exploratory with small n.
Failure of a5pp gate is not equivalence or zero effect. Negative R2 is retained.

## Baseline, engineering and validation

Replay the archived structural K4 intact baseline and archived DAN neuron lesion
on the old seed; because DAN has no external input, incident-edge deletion with
zero initialization should match the old full-neuron lesion exactly. Verify this
prediction; do not count it as a new independent cohort.
Save every edge mask/IDs, original graph, normalized weights, betweenness,
checkpoint/state features, real/null predictions, losses, per-lag accuracy/R2,
neural rank/sparsity/MBON norm/active count/temporal cosine/decay, graph counts,
removed raw and normalized strength, DAN overlap, endpoint role composition.
Record config/git/data/source hashes, environment, runtime and sampled memory.
Require used source bytes to match committed Git bytes before execution; preserve
old source snapshots and outputs unchanged. Historical artifacts stay pinned.

Each case fully replayed; independent augmented least-squares real/null refits;
frozen source-only least-squares predictions independently checked. Independent
verifier regenerates masks/weights and trajectories, recomputes metrics/statistics.
Betweenness ranking verified independently from shortest-path counts, including
synthetic graphs with ties, directionality and disconnection.
Statistics: all per-seed rows, paired effects, mean/median/sample variance,
10000 bootstrap resamples of THREE seed blocks, seed134399, paired dz when defined.
Circuits/lags/masks are not independent replicates. Plot lag curves and sensitivity.
Close the current-model III-B panel after all registered checks/reporting, even
if primary or secondary hypotheses fail. Select one next study from the findings.
