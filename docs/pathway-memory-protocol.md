# ACT III-D: prospective pathway sensitivity and specificity map

Bounded closeout of five annotated directed pathways in the current c701/c702
partial models. No new task outcomes viewed before protocol lock. III-C did not
establish intact-wiring superiority; this does not imply every pathway is equally
important. We distinguish sensitivity to removing a pathway from disproportionate
sensitivity relative to matched cuts. Neither identifies a biological memory circuit.

## Pathways and primary hypothesis

Five predeclared blocks: KC->MBON, DAN->KC, DAN->MBON, KC->APL, APL->KC.
This captures direct readout input and selected DAN/APL routes; it is NOT an
exhaustive search of all14 nonempty role blocks, individual neurons, or motifs.
Primary family DAN_MBON refit, motivated by earlier DAN lesion sensitivity.
Other families are exploratory secondary endpoints with identical decision rules.
Frozen-source-head sensitivity is separate and cannot establish a memory-critical
representation. An eligible refit family that passes both cohorts is a candidate
memory-critical pathway IN THIS COMPUTATIONAL MODEL, relative to these controls.
No claim of minimal/unique subnetwork, learned synapses or whole-brain localization.

## Graph-only matching and identifiability

For every target remove ALL edges in the directed role block. Controls preserve
EXACT edge count and negative-edge count. Match total raw absolute weight AND
total original incoming-L1-normalized absolute weight within0.5% relative error;
a1e-7 relative guard accommodates solver arithmetic. Neither the full weight
histogram nor per-neuron incoming/outgoing strength/degree is matched.
No post-cut renormalization; same neuron IDs, input and observation slots.

Binary mixed-integer program over all source edges first minimizes number of
selected target edges, subject to count/sign and two mass constraints. Require
solver status optimal and zero optimality gap, integer primal audit; save incumbent
and dual bound. Graph pilot minimum overlaps are KC_MBON1014/1474,1018/1431;
DAN_KC0/24,0/20; DAN_MBON0/189 in both; KC_APL466/512 both;
APL_KC460/512,464/512. This was selected using graph data, not memory outcomes.
Thus the two DAN controls are disjoint, KC_MBON unavoidably overlaps~69–71%,
and both APL routes overlap~90–91%. Never describe all controls as disjoint.

At the minimum overlap, generate3 controls by minimizing independently seeded
uniform(-1,1) edge costs. Require solver success with gap<=0.001, all actual
count/mass/overlap constraints checked. This is a RANDOM-COST optimization sampler,
NOT a uniform sample over matched edge subsets. No rejection by neural outcome.
Solver time cap60s per solve; fail and preserve failures if not solved. All masks,
solver diagnostics and hashes are generated/sealed/committed BEFORE task outcomes.
Independent verifier rechecks constraints and separately asks whether a solution
with overlap<=minimum-1 exists; require infeasible for positive minimum. Zero is
trivially minimal. The same MILP backend is used; not an independent solver proof.

Specificity eligibility is fixed by BOTH circuits' minimum overlap <=80%.
KC_MBON,DAN_KC,DAN_MBON eligible; KC_APL/APL_KC are sensitivity-only because
matching leaves inadequate structural contrast. Still execute/report their controls,
raw differences and all seeds. Do not call an ineligible route specific even if its
numeric gate passes.80% is a pragmatic declared screen, not a biological threshold.
This mass matching replaces prior overly restrictive joint weight bins; do not pool
with III-B to infer a stronger result. Source graph pilot showed joint-bin overlap
as high as~85–98% for direct/APL paths before the mass-matched design was chosen.

## Fixed task and decision rules

Same K4 iid train/test delayed-symbol task, source threshold5,686neurons,
3309/3241edges,48MBON observations. Input fraction0.1,amplitude0.5,gain0.9,
leak0.6,mbon_after_kc. Warmup100,train2000,test1000; smoke200/100.
Lags0,1,2,3,4,5,8,12,16,24,32; primary1,2,3,4,5,8.
Ridgealpha1, train-only moments,std floor1e-5,196 parameters per lag.
Train and test streams independent; all paired arms use identical maps and symbols.
Frozen source moments/head remain fixed; refit trains a separate external decoder.
This is not pi autonomous recall or a formal memory-capacity measure.

Each family: mean intact-target>=5pp AND control-target>=5pp, BOTH positive in
all3 paired seed blocks, intact frequency/null excess>=5pp per block and mean
R2>0. Both discovery AND fresh confirmation must satisfy the same mode's gate.
Specificity additionally requires graph eligibility above. Primary DAN_MBON refit;
other eligible refit candidates labelled exploratory even if independently confirmed.
A sensitivity-only gate (intact-target>=5pp, all3 positive, intact access) is also
reported separately; it does not require a specific matched-control effect.
Frozen confirmed effects describe decoder sensitivity, never critical representation.
No p-value selection, no pooling cohorts, no changing5pp or80% after outcomes.

21arms=1 intact+5*(target+3controls). Smoke21,discovery126,confirmation126:
273 fits/exact replays and260 frozen evaluations. Confirmation always runs.
Discovery181142–181144,confirmation191142–191144; full map/data/control seeds
in config. Three control draws and two circuits averaged inside each block;
statistical n=3 per cohort, not masks,cells,lags or graph subsets.
Per-seed raw data,paired differences,mean,median,sample variance,10k block bootstrap
(seed194399) and paired dz saved. Small-n intervals descriptive. Input/output budgets,
all hyperparameters and update procedure unchanged across targets/controls.

## Validation, artifacts and stopping

Replay historical intact baseline. Numerical smoke before full cohorts. Every case
full repeat, independent augmented-LS refit and frozen checks; independently
reconstruct train/test state dynamics and direct metrics/statistics/gates.
Graph cuts verified against precommitted mask bank; source/old result hashes
unchanged. Save symbols/features/heads/scores,weights,masks,neural rank/sparsity/
activity/decay,per-pathway sensitivity map,role-network diagram,environment,
config hash,Git revision,runtime,sampled peak RSS and all source/artifact paths.
Resource cap3600s/3GiB per command; graph generation has per-solve60s cap.
On integrity failure preserve output and stop before repair. No tuning to positive
outcomes. Stop after complete273-case panel, fresh confirmation, independent audit,
report and artifacts. No automatic extra sweep or ACT IV. If no eligible refit
candidate confirms, explicitly report that no disproportionate critical pathway
was identified by this bounded design; do not nominate the best seed or frozen hit.
