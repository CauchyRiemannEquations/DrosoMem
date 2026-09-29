# Observation location after DAN→MBON removal — prospective protocol

Question: does the previously confirmed DAN→MBON decoding deficit remain when
reading48 non-directly-stimulated KCs instead of48 MBONs? This is an ACT IV-A
entry diagnostic of readout dependence, not internal learning or biological memory.
Prior III-D outcomes motivated the question. No NEW task outcomes have been
inspected. Config and graph/input-only selection are committed before execution.

## Fixed design

Use c701/c702 existing686-neuron partial graphs, threshold5,3309/3241 edges.
Original incoming-L1 gain0.9, leak0.6 and mbon_after_kc update schedule remain
unchanged when observation indices change. KC input fraction0.1, amplitude0.5.
For each mapping exclude KCs directly stimulated by ANY of the4 input symbols;
sort candidates by numeric root ID, draw48 without replacement with the declared
observation_seed, then sort selected indices. No state/outcome-dependent selection,
no replacement observation set after failure. Save IDs and selected/candidate counts.
Composition, dynamics, degree and intrinsic predictivity need not match MBONs.
Both observations read the SAME trajectory; neither feeds back into the dynamics.

Five arms: intact,DAN_MBON cut,three precommitted disjoint matched cuts from III-D.
Use the exact archived mask blocks specified in config, regardless of old outcomes;
all3 are retained. They preserve count/sign and raw/original-normalized total mass
within0.5%, not per-neuron strength/degree or weight histograms. Do not regenerate
or retune controls. New input/sequence/observation seeds are separate from III-D;
this is fresh task replication conditional on the previously fixed graph masks.

Refit only: each site's48 features feed its own affine ridge decoder,196 parameters
per lag, alpha1, training-only mean/std, floor1e-5. No frozen transfer between
biologically different feature slots. Internal weights are never learned.
K4 iid independent streams, warmup100,train2000/test1000, lags0,1,2,3,4,5,8,12,16,24,32;
primary mean lags1,2,3,4,5,8. Current lag0 and longer lags are descriptive.
Smoke200/100,one circuit,seed201001. Discovery201142–201144 and confirmation
211142–211144,paired across2 circuits/5 arms/2 sites. Train seed+1000,test+2000,
observation+3000; bootstrap214399,10000 paired-block draws. Config is authoritative.
130 fits:10 smoke,60 discovery,60 confirmation.65 graph/task trajectories per split.
Smoke excluded from inference. Average lags then circuits within each mapping-seed;
control draws averaged within block. n=3 per cohort; do not pool cohorts.

## Hypotheses and decision rules (fixed before new outcomes)

For each block let loss(site)=accuracy(intact,site)-accuracy(DAN_MBON,site).
Primary interaction=loss(MBON)-loss(KC_unstimulated). Positive means attenuation
at alternative observation. Control-specific contrasts=mean(control)-target are
reported separately; controls can affect different target neuron populations.

Past access: frequency and shifted-label-null excess each>=0.05 for every seed
block AND mean R² versus training-frequency predictor>0, for each tested cell.
A cohort supports the narrow observation-dependence hypothesis only if ALL:
1. MBON intact, KC intact AND KC cut pass past-access gates.
2. MBON mean loss>=0.05 and loss>0 for every block.
3. KC mean loss<=0.02 (predeclared retention tolerance).
4. Mean interaction>=0.05 and interaction>0 for every block.
Require this in BOTH discovery and fresh confirmation. No post-hoc threshold,
new observation set, lag selection or tuning. Confirmation runs regardless of
initial result. If either intact site lacks access the comparison is inconclusive
at that site; if access holds but rules fail report a negative scoped result.
As secondary replication, MBON control-minus-target mean>=0.05 and all blocks>0
plus intact access confirms III-D specificity in fresh tasks (same masks).
Passing retention is a descriptive finite-seed gate, NOT a formal equivalence test.

## Artifacts, verification and stopping

Store every site's checkpoint with symbols, features, labels, fitted/null scores,
parameters, neuron IDs, graph/mask identities; all lag metrics and representation
rank/sparsity/activity/decay. Save exact full repeats and independent trajectory
updates plus augmented least-squares refits. Reproduce an archived III-D MBON
intact checkpoint before new cases; compare all common trajectory/fit arrays.
Preserve every previous result checksum. Manifest includes config hash, Git
revision, source hashes, dependency/hardware metadata, masks, timing and sampled RSS.
Code/tests/protocol committed before any smoke or main neural outcome.
Stop on nonfinite values, invariant/checkpoint disagreement or resource cap
3600s/3GiB per run/verification. Preserve failed outputs in place and use a new
directory for any justified repair. Never replace raw outcomes. Finish the bounded
cohorts only; no automatic next experiment. Report mean/median/variance/paired
differences/bootstrap95/dz, raw3-seed tables, limitations and one next experiment.

## Interpretation limits

Preserved KC access can refute total loss of LINEARLY ACCESSIBLE past information
in these sampled cells; it does not prove where information is stored or isolate
one physiological mechanism. If KC access fails, this48-cell probe may simply be
uninformative. Whole-brain, autonomous recall, formal memory capacity, real animal
memory, dopamine plasticity, all alternative populations and observation-set
ensembles remain outside scope. Previous III-B/C negative findings stay intact.
