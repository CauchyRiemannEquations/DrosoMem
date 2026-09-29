# ACT IV-A — External readout dependency: bounded completion protocol

This closes the CURRENT PARTIAL RATE-MODEL A/B/C/D comparison, not all possible
biological plasticity or whole-brain memory. There is one internal update rule:
the repository's existing supervised KC→MBON delta/row-competition rule
(KCMBONPlasticity). It is an artificial teacher semi-gradient, NOT dopamine,
STDP, a calibrated biological rule or autonomous internal learning. Older Phase5B
biological-rule failures remain independent evidence; this study cannot replace them.

## Module1: retrospective common-delay analysis

Observation-location outcomes and lag plots have already been viewed. This is an
explicitly retrospective exploratory analysis, not fresh blinded confirmation.
Source results/observation_location, unchanged root manifest pinned in config.
For each original primary lag1,2,3,4,5,8 select using DISCOVERY INTACT data only:
BOTH sites must have frequency/null excess>=5pp in every mapping-seed block
(circuits averaged) and mean R²>0. List eligible AND ineligible lags and all source
rows. Freeze that lag set for descriptive confirmation-cohort estimates; do not
reselect using lesions/interaction/confirmation outcomes. If no eligible lag,
report inconclusive, no substitute. For eligible lags show site impairment,
paired interaction and baseline gap at each lag plus equal-lag averages. No new
success declaration based on retrospective selection; original primary remains.

## Module2: fresh matched factorial study

A: frozen real graph + trained affine ridge head.
B: frozen random graph + identical trained head; role/signed-degree/incoming-weight
preserving rewiring is a second explicit control. Random preserves neuron/edge
counts and signed raw-weight multiset but not degrees, role blocks or row strength.
C: trained internal KC→MBON connections + same simple ridge head; also evaluate
without trained head to separate internal target coding from downstream refitting.
D: fixed nearest-code decoder on MBON activity, with0 learned output parameters.
This fixed external interpretation of neural codes is an operational direct-decoding
proxy, not literal readout-free biology. It never accesses targets during evaluation.

Full factorial networks real/random/role × training frozen/aligned/shifted.
Same graph per training arm, input map, symbols, observation IDs and codebook.
Shifted target control circularly rolls only the2000 training labels by1000;
labels remain shifted for all epochs, matched label counts and update budget.
Training target is the symbol2 steps in the past. This lag is DATA-INFORMED by
previous plots; it is fixed now and fresh tasks provide the new comparison.
All lags0,1,2,3,4,5,8,12,16,24,32 are stored for the trained ridge, primary lag2.
Do not compare this study's primary directly with earlier six-lag average scores.

Two existing686-cell partial graphs c701/c702,threshold5,48MBON observation.
Incoming-L1 gain0.9,leak0.6,input fraction0.1,amplitude0.5,K4 iid independent
train/test streams,warmup100,train2000/test1000. All arms use SYNCHRONOUS updates
so the pre-existing plasticity rule runs unchanged, one microstep. This differs
from previous mbon_after_kc scheduling; no cross-study causal attribution.
At every epoch reset states, feed100 warmup symbols WITHOUT weight updates,
then2000 target updates for10 fixed epochs,lr0.05,floor1e-4. No early stopping,
validation selection, learning-rate sweep or additional epochs after failure.
Existing edges/signs and per-MBON absolute plastic-edge mass are conserved;
nonplastic weights unchanged. Frozen runs identical forward passes withlr0.

A fixed artificial codebook partitions the48 MBON slots into4 disjoint groups
of12 using code_seed; each group's target activation0.25, elsewhere0.
Identical across graph/training arms, independent of input coding. Every epoch
receives target codes only AFTER the current forward step. During test internal
weights freeze; evaluation resets states and exposes only input symbols.
Nearest-code chooses minimum squared Euclidean distance, deterministic argmin tie
break, no learned thresholds or affine calibration. Ridge is alpha1, train-only
moments,std floor1e-5,196 learned coefficients per lag; no test fitting.
Recollect both train/test trajectories at FINAL weights before fitting any head.
Thus C changes representation; this is not joint head/internal optimization.

Discovery input seeds221142–221144; confirmation231142–231144. Train+1000,
test+2000,code+3000,graph+5000 (same RNG seed across named independent algorithms,
not across observations). Bootstrap236399,10000 mapping-block draws. Config is
exact specification. Smoke221001,one circuit,train200/test100,one plastic epoch.
9 smoke +54 discovery +54 confirmation =117 network/training cases,234 trained
or fixed-decoder evaluations at primary lag2. Circuits averaged within each seed,
n=3 per cohort; no pooling or treating graphs/epochs as independent samples.

## Outcomes and prospective decisions

Report per-seed accuracy,frequency/null controls,raw scores,training history,
weight change,rank/sparsity,counts/runtime/RSS and generated data identity.
Ridge past access: frequency/null excess>=5pp in EVERY block and mean R²>0.
Primary INTERNAL CODING claim on REAL graph requires in BOTH cohorts:
- aligned FIXED-code test accuracy beats training-frequency predictor by>=5pp
  in every block;
- aligned fixed-code accuracy exceeds BOTH frozen and shifted fixed-code mean
  by>=5pp, with positive paired differences in all3 blocks for each contrast.
This tests whether an artificial internal teacher produces useful fixed codes.
If it fails, do not substitute ridge success for fixed-code success.

Secondary, separately named endpoints, both cohorts required:
- useful REAL reservoir representation: frozen-real ridge passes access;
- original-graph advantage: frozen-real ridge beats random AND role means>=5pp,
  all paired differences positive, with real ridge access;
- representation improvement: real-aligned ridge beats real-frozen AND shifted
  by>=5pp, all paired differences positive, with aligned ridge access;
- external-head dependence in frozen REAL: ridge exceeds fixed-code by>=5pp,
  every block positive, with ridge access. Fixed decoder is ONE operational proxy,
  so failure never proves that every decoder needs learning.
Small-n bootstrap intervals/dz are descriptive. No p-value-based success fishing.

## Completion, stopping, verification

Finish modules even if hypotheses fail, then document A/B/C/D with explicit
success/failure/limitations; no automatic next experiment. Prospective fresh
confirmation runs regardless of discovery result. Checkpoint every final graph,
training histories and state/head/symbol arrays. Independent update implementation
must reproduce final plastic weights, masked invariants and final trajectories;
ridge uses an independent augmented-lstsq check, fixed decoding rederived directly.
Historical MBON baseline replay and prior artifact hashes are checked. Protocol,
config and code commit precede any NEW smoke or main task outcome.
Stop and retain failure on numerical/invariant disagreement or7200s/3GiB budget
per execution/verifier. New directories only. Retrospective analysis cannot change
completed criteria. Archive negative Phase4/5/5B findings and restrict conclusions
to artificial-teacher representation/decoding, not biological internal learning.
