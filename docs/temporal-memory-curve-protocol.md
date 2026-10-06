# P1: Temporal Decodability Curve — prospective protocol

Registered after the v1 audit, before any new TDC trajectories or outcomes.
[Configuration](../configs/temporal_memory_curve.json) is part of this protocol.

## Question and operational definition

How much historical input information remains decodable from the state of the
fixed fly-connectome reservoir as temporal lag increases?

TDC(lag) is the accuracy of a train-only standardized, alpha-1 affine ridge
decoder of `symbol(t-lag)` from the 48 observed MBON coordinates after
incorporating `symbol(t)`. State index 0 is after the first input. Each lag has
48×10 coefficients and 10 intercepts (490 fitted decoder parameters); fitting
all heads as separate columns of one ridge solve is mathematically equivalent
to fitting each separately. Means/scales use training state only. No recurrent
weights or input patterns are learned. No autonomous feedback is evaluated.
This is an operational decoding curve, not formal memory capacity or Shannon
information. It measures access through this decoder and observation interface.

## Fixed design

Primary graph: source-derived threshold-5 partial `legacy5`, circuit 701,
686 neurons, 512 KC, 48 MBON. Circuit 701 was established before this study;
it is not selected on TDC outcomes. Secondary: source-derived full-node-coverage
`brain5`, 138,639 neurons and 2,700,513 edges. Secondary performance cannot
rescue the primary gate. No whole-brain superiority gate is introduced.

Both use exactly the same source root IDs for stimulation and observation,
gain .9 incoming-L1 normalization, leak .6, KC stimulation fraction .1 and
amplitude .5, and the verified `mbon_after_kc` discrete schedule. Units are
symbol steps, not physiological milliseconds. Sparse rate dynamics remain fixed.

Six blocks 910001–910006 with input mapping seeds 911001–911006, train seeds
912001–912006 and test seeds 913001–913006. Every seed is fixed in the config.
No discovery-selected confirmation, seed addition, hyperparameter search or
architecture search. Primary iid uniform K10 input is generated separately for
training and test by NumPy default_rng. Each stream starts from zero state;
warmup200 is excluded. Retain 4,000 training and 2,000 test state/label rows for
every lag, using the same times `t=200..end-1`. Labels for lag1–20 are strictly
past inputs; lag0 is current-symbol sanity only. Independent test trajectories
have no teacher target access. Times remain aligned and no split-crossing
history is used. Small accidental frequency/correlation fluctuations in iid
streams are measured by controls, not treated as temporal dependence.

## Baselines and primary endpoint

For each lag: uniform chance .1; majority class chosen from that lag's training
labels; current-symbol-only training lookup table (current symbol → lag label).
The lookup reads only the current test symbol, never its past or test labels.
No sequential Markov predictor is needed for iid. Ties use the lowest symbol.
Baseline B is `max(.1, test frequency accuracy, test current-only accuracy)`.
All three are reported separately. The conservative maximum is fixed here.

Primary PASS if, on `legacy5`, every one of the six blocks at **each of lag1–5**
has accuracy minus B **≥ .10**. Otherwise FAIL. Equality passes. Lag0,
whole-brain values and later lags do not enter this gate. The threshold is
motivated by the earlier +72.72pp iid lag2 evidence; extending to all lag1–5
is a new stronger conjunction, not an assumption of success. Scientific FAIL
ends this test honestly and does not prevent a valid P2 comparison.

Report raw per-seed/per-lag correct counts and accuracy, chance-adjusted
`(accuracy-.1)/.9` without clipping, baseline excess, mean/median/sample SD.
Bootstrap resamples the six computational blocks with replacement, 10,000
draws, seed914001, percentile 2.5/97.5%, paired across lags/graphs. These are
descriptive computational intervals, not animal-population inference. No
monotonic curve or exact forgetting boundary is assumed.

## Validation, resources and execution

Smoke: first block, both graphs, warmup200, train300/test200; all lags and same
model/decoder. Smoke is excluded from main statistics and has no scientific
gate. Run independent validation before main. Main uses all six fixed blocks.
Verifier separately loads source edge/role/ID tables, rebuilds input mapping,
normalization and the rate update without calling the main model constructor
or step routine; reconstructs both streams and all lag labels; refits by
augmented least squares rather than normal equations; compares saved
trajectories, coefficients, scores, predictions, baseline counts and final gate.
Graph cache hashes and source provenance are checked. Predictions must agree
exactly; floating arrays use fixed atol/rtol 1e-9, scalar metrics 1e-12.

Every output saves protocol/config/source hashes, clean Git source commit,
environment/BLAS/package versions, stream/mapping/graph/model array hashes,
runtime, 50ms sampled RSS and Windows process peak working set when available,
raw artifacts and a checksum manifest. Verify hashes again before completion.
New paths only under `results/tdc_v2`; historical Git result blobs are sealed.
Maximum main or verification run: two hours and 4 GiB process RSS. Resource
or technical failure produces an interruption/failure record and incomplete
status, never a scientific PASS. Same-config retries get a new attempt path;
failed paths remain. No numerical tolerance/criterion change after outcomes.
P2 protocol and null count are decided only after P1 is fully documented and
validated, using resource timings rather than accuracy to set a finite budget.
No P3 physiology/spiking/plasticity study is included.
