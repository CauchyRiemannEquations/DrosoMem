# M1: state carry and history-dependent synaptic drive — prospective protocol

Date: 2026-10-08. Baseline main: `28248fc7bbf306bcc4af708343c5cb4c6017856e`.
Registered before any new arm trajectory or decoding outcome.
[Configuration](../configs/temporal_mechanism.json) is part of the protocol.
P0/P1/P2 and their negative results remain closed and immutable. This is the
next-work computational mechanism question, not P3 biology/plasticity, a new
real-wiring superiority test, or an attempt to rescue previous failed gates.

## Question

Does history-dependent synaptic transmission add materially to iid historical
decoding beyond direct state persistence in the existing rate model, while
keeping instantaneous input access and decoder budgets fixed?

The word recurrent is operational here: synapses reading the previous state.
It includes one-step delayed feedforward paths as well as anatomical feedback
cycles. This experiment does not uniquely isolate graph-theoretic cycles or
locate storage. Direct persistence also passes through retained KC-to-MBON
feedforward transmission, so carry-only is a leaky cascade, not isolated cells.

## Factorial intervention with input access retained

Let `b=.6` be the integration/drive multiplier, `a=1-b=.4`, `u_t` the fixed KC
input pattern, and `W` the unchanged gain-.9 incoming-L1 graph. Let `L` and `R`
switch direct carry and previous-state synaptic drive respectively.

For the provisional update of all neurons:

`q_t = L*a*x_(t-1) + b*tanh(R*W*x_(t-1) + u_t)`.

Make a presynaptic vector `v_t` with updated `q_t` at KC coordinates and
`R*x_(t-1)` elsewhere. Then replace MBON coordinates with:

`x_t[MBON] = L*a*x_(t-1)[MBON] + b*tanh(W[MBON,:]*v_t + u_t[MBON])`.

Other coordinates remain `q_t`. All neurons integrate once. The same-step
KC-to-MBON block of W is retained in **every** arm. W is never renormalized
after intervention and no connectivity, input amplitude or input fraction is
learned/changed. When L is off, explicitly use a zero carry vector rather than
switching leak to1: b stays .6, avoiding a confounded change of input gain.

| arm | L | R | operational role |
| --- | ---: | ---: | --- |
| full | 1 | 1 | Original `TimedReservoir` and `mbon_after_kc` rule |
| carry_only | 1 | 0 | Local persistence and current KC-to-MBON transmission |
| synaptic_only | 0 | 1 | Previous-state synaptic drive without direct self carry |
| instantaneous | 0 | 0 | Current-symbol feedforward mapping; no history access |

Full uses the unmodified validated model. The factorized expression must
reproduce it in numerical unit checks and independent replay. At zero initial
state all four arms have the same instantaneous response to each symbol.
For instantaneous, an explicit prototype for each current symbol must reproduce
the entire saved state trajectory regardless of previous inputs; this is an
analytical/numerical no-history certificate. Removing all W while observing
unstimulated MBONs is deliberately not used, as it would destroy input access.

## Fixed data, graphs and decoder

Primary source-derived `legacy5`, circuit701, 686 neurons/3,309 edges; six fresh
input/data blocks1100001–1100006. Secondary full-node-coverage threshold-5
`brain5`, 138,639 neurons/2,700,513 edges; the first three blocks by numeric order.
The secondary scope is limited by the prior measured whole-brain runtime; it
cannot rescue a failed primary result. No new null ensemble or weak-edge sweep.

Mapping seeds1110001–1110006, independent train seeds1120001–1120006 and test
seeds1130001–1130006 are fixed now. Uniform iid K10, reset each stream to zero,
warmup200, 4,000 training and2,000 test rows at every lag. Lag0 is current-symbol
sanity; historical TDC is `state(t) -> symbol(t-lag)` for all lag1–20. Times begin
after incorporating input t; labels remain within their own independent stream.

All arms/levels within a block share stream bytes, source input root IDs and the
same48 observed MBON root IDs. Fixed alpha-1 affine ridge, train-only means/SD
(SD floor1e-5), 48x10 coefficients and10 intercepts per lag, 490 parameters.
The same row-normalized W, encoder, drive coefficient and same-step block apply.
Targets never enter the state update. Only the external readout is fitted;
no recurrent/synaptic learning, autonomous feedback, physiology calibration or
biological replicate inference. Chance1/10, train-only class-frequency predictor
and current-symbol-only training lookup are reported at each lag.

## Primary endpoint and stopping interpretation

Score `S_arm` is the unweighted mean `(accuracy-.1)/.9` over lag1–20 and the
fixed paired blocks, without clipping. Reuse the P2 scalar definition and its
.03 adjusted-score material-effect scale (2.7pp raw mean accuracy); do not select
lags based on new contrasts or tune decoder/drive/carry coefficients.

Assay eligibility: independent reconstruction passes, the instantaneous
prototype certificate holds, and **every primary block/arm** has lag0 accuracy
at least .90. If this fixed input-access criterion fails, report `assay-invalid`
and leave the scientific endpoint undefined. Do not call loss of input access
evidence of no memory or strengthen stimulation after the outcome.

On an eligible assay, primary PASS requires `full - carry_only` mean score
**>=.03 and strictly positive in every one of the six primary blocks**.
Otherwise FAIL. Exact equality at .03 passes; use integer-count fractions.
If FAIL or assay-invalid, close this question at the registered result. No extra
seed, lower threshold, new lag weighting or alternate decoder is a completion step.

Report predefined secondary contrasts, with the same .03/each-block-positive
rule as separate statuses: full−synaptic_only (carry with synapses active),
carry_only−instantaneous, synaptic_only−instantaneous, and the full−carry_only
contrast in the three whole-brain blocks. These cannot replace the primary gate.
Report the interaction `full-carry_only-synaptic_only+instantaneous` descriptively.
Nonlinear dynamics/refitting prevent interpreting these differences as fractions
of a total stored memory or as unique biological mechanisms.

## Reporting and reproducibility

Preserve per-case/per-lag correct counts, labels, predictions/scores, all input
streams/features, full-state terminal vectors/trajectory SHA256, graph/input/model
hashes, decoder coefficients and moments. Report all paired block contrasts,
mean/median/sample SD and 10,000-draw paired seed bootstrap intervals with
seed1140001, percentile2.5/97.5%. Resample entire input/data blocks together
across arms/lags; these are descriptive computational intervals, not animal CIs.
Do not force curves to decrease monotonically or call TDC formal memory capacity.

Every run saves protocol/config/seeds/source Git revision and hashes, full
environment/BLAS versions, runtime, OS process lifetime peak working set and
50ms sampled process-tree RSS (shared pages may be counted twice). Four worker
processes, one numerical thread each. Maximum stage budget: two hours/4GiB tree
RSS. Main is36 arm/level/block cases, 756 lag heads and72 train/test streams.
Existing research source/data/configs and every prior result Git blob are sealed
at baseline; no historical result or checksum is overwritten.

New namespace: `results/tdc_mechanism_v1/`. Existing output paths are rejected.
Retain all incomplete/failed attempts and exception/interruption records.
Retries use the same scientific settings and fresh output paths, never new
favorable seeds. A technical bug is separately documented; no post-outcome
relaxation of scientific criteria or numeric tolerances (arrays1e-9, scalars1e-12,
exact predictions/count-based gates) is permitted.

## Execution order

1. Audit and seal baseline; commit protocol/config before new arm outcomes.
2. Implement main and separate verifier; meaningful numerical/safeguard tests.
3. Smoke: first block, both levels/all four arms, train300/test200, warmup200.
   Eight cases/168 heads; no scientific PASS/FAIL gate and excluded from main.
4. Independently verify smoke before main.
5. Main all36 planned cases, then independently rebuild source graphs/input
   maps, rate trajectories, lag labels, least-squares ridge fits, baselines,
   instantaneous prototypes, scalar/gates and descriptive intervals.
6. Write results/synthesis and update README/status/roadmap/next-work; final
   checksum/recorded-source/history audit; commit and push verified work.

Source/data reconstruction and alternative augmented least-squares fitting
reuse the prior independent verifier helpers where appropriate. New trajectory
and criterion code is separate from the runner and must not call its step,
fitter, label builder or summary implementation. Old P0/P1/P2 claims and negative
results remain unchanged. P3 spiking/plasticity and biological learning are not
part of this protocol.
