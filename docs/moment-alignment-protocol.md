# Train-only moment alignment before frozen-head transfer

Registered before aligned predictions on 2026-09-28. ACT III decoding diagnostic.
The preceding unaligned transfer failed; this follow-up is exploratory and uses
already observed cohorts, not a new independent confirmation sample.

## Question and hypotheses

Can neuron-wise mean/scale shifts explain failed R/F head transfer? R is
`renormalized` D(B)B; F is `fixed_original` D(A)B, with the same raw shuffled B.
H1: alignment improves independent-test past-symbol accuracy by at least 5 pp
over archived unaligned transfer. H2: aligned decoding accesses past information.
H3: aligned accuracy retains target-specific refit performance within 5 pp.
Failure of any hypothesis remains reported separately; no normalization tuning.

## Frozen design and adapter

Use all archived `normalization_control_{smoke,main,confirmation}` checkpoints,
R→F and F→R, 48 identical MBON observations, alphabet K=4, alpha=1 summed-loss
ridge, 11 four-output heads, 196 supervised coefficients per head. Source head
weights AND intercepts, source stored means/scales, and shifted-label null heads
are byte-identical. No new target supervised fits, recurrent updates or rollout.
This is driven past-symbol decoding, NOT autonomous recall or formal memory capacity.

Adapter receives ONLY target training features: mu_t=mean(xtrain),
sigma_t=max(std(xtrain, ddof=0),1e-5), neuron-wise. It estimates 96 unlabeled
values. On test rows compute z=(xtest-mu_t)/sigma_t, scores=z W_source+b_source.
This is algebraically equivalent to x'=mu_s+sigma_s*z followed by unchanged
source standardization. Use direct z for numerical stability; independently
check mapped-input score equivalence at atol=rtol=1e-9. No clipping, rotation,
cross-feature mixing, label access, held-out moments or target head fitting.
Thus HEAD is frozen; target preprocessing is newly adapted and is NOT frozen.
The shifted-label head receives the same adapter. Frequency control uses source
training frequencies. Scores are unbounded regressions, not probabilities.

## Cohorts, baseline and stopping rule

Main mapping seeds 34142–34146; archived confirmation 41142–41144; circuit
strata 701,702. Smoke seed34001/c701 is excluded from inference. Each archived
stream has warmup100; main/confirmation train2000,test1000; smoke200/100.
All other stream, graph, mapping seeds and identities are inherited verbatim
from source manifests. Run 20 main +12 confirmation +2 smoke directions, all
11 lags [0,1,2,3,4,5,8,12,16,24,32]. Primary past lags [1,2,3,4,5,8].
Reproduce source-only fits and all archived unaligned predictions before the
corresponding aligned prediction. Smoke first, then both complete cohorts
regardless outcome. Stop on integrity/numerical failure, 1800 seconds or 3 GiB
sampled process RSS; report partial execution, do not silently replace seeds.

## Fixed analysis and decisions

Average primary lags per run, then two circuit strata per mapping-seed block.
Treat n=5 and n=3 blocks separately; no pooling or treating time rows as samples.
For each direction separately, require both cohorts to meet a hypothesis:

- H1: mean aligned-minus-unaligned >=5 pp, with positive differences in >=4/5
  main and all3 confirmation blocks.
- H2: mean accuracy exceeds BOTH majority and aligned shifted-label control
  by >=5 pp; both margins >=5 pp in >=4/5 main and all3 confirmation blocks;
  mean R² versus source-frequency predictor >0.
- H3: mean aligned-minus-target-refit >=-5 pp, and this tolerance holds in
  >=4/5 main and all3 confirmation blocks. Portable decoding requires H2+H3.

Keep per-lag scores, predictions, raw per-circuit and seed-block tables, adapter
parameters, mean/median/sample variance, 10,000 paired-block bootstrap intervals
(seed49399), and paired dz for differences. Intervals with n=3/5 are descriptive.
Report unaligned baseline, target refit, majority, null, MSE and R² including
negative values. No p-value gate or post-result criterion changes.

## Verification and provenance

Hash all previously tracked results before execution and recheck after. Audit
graph/pattern/observation/stream/label identities; source-only independent
augmented least-squares verification; exact saved-checkpoint replay; independent
artifact verifier reconstructs labels, scores, metrics, aggregation and gates.
Test affine-distortion recovery, train-only isolation, identity, frozen parameters
and zero-variance floor. Record git/config/code hashes, source manifests,
environment, runtime and sampled RSS. Preserve every archived result.

## Limits

A successful diagonal adapter supports a mean/scale explanation within these
saved trajectories. Residual failure may reflect more complex state geometry;
it does not establish absent representation. No claim of whole-brain advantage,
real-topology advantage, autonomous memory, biological learning or fresh-seed
confirmation follows. This concerns a computational model using actual
Drosophila connectome structure.
