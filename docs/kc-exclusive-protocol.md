# ACT III-A: disjoint KCg / non-KCg subset lesions

Registered after the full-population study and BEFORE new subset trajectories.
This is an exploratory follow-up on the SAME three mapping/stream blocks, NOT
an independent confirmation. The prior full-population hypothesis failed:
intact77.303%, KCg lesion79.219%, matched lesions77.156%; extra impairment−2.064pp.
Do not revise that study's criteria or combine cohorts into a larger sample.

## Question and fixed masks

Does the negative full-population specificity result persist when the compared
lesions are disjoint? Read each existing KCg mask G and matched mask M from
results/kc_ablation. Compare lesion G\M (KCg-only) against M\G (non-KCg-only),
on the original intact graph background. The shared intersection remains active
in BOTH conditions. This changes both dose and background relative to the first
study; it does not isolate overlap as a single causal factor.

Removing the shared intersection from two equal joint-stratum histograms leaves
equal histograms. Verify equal count and EXACT (raw in-degree,out-degree,input
exposure bitmask) histogram, zero overlap, and exclusive annotation membership.
Use ALL three saved matched draws for ALL circuits/blocks, no new mask selection.
Report variable lesion sizes and removed edge/strength imbalance. No strength
matching, no re-normalization. Original graph size686 and48MBON observation slots
remain fixed. The subset cannot represent all KCg neurons (forced shared neurons
are excluded). No claim about the entire biological population.

## Budget, dynamics and endpoints

Inherit configs/kc_ablation.json: circuits701/702, mapping61142–61144,
train62142–62144,test63142–63144,train2000/test1000,warmup100,K4,
gain.9/leak.6,mbon_after_kc,alpha1,all11lags and primary1,2,3,4,5,8.
Input maps, intended input amplitude/fraction, original incoming normalization,
refit-only real/null heads,196coefficients per head remain identical.
Smoke: existing c701/s61001 streams200/100,3pairs=6conditions.
Main:3blocks×2circuits×3draws×2lesions=36conditions. All run regardless outcome.
Full trajectory/fit replay of every condition and independent saved-artifact audit.
1800seconds/3GiB sampled RSS; stop on integrity/nonfinite errors, keep partials.

Primary delta = non-KCg-subset lesion accuracy minus KCg-subset lesion accuracy.
Average primary lags→three pairs→two circuits within each block; n=3.
Support subset-specific impairment only if mean>=5pp and ALL3 deltas>0, with the
already verified intact access gate from the source study. Otherwise unsupported;
do not use an opposite-direction post-hoc success gate. Show all raw pair/seed/
lag data, cell mean/median/variance, paired dz,10000-draw block bootstrap95 with
seed64399. Source intact comparison is descriptive and uses existing checkpoints.
Do not claim p-value significance or count subsets as independent replicates.

Archive masks, source manifest/checkpoint hashes, graph identities, exact configs,
neural rank/sparsity/decay, heads/features/scores/labels, runtime/environment,
source commit. Commit protocol/config, then implementation, before execution.
Original results and previous numerical runner remain unchanged.

This examines refitted decoding of past externally driven symbols. It does not
measure autonomous recall, formal memory capacity, internal plasticity, whole
brain superiority, or a biological memory circuit.
