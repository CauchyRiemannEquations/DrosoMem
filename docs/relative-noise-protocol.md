# ACT V extension — relative ongoing noise, preregistered 2026-09-30

## Question and prior evidence

Does whole-brain recall outperform partial recall when ongoing state noise is
scaled to each model's clean observed-state variation? ACT V's absolute-noise
study at 4af5ec2 found no confirmed advantage. Its sigma .001 was approximately
.27 times the partial model's median MBON training SD, versus1.3–2.3 in whole
graphs. This motivates a separate calibration study, not replacement of that result.

## Fixed task, models and calibration

Keep ACT I rate dynamics, frozen recurrent connectivity, exact matched inputs,
48 observed MBON IDs/order, 482-parameter 48→8→10 head and train-only preprocessing.
Pi includes leading3; offset0,length200,prompt314,197 generated symbols. Incoming
L1 gain.9,leak.6,mbon_after_kc; Adam2000,LR.03,L2=1e-5,first32 targets4×.
Reuse all30 archived heads for seeds7142–7146 ×circuits701/702 ×legacy5/brain5/brain1.
Unconditionally fit18 fresh clean heads at seeds381142–381144 in the same strata.
These fresh model seeds and bootstrap seed388399 are absent from prior configs
([audit](relative-noise-seed-audit.json)).
They are computational replications on one connectome, not biological samples.

For each head calculate q=median_j(std_t(X_train[t,j],ddof=0)) using all199 clean
training feature rows and48 MBON columns, before preprocessing. Do not use the
head's floored scale or evaluation states. Require finite positive q; otherwise
stop as unidentifiable without epsilon or substitute seeds. Apply sigma=r*q
for fixed r=[.0003,.03,.3], plus zero. This grid roughly spans the earlier partial
model's registered relative doses; its values are fixed before new outcomes.
One scalar applies to every neuron, not neuron-specific calibration.

Canonical Gaussian draws use SeedSequence([372001,model_seed,circuit,1]), same
universe/order and stream as ACT V. Shared neuron IDs receive the same draw;
strengths reuse it. Fresh model seeds imply fresh streams. Every step after the
clean prompt, add sigma*Z to all states immediately before prediction and clip
to[-1,1]. All-neuron exposure and extra whole-graph coordinates remain. Equal
relative MBON scale does not match total noise energy, every coordinate's SNR,
correlations, topology, or physiological noise.

## Endpoints and decisions

Execute clean and all3 nonzero strengths in both autonomous and teacher modes:
8 trajectories/head,384 total. A zero-strength autonomous row explicitly aliases
clean output (48 aliases), never an independent replicate. Save all197 predictions,
probabilities, observed states, position/region accuracy, first error, PMS,
confidence, active counts, sparsity, rank, temporal similarity, clean-state MSE,
clipping, q/sigma, checkpoints/training history, graph IDs and environment.

PMS excludes the3 prompt symbols. Capped retention=min(PMS/PMS_clean,1).
PMS_clean=0 makes the relevant gate inconclusive, without dropping the head.
Curve indices equally average the3 nonzero points, then average the two circuit
strata within seed. Keep discovery n5 and confirmation n3 separate.

Primary: brain1−legacy5. Require in EACH cohort mean curve-retention gain>=.10
AND mean raw curve-PMS gain>=2 digits, each positive in>=4/5 discovery and3/3
confirmation seed blocks. Only both-cohort passage confirms advantage. Secondary
brain5−legacy5 and brain1−brain5 use the same fixed gate and never replace primary.
Report all paired differences, mean,median,sample variance,10000 paired bootstrap
intervals,paired dz,wins/ties/losses. No p-value gate, optional stopping or tuning.
Teacher accuracy diagnoses correct-input decoding separately; not autonomous memory.
No formal capacity, biological noise, or pure topology-causal claim is licensed.

The interpretation question is whether the prior conclusion survives this single
calibration. A changed outcome does not invalidate the absolute-noise result;
an unchanged outcome does not prove amplitude irrelevant in all settings.

## Verification and stopping

Before main, real horizon8 smoke at7142/c701 for all3 graphs, both modes and all
doses, plus actual zero-noise rollout. Check q/sigma, exact clean checkpoint
replay, paired IDs, target-free autonomous path, immutable weights/head and output.
Use tests for train-only raw scale, zero/degenerate handling and decision gates.
Reuse the independent ACT V scalar reference for every legacy5 trajectory plus
whole-graph discovery7142/c701 and fresh381142/c702:160 independent full-horizon
trajectories. Refit first fresh seed in all6 conditions exactly. Independently
recompute all metrics, aggregation/gates and hashes; preserve every old result byte.
Not every whole-graph trajectory receives independent dynamical replay.

Two workers,one numerical thread each;900s/3GiB per worker tree,5400s main and
3600s verification. Stop and preserve partial artifacts on integrity/resource
failure; never shorten main horizon, tune dose, replace seed or relabel failure.
Commit protocol/config/code before smoke/main outcomes. After verification,
document all results and select one separately preregistered follow-up. The user
authorized continuing that next study if this pipeline has no integrity problem.
