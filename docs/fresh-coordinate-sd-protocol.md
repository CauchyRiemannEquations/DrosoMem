# Fresh-model coordinate-SD structural confirmation

Registered before smoke or main outcomes. This is one new, bounded computational
experiment following the closed coordinate-SD calibration study. No previous
artifact or decision is changed by its result.

## Question and decision

With **newly fitted models, new random-sequence seeds and new control-graph
rewirings**, does intact partial wiring outperform a signed-degree-preserving
control during autonomous recall under coordinate-SD observation noise?

Primary task: seeded iid random digits. Primary control: signed-degree-preserving
rewiring. For each model/data seed block, average the two circuits, three fresh
noise streams and three fixed strengths, separately for intact and degree. Define
the paired advantage as intact minus degree for first-error prefix symbols and
retention (prefix divided by that model's own clean prefix, capped at one).
The hypothesis passes only if the advantage is at least 2 symbols and 0.10
retention in **both** discovery and confirmation cohorts, and both metrics are
strictly positive in at least 4/5 and 3/3 seed blocks respectively. A clean
prefix of zero makes that model's retention undefined; propagate it to the
cohort decision as inconclusive. A failed conjunction is a negative result.
The thresholds match the previous practical-magnitude convention, not a
biological standard or a claim of equivalence.

Secondary, descriptive comparisons: role-preserving control, trained pi prefix,
and the prior own-median versus direct coordinate-SD incremental gap. These
cannot replace the primary outcome. Report signed paired differences, all seed
blocks, mean/median/sample variance, paired dz and a 10,000-resample seed-block
percentile interval. Cohorts are never pooled for the primary decision.

## Fixed design and budget

Two source-derived 686-neuron left mushroom-body circuits (701,702), threshold 5,
identical source files and graph-building algorithm. The intact graph itself is
fixed source anatomy; **new graph seeds apply to each degree/role rewiring**.
Random and pi tasks each use K=10, length 200, prompt length 3 and horizon 197.
The same 48 MBONs are observed; the 48→8→10 head has 482 learned parameters.
All recurrent weights remain fixed. Train each fresh head for 2,000 Adam updates,
LR .03, L2 1e-5 and fixed 32-target ×4 prefix weighting. Gain .9, leak .6,
incoming-L1 normalization and `mbon_after_kc` timing remain unchanged.

The config fixes disjoint discovery model/data blocks 680001–680005 /
690001–690005 and confirmation blocks 681001–681003 / 691001–691003.
Control rewirings use graph seeds 698000+; fresh noise streams 699001–699003,
and smoke 699901–699903. All differ from earlier study seeds. Run every
prespecified cell: 8 blocks × 2 tasks × 2 circuits × 3 topologies = 96 **new
fits**. No seed selection, refit or strength tuning after outcomes.

The three strength values are .0003, .03 and .3. Let s_i be the ddof-0 SD of
coordinate i across 199 teacher observations, q its median and u its RMS.
Coordinate calibration uses amplitude r s_i; own calibration uses
r q s_i/u. Noise is Gaussian, time-white and independent by coordinate,
paired across topology and calibration by sorted MBON root ID. Observations are
clipped to [-1,1]. These controls are synthetic and do not equate realized
clipping, covariance or physiological noise.

Run a one-case random smoke with an 8-step horizon before the main panel.
Main and independent verification each have a 7,200-second wall budget and
3-GiB sampled process RSS budget. Stop and preserve partial output on any
nonfinite value, identity mismatch, budget breach or replay failure. Never
select only completed cases for inference.

## Evidence and interpretation

Record every clean and noisy **actual autonomous trajectory** (not just a
teacher certificate), all generated symbols, probabilities, observed features,
energy and clipping; also record first-error teacher certificates separately.
Save training checkpoint, data/graph identities, training history, fixed config,
source hashes, per-case manifest and raw block tables. Independent verification
must rebuild graphs and teacher states, check head parameters and paired noise,
replay all main trajectories through an independent recurrence/head calculation,
recompute metrics and decision, and check saved hashes. Selected exact fresh
head refits check deterministic training. No prior result files are overwritten.

These are computational model seeds, not independent flies. Prefix bits are not
Shannon or formal memory capacity. Fitted-sequence recall is not prediction of
unseen digits. A positive criterion would be limited to this model, source
connectome, tasks, head and synthetic-noise design; a negative result does not
rule out other biological or computational mechanisms.
