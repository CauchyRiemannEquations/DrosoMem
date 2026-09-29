# ACT III-A: frozen intact readout on KC lesion states

Registered BEFORE computing any intact-to-lesion prediction. Previous refit
outcomes are already known: intact77.303%, full KCg lesion79.219%, matched77.156%;
disjoint KCg/non-KCg76.957%/77.072%. This is an exploratory SAME-cohort follow-up,
not independent replication, and must not alter either prior specificity gate.

## Question, hypotheses and baseline

Primary question: how much does condition-specific decoder refitting compensate
for removal of the full annotated KCg population? Compare the saved intact head
on the SAME target test states with the existing condition-specific refit head.
H1 (primary, full KCg only): mean(refit accuracy − frozen accuracy)>=5pp and
ALL3 paired mapping/stream blocks>0, provided target refit passes past access:
every block exceeds majority and shifted-null by>=5pp and mean R²>0.
Failure of either part means H1 unsupported. It is decoder adaptation, not
learning inside the connectome, even if this criterion is met.

Baseline: source intact checkpoints c701/702,s61142–61144 plus smoke c701/s61001.
Before transfer verify source-only ridge refits / independent augmented least
squares and exact self-prediction, including all source metrics. Preserve all
source and lesion manifests, arrays and hashes. Reuse the locked environment.

## Frozen operation and controls

For each paired seed/circuit use the intact head from results/kc_ablation.
Freeze source mean,scale,weights,bias and shifted-label null head; predict
((target features − source mean)/source scale) @ source weights + source bias.
No target standardization, moment alignment, coefficient update or adaptation.
Target training states may be used for descriptive drift/training metrics only.
Target labels and refit weights cannot enter the frozen prediction function.

Evaluate all full lesions gamma,matched0,matched1,matched2 (24main+4smoke) and all
disjoint gamma0/nongamma0,gamma1/nongamma1,gamma2/nongamma2 (36main+6smoke).
Total70transfers,7unique source self-checks; no new reservoir trajectories or
new target fits. Comparator target refits are the existing saved heads.
Audit identical intended input maps, observed IDs, train/test symbol streams,
labels,readout budget. Effective input maps correctly differ after lesion.
All48MBON slots,196coefficients per lag head,alpha1,gain.9/leak.6,original
normalization fixed before lesion; no changes to graphs or masks.

Sources: results/kc_ablation and results/kc_exclusive, pinned manifest hashes in
config. Use source blocks mapping61142–61144,train62142–62144,test63142–63144,
matching64142–64150,circuits701/702. Smoke61001/62001/63001,matches64001–64003,c701.
K4 independent pseudo-random streams,train2000/test1000,warmup100;smoke200/100.
All11lags0,1,2,3,4,5,8,12,16,24,32; primary1,2,3,4,5,8.

## Secondary diagnostics fixed before outcomes

For each of four groups (full gamma/full matched mean/subset gamma/subset
non-gamma), report frozen/refit/source accuracy,refit−frozen,source−frozen,
majority,source shifted-null,R²,train/test MSE and train accuracy. For each group:
frozen access requires every block majority and null margin>=5pp,mean R²>0;
retention requires mean and all3 (refit−frozen)<=5pp. Portable only if both pass.
Report H1-style refit benefit for other groups as secondary, not extra primaries.

Report frozen specificity delta = matched frozen accuracy − gamma frozen
accuracy separately for full and disjoint families. Secondary specificity gate:
mean delta>=5pp and all3>0, plus intact baseline access (same definition).
Compare descriptively with the prior refit differences. This cannot by itself
identify greater information loss; frozen failure can reflect decoder mismatch.
Do not pool families or treat either as new seeds. If no support, report it;
do not tune, select masks or redefine the criterion.

## Analysis and stop rule

Average primary lags→three control draws if applicable→two circuits within each
of3 mapping blocks. Keep individual circuit/draw/lag raw rows. Report block
mean,median,sample variance,bootstrap95(10000draws,seed65399),paired dz for
differences. No p-value gate; n=3 is small and not30/60/1000.
Maintain real/null scores,train scores,labels,source parameters,referenced graph/
feature/checkpoint hashes,drift diagnostics,config/code/environment identity,
runtime and50ms sampled RSS. Reuse neural diagnostics with explicit provenance.
Smoke is engineering only. Run all70 regardless performance. Stop on integrity
or nonfinite failure,1800seconds or3GiB sampled RSS; preserve partials.
Commit protocol/config then implementation before evaluation. Require exact
replays, independent source-only least squares and a separate direct-formula
verifier for scores,metrics,raw tables,aggregations,bootstrap and gates.

No autonomous recall/Pi Memory Score or formal memory-capacity claim follows.
No inference about actual fly learning or biological necessity of KCg follows.
