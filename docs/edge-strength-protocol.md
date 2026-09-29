# ACT III-B follow-up: match effective normalized edge strength

Prospective follow-up motivated by the completed raw-weight panel, not a revision
of its success criterion. That panel removed mean normalized L1 88.023 for DAN
versus58.774 for signed/raw-bin controls (~50% more) despite raw L1 6651.5/6689.9.
The original between/betweenness aggregation bug is separately corrected with
raw artifacts preserved. It does not motivate this new strength matching.
No new outcomes for this follow-up have been observed when this protocol is locked.

One question: does the DAN refit effect meet the same5pp criterion with jointly
matched sign,raw-weight bins AND effective normalized-weight bins?
No additional hyperparameter/weighting/dynamics search. This is the final bounded
III-B control package; close the stage with all results, positive or negative.

## Exact graph-only matching

Use the same c701/c702 source graphs,threshold5,686 neurons and48 MBON slots.
Compute original incoming-L1 weights with gain0.9 before removal. Match counts
in every joint stratum of (raw sign, raw magnitude bin, normalized magnitude bin).
Raw bins [5,10),[10,20),[20,50),[50,100),[100,infinity).
Normalized bin integer = floor(abs(normalized_weight)*1000 +1e-9); width0.001.
The1e-9 is only a boundary-rounding guard, not a tolerance on matching counts.
Target all DAN incident edges552/524,3 random masks sampled without replacement
within each stratum from ALL edges. Target edges remain in pools. No rejection
sampling, no trial selection, no relaxing bins after outcomes. Exact total edge
count/sign/raw-bin histogram and normalized-bin histogram are preserved.
Removed normalized L1 must differ by less than target_edge_count*0.001+1e-9,
at most0.618% of target L1. This is approximate strength matching, not exact
per-neuron incoming/outgoing strength preservation. Endpoint roles not matched.
Graph-only audit predicts target overlap65.3%/67.7%, with minimum52.7%/56.9%.
Do not call controls disjoint or non-DAN; report actual overlaps. Higher overlap
reduces the structural contrast, so failed5pp confirmation is not no-effect evidence.

## Execution, fixed design and criterion

Five conditions: intact,DAN,dan_control0/1/2. Smoke5,discovery30,confirmation30,
65 fits/exact replays and52 frozen evaluations. Use fresh mapping/input/mask
seeds in both cohorts; do not reuse observed panel seeds. All confirmation arms
run regardless of discovery. Cohorts separate, not pooled for a pass.
Discovery141142–141144,train142142–142144,test143142–143144;
confirmation151142–151144,152142–152144,153142–153144. Full mask seeds in config.
Unchanged K4 random-stream delayed-symbol task,100 warmup+2000/1000 train/test,
lags0,1,2,3,4,5,8,12,16,24,32;primary1,2,3,4,5,8. Smoke200/100.
Same input0.1/0.5,leak0.6,mbon_after_kc,ridge1,train-only moments/std floor1e-5,
196 coefficients per lag. Direct input/observed IDs/remaining weights unchanged.
No renormalization after cuts; all neurons retained; frozen source heads unchanged.

Primary H-refit: mean intact−DAN >=5pp AND mean control−DAN >=5pp, both positive
in all3 seed blocks, intact frequency/cyclic-null margins>=5pp per block and
mean R2>0. SAME refit gate in both cohorts required. Frozen analogous secondary.
No p-value success rule; no pooling these results with the original raw-bin panel.
Per-seed results,paired differences,mean/median/sample variance,10000 bootstrap
block resamples seed154399,paired dz when defined; no masks/lags/circuits as n.

## Validation and stopping

Protocol then implementation committed before new outcomes. Replay archived
intact baseline and original DAN edge case first. Every new condition full replay,
independent augmented least-squares refit/frozen inference, masks independently
regenerated,normalized L1 bound verified,trajectories/metrics/statistics audited.
Preserve all old results and original failure record. Store source Git-byte check,
all raw outputs/checkpoints/manifests/diagnostics/environment/graph strengths.
Runtime cap3600s,RSS3GiB per command. On integrity failure preserve output and
repair before continuing; negative scientific gates cause no tuning or extra seeds.
Stop after65 conditions and independent validation/report. Current-model III-B
closeout explicitly leaves whole-brain,full dose curves,real biological modules,
autonomous recall and III-C/D/IV untested. Select one next information-rich study.
