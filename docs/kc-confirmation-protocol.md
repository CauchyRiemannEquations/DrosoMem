# ACT III-A: fresh-seed confirmation of frozen/refit lesion decoding

Registered before any new-seed states, fits or transfer scores. Discovery
results/kc_frozen (pinned manifest in config) showed full KCg26.250% frozen,
79.219% refit; matched26.377%/77.156%. Primary refit-benefit gate passed; frozen
access/retention and KCg specificity failed. Preserve all discovery criteria.

## Primary question and decision

Does the full KCg refit benefit replicate on three fresh paired mapping/stream
blocks? Use exactly the discovery H1: mean(refit−frozen accuracy)>=5pp,
ALL3 block differences>0, and KCg refit past access (ALL3 margins over majority
and its shifted-label null>=5pp; mean R²>0). Confirmation is supported only if
this new cohort passes AND the pinned discovery H1 passed. Report cohorts
separately; never pool six blocks to rescue a failed confirmation.
This is decoder refitting, not internal connectome learning.

## Fixed design and seeds

Main mapping71142–71144,train72142–72144,test73142–73144.
Matched RNG seeds74142–74150,three per block in config order.
Smoke mapping71001,train72001,test73001,matched74001–74003.
Seed audit checks all tracked config/result JSON seed-named fields at4272cb0:
5888files,333distinct prior values,no candidate collisions. See the saved audit.
Audit scope excludes documentation and arbitrary non-seed numerical fields.

Same source-derived partial circuits701/702,686nodes,512KC,48MBON observations,
threshold5,edges3309/3241. These are existing biological graph strata, not new
biological replicates. KCg annotation and matching are unchanged: roleKC with
cell_type startingKCg,254/226neurons. Intact,full gamma,matched0/1/2.
Each matched mask samples ALL KC,including KCg,with exact joint raw in-degree,
out-degree,and4-symbol input-exposure-bitmask histogram. Report overlaps and
unmatched removed edge/strength quantities. No subset follow-up in confirmation.

K4 uniform pseudo-random streams,train2000/test1000,warmup100,independent train/
test seeds. Smoke200/100,c701 only. input fraction.1/amplitude.5,gain.9/leak.6,
incoming-L1 before lesion,mbon_after_kc. Zero removed rows/columns/input; surviving
weights unchanged,no renormalization; verify removed states stay zero.
All11lags0,1,2,3,4,5,8,12,16,24,32; primary1,2,3,4,5,8.
Real/null ridge alpha1,48features,196coefficients per lag head,train std floor1e−5.
Each condition gets its own refit. Null labels cyclically shifted half train.

Frozen transfer uses paired intact source mean/scale/real/null weights/bias
UNCHANGED. No target normalization, moment alignment or coefficient adaptation.
Target labels cannot enter frozen inference. Compare on identical target test
states to the newly saved target-refit head. Fresh input maps differ from
discovery; within a paired block intended input maps and observation IDs match.

## Baseline, execution and engineering gates

First reproduce archived structural_k4 real c701/s34142 complete checkpoint
and metrics. Then5new smoke conditions/full replays and4frozen smoke transfers;
then30new main conditions/full replays and24frozen main transfers. Each fresh
intact head must exactly predict its own stored scores;7source self-checks.
Run all conditions/seeds regardless outcomes, no seed replacement or tuning.
Retain partial artifacts on error. Stop on nonfinite/integrity failures or
combined1800seconds/3GiB50ms sampled RSS across both stages. Commit protocol/
config before implementation, commit implementation before execution.

Preserve numerical engines kc_ablation,kc_frozen and normalization_transfer.
Archive graph/masks,input maps,features,symbols,labels,real/null/refit/frozen
scores,heads,training/test metrics,neural rank/activity/sparsity/cosine/decay,
runtime/RSS/environment/git/config/source/artifact identities. Independent
mask/graph/stream/refit audit and direct-formula frozen metrics/statistics audit,
exact full replay,independent augmented least squares. Check all old result hashes.

## Secondary analyses (not additional primary claims)

For full gamma and matched mean, carry forward frozen past access (ALL3
frequency/null margins>=5pp,mean R²>0),5pp retention (mean and ALL3 refit−frozen
<=5pp),portable=access AND retention,matched-group refit benefit by the H1 rule.
Report frozen specificity: matched−gamma mean>=5pp and ALL3>0 plus intact past
access; refit specificity uses the prior KC ablation rule with the same5pp.
Also report intact−gamma and intact−matched refit differences without changing
the primary question. Opposite-direction improvements are descriptive only.

Average primary lags→three matched draws→circuits within each mapping block.
n=3,not number of masks,circuits,time rows. Preserve per-draw/per-circuit/per-lag
raw results,mean,median,sample variance,bootstrap95(10000draws,seed74399),paired
dz and directions for differences. Smoke excluded. No p-value success gate.

Failure means the predefined confirmation is unsupported; do not tune or use
discovery to change that. If supported, claim only replication across new
pseudorandom input/mapping/mask realizations on these SAME biological circuits.
No whole-brain benefit,autonomous recall,formal memory capacity,biological KCg
necessity,actual fly learning or internal plasticity claim follows.
