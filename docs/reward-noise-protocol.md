# ACT IV — Frozen final-weight noise-match diagnostic

Motivation: the completed reward-direction diagnostic found positive noisy forward
effects in6/6seed blocks but a clean reversal in one discovery seed. That does NOT
explain the older learning failure. This separate preregistered experiment asks
whether evaluating the OLDER final weights with the training noise restores useful
fixed-code learning benefits. The direction study and older learning study have
different input/code seeds; do not treat their rows as paired observations.

## Frozen artifacts, conditions and finite scope

Pin results/local_reward root manifest in config. Reuse all39 final checkpoints:
3smoke,18discovery,18confirmation. No retraining, model/head updates, new sequence,
best-checkpoint selection, noise-level sweep or learning-rate change. Existing
686-cell c701/c702,48MBON,K4,lag2,train2000/test1000,warmup100,leak.6, all original
input/code/weight identities. Frozen,contingent,yoked remain distinct.

Evaluate two conditions: clean noise0 (exact original replay) and noise SD.02 at
MBON drive, the SAME value used in original training.32 independent noise streams
per main checkpoint,2per smoke. SeedSequence([evaluation_seed,replica]) generates
each stream; identical streams shared across arms/circuits of an input-seed block.
Zero-noise scores/features must exactly match each original checkpoint before new
noisy results can be interpreted. Noisy scores are uncalibrated negative distances.
Use ALL test symbols,original lag2 labels; evaluation weights remain fixed.

New noise seeds281001smoke,281142–281144discovery,291142–291144confirmation,
mapped explicitly to original input seeds241001,241142–241144,251142–251144.
Bootstrap298399,10000draws. These are new Monte Carlo noise draws, NOT fresh model
training seeds or a new independently trained confirmation study. Original clean
outcomes have been viewed; noisy test outcomes have not. This is source-informed
diagnostic re-evaluation of two previously fixed cohorts.

## Metrics and unchanged interpretation boundary

Primary accuracy = mean of32 noisy-replicate accuracies. Average two circuits
within each seed, n=3per cohort. Noise samples do not inflate n. Save every replica's
scores and accuracy,case/seed tables,mean/median/variance/bootstrap95/paired dz.
Training-frequency prediction uses original TRAIN label frequency, not test mode.
Define benefit against each control independently: contingent minus frozen/yoked.
Define interaction = benefit(noisy) minus benefit(clean), paired by original seed.

Noisy coding success requires, in BOTH original cohorts:
- contingent noisy accuracy >=frequency+5pp in every seed;
- mean benefit>=5pp against EACH control, positive for every seed.
These are the preceding study's coding thresholds; original clean decisions stay.
Noise interaction requires BOTH controls' mean interaction>=1pp, positive for every
seed in EACH cohort. Predeclare the joint explanation criterion: noisy coding AND
noise interaction both confirmed. Otherwise do not claim removal of training noise
accounts for the old failed primary. Added noise can affect every arm; raw accuracy
improvement alone does not establish a learning-specific effect.

If either gate fails, report all mixed/negative results and do not change noise,
repetitions,learning rule or thresholds. No p-value decision. This does not test
robustness curves (ACT V), autonomous recall or biological memory. Even joint
success would be specific to this model/interface and noise condition.

## Execution and validation

Protocol/config/source commit before noisy smoke/main. Smoke first and excluded
from main inference.39clean+1158noisy=1197trajectories,78aggregate checkpoint/mode
rows.7200s/3GiB per runner/verifier; stop on nonfinite/budget/replay/hash failure,
preserving partial artifacts. Independent scalar evaluation reconstructs every
score,clean state,accuracy,paired interaction and decision. Store source checkpoint
and weight identities,all new scores,config/Git/source/environment metadata,
runtime/RSS and preservation hashes for all previous result files. Final weights
remain in their pinned source artifacts, not copied or overwritten.
