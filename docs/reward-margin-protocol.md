# ACT IV: fixed-code margin directional sensitivity

Preregistration. Commit config/protocol/code/tests before real outcomes.
This is a NEW diagnostic endpoint; it does not replace or rescue the inconclusive
finite-accuracy [trajectory study](reward-trajectory-results.md).
[Completion boundary](act4-completion-plan.md) is also fixed before outcomes.

## Question, sources and fixed design

Does the local update direction point toward a larger correct-code margin, and does
that directional utility weaken during unchanged learning? Reuse all50 source
checkpoints at epochs0/1/5/10 (smoke0/1) and their six stored unit directions.
Source `results/reward_trajectory`, manifest
`0c61a3b4637033f66a3a7f0b1e3860ea517328374c521383e552b45ca218104b`.
Original learning source `results/local_reward`, manifest
`33c04aef89209acc8ca16e9c9e4ba7ac7f3eaf7218e2ebcfdc6468d541582f31`.
Canonical config hash:
`95b1cc1bc72940c439739e487e5028be63ed47967e37672b2b79b13a44fd55f3`.

Add fresh paired input seeds341142/341143/341144 on BOTH c701/c702, unconditionally
(not selected after seeing archived results). +1000 train sequence,+2000 test,
+3000 fixed codebook,+4000 training noise,+5000 proposal noise,+6000 random directions,
+7000 evaluation noise. [Seed audit](reward-margin-seed-audit.json).
Archival cohorts keep their original names/seed streams, paired THROUGH TIME.
Fresh cohort is independent task/input/code/noise replication on the same two graphs;
not independent biological/connectome replication.

686-neuron partial graphs,threshold5,3309/3241 edges,48 MBONs,K4 iid symbols,lag2,
warmup100,train2000,test1000,10 epochs,normalization incoming-L1/gain.9/leak.6.
Fixed nearest-code decoder has zero fitted parameters. Rule: scalar reward,
KC→MBON rate covariance eligibility(.8),activity/baseline EMA(.05),learning rate.05,
training noiseSD.02. Original initial-magnitude floor1e-4 followed by original row
mass restoration. No target vector/gradient enters learning. No readout refits.
Fresh runs reproduce the SAME procedure; capture0/1/5/10,freeze at each checkpoint,
reset state/trace/baseline and estimate10-pass average one-step local proposal,
restoring checkpoint after every proposed update. Preserve original floor/budgets.
Five current-magnitude-weighted Gaussian row-tangent random directions, same raw
Gaussian draws across epochs, normalize local/random vectors to unit L2. They are
not isotropic and do not match every per-edge allocation property.

## Margin and analytic tangent dynamics

Score_k(x)=−mean_j (x_j−code_kj)^2, fixed independent codebook.
Margin=score_true−mean(other THREE code scores). No temperature or tunable smoothing.
This smooth diagnostic is not accuracy, a calibrated probability or memory capacity.
Its state derivative is 2/48 × (code_true−mean(other code vectors)).

For each unit magnitude direction u, V has signed plastic entries sign(W)*u,
zero elsewhere. Define W(a)=W+a×s×V with s=.01×ORIGINAL plastic-magnitude L2.
Compute d(margin)/da at a=0 analytically, without forming any changed graph.
s is a reference UNIT SCALE, not a realizable finite probe radius. Do not imply
that a=1 is feasible or that the linear prediction describes finite accuracy.

h_t=(1−leak)h_prev+leak tanh(W h_prev+input+fixed noise).
G_t=(1−leak)G_prev+leak(1−tanh(drive)^2)(W G_prev+s V h_prev), G_initial=0.
All noise samples are independent of W and paired through time/directions.
No derivative is passed to any learning routine. Hash weights before/after analysis.
Save full scores,margins,per-time six-direction sensitivities,replicate means,
clean observed states,checkpoints,inputs,training/proposal logs for fresh cases.

## Primary analysis and locked criteria

Primary:32 training-noiseSD.02 realizations per checkpoint. Secondary clean1.
Smoke2 noisy+1 clean, excluded from inference. Use source evaluation-noise streams
for archived cases so original baseline scores must match EXACTLY.
Total19 blocks(13archival+6fresh),74checkpoints,2382 baseline trajectories,
14292 directional mean sensitivities,888 direction/mode metric rows.
Noisy replicas integrate nuisance noise; n=3 input seeds per cohort after averaging
TWO circuits FIRST. Do not pool discovery,archival confirmation,fresh confirmation.
Bootstrap10000 seed-block resamples,seed358399. Report all seed means,paired
differences,mean,median,variance,percentile95% CI and standardized paired mean/sd.
With n=3, intervals are descriptive, not strong population inference.

At each epoch: local=mean margin slope in local direction;
above_random=local−mean(five random directional slopes).
Numerically resolved utility requires local and above_random each mean>1e-10 and
ALL THREE seeds>1e-10. This epsilon is numerical resolution, not biological/material
importance. Report magnitudes in score units per reference1% displacement.
Primary attenuation: epoch0 utility AND (epoch0−epoch10) for BOTH local and
above_random has mean>1e-10 and ALL seeds>1e-10.
Secondary persistence: utility at BOTH epoch0 and epoch10.
Each endpoint requires passing ALL THREE cohorts, including unconditional fresh
confirmation. The two endpoints can coexist. Epoch1/5 are descriptive; no best
checkpoint selection. Clean findings cannot rescue primary. No p-value gate.
Failure does not prove no local learning or impossibility of other rules.
A positive margin result does not reverse original failed coding/accuracy criteria.

## Verification, stops and completion

Before outcomes: synthetic central differences with fixed epsilon grid1e-3,3e-4,
1e-4; analytic forward tangents versus independently implemented reverse adjoint.
Check common noise, mean reduction, sign and graph invariance. Test seed-level
inference and fresh-cohort requirement. Then commit the protocol.
Real-data independent verifier: all scalar scores exact, clean states exact;
reverse-mode adjoints independently verify ALL14292 trajectory-average slopes
(atol1e-12,rtol1e-9; max absolute discrepancy also≤1e-12=epsilon/100).
It does NOT independently replay every per-time tangent entry; those full traces
are preserved, covered by synthetic derivative tests, and their averages checked.
Fresh training checkpoints and frozen proposals are independently reconstructed.
Preserve prior result SHA256 values and source revision hashes. Fresh output dirs.
Stop on nonfinite values,zero direction norm,provenance/score/derivative mismatch,
7200s or3GiB sampled process RSS per runner/verifier;one BLAS thread.
Execute smoke first; any failed assertion stops study and retains failure artifact.
No tuning or outcome-dependent seed additions. Stop the bounded study after all
specified outputs/verifications, regardless of scientific outcome.

Commands from repository root, isolated pinned Python environment:
```powershell
$env:PYTHONPATH='src'
../venv-act1/Scripts/python.exe -m pytest -q
../venv-act1/Scripts/python.exe scripts/reward_margin.py --out results/reward_margin
../venv-act1/Scripts/python.exe scripts/verify_reward_margin.py results/reward_margin --out results/reward_margin_validation
../venv-act1/Scripts/python.exe scripts/plot_reward_margin.py results/reward_margin --out results/reward_margin_analysis
```

Claims restricted to computational models using actual Drosophila connectome
structure. Continuous-input delayed decoding is not autonomous recall; changes
in code margins are not physiological validation or increased memory capacity.
