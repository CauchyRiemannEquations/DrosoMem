# ACT IV: local direction through an unchanged learning trajectory

Status: preregistration; commit this protocol, config, runner, independent verifier
and tests before outcomes. A bounded mechanistic diagnostic, not a new learning rule.

## Question and prior evidence

Does useful local reward direction attenuate from epoch 0 to 10 of the SAME
training trajectory? Prior [initial-direction](reward-direction-results.md) and
[noise-matching](reward-noise-results.md) studies used different task seeds, so
cannot be spliced into a learning curve. Neither passed its full criterion.
This study reuses the original local-reward model/task seeds through time;
new proposal/evaluation noise is NOT fresh independent training confirmation.

The next-work note initially suggested symmetric probes. Before new outcomes,
a read-only audit of archived final weights found many KC→MBON magnitudes near
1e-4 of original. Symmetric perturbations of a useful restoring direction could
therefore force near-zero radii. We explicitly replace that proposed design with
ONE-SIDED forward probes and signed random controls. No favorable orientation
selection. The historical direction result and its ± statistic remain unchanged.

## Fixed model and task

Config: `configs/reward_trajectory.json`, canonical SHA256
`f0a2d83e03d1970616b3224ec0fb244f07c329561b14b736e60325a474b5bd37`.
Source: `results/local_reward`, root manifest SHA256
`33c04aef89209acc8ca16e9c9e4ba7ac7f3eaf7218e2ebcfdc6468d541582f31`.
Use ALL thirteen real/contingent source cases, including smoke.
Main: circuit seeds701/702, input seeds241142–241144(discovery) and251142–251144
(archival confirmation cohort). Average two circuits before inference: n=3 per
cohort. Smoke c701/s241001 is excluded from inference. New proposal, direction,
evaluation seeds are explicitly mapped in config and audited in
[seed audit](reward-trajectory-seed-audit.json).

686-neuron actual-connectome partial rate model, threshold5,3309/3241 edges,
1431/1474 plastic KC→MBON edges,48 observed MBONs. Incoming-L1 normalization,
gain.9,leak.6,one synchronous tanh step. K4 iid integer inputs; target lag2.
Warmup100,train2000,test1000;10 epochs (smoke200/100 samples,1 epoch).
Identical source input bank, codebook, symbols and training noise. Fixed nearest
code decoder has ZERO fitted parameters. This is continuously driven delayed-symbol
decoding; there is no autonomous recall, Pi Memory Score or capacity claim here.
Rule remains scalar reward, covariance eligibility(.8),activity/baseline EMA(.05),
learning rate.05,noiseSD.02,original-magnitude floor1e-4 followed by ORIGINAL row
mass restoration. The floor is applied BEFORE projection; it is not a strict
final absolute lower bound. Do not retune or add biological rules.

## Capture and frozen local proposals

Exactly replay original contingent training; save weights at epochs0,1,5,10
(smoke0,1), histories, actions,rewards,advantages and trace norms. Final weights,
training histories and clean final test features must equal original archives.
For each checkpoint, reset state/trace/baseline and run10 frozen diagnostic passes
(smoke1), with fresh common proposal-noise stream through TIME. Apply each local
one-step update temporarily, accumulate its magnitude displacement, then restore
the checkpoint before the next forward step. Keep original initial floor references
and row budgets, rather than reinitializing them at each checkpoint. These are
reset-state checkpoint diagnostics, not the live online eligibility trajectory.

Project the mean displacement into row-sum-zero tangent directions (remove
roundoff residual using CURRENT magnitudes). Generate five Gaussian random vectors
weighted by CURRENT magnitudes, tangent project, normalize each to unit L2.
Random stream is identical across time, but current-magnitude weighting changes
axes. Controls are not isotropic or matched for per-edge allocation/locality.
Store raw mean, normalized vectors, complete proposal logs and norms.

Nominal absolute radius is .01 × ORIGINAL plastic-magnitude L2 norm.
Use ONE actual radius for all epochs/directions in a circuit/input block:
min(nominal, .5 × minimum(current_magnitude / -unit_direction) over negative
components, all directions and all epochs). This preserves positivity and at least
half the current magnitude, but does not impose a new original-floor constraint on
probe graphs. Positive components may restore small weights. Preserve signs,
nonplastic weights and per-postsynaptic row mass. If any direction norm is zero,
radius=0 and mark non-identifiable. Radius<nominal/10 is resolution-limited;
if ANY main circuit/input block in a cohort is limited, temporal endpoints in that
cohort are not identifiable. Report scores and radii regardless. Never enlarge
radius, omit checkpoints or replace directions after seeing results.

## Evaluation, hypotheses and decision rules

Seven graphs/checkpoint: unperturbed, local forward, five random forward.
Evaluate original held-out iid test sequence, same codebook. Primary: training
noiseSD.02,32 common noise realizations through TIME and across ALL arms. Secondary:
clean1 realization. Smoke2 noisy+1 clean. No weight learning during evaluation.
Save per-realization full scores/predictions recoverable by argmax, accuracies,
clean MBON trajectories, graph checkpoints, graph/hash/environment provenance.
Noisy replicates are Monte Carlo integration, not independent model sample size.

For each seed, average circuit accuracies FIRST. At each epoch:
- forward_gain = local accuracy − unperturbed accuracy;
- above_random = local accuracy − mean of five random accuracies.
Utility gate: both contrasts have mean≥0.10 percentage point and are strictly
positive in ALL THREE seeds of a cohort. Evaluate every predefined epoch.
Primary attenuation hypothesis: identifiable + epoch0 utility + BOTH contrasts
(epoch0 − epoch10) mean≥0.10pp and strictly positive in ALL THREE seeds.
Secondary persistence: identifiable + epoch0 utility + epoch10 utility.
Both endpoints require passage in BOTH original cohorts, without pooling.
They are not mutually exclusive: a signal can weaken while remaining useful.
Epoch1/5 describe the trajectory; cannot select a best/worst time.
Clean results cannot rescue a failed noisy endpoint. Failure to satisfy the gate
is not proof of absence; resolution-limited failure is inconclusive by design.
No p-value success rule. Save per-seed,paired differences,mean,median,sample variance,
10,000-seed-block bootstrap percentile95% CI and paired standardized mean/sd;
bootstrap seed338399. n=3 CIs are descriptive and have limited coverage resolution.

Representation diagnostics (descriptive): centered effective rank, neuron-state
standard deviation,saturation,local direction cosine to epoch0,proposal norm,
fraction of plastic magnitudes<.001 of initial. Rank change alone does not establish
information loss or explain causal learning failure. No classifier refit is added.

## Budget, stop and verification

50 checkpoints,350 graphs,11,130 evaluation trajectories,700 graph/mode metric rows
including smoke. Stop on nonfinite values, provenance/replay mismatch, resource
limit or invariant failure; save failure and do not overwrite. Stop entire study
rather than continue past a failed smoke. Smoke executes first. Hard limit7200s,
3GiB sampled process RSS per runner/verifier,one BLAS thread. No sweeps/restarts to
find positive results. Exact original final replay is baseline reproduction.
Independent manual local-rule replay reconstructs each intermediate snapshot,
frozen proposals and ALL scalar evaluation scores; validates radius, graph
constraints, aggregate statistics and predeclared gates. Protect prior files by
SHA256 before/after. New folders only. Commit protocol before any outcomes.

Commands (repository root, existing isolated environment):
```powershell
$env:PYTHONPATH='src'
../venv-act1/Scripts/python.exe -m pytest -q
../venv-act1/Scripts/python.exe scripts/reward_trajectory.py --out results/reward_trajectory
../venv-act1/Scripts/python.exe scripts/verify_reward_trajectory.py results/reward_trajectory --out results/reward_trajectory_validation
../venv-act1/Scripts/python.exe scripts/plot_reward_trajectory.py results/reward_trajectory --out results/reward_trajectory_analysis
```

Interpretation restricted to this computational model using actual Drosophila
connectome structure. Internal weight change is established by replay; successful
biological learning, fly memory, whole-brain benefit, autonomous recall or formal
memory capacity are not established by this diagnostic.

Preflight note: a synthetic floor test initially set learning rate to zero, which
intentionally bypasses all updates including floor restoration in the existing
learner. The fixture now uses a negligible nonzero rate to exercise the floor.
The experimental rate remains .05; no real outcomes had been generated.
