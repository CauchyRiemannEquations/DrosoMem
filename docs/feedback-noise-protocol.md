# Observation allocation under autonomous feedback

Prospective execution and new-noise protocol, with a separately labeled archived
prefix reanalysis. Commit before extracting new prefix endpoints or executing
rollouts. Base revision0e01387. This follows the completed ACT V; do not reopen its
completion or overwrite its negative results. This extension executes only this study.

## Question and hypotheses

Does training-SD-proportional observation noise, at equal expected raw energy,
improve exact autonomous prefix over flat-coordinate noise? The prior teacher
accuracy improvement does not answer this question. Primary is within brain1;
legacy5 and brain5 are prespecified secondary conditions. It is not a primary
whole-versus-partial superiority test, nor an internal-learning experiment.

Keep all48 archived heads, two inherited cohorts7142–7146 and381142–381144,
circuits701/702 and legacy5/brain5/brain1. Same input IDs,48 observed MBONs,
482 head parameters,π offset0/length200/prompt314,197 generated symbols.
Build original rate models with incoming-L1 gain.9/leak.6/mbon_after_kc.
No head, preprocessing or internal-weight adaptation. No new model seeds.
Require parent manifest SHA256 and independent validation in config to match.

## Intervention and information access

Use q=median(clean-training coordinate SD), a_j=SD_j/RMS(SD), ddof0, on199 rows.
Freeze these values. Relative doses remain[.0003,.03,.3]; same fixed48 coordinates.
At each prediction use clip(state_observed + rq*z,−1,1), or
clip(state_observed + rq*a*z,−1,1). This noise does NOT modify the neural state.
After argmax, feed the **generated** symbol to the unchanged neural model.
The autonomous function accepts only model, observation indices, head, prompt,
horizon, precomputed noise and amplitudes; no target/reference sequence.
Score afterward. Teacher accuracy and state traces are separate diagnostics.

Expected pre-clipping noise energy is48(rq)² in each arm of a given model.
Realized energy, clipping and energies between graphs need not match. Save all
realized energies and distinguish actual reservoir observations from noisy head
features. An effect here is not rescue of all-neuron internal state perturbation.

Fresh noise seeds418001/418002/418003, PCG64 SeedSequence([noise_seed,model_seed,
circuit,2]),197×48 normals in sorted observed-ID order then restore head order.
Same draws across arms/doses/graphs for a model/circuit. All three streams run
unconditionally. Noise draws are repeated measures, not additional model samples.

## Prefix certificate and actual rollouts

Before the first error, autonomous inputs equal correct teacher inputs. Determinism
and identical time-indexed observation noise imply equal state, head probabilities
and output through the first error. Thus the teacher trace gives an exact-prefix
certificate, not the autonomous output after the first error. Numerical equality
must be checked where actual trajectories are executed; no full autonomous sequence
may be fabricated from this certificate.

1. Archived reanalysis: all48×4saved streams×3doses×2arms=1152 certificates from
   saved allocation predictions. Label retrospective; exclude from primary decisions.
2. Prospective fresh prefix: all48×3new streams×3doses×2arms=864 certificates.
   Save fresh teacher features/predictions/probabilities and correctness. No new
   neural trajectory is needed for these, since clean teacher states already exist.
3. Actual autonomous paths: for first fresh stream418001 only, all48×3doses×2arms
   =288 full197-step rollouts, plus48 actual clean/zero replays:336 neural paths.
   Save full generated sequence, probabilities, raw/noisy observations, active
   neuron counts, norms, clipping and position/region accuracy. Compare first-error
   position and all pre-error states/probabilities with the certificates.

This subset is fixed before outcomes. The other two streams have certified prefix
endpoints but no claimed post-error autonomous accuracy. Verify frozen graph/head
identities and clean baseline output. A clean rollout is an actual zero-injection
test; other zero rows must not be counted as new trajectories.

## Metrics and fixed decisions

PMS is generated exact prefix before first error, excluding prompt, capped197.
Report first error1-based or null if censored, L*log2(10) recalled-prefix bits,
raw PMS and capped retention min(PMS/cleanPMS,1). A zero clean score makes retention
undefined and the relevant gate inconclusive; no epsilon or exclusion.

Primary fresh-noise endpoint: for each model seed equally average3doses×2circuits
×3noise draws separately within each arm, then compare allocated−flat.
Require mean gain>=2symbols AND capped-retention gain>=.10, both positive in
>=4/5 discovery or3/3confirmation model-seed blocks. Confirm only if both cohorts
pass. Secondary legacy5/brain5 use the same rule. Archived endpoints are descriptive
only, and a failed primary cannot be replaced by a secondary or a selected dose.
Full-rollout accuracy/teacher gap, prefix32 and raw-state diagnostics are descriptive
secondary outcomes, not alternate success gates. No p-value decision.

Save all cases/doses/streams, per-seed paired differences, mean/median/sample
variance,10000 seed-block bootstrap intervals(seed418399), paired dz and sign counts.
Keep cohorts separate. These previously analyzed model cohorts are not new independent
holdouts; fresh exposure confirmation is narrower than new-model generalization.

## Verification and finite budget

Synthetic tests must cover target-free autonomous signature, no observation-noise
writeback, feedback timing, zero behavior, clipping, prefix equality and divergence
after first error, frozen parameters and unchanged joint gate. Reuse tested canonical
allocation/noise routines without modifying their archived source.
Smoke:7142/c701 in allthree graphs, first8 positions, allstateless certificates and
allseven actual paths permodel. Save separately; no main inference.

Independently reconstruct all2016 certificate endpoints, every metric/statistic,
all zero and parent/hash checks. Independently replay all16partial cases and all
paths for whole cases(7142,701) and(381142,702):20cases×7=140full paths.
Also independently replay every smoke path. Numerical tolerancesatol1e-12/rtol1e-10,
exact predicted symbols. Check all saved main paths for pre-error certificate
agreement, without claiming independent full replay for the unselected whole paths.

Two workers maximum, one numerical thread each;900s/3GiB worker process-tree RSS,
7200s main supervisor and3600s verifier. Sample tree RSS at.2s and verifier at.05s.
Stop and preserve failures on nonfinite/integrity/resource problems. No additional
seed, dose, shorter horizon or criterion change to rescue a result. Finish with
plots, reports, updated status and one next proposal, whether positive or negative.
