# Energy-matched orientation control for frozen-head distortion

Registered 2026-09-29 AFTER the low-mode-dominance hypothesis failed and BEFORE
orientation reference values or randomized control predictions were computed.
Second authorized task. New exploratory question; does not revise the preceding
negative hypothesis or choose a different mode cutoff.

Question: is observed residual score distortion unusually large because paired
state differences align with sensitive directions of the frozen readout?
Reuse all34 residual_modes cases, same low/high split24/24, streams, seeds,
primary6lags, all11lags and smoke exclusion. No new trajectories, head/adapter
fits, clipping or label-based choices. Source train basis stays fixed.

For residual Q in source mode coordinates and H=VᵀW, apply a random signed
permutation independently within each fixed group of24 modes. One transform is
shared across all test rows and lags of a case. This preserves every row's group
norm and the residual covariance eigenvalues within each group (also full
covariance spectrum under the joint orthogonal transform), but destroys its
orientation relative to H and individual source singular values. It does not
preserve neuron identities, marginal per-mode variances, graph topology or neural
trajectories. These are geometric controls, NOT connectome rewiring controls.

Compute the EXACT expectation over independent uniform permutations/signs:
E_ref = mean(||Q_high||²) ||H_high||_F²/24
      + mean(||Q_low||²) ||H_low||_F²/24, separately for each4-output lag head.
Cross terms have zero expectation. This is the primary comparator and has no
Monte Carlo estimation error. Supplement with16 predetermined transforms,
seed51400–51415, reused for every case, and save per-replicate energies; no
selecting a favorable transform, no Monte Carlo p-value or sampling gate.

Primary metric log2(E_actual/E_ref). Positive means more amplification than
energy-matched isotropic orientation; negative means less. Also report raw ratio,
actual/reference energy and draw distribution, inherited accuracy/R² for context.
Undefined denominator<=1e−24 fails the claim without discarding a case.
Average primary lag log-ratios, then circuits per mapping block, cohorts separate.
H: preferential alignment is supported for a direction only if BOTH cohorts
have mean log2 ratio>=1 (geometric ratio>=2), positive blocks>=4/5 main/all3
confirmation. Preserve negative ratios and failed criteria. No interpretation
of non-significance as equivalence. Bootstrap10000 blocks seed51399; per-seed
raw,mean,median,sample variance,bootstrap95, paired standardized log-ratio.

Reproduce first-study checkpoints/algebra and moment-aligned baselines before
each control. Verify row norms, covariance spectra and analytic expectation
using an exact exhaustive 2D synthetic sign/permutation ensemble; independent
artifact verifier reconstructs draws and uses covariance traces for expectation.
Smoke→main→archived confirmation regardless outcome. Runtime1800s, sampled
RSS3GiB; stop on integrity failure and preserve partial artifacts. No change to
source results or source hashes. Record code/config/git/environment provenance.

This tests orientation relative to this decoder, not memory capacity, probability
calibration, biological circuitry, causal neural ablation, recurrent learning,
whole-brain or actual-topology advantage. Cohorts are previously observed; no
new independent confirmation. Whatever the outcome, do not add another rescue
condition in this task.
