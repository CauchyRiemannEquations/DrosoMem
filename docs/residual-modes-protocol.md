# Source singular modes and residual decoder amplification

Registered 2026-09-29 before decomposition. Exploratory ACT III diagnostic;
both cohorts have already been observed. No new confirmation claim.

Question: after train-only moment alignment, do low-variance source directions
disproportionately account for frozen-head score distortion?

Reuse all34 directions in results/moment_alignment: smoke2 (34001,c701),
main20 (34142–34146,c701/702), archived confirmation12 (41142–41144,c701/702).
Both R→F/F→R; K4,48MBON; all11 original lags. Primary lags1,2,3,4,5,8.
No new head, alignment, neural trajectory, labels, gain or hyperparameter fit.
Baseline: exact replay of saved source-within and moment-aligned real/null scores.

Source training z=(x−source_mean)/source_scale. Thin SVD z=U S Vᵀ, singular
values descending. Basis fitting uses source training features only. Partition
fixed ranks1–24 (high) and25–48 (low), never select a cutoff after outcomes.
Test residual D=target_test_standardized_by_target_train−source_test_standardized.
Q=D V, H=Vᵀ W_source. For each lag with four outputs, C_k=Q_k H_k and
T=sum_k C_k=D W. T is aligned-minus-source-within scores, not error versus truth.

Store all mode energies, basis, singular values, projections and score differences.
State energy e_k=mean(Q_k²), diagonal score energy d_k=mean(||C_k||²), signed
score attribution a_k=mean(C_k·T). sum a_k=mean(||T||²), whereas sum d_k need
not equal actual energy because cross-mode terms can cancel or amplify.
For low/high groups store E_L=mean(||sum_low C||²), E_H and cross=2 mean(T_L·T_H).
Verify E_total=E_L+E_H+cross. Signed low share=sum_low a_k/E_total may be negative
or >1; never clip or call it a probability. State low share=sum_low e_k/sum e_k.
Also store low/high gain E_group/state_energy_group, low train variance share,
boundary relative singular gap, score distortion, inherited accuracy and R².
If required total energy<=1e−24, mark undefined and fail the hypothesis rather
than silently discard a run. Boundary gap<=1e−8 is flagged as basis-sensitive.

Primary hypothesis: low-mode concentration is supported for a direction only
if BOTH cohorts have mean signed low score share>=.75, mean low state share<=.5,
and signed-minus-state share>0 in >=4/5 main/all3 confirmation mapping blocks.
Flagged boundary degeneracy precludes this interpretation. Report failures.
Average primary lags, then circuit strata, then seed blocks (n5/n3 separately).
Report every seed, mean/median/sample variance,10000 paired-block bootstrap95
(seed50399), and paired dz only for signed-minus-state differences. No p-value gate.

Smoke first, then full main and confirmation regardless result. Stop on integrity
failure, nonfinite decomposition,1800 seconds or3GiB sampled RSS; retain partials.
Tolerance for algebraic equivalence: atol1e−9,rtol1e−8. Exact same-code replay
must be bitwise. Independent verifier uses feature-space projectors, checks
orthonormality/covariance diagonalization and all metric/statistic/gate identities.
Synthetic tests cover cancellation, sign invariance and source-only basis fitting.
Hash every prior tracked result, record git/config/code/source hashes and environment.

Limits: an algebraic attribution of decoder distortion, not a causal neural
ablation, absence of representation, autonomous recall or recurrent learning.
Low variance modes mix neurons; they are not biological neuron populations.
The question concerns a computational model using actual Drosophila connectome
structure and its rewired controls. Preserve all previous negative findings.
