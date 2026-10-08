# M5: finite-lag functional symbol-contrast sensitivity

Date: 2026-10-08. Prospective protocol to be committed before new source-graph
trajectory, sensitivity, replacement or decoding outcomes.
[Config](../configs/temporal_functional_sensitivity.json).
Baseline main `4a72db553f80317c282f48561e33b5edbfc3a765`;
new namespace `results/functional_sensitivity_v1/`. Prior evidence is immutable.

## Question and registered hypothesis

How does the actual signed/nonlinear effect of a past input contrast on the
observed reservoir state change with temporal lag, and how does this differ
from structural path support and iid linear decoding?

Primary hypothesis: on the fixed zero-carry original partial graph, normalized
RMS observed sensitivity at lag5 is at most one tenth of immediate sensitivity
in **every one of six fresh blocks**. This is an operational order-of-magnitude
attenuation test, not a memory-improvement or memory-failure test. Drive0.6 and
incoming-L1 gain0.9 suggest contraction, but contraction of the entire state
does not guarantee this ratio after an observed-coordinate projection. M3's
short-lag results motivate the fixed lag5; no new outcomes inform the choice.

Derivative correctness is a separate technical eligibility requirement. A
scientific FAIL remains FAIL after successful validation. No favorable lag,
extra seed, weaker threshold, different decoder, pulse direction or dynamics
will be selected after results. P3 physiology/spiking/plasticity is excluded.

## Fixed graph, inputs, observations and decoding context

Reuse original source-derived circuit701 legacy5 primary and brain5 secondary;
no DAG intervention or new rewiring. Same labeled source neurons, current
KC→MBON block, gain0.9 incoming-L1 normalization, no post-construction rescaling.
Use M3 intact_synaptic semantics: direct carry0, previous-state synaptic drive
on, drive0.6, mbon_after_kc. Do not replace carry removal with leak1.
Only the external alpha1 ridge head learns; recurrent weights remain fixed.

Six blocks1500001–1500006; mappings1510001–1510006, iid train1520001–1520006,
iid test1530001–1530006. Whole secondary uses the firstthree paired mappings
and streams. K10; each symbol stimulates51 of the same512 KC input neurons at
amplitude0.5. Same48 observed MBON IDs. Train and test reset independently,
warmup200 each, train4000/test2000 scored samples. Decode
`state(t)->symbol(t-lag)` for lag0..20 with490 affine ridge parameters per lag,
train-only means and SD floor1e-5. Chance0.1, train class-frequency and train
current-symbol-only lookup are reported. Lag0 is a separate sanity outcome.
No autonomous feedback, biological replicate inference or formal capacity claim.

Fresh TDC is descriptive context, with no second scientific pass criterion.
It is not identified with sensitivity norm, and tiny signals can remain
linearly decodable through coordinate scaling. Never differentiate argmax
accuracy or equate norm attenuation with loss of information.

## Pulse and operational sensitivity definition

Fix32 test anchors
`tau = warmup + floor(linspace(0,test_samples-1-20,32))`.
All anchors and their20 future steps are within the independent scored test
stream. They are probes within one computational block, not32 independent
models or animals. Record the exact pre-pulse full state at tau−1.

For actual symbol s at tau, fix alternative a=(s+1)%10 and contrast
`d=p_a-p_s`. At this ONE step use `u_tau(eta)=p_s+eta*d`; all future inputs and
the exact pre-pulse state remain unchanged. Differentiate at eta0. This is a
derivative of a continuous extension of the encoder, not of a categorical
random variable. Negative eta in centered finite differences is explicitly an
artificial off-code input. Eta1 is a valid replacement of that one symbol.
No persistent mutation of encoder patterns is allowed.

Let v_p,l be the full-state derivative at tau+l along the UNPERTURBED trajectory.
Per-probe normalized observed gain is
`g_p,l=||v_p,l[observed]||_2/||d_p||_2`.
Pooled gain is `G_l=sqrt(mean_p(g_p,l^2))`; relative gain is `R_l=G_l/G_0`.
Normalize each probe before pooling, not a pooled unnormalized ratio.
No zero-direction replacement or denominator floor. Zero contrasts or
`G_0<1e-8` make the assay invalid, not a reason to remove probes/add seeds.
The small G0 requirement is a predefined normalization-resolution check.

Primary PASS iff all eligibility checks pass and `R_5<=0.10` in all six primary
blocks. Evaluate equivalently `S_5<=S_0/100`, where
`S_l=(1/32) sum_p [(sum_i v_p,l,i^2)/(sum_j d_p,j^2)]`.
For the official decision compute these sums as exact rational arithmetic on
the saved float64 values (Fraction.from_float), with threshold1/100 exactly.
Equality passes. Independently reconstructed derivatives must also give the
same decision; disagreement near a boundary is validation failure, not rounding
into success. Whole firstthree use the same rule separately, never pooled as
nine independent input replicates. Smoke endpoint is null.

## Dynamics derivative and mathematical reference

With b0.6, first provisional update is `f=b*tanh(W*x+u)`. Final non-MBON state
is f; MBON uses `b*tanh(W_M*m+u_M)`, where m equals previous x except its KC
coordinates equal newly updated f_KC. Keep explicit zero-carry additions when
matching frozen floating-state trajectory digests.

For a previous-state tangent v and an input direction e (d at pulse,0 later),
first tangent is `r=b*(1-tanh(W*x+u)^2)*(W*v+e)`. Replace KC coordinates of v
with r_KC, then MBON tangent is
`b*(1-tanh(W_M*m+u_M)^2)*(W_M*mixed_tangent+e_M)`.
Use actual final MBON preactivation for its derivative, not provisional MBON
values, and retain the current KC→MBON chain rule.

Compute an unsigned envelope using identical unperturbed derivative factors,
absolute W and absolute initial d, with the same schedule. Coordinatewise
`abs(v)<=envelope` must hold within atol1e-12+rtol1e-12 descriptive reduction
tolerance. This envelope is a bound involving path magnitude/direction; its
ratio with signed sensitivity is not a unique inhibitory or cycle mechanism.

Because each incoming absolute row sum is <=0.9 and tanh is1-Lipschitz,
the scheduled zero-carry map has global infinity Lipschitz bound q0.54.
The KC coordinates inserted into the MBON mixed state themselves contract by
at most q, so MBON also obeys this bound. Since max(abs(d))<=0.5,
`||v_l||_infinity<=0.3*0.54^l`. The same upper bound holds for a finite eta1
replacement difference. Verify these bounds for every probe/lag. They are
upper bounds, not exact decay or biologically calibrated time constants.
Observed R need not decrease monotonically as hidden signals reach MBONs.

## Independent validity checks and controls

Fix finite-difference probe indices0,10,21,31 and epsilons1e-4,1e-5. Start +/-
trajectories from the authenticated exact same pre-state, inject pulse once,
and replay all21 lags. On EVERY full-state coordinate require
`abs(centeredFD-JVP)<=2e-9+2e-5*abs(JVP)` for both epsilons.
Record full-state error/excess statistics, observed FD vectors, trajectory
digests and selected complete tangent/envelope arrays. At tiny long-lag
derivatives the absolute tolerance dominates; no resolved nonzero influence
may be claimed solely from derivatives below that numerical resolution.

The separate verifier uses its own manually scheduled trajectory and a
complex-step derivative: imaginary pulse `1e-20*d`, derivative imaginary
state/1e-20. No main Jacobian/trajectory/fitter is imported. Compare every saved
observed tangent for all32 anchors and the selected4 FULL-state tangents at
all21 lags using atol1e-12+rtol1e-9. This independent analytic extension is
supported by smooth tanh; it does not supply biological validation. Also
independently replay centered finite differences and their full-vector gate.

Instantaneous reference disables prior-state drive but retains the same
current KC→MBON block, b and input mapping. Analytic prototypes and actual
replay must show EXACT zero past sensitivity/replacement influence after
lag0, at all coordinates. Its immediate derivative and response are recorded;
do not require they equal history-bearing responses or cross-graph amplitudes.
No trained instantaneous readout is needed for this analytical negative control.

Each eta1 valid replacement is replayed for all32 anchors/all21 lags. Save
observed finite differences, full-state norms, final states/digests and ratios
to local tangents descriptively, without a new success threshold. Sign/support
panel for the contrast's nonzero source set is boolean structural reachability,
not a performance curve or proof that every input/output contributes.

Synthetic safeguards before source outcomes cover same-step chain rule, signed
path cancellation despite support, delayed reemergence in observed coordinates,
instantaneous zero-history, eta1 single-pulse replacement, complex-step/FD
agreement, both epsilon gates, contraction/envelope bounds, zero-direction or
unresolved denominator invalidity, exact all-seed/equality rules, labels,
train-only ridge normalization and smoke-null. Fix these tests and tolerances
before new source-graph outcomes; do not tune them to a failed main result.

## Resources, provenance, validation and stopping

Run protocol/config commit, immutable baseline seal, implementation/source
freeze, synthetic safeguards, smoke, independent smoke, fixed main,
independent main, synthesis and final audit in that order. Do not edit any
source while safeguards/source capture or experiments run. Capture clean
tracked tree and stable HEAD before/after hashing, authenticate each tracked
SHA against recorded Git blobs, then recheck source unchanged before sealing.

Use4 workers with1 numerical thread each, per-stage7200 seconds and4GiB
aggregate sampled process-tree RSS budget. Save individual process RSS/OS
peaks plus50ms aggregate sampling; shared pages may be counted multiple times.
No adaptive budget, probe/graph/lag/epsilon/seed increase after outcomes.
Save full baseline trajectory digests, observed features, pre-states32×N,
all signed/envelope observed vectors and selected4 full-vector archives rather
than dense N×N Jacobians or all6000 full states. Split selected probe archives
to avoid one oversized artifact. No neuron copies or unrolling changes.

Each stage/config/seeds/protocol/source/environment/input/data/graph hash,
resource record, manifest, raw metrics, summaries and verifier checks is kept.
Directories use exist_ok=False; failures/interruptions are sealed and retained.
Technical same-settings repairs may use a fresh directory, with explicit
failure registry; scientific failures never trigger new settings or seeds.

The independent verifier reconstructs source graph/normalization/mappings,
full state traces, labels, SVD ridge heads/predictions, prototype reference,
complex derivatives, envelope/support, finite differences/replacements and
exact saved-float criterion. Report per-block paired curves, mean/median/SD
and descriptive10,000-draw block bootstrap (seed1540001), not animal inference.

Final artifact audit protects all44,147 prior result-file Git identities,
older source/data/config/protocol/tests and the existing source-validation
failure snapshot. Sparse identities are not falsely described as new replays.
Keep P2 failed material gates, M1 assay-invalid and M4 strict INFEASIBLE intact.
No unique cycle/storage/real-wiring/whole-brain superiority or physiological
learning claim follows from any M5 outcome.
