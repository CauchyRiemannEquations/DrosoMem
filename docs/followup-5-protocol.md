# Follow-up 5: physiological constraint and internal learning

Prospectively fixed before new model outcomes. One question: can a single
Hige-2015-constrained γ1/pedc KC→MBON11 depression rule produce selective,
detectable neural responses and useful fixed-decoder sequence information,
while respecting timing and compartment controls?

The primary Hige et al. 2015 in-vivo patch study reports approximately 90%
reduction of the conditioned odor-evoked synaptic current, forward but not
backward timing sensitivity, and compartment specificity. This is a published
**current** observation, not a measurement of this model's weight parameter.
Do not digitize a figure or claim parameter identification from one number.
Use this qualitative/scale anchor as an external plausibility check and report
the distinct measured quantities side by side. Citation:
https://pmc.ncbi.nlm.nih.gov/articles/PMC4674068/.

Use the pinned two 686-neuron circuits, the conservative mapped PPL101 and
MBON11 identities from `configs/compartments.json`, and the existing Phase 5B
local KC→MBON rule as an implementation reference. Three fresh seeded K4
stimulus maps; one training CS+ and separate CS−; no model receives a correct
sequence label, trained readout gradient or reward. Prespecified eta .1,
eligibility 1,000 ms and 10% weight floor from Phase 5B are fixed. Compare
paired forward, backward, no DAN, wrong-compartment DAN, and no-plasticity
arms, all from the same initial weights and inputs. Measure changed edge
support, CS+/CS− KC synaptic drive into mapped MBON11, and actual MBON11
activity in a held-out probe. A second simple fixed anatomical readout and a
new alpha-1 fitted lag-2 ridge are reported separately to distinguish
internal change from external decoding. No hyperparameter fitting to the
reported 90% current magnitude is allowed.

**Primary functional criterion:** both circuits and all three seed blocks
must have a nonzero pre-conditioning CS+ MBON11 response, ≥20% selective
post-conditioning CS+ response suppression with ≤10% CS− change, and exact
zero KC→MBON weight change under all four controls. If baseline activity is
zero, response suppression is undefined and the gate fails. Secondary fixed
anatomical decoding must improve over no-plasticity by ≥5 points in both
circuits and all seeds to support useful internal learning; fitted ridge
cannot substitute for this gate. The physiological current benchmark is
reported as external comparison, not a success threshold on incomparable
model units.

One smoke block then 3 seeds × 2 circuits × 5 arms = 30 cases. Budget 3,600
seconds/3 GiB sampled RSS for execution and verification each. Preserve exact
pre/post weights and traces, all inputs and state trajectories. Independently
recompute the local weight update and every response/decoder metric. Null or
negative outcomes close this bounded rule assay; do not search eta or stimulus
seeds after seeing results. Whole-brain biology remains a separate question.
