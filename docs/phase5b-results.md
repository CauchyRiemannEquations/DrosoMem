# Phase 5B: local dopamine rule works; response criterion fails

2026-09-27. The gamma1/pedc implementation changes existing KC-to-MBON11
connections only when recent KC activity is followed by actual PPL101 spikes.
All locked causal/locality controls pass in both circuits. **The separate neural
response criterion fails:** both isolated test patterns evoke zero MBON11 spikes
before conditioning, so selective post-conditioning suppression cannot be assessed.
Pi transfer shows no improvement: mean recall falls from **3 to 2 digits**.
Keep the existing rate-model research baseline.

[Locked protocol](phase5b-protocol.md) · [Type registry](../configs/compartments.json) ·
[Configuration](../configs/phase5b.json) · [Artifacts](../results/phase5b) ·
[Criteria](../results/phase5b/summary.json).

## What this implements

The verified FlyWire labels resolve one PPL101 and one MBON11 to gamma1/pedc per
circuit, and one PPL106 plus two MBON14 cells to alpha3. Another 168 DAN/MBON
cells remain unmapped in this conservative registry. No locations or compartment
assignments were invented for them. All KCs with existing edges to MBON11 are
eligible; this includes the combined gamma1/pedc domain rather than assuming only
gamma KCs innervate it. Registry evidence is [Li et al. 2020](https://elifesciences.org/articles/62576).

The causal direction, local depression and absence of a postsynaptic-spiking
requirement follow qualitative findings in
[Hige et al. 2015](https://doi.org/10.1016/j.neuron.2015.11.003). The 1,000 ms
eligibility constant, learning rate 0.1 and 10% weight floor are explicit model
choices. Neither their numerical values nor the last-spike eligibility equation
are presented as measurements or a reproduction of receptor kinetics.

In contrast to Phase 5A's classifier-feedback rule, the new rule observes actual
KC and mapped DAN spikes inside Brian2. It receives no correct digit, classifier
gradient or scalar correctness reward. Changes affect subsequent transmissions
online. Existing ordinary signed DAN transmission remains an approximation;
alpha3 receives no inferred copy of the gamma1 learning rule.

## Locked conditioning results

One fresh pattern seed (6142), two overlapping 686-neuron circuits, six arms each.
CS+ is digit 3 and CS- digit 1; no responsive pattern or seed was selected after
observing the silent probes. Each independent arm begins from original weights.

| Measurement | Circuit s701 | Circuit s702 |
|---|---:|---:|
| Existing candidate KC-to-MBON11 edges | 179 | 134 |
| Edges changed by paired conditioning | 14 | 11 |
| Mean depression of CS+-active candidate edges | 28.12% | 28.12% |
| Mean depression over all candidate edges | 2.20% | 2.31% |
| Matching DAN spikes during paired conditioning | 4 | 4 |
| Edges changed outside the selected compartment | 0 | 0 |
| Inactive candidate edges changed | 0 | 0 |
| CS+ MBON11 probe spikes, before -> after | 0 -> 0 | 0 -> 0 |
| CS- MBON11 probe spikes, before -> after | 0 -> 0 | 0 -> 0 |

**All five controls show exactly zero weight changes:** no dopamine, backward
pairing, dopamine without KC input, PPL106 stimulation in the other compartment,
and blocked learning. Backward, dopamine-only and blocked arms each evoke four
PPL101 spikes; the wrong-compartment arm evokes four PPL106 spikes and no PPL101
spikes. The negative controls therefore distinguish missing causal KC activity,
wrong timing, wrong modulator identity and disabled plasticity.

Independent closed-form reconstruction from recorded spikes agrees with online
final weights within **8.88e-16 contact units**, below the locked 1e-10 limit.
Signs and sparse edge support are preserved. Passing these checks validates the
implemented rule and its routing; it does not validate its biological parameters.
Zero baseline probe responses are recorded as unavailable suppression ratios and
failed response criteria, never as a 0% or 100% learning effect.

## Separate pi transfer

Conditioning teaches an association with the artificial digit-3 stimulus. It
does not teach pi. The following experiment asks whether the resulting graph
helps or interferes with the separately supervised pi readout. Each fresh fit
uses 482 parameters and 6,000 updates with the same fixed-32 weighting.

| Circuit | Initial contacts, frozen/refitted head | Conditioned contacts, frozen head | Conditioned contacts, refitted head |
|---|---:|---:|---:|
| s701 | 4 / 4 | 2 | 2 |
| s702 | 2 / 2 | 2 | 2 |
| Mean | 3 / 3 | 2 | 2 |

These are first-error scores excluding `314`; all 197 generated digits are saved.
Teacher-forced accuracy is separately available in
[pi_transfer.csv](../results/phase5b/pi_transfer.csv). One seed and two samples of
one brain do not provide a broad estimate of intervention efficacy.

A descriptive inspection of the saved training arrays further limits the
silence finding: s701 MBON11 emits 102 spikes during the varied pi sequence
before conditioning and 96 afterwards; s702 emits none in either case. Thus the
isolated CS probes being silent does **not** mean that every MBON11 is permanently
inactive under every input. In s701, four neurons' training count sequences change;
in s702, the complete training count array is unchanged despite altered contacts.

## Reproduction and implementation corrections

The design was locked in `5f4f2ae`; finalized execution source is `ea4e837`.
The first attempt stopped before publishing any condition because an exact check
converted stored SI weights back to mV and encountered a 3.55e-15 round-trip
difference. The correction compares the identical SI conversion used on write;
a regression test was added. No learning constant, stimulus, seed or criterion
changed. The incomplete first-attempt ledger was retained in ignored scratch.

The main run checkpointed the first condition and resumed the remaining eleven.
An initial verification correctly rejected a test log that was still being
written when its hash was captured. After the test process finished, ordinary
resume finalized the manifest; experiment checkpoint arrays remained unchanged.

The independent complete rerun reproduces **all 12 conditioning cases and all
eight 197-digit pi rollouts exactly**. Every saved array and JSON payload matches:
conditioning spikes, gate events, adapted weights, baseline/post probes, training
states, newly fitted head arrays, teacher predictions and recall spikes. Each
complete rebuild performs six fresh readout fits and twelve independent
closed-form LTD checks. See the [verification record](../results/phase5b/verification.json).

The full LIF suite passes **181 tests** (29 upstream Pyparsing deprecation
warnings). The unchanged rate environment passes **148 tests**, with three
optional LIF test modules explicitly skipped. Both logs are archived. Use the
separate LIF dependency lock and the [protocol commands](phase5b-protocol.md#audit-and-reproduction).

## Completion scope and next research work

The source-constrained gamma1/pedc Phase 5B implementation and controlled assay
are complete; the functional response criterion failed. This is not a validated
all-compartment dopamine model, a physiological calibration, or reward-only pi
acquisition. Those extensions remain unestablished.

Phase 6 remains unfinished. Its first work should audit input retained by the
512-of-2,580-KC sampling, benchmark sparse memory/runtime, and test whether the
input and activity limitations persist as graph coverage grows. Cell sampling
alone does not quantify lost synaptic weight or prove the cause of silent probes;
that requires a measured, separately locked comparison. Preserve the failed
response result and complete those prerequisites before claiming a whole-brain
model or improved sequence recall.
