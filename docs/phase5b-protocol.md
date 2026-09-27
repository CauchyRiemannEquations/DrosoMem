# Phase 5B: compartment-local dopamine-dependent depression

Editorial update: research terminology only; the registered numerical design is
unchanged. Original protocol bytes are preserved in Git history; see
[historical reproduction](historical-reproduction.md).

Locked before circuit outcomes, 2026-09-27. Commit code, tests, registry and
`configs/phase5b.json` before the main run. Preserve failures without tuning.

## Evidence and conservative mapping

[Li et al., eLife 2020](https://elifesciences.org/articles/62576),
[figures](https://elifesciences.org/articles/62576/figures), associates PPL101 and
MBON11 with gamma1/pedc. Figure 2C associates PPL106 and MBON14 with alpha3.
`configs/compartments.json` records these exact type names. Apply them to the
already checksum-verified FlyWire annotations; preserve root IDs as strings,
cross-check role, side and hemibrain type, and fail on missing/conflicting labels.
No substring guesses, assignments to ambiguous types or fabricated spatial
synapse coordinates. This is a type-level crosswalk, not a new segmentation.

[Hige et al., Neuron 2015](https://doi.org/10.1016/j.neuron.2015.11.003),
[author-hosted paper](https://www.janelia.org/sites/default/files/Labs/1-s2.0-S0896627315009824-main.pdf),
supports stimulus-specific KC-to-MBON depression in gamma1/pedc when sensory
activity precedes DAN activation. Its backward control does not induce the same
depression; induction does not require MBON spiking. The experiments also show
that induction rules differ between compartments. Accordingly, this study models
gamma1/pedc depression only. Alpha3 is an anatomical negative control, not assigned
the gamma1 learning rule.

The paper's short forward schedule pairs a 1 s sensory stimulus with four light
pulses at 0.2, 0.7, 1.2 and 1.7 s. Its backward schedule starts sensory stimulation
0.5 s after the final pulse. We retain those timing relationships but replace
odor with an artificial KC pattern and each light pulse with a direct voltage
kick to a mapped DAN. A light pulse's biological spike train is **not** equated
to the model's evoked spike. Actual model spikes are recorded and gate updates.

## Explicit model assumptions

Retain Stage B/C's LIF equations, signed raw contacts, 0.1 ms timestep, 100 Hz
KC voltage kicks and uniform source parameters. Retain ordinary signed DAN
transmission as an explicit approximation; add a separate modulatory mechanism.
This is not a dopamine concentration, receptor-kinetic or release-site model.

For existing excitatory KC j -> MBON11 edges only, let s_j be the most recent
KC spike tick. At each actual PPL101 spike tick t:

```text
e_j(t) = exp(-(t-s_j)*dt/tau_e), or 0 if KC j has never spiked
w_ij <- max(0.1*w_ij_initial, w_ij*exp(-eta*e_j(t)))
```

Set tau_e = 1,000 ms and eta = 0.1 per DAN spike. These values, the last-spike
eligibility kernel and multiplicative bounded update are **engineering choices**,
not estimates from the paper. Simultaneous KC/DAN spikes have eligibility one.
No label, classifier gradient, scalar correctness reward or MBON spike enters
the rule. Apply updates in Brian2's end slot after current-tick transmission.
Only subsequent transmissions see changed weights. Other weights, signs and edge
support remain unchanged; do not renormalize rows and undo genuine depression.
Reset states, eligibility and weights before every independent conditioning arm.

## Design and controls

Use both existing 686-cell real MB circuits; fresh fixed seed 6142 creates KC
patterns. CS+ is digit 3 and CS- digit 1, chosen before outcomes. Their overlap
is retained, not edited to improve selectivity. These sampled circuits overlap
and come from one fly; they are not independent biological subjects.

Six 3-second conditioning arms per circuit:

| Arm | KC stimulus | DAN kicks | Learning |
|---|---|---|---|
| no_dopamine | CS+ at 0–1 s | None | Enabled |
| paired | CS+ at 0–1 s | PPL101, forward schedule | Enabled |
| backward | CS+ at 2–3 s | PPL101 at 0, 0.5, 1, 1.5 s | Enabled |
| dopamine_only | None | PPL101, forward schedule | Enabled |
| wrong_compartment | CS+ at 0–1 s | PPL106, forward schedule | Gamma1 rule only |
| blocked | CS+ at 0–1 s | PPL101, forward schedule | eta = 0 |

After conditioning, copy adapted contacts into a fresh frozen LIF network. Probe
CS+ and CS- separately for one second, each from reset, without DAN stimulation
or learning. Baseline probes use the initial weights. Measure actual MBON11
spikes, not a decoder score substituted for neural suppression.

## Prespecified checks and interpretation

Numerical requirements: analytic one-edge updates, causal timing, compartment
mask, inactive KC null, zero-learning null, floor/sign/support preservation,
independence from MBON spiking, actual Brian2 online-versus-offline agreement,
deterministic reset, and no-DAN equivalence to the frozen LIF backend.

For every main case, independently reconstruct final weights from recorded spike
times using a closed-form cumulative eligibility sum per edge. Maximum absolute
contact-weight error must be <= 1e-10. All non-target edges must be byte-identical.

Per-circuit rule criterion: paired stimulation evokes PPL101 spikes and reduces
the mean weight of CS+-active mapped edges by at least 5%; inactive mapped edges
and all five other arms have exactly zero weight changes. The 5% threshold is a
computational detection criterion, not a biological calibration. If recurrent
activity breaks a null assumption, record failure rather than suppress spikes.

Separate response criterion: both pre-conditioning probe responses must be
nonzero; paired CS+ suppression must be positive and greater than CS- suppression.
A silent baseline makes this criterion fail as unassessable. Do not divide by an
arbitrary epsilon, inject extra currents or choose another responsive seed.

## Separate pi transfer diagnostic

For no_dopamine and paired only, train a baseline head on initial contacts and
evaluate the adapted network with (1) that frozen head and (2) a newly fitted
head. Use the unchanged 482-parameter readout, eight hidden units, 6,000 updates,
fixed-32 / 4x weighting, learning rate 0.03 and L2 1e-5. Both fits use seed 6142,
initialization 0 and their training-only normalization. This produces six fresh
fits (two initial, four post-conditioning) and eight evaluations.

Train on 199 next-digit pairs from 200 pi digits. Generate all 197 outputs after
prompt `314`, with targets confined to the scorer; report first-error recall and
teacher-forced accuracy separately. The dopamine conditioning has no pi targets
and teaches an association with digit 3, **not the pi sequence**. This diagnostic
only measures transfer/interference. No claim of dopamine-only pi learning or
recall improvement is established by it, and no promotion criterion is defined.

## Audit and reproduction

Use the unchanged isolated LIF environment and `requirements-lif-lock.txt`.
Archive the exact mapping, config/source/data/registry/runtime fingerprints,
all conditioning spikes and gate events, adapted sparse arrays, pre/post probes,
pi training states, all head arrays, predictions and recall spike trains. Atomic
condition checkpoints support strict resume. Rerun all conditioning, probes,
readout fits and rollouts and compare **every saved array and payload exactly**;
independent closed-form LTD checks run in both executions. Verify pi with the
independent Decimal implementation before training.

```powershell
$env:PYTHONPATH='src'
python -m pytest -q
python -m flying.training.phase5b --output outputs/phase5b
python -m flying.training.phase5b --output outputs/phase5b --resume
python -m flying.training.phase5b --output outputs/phase5b --verify
```

This closes only a source-constrained gamma1/pedc Phase 5B implementation and
controlled assay. All-compartment rules, physiological rate fitting, reward-driven
pi acquisition, additional dopamine mechanisms and whole-brain Phase 6 are outside
this study. Keep those limits visible even if the model-rule checks pass.
