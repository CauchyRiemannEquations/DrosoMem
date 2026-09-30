# Limitations and interpretation boundaries

DrosoMem is exploratory computational research software. The repository is
designed to preserve both positive and negative results, but its outputs should
not be promoted beyond the scope of the implemented experiments.

## Biological interpretation

- The project uses **Drosophila connectome-derived structure**; it does not
  simulate a living fly in full biological detail.
- A connectome constrains who is connected to whom. It does not by itself
  specify all membrane dynamics, neuromodulation, gene expression, morphology,
  physiological synaptic efficacy, learning history or behavior.
- Rate-based dynamics and the optional LIF experiments are modeling choices,
  not validated reconstructions of biological memory mechanisms.
- Source-derived synapse counts and signs are transformed for computation.
  They should not be interpreted as measured physiological synaptic strengths.

## What "memory" means here

- π is a **trained sequence benchmark**. Recalling a trained prefix is not
  prediction of unseen π digits and is not evidence that a living fly can
  memorize π.
- The repository separates **representation** (past input affects network
  state), **decoding** (an external readout extracts useful information) and
  **internal learning** (the recurrent network itself changes with experience).
  Evidence for one does not establish the others.
- Strong performance after fitting a readout may reflect decoder capacity,
  training emphasis or state accessibility rather than increased biological
  storage capacity.
- Prefix-recall scores stop at the first error and can be censored by the
  evaluation horizon. They are not Shannon capacity estimates.

## Data and sampling

- The project is based on a public FlyWire FAFB v783 connectome release and
  deterministic or seeded subsets of that source.
- A seed is a computational initialization or sampling choice, **not** an
  independent animal.
- Several subsets are intentionally selected, hub-biased or anatomically
  restricted. They are not population-representative samples of flies.
- Results from one connectome release cannot by themselves establish
  across-animal biological generality.

## Controls and statistical scope

- Some confirmatory analyses reuse trained heads, graphs or previously fixed
  task/model seeds by design. Those replays are useful reproducibility checks
  but are not fresh independent training cohorts.
- Rewired controls answer only the question defined by what the rewiring
  preserves. Degree-preserving, role-preserving and weight-shuffled controls
  test different hypotheses.
- Negative results are retained. In particular, current results do not support
  a general claim that intact connectome wiring or the whole-brain graph is
  always superior for sequence recall.
- Multiple comparisons, bounded seed counts and project-specific stopping rules
  limit broad statistical generalization.

## Noise and robustness

- Observation noise used in the robustness extensions is synthetic and
  computationally calibrated. It is not a measurement of in-vivo physiological
  noise.
- Matching noise to coordinate or graph-relative scales is a control on the
  computational comparison, not a validation of biological realism.
- Strong-noise autonomous recall can remain weak even when a relative
  comparison passes a pre-registered project criterion.

## Software and reproducibility

- Reproduction depends on the pinned data revisions, configuration files,
  package versions and random seeds documented in the repository.
- Hardware, numerical libraries and floating-point behavior can introduce small
  differences. Verification scripts should be used instead of relying only on
  a single printed metric.
- The repository contains a large volume of committed outputs. A release
  archive can therefore be large even when only a small subset is needed to
  understand the method.

## Licensing

- DrosoMem source code is distributed under the repository's MIT license.
- FlyWire-derived public data remain subject to the upstream **CC BY-NC 4.0**
  terms and attribution requirements. The MIT code license does not relicense
  those data.
- See [`DATA_SOURCES.md`](DATA_SOURCES.md) and
  [`data/README.md`](data/README.md) before redistributing derived data.

## Appropriate summary claim

A conservative description of the project is:

> DrosoMem studies sequential recall and decoding in computational models that
> use Drosophila connectome-derived structure, with reproducible controls for
> sequence, structure, readout behavior and synthetic noise.

Claims about biological memory mechanisms, whole-brain superiority, unseen
sequence prediction or physiological validity require evidence beyond the
current repository.
