# Stage B/C: small-circuit LIF validation and pi feasibility

Editorial update: research terminology only; the registered numerical design is
unchanged. Original protocol bytes are preserved in Git history; see
[historical reproduction](historical-reproduction.md).

Locked before circuit outcomes, 2026-09-27. Config: `configs/stage_bc.json`.
Commit this protocol, implementation, tests and separate dependency lock before
executing the four main graph conditions. No parameter sweep or seed selection.

## Sourced model and units

The parameter source is the authors' [pinned model.py, commit 91bdd1e7](https://github.com/philshiu/Drosophila_brain_model/blob/91bdd1e7dcf193f3e7ca5a8933497fcef63b7960/model.py),
associated with [Shiu et al., Nature 2024](https://www.nature.com/articles/s41586-024-07763-9).
Uniform defaults are model assumptions, not measurements for each bundled cell.

| Quantity | Adopted value | Status in source |
|---|---:|---|
| Rest / reset / threshold | -52 / -52 / -45 mV | Uniform defaults; cites Kakaria and de Bivort |
| Membrane time constant | 20 ms | Uniform default; same citation |
| Synaptic decay | 5 ms | Uniform default; cites Juergensen et al. |
| Refractory interval | 2.2 ms | Uniform default; cites Lazar et al. |
| Transmission delay | 1.8 ms | Uniform default; cites Paul et al. |
| Weight per signed contact | 0.275 mV | Explicit free parameter |
| Direct input voltage kick | 68.75 mV | Contact scale times source stimulation factor 250 |

Equations: `dv/dt = (v_rest - v + g)/tau_m`, `dg/dt = -g/tau_s`.
Both variables have voltage units. Here `g` is exponential synaptic drive, not
conductance. Threshold is strict `v > v_threshold`; reset restores `v` and zeros
`g`. Incoming weights use raw signed contacts, with no reservoir normalization.

[Brian2 2.10.1 refractory semantics](https://brian2.readthedocs.io/en/2.10.1/user/refractoriness.html)
freeze both marked variables and suppress synaptic writes during refractoriness.
The implementation preserves this behavior. Its explicit clock follows the
[documented schedule](https://brian2.readthedocs.io/en/2.10.1/user/running.html#scheduling):
integration, thresholds, synapses, resets. End-slot state samples correspond to
the integrated interval; spike timestamps use the clock's interval start.

## Deliberate task adaptations

- Use the two existing 686-neuron sampled MB circuits (512 KC, 48 MBON, 125 DAN,
  one APL), with their existing data checksums. This is not a reproduction of the
  paper's full-brain sensorimotor experiments.
- Seed 5142 creates ten fixed masks, each stimulating 51 KCs. A digit lasts 50 ms
  and receives synchronous kicks at 0, 10, 20, 30 and 40 ms. These deterministic
  100 Hz pulses replace source Poisson stimulation; they are an engineering code
  for digits, not natural sensory activity or a fitted biological stimulus.
- All KCs are input-eligible and therefore have zero refractory interval,
  extending the source's stimulated-cell exception. Other roles retain 2.2 ms.
- Primary timestep is 0.1 ms. Delays and pulses are integral clock steps.
- Primary features are spike counts over each digit window at the 48 MBONs.
  No membrane feature fallback, clock, position, target lookup or per-digit state
  reset is allowed. Only the readout is trained; recurrent contacts remain fixed.

## Numerical acceptance before interpretation

1. Analytic passive voltage/drive decay; signed impulse and `W[post, pre]`
   direction; delay; strict threshold/reset; refractory masking; zero-edge null;
   unit rejection; batched versus sequential input; deterministic reset.
2. Independent NumPy/SciPy exact-discrete reference, coded without Brian2: compare
   every state and spike for a signed recurrent toy at 0.1, 0.05 and 0.025 ms.
3. For every real/shuffled circuit, compare the first three digits at 0.1 ms:
   identical spike indices/ticks and maximum voltage/drive error below 1e-8 mV.
   Stop the corresponding main comparison on reference failure.
4. On each circuit's first 20 digits, run 0.1, 0.05 and 0.025 ms. For each adjacent
   pair, MBON count-array absolute difference divided by total finer-grid counts
   (denominator at least one) must be at most 5%. Report both ratios and role
   rates. Failure limits temporal convergence; it does not authorize retuning.
   This short feature check does not establish full-rollout timestep invariance.

## Paired descriptive comparison

Four conditions: two circuits times real and within-role degree-preserving
shuffled graphs. One fresh encoder/shuffle seed (5142), one initialization (0).
For each, compare LIF to existing incoming-L1 gain 0.9, leak 0.6,
`mbon_after_kc` rate dynamics. Reuse the same graph and KC masks, but rate input
amplitude is 0.5 and its states are dimensionless. Therefore this is a feasibility
comparison, not an isolated causal test of spiking versus rate dynamics.

Each of eight readouts has 482 parameters, eight hidden units, 6,000 Adam updates,
learning rate 0.03, L2 1e-5, and the existing fixed-32 / 4x loss weighting.
Train the first 200 pi digits as 199 next-digit pairs. Report teacher-forced
accuracy separately from 197 autoregressive outputs after prompt `314`; first
error ends the score and supplied prompt digits are excluded. Continue the full
rollout after the first error for reproducibility. No best-head selection or
promotion criterion is defined by this small cohort.

Save all training states, LIF spike indices/ticks for training and recall, heads,
predictions, diagnostics, histories, graph hashes and package/code/data/protocol
fingerprints. Atomic per-condition checkpoints support strict resume. Rebuild
all eight training-state arrays, replay all eight saved heads, and independently
refit all eight heads with exact equality in the recorded environment. Verify
pi against the independent Decimal generator. Keep failed biological or recall
claims distinct from successful numerical validation.

## Reproduction and scope

Create a separate Python 3.12 environment and install `requirements-lif-lock.txt`.
The old rate environment and `requirements-lock.txt` remain intact: Brian2's
SymPy dependency requires mpmath below 1.4. Use one BLAS thread and Brian2's NumPy
backend, without C++ compilation.

```powershell
$env:PYTHONPATH='src'
python -m pytest -q
python -m flying.training.stage_bc --output outputs/stage_bc
python -m flying.training.stage_bc --output outputs/stage_bc --resume
python -m flying.training.stage_bc --output outputs/stage_bc --verify
```

Completion closes only sourced small-circuit Stage B/C implementation, numerical
validation and descriptive comparison. It does not establish cell-specific
parameter validity, natural digit coding, dopamine plasticity (Phase 5B),
whole-brain performance (Phase 6), or improved sequence recall. Preserve the
existing baseline while those claims remain untested or unsupported.
