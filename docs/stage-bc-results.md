# Stage B/C: small-circuit LIF result

2026-09-27. The sourced LIF implementation passes the locked numerical checks,
but this spike-count configuration is a weak pi recall model. On the two real
circuits, mean first-error recall is **3.0 digits**, versus **33.5** for the paired
rate baseline. Keep the existing rate baseline. This is one fresh encoder/head seed,
not a broad performance estimate or evidence that all spiking models are worse.

Protocol and implementation were committed as `d9c69b4` before circuit outcomes.
[Locked protocol](stage-bc-protocol.md) · [Configuration](../configs/stage_bc.json) ·
[Raw artifacts](../results/stage_bc) · [Summary](../results/stage_bc/summary.json).

## What was run

Two 686-neuron MB subsets, each with real and role-shuffled connections; seed
5142; one 482-parameter readout per graph/dynamics condition. All eight readouts
receive 6,000 updates with the existing fixed-32 / 4x loss weighting. Train 199
next-digit pairs from 200 pi digits and generate all 197 following digits from
prompt `314`, without giving the generator targets. Scores exclude the prompt
and stop at the first error.

| Circuit / graph | LIF recall | Rate recall | LIF teacher-forced accuracy | Rate teacher-forced accuracy |
|---|---:|---:|---:|---:|
| s701 real | 2 | 35 | 37.69% | 94.97% |
| s701 shuffled | 2 | 18 | 32.66% | 69.35% |
| s702 real | 4 | 32 | 43.72% | 84.92% |
| s702 shuffled | 2 | 35 | 41.21% | 92.96% |

The graphs and input masks are paired. LIF uses raw contact strengths and
dimensionful spiking dynamics; the rate baseline uses normalized weights and
dimensionless states. The comparison does not isolate spiking as the cause.
No seed, timestep, input pulse rate or readout view was selected after outcomes.

## Numerical evidence

- Analytic passive decay, signed recurrent toy reference, direction and delay,
  strict threshold/reset, refractory suppression, dimension rejection, zero-edge
  null, and batched/sequential/reset agreement all pass.
- Every circuit's first three digits exactly match independently calculated
  spike indices and clock ticks. Maximum membrane error is **6.68e-13 mV** and
  maximum synaptic-drive error is **2.84e-13 mV**, below the locked 1e-8 mV limit.
- The first 20 digits were run at 0.1, 0.05 and 0.025 ms. All eight adjacent-grid
  comparisons pass the 5% MBON count-difference criterion; maximum is **3.97%**.
  This establishes only the specified short feature test, not full autoregressive
  recall invariance or a continuous-time solution for spike events.

## Activity and interpretation

Only **8–9 of 48 MBON count features vary** during training. Real-circuit mean
MBON activity is 2.37 and 2.75 Hz, with 68 and 97 distinct feature vectors across
199 positions. Across all four conditions, mean KC activity is 9.96 Hz, APL is
100 Hz and DAN activity is zero. KC's population mean includes unstimulated cells;
the input pulses themselves repeat at 100 Hz.

These observations are consistent with an information bottleneck in the chosen
spike-count representation, but do not prove its cause or identify a remedy.
The uniform source parameters and synchronous artificial digit code are not
cell-specific physiological calibration. No compartment-specific dopamine rule
was implemented, and silent DAN nodes do not establish dopamine learning.

## Reproduction and boundaries

All **eight saved heads reproduce all 197 recall digits exactly**, with exact
reconstruction of their training-state arrays. The LIF training and recall
spike indices/ticks also match. **All eight independent refits reproduce the
saved readout digests**; both pi generators agree. See the
[verification record](../results/stage_bc/verification.json).

The full LIF-environment suite passes **166 tests**. Brian2 emits 28 upstream
Pyparsing deprecation warnings; no test failures occurred. Numerical tests use
the real Brian2 backend. LIF-only tests skip explicitly when that optional
dependency is absent.
The unchanged rate environment separately passes **148 tests**, with the two
LIF-only test modules explicitly skipped. Both test logs are archived with the
results.

Use a separate Python 3.12 environment with `requirements-lif-lock.txt`, then
follow the [protocol commands](stage-bc-protocol.md#reproduction-and-scope).
The original environment and dependency lock remain unchanged. The first graph
condition was checkpointed separately; `--resume` reused it before completing
the remaining three. Changed contexts and corrupt artifacts are rejected before
simulation.

The completed scope is a sourced small-circuit Stage B/C implementation and
numerical/feasibility comparison. It does not close **Phase 5B** (source-supported
compartment mapping and dopamine-dependent learning) or **Phase 6** (whole-brain
data, scaling and simulation). Those remain the next original research work.
A poor recall result does not invalidate the numerical checks, and passing those
checks does not establish improved sequence memory.
