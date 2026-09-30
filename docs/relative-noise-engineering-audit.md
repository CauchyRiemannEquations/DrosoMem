# Relative-noise smoke CSV discrepancy, corrected before main

The original three-graph smoke at2f5d62c completed simulation, but the independent
verifier stopped at a sigma calibration assertion: the default pandas CSV reader
changed a written float by approximately7.54e-17 (relative6.74e-13). This exceeded
the already registered numeric check despite unchanged binary trajectory arrays.
The failed check is preserved in results/relative_noise_smoke_validation.

The correction uses float_precision='round_trip' when reading new study tables.
The tolerance, calibration formula, doses, seeds, dynamics, heads and scientific
criteria remain unchanged. A small-sigma round-trip regression test was added.
Repeat smoke and verification use new directories with the corrected source
revision. Compare all saved numerical arrays against the original smoke; never
rewrite old outputs or their manifests. The main starts only after this passes.
