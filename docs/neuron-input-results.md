# ACT III-A direct-input silencing control

Executed under protocol commit `3773074`, implementation `5e13acd`.
[Protocol](neuron-panel-protocol.md), [config](../configs/neuron_panel.json).
This is package1 of the bounded neuron-population closeout, not the whole panel.

All28 conditions (4 smoke,24 main) exactly replayed. Independent verifier rebuilt
56 train/test trajectories, audited616 lag-metric rows, masks, preserved recurrent
weights, source-only frozen decoding and target refits. Archived baseline exactly
reproduced;8,811 prior result files unchanged. Runtime39.45s, sampled peak167.63MiB.
No hyperparameters changed. Full results: [manifest](../results/neuron_input_control/manifest.json),
[verification](../results/neuron_input_validation/checks.json).

Primary past-lag input-only minus full-lesion accuracy (percentage points):

| Mask / mode | seed71142 | seed71143 | seed71144 | Mean | Bootstrap95 interval |
|---|---:|---:|---:|---:|---|
| KCgamma frozen | +0.216667 | -0.516667 | +0.041667 | -0.086111 | [-0.516667,+0.216667] |
| KCgamma refit | +0.591667 | +0.458333 | +0.200000 | +0.416667 | [+0.200000,+0.591667] |
| Matched frozen | -0.738889 | -0.591667 | -0.322222 | -0.550926 | [-0.738889,-0.322222] |
| Matched refit | +1.150000 | +1.250000 | +0.911111 | +1.103704 | [+0.911111,+1.250000] |

All four registered5pp gates fail. The primary KCgamma refit difference is
positive in all3 blocks but subthreshold; do not call it zero or equivalence.
Features are not exactly equal: recurrently driven activity is possible after
direct input removal. Retaining these connections does not restore frozen-head
decoding to refit performance. This is conditional on identical removed input;
it does not establish that recurrent connections generally do not matter.

Same masks/streams as the previous confirmation intentionally reused,48 MBON
slots retained. Two circuits averaged within each of3 seed blocks, not independent
biological replication. These are independent-stream delayed-symbol scores,
not autonomous sequential recall or biological learning.

All per-seed values, means/medians/variance/bootstrap/paired dz are in
[seed table](../results/neuron_input_control/seed-blocks.csv) and
[summary](../results/neuron_input_control/summary.json); diagnostics/checkpoints
remain in each case directory. Both frozen and refit outputs preserved.

```powershell
$env:PYTHONPATH='src'
python scripts/neuron_panel.py --package input --out outputs/neuron_input_new
python scripts/verify_neuron_panel.py outputs/neuron_input_new --out outputs/neuron_input_check_new
```

Next registered package: five population targets, three matched masks each,
two positive controls, intact reference; all fresh confirmation conditions run
regardless of discovery outcomes. No extra sweep.
