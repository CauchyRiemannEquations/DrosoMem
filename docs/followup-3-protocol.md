# Follow-up 3: length × alphabet × sequence family

Prospectively fixed before the new experiment's outcomes. One question: under
matched partial-connectome dynamics and an external readout, does trained
sequence recall across length and alphabet remain above simple prediction
controls, and does an independent delayed-symbol test show past information
beyond a predictor using only the current symbol?

Use two 686-neuron threshold-5 circuits (701/702); model seed blocks
810001–810003, separate dataset and input seeds; 3 lengths (64,128,256) ×
3 alphabets (4,10,16) × 3 families (iid uniform, sticky first-order Markov
with .7 self-transition, and period-8 motif with seeded symbols). Each
length is the exact prefix of the corresponding 256-symbol seeded sequence.
No task or seed selection after scores. The same 48 MBONs, gain .9, leak .6,
incoming-L1, `mbon_after_kc`, 48→8→K nonlinear head, 300 full-batch Adam
epochs, train-only standardization and three-symbol prompt apply in all cells.
This is 162 model/readout cells. Save all generated digits, not only the first
correct prefix. Majority and first-order Markov predictors fit the same
training sequence and generate their own outputs from the same prompt.

For an independent history control, use a separate 2,000-symbol training and
1,000-symbol test stream (plus 100 warmup) at N128 for each K/family/seed/
circuit. Refit an alpha-1 ridge on lag-2 MBON state; compare with train-only
frequency and reverse Markov-1 predictors of the lag-2 symbol from the current
symbol. For motif, also report a phase-8 oracle as a predictability diagnostic,
not an allowed model input. The delayed probe tests representation and must not
be conflated with autonomous recall.

**Primary gate:** in both the first two discovery seeds and the held-out third
confirmation seed, at N128/K10 iid and Markov, mean lag-2 MBON accuracy across
the two circuits exceeds the stronger frequency/reverse-Markov control by at
least 5 percentage points; all four cohort/family comparisons must pass.
This stringent conjunction is not a capacity estimate. Report the entire
length × K × family grid, signed per-seed contrasts and any failures regardless
of the primary gate. The motif phase oracle, learned-sequence autonomous
first-error prefix and simple-predictor prefixes are secondary. No benchmark
winner is chosen post hoc.

Budget: one smoke cell then all 162 cells and 54 independent-stream lag-2
probes; 3,600 seconds and 3 GiB sampled process RSS for each of execution and
verification, one process/numerical thread. The verifier independently
reconstructs sequences and predictions, the delayed ridge, checks all
manifests and replays every trained-sequence autonomous path. Incomplete
outputs are preserved, never counted as finished cells.

