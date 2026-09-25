# Input/output timing diagnosis

Question: is missing current-digit access at MBON a material bottleneck for trained-prefix recall? This is a prespecified fixed-reservoir scheduling comparison, not another connection-learning experiment.

Keep both existing 686-neuron circuits, all edge weights/signs, KC input, exactly 48 MBON features, the 200-digit prefix and affine softmax hyperparameters. Cross three fresh seeds 842–844, two normalizations, real/role-degree-shuffled graphs and three schedules: **72 π evaluations**. For each condition independently decode iid inputs at lags 0/1/2/3/5/8 with 2,000 training and 1,000 test items after 100 warmup items: **432 lag measurements**. Shared seeded inputs/readout budgets across conditions; no π labels enter the random-input probe.

- `sync_one`: existing one synchronous update; MBON uses old KC and cannot access the new digit at that step.
- `mbon_after_kc`: every neuron integrates once. KC/DAN/APL use old states; MBON alone reads newly updated KC on existing KC→MBON edges, retaining old states for other presynaptic roles. No extra features, skip connections, positional input, trainable connections or label access. This is a computational update-order intervention, not an established biological schedule.
- `sync_two`: existing two synchronous updates with input held; allows current input to propagate but also advances every neuron's memory decay twice. This comparison contextualizes the staged result and does not isolate latency alone.

All readouts fit on their own training states with identical 400-epoch Adam/.03/L2=1e-5 settings. Recall resets state, gives `314`, freezes everything and generates 197 digits. No best seed or hyperparameter is selected. Record teacher-forced training accuracy separately.

Additionally verify an exact implication of the scoring protocol: in a deterministic model with identical reset/prompt, before the first free-running error every generated input equals the teacher-forced input. Therefore first-error free-recall score MUST equal the initial consecutive-correct teacher-forced segment beginning after the prompt. Instability after feeding a wrong digit cannot explain an earlier first-error score. This equivalence is not a claim about noisy states, alternate prompts, later error recovery or full-rollout accuracy.

Interpretation: greater lag-0 decoding confirms current-input access. A recall gain with retained delayed-input decoding supports a scheduling bottleneck, not greater biological memory or learned synaptic improvement. If lag-0 access improves but recall does not, latency alone is insufficient under this fixed fit. Compare real and shuffled graphs before anatomical claims. Any discovered advantage needs independent confirmation before a robustness claim. Whole-brain scaling stays outside this experiment.
