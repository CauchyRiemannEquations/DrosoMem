# Early-prefix loss weighting: locked diagnostic

Protocol written before execution, building on commit 9c9f914.
Question: does prioritizing early clean-trajectory predictions extend error-free recall?

Use two existing 686-neuron circuits, incoming-L1 normalization, mbon_after_kc,
real and role-degree-shuffled graphs, fresh seeds 1042–1046 and all initializations
0–2. Twenty state conditions × three initializations × two objectives = 120 fits.
Each paired fit uses identical frozen states, unweighted train-only feature
statistics, initialization, 482-parameter head and 2,000 Adam updates at .03.
No tuning or best-run selection is permitted.

Uniform weights are 1. The intervention sets weight 4 at training rows 2:34,
corresponding to target digit indices 3–34 inclusive: the first 32 generated
positions after the supplied 314. Rows predicting prompt digits retain weight 1.
Normalize cross entropy by the sum of weights; keep L2 unchanged.
This changes training priorities; no time index or target enters the generator.

Primary outcome: paired difference in consecutive correct generated digits,
horizon 197. Average all three initializations within each condition before
reporting wins/ties/losses across ten real and ten shuffled conditions.
Also report early-window and later-position accuracy, unweighted full training
accuracy/cross entropy, full per-position teacher predictions, every generated
string and complete saved heads. Report how many conditions improve for all
initializations. The selected schedule is exploratory from the previous study;
fresh seeds do not create independent biological replicas.

Verify normalized weighted gradients numerically, all-one weight equivalence,
invalid weights and exact target indexing. Rebuild all 20 states and replay all
120 saved heads. Independently refit the first real condition's six paired heads.
An early-prefix gain with worse later accuracy indicates error redistribution,
not more total memory, biological learning, anatomical superiority or unseen π
prediction. Whole-brain scaling remains deferred.
