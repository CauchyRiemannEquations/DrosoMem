# Parameter-budget-controlled readout experiment

Question: can a small nonlinear decoder use the same frozen 48-MBON states more effectively than the current affine classifier?

Preserve both 686-neuron circuits, weights, encoder, MBON observation set and 200-digit prefix. Use fresh graph/input seeds 942–944, spectral/incoming-L1 normalization, real/role-degree-shuffled topology and the two timing schedules sync_one/mbon_after_kc. Both readouts receive EXACTLY the same states and train-only means/scales per matched condition.

- Affine: 48→10 softmax, **490** fitted parameters, zero initialization as before.
- Nonlinear: 48→8 tanh→10 softmax, **482** parameters. No recurrence, persistent state, position input or lookup table in the readout. All coefficients train by full-batch Adam.

Use learning rate .03, L2=1e-5 on nonbias weights and checkpoints at 400 and 2,000 epochs for both. The prespecified primary comparison is 2,000 vs 2,000; 400 is a training-duration diagnostic. The nonlinear readout uses ALL three prespecified initializations, with streams seeded by [graph/input seed, 9901, initialization]. Never pick the best initialization, epoch or seed by recall. Report every run, the mean across the three initializations per paired state condition, and how many conditions improve for all three initializations.

There are 48 fixed-state conditions, 48 affine fits at each epoch budget and 144 nonlinear trajectories with two saved checkpoints: **384 evaluations**. The two nonlinear checkpoints share a trajectory; affine duplicates across initialization are not inserted into the raw data. Three nonlinear initializations of one state condition are not three independent graph samples.

Primary outcome is consecutive correct generated digits after `314`, horizon 197. Also log train next-digit accuracy, cross entropy and the teacher-forced initial correct segment. That segment must agree with the free-recall first-error score under this deterministic reset. Evaluation freezes all parameters and exposes no target to the generator.

Matching parameter counts does not equalize functional capacity, optimization difficulty, initialization, effective regularization or computational cost. The tanh bottleneck can be less expressive for some tasks even though it is nonlinear. A gain would show that the fixed neural states are more useful to this decoder; it would NOT show improved synaptic learning or biological-wiring superiority. A negative result after 2,000 epochs would not prove that the states contain no usable information or that every nonlinear decoder fails.

No unseen π prediction, whole-brain simulation, weight learning, architectural search or automatic follow-up sweep is included. Any apparent advantage is exploratory until checked independently; report frozen/shuffled comparisons and initialization sensitivity before suggesting a next phase.

Verification reconstructs every fixed state and replays all 384 saved heads. A prespecified refit subset is the first circuit, first seed, real graph, incoming-L1, sync_one: both affine budgets and all three nonlinear initializations at both checkpoints (8 checkpoints). This distinguishes complete replay from complete retraining; only this subset is retrained during verification.
