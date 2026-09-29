# ACT V — frozen-readout robustness, preregistered 2026-09-30

## Question and completion boundary

Does whole-brain connectivity preserve autonomous recall under computational
perturbation better than the partial graph, with the same observation budget?
Complete all five specified perturbation curves, fresh-seed confirmation,
numerical verification, diagnostics and reporting regardless of sign. This closes
the current rate-model ACT V experiment programme; physiological calibration,
other dynamics/tasks and robustness-trained heads remain untested extensions.
Do not change ACT I clean conclusions or ACT IV learning conclusions.

## Audit and sources

ACT IV ended at bca3a0b. ACT I contains 50 matched fits; take **all 30** existing
legacy5/brain5/brain1 fits, no selection by score. Source root SHA256 is pinned in
[config](../configs/act5_robustness.json). Prior Phase2 robustness studied 72 partial
real/shuffled heads, three perturbations and 16,128 recalls; its failures remain.
It did not compare whole graphs, neuron dropout or weight perturbation.
Existing graph cache and exact source IDs are available. No new biological data.

## Models, task and fresh confirmation

Discovery: seeds 7142–7146, circuits 701/702, three graph conditions (30 heads).
Confirmation: seeds 371142–371144, same two circuits and conditions (18 fresh
heads), unconditionally run. These new seed values and perturbation/bootstrap
seeds have no occurrence in pre-existing configs. This is computational replication
on one connectome, not new biological specimens.

Use unchanged ACT I configuration: pi including leading 3, offset0/length200,
prompt314, horizon197; incoming-L1 gain.9, leak.6, mbon_after_kc; exact matched KC
stimulation and 48 MBON identities/order; 48→8 tanh→10 head,482 fitted parameters;
train-only scaling, Adam2000/LR.03/L2=1e-5, first32 generated targets weighted4×.
Only fresh clean heads are fitted. Do not refit or recalibrate after perturbation.
Report original training history/loss and clean teacher accuracy separately.

## Fixed interventions and pairing

Each family has zero plus three nonzero levels in config. One perturbation
realization per model/circuit, paired across graphs and nested across strengths.
The independent seed-block units are five discovery and three confirmation
seeds; average the two circuit strata first. There is no Monte Carlo precision
claim from one exposure per model. Fresh confirmation changes input/head seeds
and perturbation streams. Do not pool cohorts.

Canonical universe is the pinned brain1 ordered neuron IDs and CSR edge order.
For every (model seed,circuit) use SeedSequence([372001,seed,circuit,tag]). Generate
draws over that same universe and select the condition's mapped neuron/edge IDs.
Shared neurons/edges receive identical draws across graphs. Extra neurons/edges
are also exposed, so equal per-unit dose is not equal total disturbance energy.
No new input or observation neurons are added.

1. **pulse**, tag1: Gaussian SD1e-6,1e-4,1e-3 added to all states once immediately
   after the clean prompt, before first autonomous prediction (initial recall
   state noise, not initial pre-prompt noise). Clip to[-1,1].
2. **ongoing**, tag1: the same Gaussian stream, fresh vector before each prediction,
   SD1e-6,1e-4,1e-3. Prompt clean. First draw equals pulse.
3. **edge_dropout**, tag3: uniform draw per canonical edge, remove when U<p,
   p=.001,.005,.01. Bernoulli fraction, not exact count. Nested masks, surviving
   normalized weights unchanged; no renormalization. Applied before prompt.
4. **neuron_dropout**, tag2: uniform per canonical neuron, clamp when U<p,
   p=.001,.005,.01. All roles eligible, including stimulated and observed neurons.
   Clamp after every update, including KC interim and MBON staged update, from
   prompt onward. The head still has 48 entries, dead observations are zero.
   Thus this includes input/output loss, not only disruption of memory storage.
5. **weight_noise**, tag4: W'=W*exp(sigma*Z−sigma²/2), sigma=.001,.01,.05 in
   log-multiplier units. Same normal draw per canonical edge across strengths.
   Sign and support preserved, row mass not restored. Applied before prompt.

No interventions are combined. State noise is in model units, with observed-state
scale ratios reported; none of these doses is physiologically calibrated.

## Saved evaluations and diagnostics

Per head execute clean autonomous and teacher-forced paths, plus15 nonzero
autonomous paths and5 teacher-forced paths (maximum dose per family):22 actual
trajectories,1056 total. Save every197-symbol output/probability/48-state trace,
position accuracy, region accuracy, prefix score/first error and confidence.
The autonomous simulator receives only prompt and horizon, never targets.
Teacher forcing is a separate labeled function supplying correct previous symbols.
Maximum-dose teacher forcing diagnoses decoding/feedback sensitivity; it is not
a full teacher-forced dose curve and does not replace autonomous results.

Zero intervention construction is checked separately for all five families at
every head. Zero rows explicitly reuse clean output (240 aliases, not240 new
trajectories); synthetic tests and real smoke verify actual zero rollouts.
Record active-neuron counts, MBON sparsity/effective rank, temporal cosine,
deviation from clean features, clipping, affected IDs/counts and weight hashes.
Clean teacher-state32-step zero-input decay is saved; no biological time scale.
Save clean checkpoints/fresh training logs, configs, graph/data identities,
source commit/hashes, environment, runtime and .05/.2s sampled RSS. No full
T×whole-neuron trajectory or dense N×N array.

## Endpoints, statistics and fixed decisions

Keep Pi Memory Score: exact generated prefix before first error, prompt excluded,
capped197. For each nonzero dose define capped retention R=min(PMS/PMS_clean,1).
If clean PMS=0, retention is undefined and the corresponding robustness/advantage
gate is **inconclusive**, never silently exclude that model or divide by epsilon.
The curve index is the equal-weight mean of R at the three registered nonzero
doses; it is a grid-specific index, not a continuous AUC or formal memory capacity.
Also report raw PMS and its equal-grid mean, uncapped ratios and full accuracy.

Primary comparative hypothesis: **brain1 versus legacy5 on ongoing noise**.
For each cohort require mean paired curve-retention advantage>=.10 AND mean
paired raw curve-prefix advantage>=2 digits, each positive in >=4/5 discovery
or3/3 confirmation seed blocks. Confirm only if both cohorts pass unchanged.
Prespecified secondary comparisons use the same gate for brain5−legacy5 and
brain1−brain5, and for the four other families. Report all; none replaces a failed
primary endpoint. Weak-edge contrast also changes row normalization and therefore
does not isolate a pure weak-edge causal effect. No p-value success gate.

Absolute robustness at the **largest** registered dose of each family requires
mean capped retention>=.8, >=4/5 or3/3 seed blocks individually>=.8, AND retention
of a32-symbol prefix in>=80% of originally32-completing heads. Empty eligible set
fails this completion component. Confirm per graph/family only in both cohorts.
Failure at the largest dose cannot be rescued by selecting a weaker dose.

Save per-head and per-seed tables, all paired differences, mean/median/sample
variance,10000 paired seed-block bootstrap intervals (seed378399), standardized
paired effects and wins/ties/losses. Small computational samples: intervals are
descriptive. No outcome-based exclusion, best-seed report or optional stopping.

## Verification, smoke and resource stopping

Before main, run the existing archived head seed7142/c701 at horizon8 in all
three graphs, every family/strength and both evaluation modes as applicable.
Smoke validates mechanics and engineering only, excluded from inference.
Test injection timing, clipping, staged neuron clamping, ID remapping, nested
masks, sign preservation, immutability and criteria/undefined-baseline handling.

Independently implement scalar dynamics/head scoring. Rebuild every legacy5
trajectory, plus all trajectories for discovery7142/c701 and confirmation371142/
c702 in both whole graphs. Compare generated symbols exactly and numerical states/
probabilities at atol1e-12,rtol1e-10. Real smoke is also independently replayed.
Refit first fresh seed in all graph/stratum conditions and compare parameters.
Verify all saved metrics, aggregates/gates, source and artifact hashes, including
every old result file. This is deliberately a full checksum/statistical audit plus
specified numerical replay subset, not a claim that every whole trajectory was
independently replayed. Archive all failures and stop on integrity mismatch.

Two workers maximum, one numerical thread each;1200s/3GiB per case,10800s main
supervision and7200s independent verification. Resource failures retain artifacts
and mark incomplete; no reduced horizon, changed criterion or replacement seed.
An engineering-only cycling-symbol benchmark (no head/scores) measured roughly
3.3s/8.5s per200 steps for brain5/brain1. Run fixed scope, no parameter sweep.

## Interpretation and stopping after results

Positive claims require the already declared fresh confirmation. Negative findings
lead to the saved maximum-dose teacher/state diagnostics, not training changes.
Do not infer complete loss of internal information from frozen-head failure.
Report all five curves, old failures, limitations and one next experiment. Finish
the current-model ACT V programme regardless of success; no automatic shutdown
is requested for this work session.
