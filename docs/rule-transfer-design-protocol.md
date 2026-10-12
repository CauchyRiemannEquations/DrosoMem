# M10: mathematical-sequence task-design validity preflight

Date: 2026-10-13. This finite **design** protocol precedes new dataset/control
outcomes. Baseline main: `81ad81f25c445c8964599d35f52e0bef51ce4755`.
[Config](../configs/rule_transfer_design.json), [prior audit](rule-transfer-prior-audit.md).
No neural fitting, connectome comparison, internal learning or P3 is included.
READY means that the registered task checks pass, not learned rule transfer.

## Question and known information

Can one bounded symbolic benchmark distinguish new initial conditions under
seen rules from application of held-out rule parameters, without exact training
prefixes or rotated training cycles solving the latter? Does its known-form
reference solve all valid structured instances while registered trivial
predictors and iid/shuffled negative controls fail to approach that reference?

The formula and modulus are deliberately supplied to the analytic reference.
For nonconstant triples over prime modulus 11, an affine rule is uniquely
identified by `a=(x2-x1)/(x1-x0) mod 11`, `b=x1-a*x0 mod 11`.
This algebraic property is known before execution, not a discovered result.
The design follows the known M9 ceiling; it is not a blind study of all task
properties. All outcomes and any failed design checks will be retained.

## One family, exhaustive finite splits

Use integer symbols 0..10 and `x[t+1]=(a*x[t]+b) mod 11`. Train multipliers
are 2..6; parameter-test multipliers are 7..10. Every multiplier uses every
intercept b=0..10. Multiplier 1 and constant/fixed-point episodes are excluded
by definition before outcomes, avoiding degenerate periodic/constant examples.
This is a bounded affine-family test, not arbitrary unseen mathematical rules.

Let fixed point `p=b/(1-a) mod 11`; start at `x0=p+y mod 11`.
For each of the 55 training rules, y=1..5 supplies training instances, y=6..10
supplies seen-rule instance tests. All 44 held-out rules use y=1..10.
There are exactly 275/275/440 episodes. Generate 11 symbols per episode:
3 observed prefix symbols and 8 continuation targets. Enumerate a,b,y ascending;
do not search seeds, families, split assignments or horizon after outcomes.

Instance tests can share rotated cycles with training, and measure a new phase
of a seen rule. Only the parameter test measures held-out coefficient transfer.
Different held-out multipliers need not have unseen cycle lengths. The task
does not test transfer outside the affine family or outside this modulus.

Export learner arrays with only prefixes/targets; keep a,b,p,y in separate
audit metadata. Test targets are scorer-only. No rule ID, split-specific seed,
absolute position, equation coefficients or orbit lookup is a learner input.
Any future fitted experiment must enforce this API and train-only preprocessing.
The training exposure is 8 next-symbol rows per episode (2200 total), with
actual within-episode histories; no transitions cross episode boundaries.

## Controls and knowledge budgets

All controls are reported, with no best-control or positive-family selection.
Fit frequency and suffix count tables on the same 2200 training transitions.
Tables of order 1/2/3 back off to the longest observed shorter suffix, then
global frequency; ties select the lowest integer. Three-symbol history is a
finite-state/memorized-context reference, not a mathematical-rule learner.
Archive tables and counts, including their variable storage sizes.

Current copies the last input; repeat-difference extrapolates its last modular
difference. They fit zero parameters; the latter knows the modulus but assumes
constant difference. The known-form oracle fits zero parameters, receives the
affine family formula and modulus, infers a,b from only the first 3 symbols,
then holds that rule fixed through its rollout. If its first difference is zero,
it declares that prefix unidentifiable and copies the last symbol. A nonzero
first difference also permits a=0 in negative-control inputs. Its supplied
functional knowledge is privileged; its success is an assay ceiling, not a
same-budget learned model result or evidence of biological mathematics.

Two negative families retain identical split sizes, prefix/horizon and exposure:
iid uniform integers using the three registered PCG64 seeds; and within-episode
shuffles of all 11 affine symbols, preserving each episode's full multiset.
Use each registered shuffle RNG once in episode order. Shuffling can accidentally
produce valid local triples; do not filter them. These are diagnostic controls,
not independent biological cohorts or a randomized held-out affine split.

## Separate scoring modes

Report first-next accuracy, all-8 teacher-forced next-symbol accuracy,
target-free autonomous exact-prefix length and full 8-symbol completion.
Teacher mode exposes actual past test symbols only. Oracle uses the initial
3-symbol inference in both modes; it never re-infers from teacher targets.
Rollout mode exposes only the 3-symbol prefix and the predictor's own generated
outputs. Training-split rollout describes trained-instance recall, instance-test
rollout seen-rule application, parameter-test rollout held-out coefficient
application. No teacher-forced score substitutes for autonomous continuation.
Report raw integer counts and exact rational rates. Exhaustive finite episodes
are correlated phases/rules, not independent samples; no p-values or bootstrap
biological confidence claims. No new reservoir outcome will be generated here.

## Registered validity decision and stopping rule

READY requires all of the following, otherwise INVALID with every outcome kept:

1. Exact split sizes and rule/start assignments; all structured prefixes have
   nonzero first difference and one uniquely inferred correct rule.
2. No identical 3-symbol prefix or complete directed cycle up to rotation between
   training and parameter test. No parameter-test teacher window of length 3
   appears in training. Record instance-test cycle overlap without rejecting it.
3. Every structured oracle episode produces all 8 targets exactly.
4. EACH non-oracle control's parameter-test first-next accuracy is at most 1/4.
5. Oracle first-next accuracy is at most 1/4 on EACH negative parameter-test
   family; all negative-family outcomes and unidentifiable prefixes are retained.
6. A separate verifier reconstructs inputs/splits/cycles, training tables,
   predictions and every exact metric without importing the producer's
   generator, predictors or scoring code. Artifact hashes/source identity and
   historical protected Git identities agree.

Do not change the quarter thresholds, split, rule set or controls after observing
a failure. These thresholds concern task separation, not a future neural
performance hypothesis. Both producer and verifier have 300 seconds/1 GiB
per-process RSS caps, with sampled RSS and recorded OS peak/environment.
Output directories must be new; capture failures and never overwrite attempts.
Use `results/rule_transfer_design_v1/` only for new artifacts. Historical source,
protocols/results/checksums/failures stay unchanged; four current overview files
may receive dated handoffs. Protocol/config are committed before execution.

## Performance handoff, explicitly outside this preflight

A future experiment needs its own committed model protocol before fitted
outcomes. The proposed common budget is 48 observed coordinates and an affine
11-way head (539 nominal learned scalars), train-only mean/SD with floor 1e-5,
alpha 1, 2200 training transition rows, frozen recurrent weights, reset per
episode and frozen heads before test. A 3-slot one-hot shift reference (33
active coordinates, padded to 48) can match its state observation/head budget.
Lookup tables and formula oracle have different knowledge/storage budgets and
must be reported separately. Select and freeze source circuit, K11 encoder,
control circuits, seeds, exact advantage/validity gates, resources and independent
trajectory/refit verification prospectively. M10 READY does not register or
execute that neural comparison, confirm transfer, or repair M9/P2/M7 failures.
