# DrosoMem research roadmap

## Latest — 2026-10-02

The [fresh-model coordinate-SD structural confirmation](fresh-coordinate-sd-results.md)
is complete, with 96 new fits and full independent replay of 1,824 paths.
The registered intact-over-degree random-digit criterion fails in both new
cohorts. The earlier reused-head calibration study remains a separate closed
result. Further whole-brain, physiological or internal-learning work needs a
new prospective protocol; this study does not trigger a calibration sweep.

DrosoMem studies computational models using actual Drosophila connectome structure.
It does not demonstrate a living fly memorizing pi. Always distinguish
**representation** (past inputs affect state), **decoding** (a head extracts useful
information), and **learning** (experience modifies recurrent connections).

## Current position — 2026-09-30

**The bounded observation-noise extension is closed, including coordinate-SD calibration.**
[Programme synthesis](additional-research-closeout.md), [last experiment](coordinate-sd-noise-results.md),
[prospective protocol](coordinate-sd-noise-protocol.md), [closeout audit](../results/coordinate_sd_noise_closeout/audit.json).
The final random/degree incremental-gap criterion is **not confirmed**.
96 existing partial heads, zero new fits; all 2,688 actual paths independently replayed.
2,592 labeled noisy settings include 288 common/own intact repeats (2,304 distinct fresh settings).
Six extension studies comprise 5,992 actual evaluation paths and 5,726 independent full replays;
these are execution counts, not distinct models. Earlier negative results remain unchanged.
No execution or verification remains in this bounded extension. Physiology, new dynamics,
whole-brain rewiring and fresh-model generalization remain separate unexecuted research.
One future proposal is fresh model/data/graph seeds under coordinate-SD calibration;
it is not registered or executed. No further calibration sweep is included in this closeout.

| Axis | Completed scope | Remaining scope |
|---|---|---|
| ACT I | Matched partial/expanded/whole graphs, thresholds5/1; no established whole-brain recall advantage | Broader tasks/dynamics require separate protocols |
| ACT II | Four-family pilot,length/alphabet scopes,context and delayed-state diagnostics | Additional families and full length-by-K design |
| ACT III-A | Current partial-model population panel,input controls,frozen/refit and fresh confirmation | Whole-brain/downstream and autonomous-recall localization |
| ACT III-B | DAN/count/raw-bin and normalized-bin controls;weak/strong/random/betweenness/within-role/between-role panel | Whole-brain,detected biological modules,full dose curves remain untested extensions |
| ACT III-C | Four controls including exact per-neuron incoming-weight and role/signed-degree preservation;169 cases | Whole-brain, uniform-ensemble mixing and broader autonomous tasks; current partial K10 comparison is completed separately |
| ACT III-D | Five preregistered pathways, matched removal controls, sensitivity map and fresh confirmation;273 cases | Minimal/unique subnetwork, motifs, whole-brain localization and separation of storage from readout access |
| ACT IV | Earlier negative biological/local-rule results; IV-A A/B/C/D, observation diagnostics and scalar/local39-case experiment plus direction/noise/trajectory/margin diagnostics and explicit closeout completed | Unanswered extensions: physiological and whole-brain local learning, literal decoder-free biology; no claim of successful internal memory formation |
| ACT V | Five perturbation curves, relative-noise calibration, instantaneous-observation and matched-energy coordinate allocation; frozen heads, declared fresh seeds/draws, final audit | Physiological calibration, other dynamics/tasks and temporal noise; autonomous and synthetic spatial-correlation studies are separately completed |

현재 두 686-neuron 부분 모델의 **III-A/B/C/D의 제한된 연구 단락을 종료**했다.
완료는 고정한 실험·검증을 끝냈다는 뜻이다. 전체 뇌의 기억 위치나 내부 학습을
규명했다는 뜻은 아니다. 관측 위치·공통 지연 진단과 ACT IV-A의 현재 A/B/C/D 비교도 완료했다.
인공 정답 교사로 내부 코드를 바꾼 것과 생물학적 학습은 구분한다.

## ACT I — Can a Fly Brain Remember?

Close the matched partial/whole-brain question first: legacy5, left5, left1,
brain5, brain1; fixed exact input IDs, 48 observed MBON IDs and 482 readout
parameters. [Protocol](whole-brain-memory-protocol.md),
[results](whole-brain-memory-results.md), [status](research-status.md).
Separate expansion, weak-edge restoration and observed representation diversity.
Do not tune a negative result into a positive headline.

## ACT II — Beyond pi

The initial four-family pilot is now executed: [protocol](sequence-memory-protocol.md),
[results](sequence-memory-results.md). Both networks support the prespecified
limited beyond-pi claim on these trained instances; whole-brain superiority is
not established. The [random length curve](length-scaling-results.md) is also
complete at N32–512, with no whole-brain superiority. The subsequent
[N512 prefix-loss-mass diagnostic](prefix-mass-results.md) confirms prefix gains
on fresh seeds but fails the predefined later teacher-forced cost tolerance in
both graphs. It changes decoder allocation on identical reservoir trajectories.
The [independent-stream delayed-symbol comparison](delayed-symbol-results.md)
is now complete: partial60.42% / whole-brain51.10% past-lag accuracy, with
20 exact repeats. Whole-brain superiority fails. The fixed-N128
[alphabet comparison](alphabet-memory-results.md) is also complete:80 fits,
80 exact replays and8 independent refits. Both graphs pass trained-prefix gates
at all four K, but no K establishes whole-brain superiority. Recall grows from
K2 toK16, so the registered load-drop endpoint fails. The subsequent
[finite-context diagnostic](context-memory-results.md) is complete:120 controls
on20 unique tasks, with both registered endpoints passing. Context distinctiveness
is a supported explanation candidate, not an identified neural cause. Next is one
[fixed-K intervention](context-intervention-results.md), now complete. Confirmed graphs: none. The selected [blocked-state diagnostic](frozen-state-probe-results.md), [cross-arm transfer](cross-arm-probe-results.md) and subsequent [K4 structural control](structural-k4-results.md) are complete. The latter finds no real-wiring advantage; the subsequent normalization control is also complete with failed material-effect confirmation. Frozen-head transfer and train-only moment alignment are complete; the residual mode/orientation diagnostics are also complete, and fresh-seed alignment confirmation is complete with replicated improvement but failed portability; KCγ refit ablation and a disjoint follow-up are complete without specificity support. Frozen-head lesion decoding is complete with a large refit benefit but no specificity; fresh paired-seed confirmation is complete. Input-only silencing and the broader population panel with fresh confirmation are now complete; see the current-model ACT III-A closeout above.
Full length-by-alphabet scaling and additional families remain open. The paragraph
below describes the staged scope, not an assertion that all stages are complete.

After ACT I, begin with only pi, seeded random digits, shuffled pi (same multiset)
and a periodic sequence, comparing whole brain to the strongest established
baseline/control. Later add e, sqrt(2), motifs and low-order Markov sequences.
Treat symbols as integers; expose K=2,4,10,16 rather than silently coercing decimal
pi. Pi digits are not assumed mathematically random. Extend SequenceDataset
without changing archived ACT I behavior.

Length curves: 32,64,128,256,512 and 1024 only if feasible. Refit separately at
each length and show the recall-collapse region. Retain Pi Memory Score and add
L log2(K) recalled-prefix bits; do not call it Shannon capacity or the formal
reservoir-computing memory capacity. Structured/Markov data additionally require
statistical-prediction controls because predictability can mimic memory.

## ACT III — Where does the model store useful information?

The bounded current-model neuron, edge and structural-control panels are now executed: see [III-A](neuron-panel-results.md) [III-B completion scope](edge-panel-results.md) [III-C](structural-controls-results.md) and [III-D](pathway-memory-results.md). Downstream populations absent from the partial
graphs and whole-brain localization remain untested. The broader blueprint follows.

Test annotated KC/MBON/DAN/APL and downstream groups, hubs and matched random
groups. Keep frozen-head ablation distinct from post-ablation readout refitting.
Remove matched edge counts by strength, randomness, module boundary or graph
centrality, respecting whole-brain compute limits.
Use multiple controls: degree-preserving rewiring, weight permutation, role-block
preserving shuffle and matched random graphs. Explicitly list preserved and
destroyed properties; random graph alone is not sufficient.
Deliver sensitivity maps and candidate memory-critical subnetworks **in this
computational model**, not biological memory-circuit claims.

## ACT IV — Can internal connections learn?

Only after I–III. One hypothesis and one rule at a time: dopamine-gated plasticity,
local eligibility traces, STDP-like or compartment-specific learning.
Compare frozen real reservoir + trained head, random reservoir + trained head,
plastic reservoir + simple head, and possible direct internal decoding.
Preserve Phase 5B's negative recall/response findings. Separate external decoder
learning from internal synaptic learning.

### ACT IV-A completion boundary

A frozen real + learned ridge; B random and role-preserved + learned ridge;
C artificially taught existing KC→MBON + simple ridge/fixed code; D zero-trained-
parameter nearest-code proxy were all executed under one matched protocol.
The existing supervised delta rule is a computational control, not dopamine or
strictly local reward learning. Fixed-code improvement shows internal recoding;
near-ceiling ridge access does not establish additional memory capacity. Literal
readout-free biology, whole-brain learning and stronger biological rules remain open.

### First scalar-reward follow-up

The target vector was removed in a separate preregistered local rate-covariance
experiment with frozen and yoked rewards. Fixed-code primary and representation
improvement fail; historical artificial-teacher success is not erased. This is
neither physiological validation nor a general impossibility result for local rules.

### ACT IV current-model completion

The [predeclared completion plan](act4-completion-plan.md) and
[question-by-question closeout](act4-results.md) now close the specified partial-model
programme. The last margin diagnostic includes independent fresh task seeds and
reverse-adjoint checks. Other candidate rules were not automatically required sweeps;
STDP and broader physiological extensions remain untested. Numerical completion,
margin direction, fixed-code accuracy and autonomous memory are different claims.

## ACT V — Robustness (independent axis)

The [current five-family panel](act5-robustness-results.md) is executed and checked.
30 archived clean heads and18 fresh heads were evaluated without perturbation
refits. Previous partial real/shuffled robustness failures remain preserved.
This completes the declared rate-model scope, not physiological validation.
The subsequent [relative-dose study](relative-noise-results.md) and
[observation diagnostic](observation-noise-results.md) are also complete. The first
uses new confirmation seeds; the second reuses both cohorts and is exploratory.
Do not treat its teacher-forced diagnostic as recovery of autonomous memory.
The final [coordinate-allocation control](allocation-noise-results.md) and
[ACT V closeout](act5-final-results.md) complete the prospective final extension.
Fresh noise repetitions are not new model cohorts or physiological replicates.
The principles below remain the blueprint for future extensions.

Use predefined curves for initial/ongoing noise, edge/neuron dropout and weight
perturbation. Preserve the existing clean-versus-perturbed failures. Do not pool
perturbed outcomes with ACT I's clean main comparison or present computational
noise amplitudes as biological calibration.

## Execution discipline

Protocol before outcomes; explicit stopping rules and finite seeds; fresh-seed
confirmation of positive discoveries; diagnostics after negatives. Save every
seed, failure, checkpoint, configuration, code/data hash, package lock and
resource measurement. Report paired differences, mean, median, variance,
descriptive uncertainty and effect sizes without treating tiny computational
cohorts as biological replication. No success-metric changes after outcomes.
