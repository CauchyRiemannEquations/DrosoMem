# DrosoMem research roadmap

## Proposed post-M6 research programme — 2026-10-09

**Status: planning only.** [M6](counterfactual-tracking-results.md) is closed:
all384/384 primary and192/192 paired whole secondary lag2 symbol replacements
were tracked by the OLD frozen readout, with0 newly trained parameters.
This establishes a causal input-to-state-to-readout response in the specified
computational system. It does **not** establish a living fly's memory, biological
learning, an anatomical storage site, a unique cycle effect, formal capacity,
or superiority of real connectome wiring. P2's real-wiring material gates
remain **FAIL**, M1 remains **assay-invalid**, and M4's exact same-node DAG
control remains **INFEASIBLE**. Earlier outcomes and artifacts are not reopened.

| Order | Research direction | Status / boundary |
| --- | --- | --- |
| 1 | Frozen-readout tracking under artificial observation noise (working label **M7**) | **Proposed only**; no protocol registered, no runs or results |
| 2 | Structural specificity: real connectome vs matched rewiring and simple delay controls | **New future proposal**; distinct study and predeclared feasible controls required |
| 3 | Mathematical-sequence rule transfer beyond trained-string recall | **Exploratory future proposal**; not evidence of animal or human mathematical learning |
| 4 | Public research synthesis, software archive and possible manuscript | **Documentation/release plan**; no DOI or publication claimed; preparation can run alongside research |

### Priority 1 — artificial observation-noise tracking (working label M7)

**Question:** Does the clean M6 paired lag2 tracking survive one fixed,
nonzero **artificial observation-noise** condition when the same original
graph, encoder, M5 model, scaler and frozen readout are retained?

1. **Audit before designing:** compare the closed ACT V, observation-noise,
   noise-allocation and later extension protocols/results. Write down the
   genuinely new paired *past-symbol replacement* question; do not repeat
   earlier analyses or reinterpret their negative endpoints.
2. **Register before observing results:** one noise allocation/dose, fresh
   independent held-out input streams, exactly specified paired noise
   realization, lag2 target/replacement, finite model/head/seed/anchor budget,
   unconditioned joint tracking metric, quantitative PASS/FAIL threshold,
   and stopping/resource rules. Do not choose a favorable setting afterward.
3. **Keep causal controls:** the original and replacement worlds share their
   prefix, future symbols and appropriately paired observation noise; all
   anchors count, including original errors. Retain frequency-only,
   current-only and instantaneous-reference controls. Compare to sealed clean
   M6 as a separate reference, not as a rescore of its registered outcome.
4. **Verify and report:** frozen parameters and no new training; source/hash
   authentication, independent trajectory and prediction replay, paired raw
   scores and failure cases. Failure under the chosen noise is an informative
   result, not permission for a noise/seed/head sweep.

This is a **proposed** M7 label, not an existing registered M7 experiment.
Noise is synthetic, not a physiological calibration.
[Latest operational handoff](next-work.md).

### Priority 2 — how much is specific to the connectome wiring?

**Question:** Is M6-style historical-value tracking better in the original
connectome-derived graph than in relevant *feasibly matched* alternatives?

- Compare real-wiring, role/degree/weight-aware rewiring ensembles, and
  simple feedforward delay or otherwise minimal-history computational controls
  under transparent matched input, observation and readout budgets.
- Fix the task, pairing, graph ensembles, train/test separation, seeds,
  effect-size requirement and validation plan *before* outcomes. Preserve
  both current-symbol accessibility and historical decoding measurements.
- Do **not** call an unmatched graph a pure test of feedback cycles. M3 shows
  short-lag feedforward sufficiency; M4 shows that exact all-lag structural
  support preservation by a same-node DAG is impossible in the tested scope.
  State which connectivity, weight and path properties each feasible control
  does and does not preserve.
- Carry forward P2 and earlier structural negative results. A generic
  delay-line also tracking lag2 would limit any claim of connectome-specific
  memory; an advantage would need independent confirmation, not post-hoc
  selection of an accommodating null.

This is **not registered or executed**. It asks a distinct structural
question rather than repairing the failed P2 hypothesis.

### Priority 3 — mathematical sequence rules, not just recalled symbols

**Question:** Can a trained decoder or explicitly defined learner transfer
regularities of arithmetic/geometric/periodic sequences to unseen examples,
rather than reproduce already trained digits or merely access past inputs?

- Define valid input encodings, tasks and held-out splits across sequence
  instances and parameters; separate **past-symbol decoding**, **next-symbol
  prediction** and **unseen-rule generalization** as different endpoints.
- Compare with trivial delay, frequency, n-gram/finite-state and appropriately
  budgeted neural baselines. Include random and shuffled-sequence controls to
  prevent easy periodic patterns from masquerading as learned mathematics.
- If recurrent/internal weights are to change, introduce a separate prospective
  learning protocol. Success with a fitted output head alone does **not**
  establish internal synaptic learning, and computation on fly-derived wiring
  does not directly describe how humans learn mathematics.

This is a **distinct exploratory proposal**; it is not part of M7 and has no
results or promised outcome.

### Priority 4 — synthesis and responsible research release

- Prepare a concise Korean/English synthesis of what was confirmed, what
  failed (including P2 and local plasticity), what was assay-invalid, and
  what remains biological speculation. Keep representation, decoding,
  autonomous recall and internal learning clearly separate.
- Validate test/replay instructions, manifests, checksums, source/data
  attribution and release archive size. Follow [Zenodo guidance](ZENODO.md)
  and [citation metadata](../CITATION.cff); code's MIT license does not
  override FlyWire-derived data terms.
- Consider a pinned GitHub software release followed by a Zenodo **Software**
  record/DOI after metadata checks, then a separately curated dataset or
  research preprint if appropriate. This is a plan, **not a claim that a DOI,
  paper or release already exists**. Documentation work may proceed in
  parallel with future experiments.

**한국어 요약:** M6까지의 결과는 확정·보존한다. 다음 우선순위는
(1) 고정 출력층을 사용한 과거 숫자 추적의 인공 잡음 강건성(M7 가칭),
(2) 실제 초파리 연결망과 재배선망·단순 지연 회로의 공정한 비교,
(3) 기존 숫자열 암기와 구별되는 수열 규칙의 새로운 사례 전이,
(4) 성공·실패를 모두 담은 보고서 및 재현 가능한 공개 자료다.
1~3은 **아직 수행하지 않은 제안**이며, 실제 초파리나 인간의
수학 학습이 입증됐다는 뜻이 아니다.

---

## Current — M6 frozen tracking closed (2026-10-08)

[M6](counterfactual-tracking-results.md) prospectively confirms unconditioned
lag2 tracking after an ACTUAL one-symbol replacement: primary384/384,
secondary192/192, with identical prefix/future and no new fitted parameters.
Current/frequency/instantaneous unchanged paired controls have joint0 exactly.
This is model input→state→frozen-readout tracking, not biological learning/site.

| Stage | Status | Interpretation boundary |
| --- | --- | --- |
| P0/v1 | Sealed | Original claims/negatives preserved |
| P1 iid TDC | Closed; short-lag PASS | Computational historical-input access |
| P2 null ensembles | Closed; material FAIL | No general real-wiring advantage |
| M1 carry/synaptic increment | Closed; assay-invalid | Original endpoint undefined |
| M2 current/history tradeoff | Closed; specified PASS | No universal memory cost |
| M3 cycle-free sufficiency | Closed; PASS | No unique cycle contribution |
| M4 strict-control feasibility | Closed; INFEASIBLE | Other causal designs remain open |
| M5 symbol-contrast sensitivity | Closed; attenuation PASS | Norm reduction is not information loss |
| M6 frozen-readout tracking | Closed; lag2 PASS | No noise robustness or biological inference |
| Next observation-noise tracking | Proposal only | Audit prior ACT V scopes; one fixed artificial dose, paired controls/fresh streams |
| P3 physiology/spiking/plasticity | Outside this work | Separate biological targets/protocol |

Next candidate asks whether the clean paired tracking result survives one
predefined artificial observation-noise setting with frozen heads. Audit earlier
closed noise studies first; do not repeat them or repair old failures under a
new label. No automatic noise/seed/lag/decoder search or physiological calibration.
All44,571 prior results and source-capture failures remain preserved.
[Current handoff](next-work.md). Older dated snapshots retain historical scopes.

## Current — M5 functional sensitivity closed (2026-10-08)

[M5](temporal-functional-sensitivity-results.md) verifies the registered
order-of-magnitude lag5 attenuation criterion in all six primary blocks; mean
response ratio0.66135%, secondary0.01635%. It also verifies fresh iid decoding
and distinguishes local tangents from actual one-symbol replacements.
Attenuation, structural support and decoded information are different metrics.

| Stage | Status | Interpretation boundary |
| --- | --- | --- |
| P0/v1 | Sealed | Original claims/negatives preserved |
| P1 iid TDC | Closed; short-lag PASS | Computational historical-input access |
| P2 null ensembles | Closed; material FAIL | No general real-wiring advantage |
| M1 carry/synaptic increment | Closed; assay-invalid | Original endpoint undefined |
| M2 current/history tradeoff | Closed; specified PASS | No universal memory cost |
| M3 cycle-free sufficiency | Closed; PASS | No unique cycle contribution |
| M4 strict cycle-control feasibility | Closed; INFEASIBLE | Other causal designs remain open |
| M5 symbol-contrast sensitivity | Closed; attenuation PASS | Norm reduction is not information loss |
| Next frozen-readout counterfactual tracking | Proposal only | Fresh streams, valid one-symbol replacement, identical future inputs |
| P3 physiological spiking/plasticity | Outside this work | Separate biological targets/protocol |

Next candidate: does an already-trained lag decoder follow a changed ACTUAL
past symbol on fresh held-out streams? Register fixed lag/criterion, head/model
budget, fresh streams and controls before counterfactual decoding outcomes.
Do not rescore old primary endpoints, select favorable seeds/lags/decoder,
promote small numerical tails into resolved biological memory, or add P3.
All44,147 prior result identities and existing failures remain preserved.
[Current handoff](next-work.md). Earlier dated snapshots retain their history.

## Current — M4 exact-control feasibility closed (2026-10-08)

[M4](cycle-attribution-feasibility-results.md) independently verifies registered
INFEASIBLE for the strict exact cycle-removal control. Partial degree/role counts
force cycles; whole degree-only tests are UNRESOLVED. Relevant-cycle witnesses
in all60/30 symbol cells prevent complete all-lag path-support preservation in
any same-node DAG. No new memory-performance or biological result is claimed.

| Stage | Status | Interpretation boundary |
| --- | --- | --- |
| P0/v1 | Sealed | Original claims/negatives preserved |
| P1 iid TDC | Closed; short-lag gate passes | Computational past-input access |
| P2 null ensembles | Closed; both material gates fail | No general wiring advantage |
| M1 carry/synaptic increment | Closed; assay-invalid | Original endpoint undefined |
| M2 current/history tradeoff | Closed; specified joint PASS | No universal memory cost |
| M3 cycle-free sufficiency | Closed; PASS | No unique cycle-effect claim |
| M4 strict cycle-control feasibility | Closed; INFEASIBLE | Exact ideal rejected; other causal designs remain open |
| Next finite-lag functional sensitivity | Proposal only | Signed/nonlinear influence versus structural support and decoding |
| P3 physiological spiking/plasticity | Outside this work | Separate biological targets/protocol required |

Any next study must register its finite lags, fresh streams, derivative/reference
checks, budgets and endpoints before outcomes. Do not relax M4's exact gate into
success, sweep degree/rank/decoder choices, or reinterpret M3's confounded gap.
The29 valid safeguards and preserved provenance-invalid first guard are documented;
all43,983 prior result identities remain sealed. [Current handoff](next-work.md).
Earlier dated snapshots below retain their historical outcomes/proposals.

## Current — M3 completed (2026-10-08)

[M3](temporal-cycles-results.md) prospectively verifies **cycle-free delayed
feedforward sufficiency**, with current input paths retained and direct carry0.
Primary lag2 decoding99.9833%, minimum baseline excess88.5pp across six blocks;
whole firstthree secondary also passes. All27 cases/54 exact state traces/567
refits and9 finite-history certificate pairs validate. Current accuracy100%.
The mask changes degree/weight/path structure, so pure cycle attribution remains
open; the bounded sufficiency question is closed.

| Stage | Status | Interpretation boundary |
| --- | --- | --- |
| P0/v1 | Sealed | Original claims/negatives preserved |
| P1 iid TDC | Closed; short-lag gate passes | Computational past-input access |
| P2 null ensembles | Closed; both material gates fail | No general wiring advantage |
| M1 carry/synaptic increment | Closed; assay-invalid | Original endpoint undefined |
| M2 current/history tradeoff | Closed; registered joint PASS | Specific readout/intervention, no universal memory cost |
| M3 cycle-free sufficiency | Closed; primary/secondary PASS | Delayed acyclic propagation can support lag2; no unique cycle-effect claim |
| Next cycle-attribution methodology | Proposal only | Match deleted degree/strength/path effects while preserving current access |
| P3 physiological spiking/plasticity | Outside this work | Requires separate biological targets/protocol |

No seed/order/threshold/decoder sweep remains in M3. Further attribution requires
a new prospective validity argument, not relabeling this result or M1.
[Current handoff](next-work.md). Earlier snapshots retain their original outcomes.

## Current — M2 completed (2026-10-08)

The [joint current/history study](temporal-tradeoff-results.md) is executed and
independently verified under its [prospective protocol](temporal-tradeoff-protocol.md).
Primary carry_only−instantaneous has a6.562pp historical gain and15.025pp current
loss, both directions across all six fresh blocks; joint criterion PASS.
Whole firstthree secondary also passes its own joint criterion. Full and
synaptic_only preserve both high current and historical access, limiting the
claim to this specified intervention. M1's original invalid assay stays sealed.

| Stage / question | Status | Boundary |
| --- | --- | --- |
| P0 / v1 | Sealed | Original evidence and negatives preserved |
| P1 iid TDC | Closed; short-lag gate passes | Computational past-input access |
| P2 null ensembles | Closed; both material gates fail | No general wiring advantage |
| M1 carry / synaptic increment | Closed; assay-invalid | Original endpoint undefined |
| M2 joint current/history change | Closed; primary and secondary PASS | Specific linear-decoding intervention comparison; no universal memory cost |
| Next: delayed feedforward / feedback cycles | Proposal only | Separate fixed controls/protocol needed; not executed |
| P3 physiology / spiking / plasticity | Outside this work | Biological calibration and learning claims remain open |

No extra seeds, decoder changes or carry/gain sweep remain in M2. Future cycle
work must preserve input access and register its own question/validity/budget
before new outcomes; M1's R switch includes both delayed feedforward and
feedback transmission. [Current handoff](next-work.md). Earlier dated snapshots
below retain their original outcomes.

## Current — 2026-10-08

The next-work state-carry/history-dependent transmission question has now been
[registered](temporal-mechanism-protocol.md), executed and
[independently verified](temporal-mechanism-results.md). Its outcome is
**assay-invalid**: carry_only lag0 accuracy84–86% fails the fixed90% current
input-access requirement. A descriptive+17.815 pp partial historical gap does
not yield a registered success. The whole secondary gap is+15.970 pp, also
ineligible. All four arms and all fixed blocks are retained, including the
initial verifier memory failure and technical retry. Earlier stages are closed.

| Stage / question | Status | Boundary |
| --- | --- | --- |
| P0/v1 | Sealed | Historical claims and negatives unchanged |
| P1 iid TDC | Closed; short-lag gate passes | Computational historical access |
| P2 matched null ensembles | Closed; both material gates fail | No general real-wiring advantage |
| M1 carry / previous-state transmission | Closed; assay-invalid | Material mechanism endpoint undefined;36 cases/72 exact replays/756 refits |
| Future current/history tradeoff assay | Proposal only | New question, budget and prospective validation; cannot revise M1's eligibility |
| Future delayed feedforward / feedback-cycle distinction | Proposal only | M1 does not isolate anatomical feedback cycles |
| P3 physiological spiking/plasticity | Outside completed scope | Requires separate biological targets/protocol; unexecuted here |

No extra seed, threshold change, decoder replacement or coefficient sweep is
a completion step. Future work must preserve this outcome and ask its own
question before observing new outcomes. [Current handoff](next-work.md).
The earlier dated roadmap below retains its historical conclusions.

## Current — 2026-10-06

The finite [v1/P1/P2 programme](temporal-memory-synthesis.md) is complete and
independently verified. Research completion does not turn a failed hypothesis
into success. New evidence and manifests are isolated under `results/tdc_v2/`.

| stage | fixed question | outcome / boundary |
| --- | --- | --- |
| P0 | Seal current claims/provenance without tuning past results | [Closed](drosomem-v1-closeout.md);39,608 historical result blobs preserved |
| P1 | Independent iid temporal decodability versus lag | [Closed, criterion passes](temporal-memory-curve-results.md); all 6 blocks at lag 1–5 |
| P2 primary | Partial real versus 20 matched null graphs | [Closed, advantage fails](temporal-null-ensemble-results.md); real below all 20 |
| P2 secondary | Whole-brain real versus 10 matched nulls | Closed, material gate fails; small+0.985 pp direction retained |
| P3 | Physiology-calibrated spiking/plasticity | Excluded; unexecuted in this work, not a remaining P0/P1/P2 step |

The supported claim is computational past-input access, not living-fly memory,
anatomical storage, formal capacity or general real/whole-brain superiority.
Any future study must ask a distinct question, register its budget/criteria
before outcomes, preserve this programme and avoid a success-seeking seed or
threshold loop. [Handoff](next-work.md). Earlier dated roadmaps below are history.

## Latest — 2026-10-02

The [full length × alphabet × family grid](followup-3-results.md) is complete;
the independent-input historical-symbol criterion passes for iid and Markov
streams, while periodic motif replay is not evidence of novel-motif prediction.
The [whole-brain scale and DAN→MBON intervention](followup-4-results.md) is
complete: no whole-brain decoding advantage, but the matched pathway cut
criterion passes. A [separate whole-brain rewiring null](followup-4b-results.md)
does not confirm a real-wiring advantage against one degree- and
role-preserving graph. The [local dopamine-rule and physiology comparison](followup-5-results.md)
is complete: local weight changes pass causal controls, but baseline MBON
silence prevents the functional response gate and there is no fixed-decoder
benefit. Additional graph-null ensembles or response-bearing physiological
models would need new protocols. Older positions below are historical.

The [DAN→MBON temporal-pathway intervention](temporal-pathway-results.md) is
complete: 12 paired state panels, 96 fresh ridge fits and independent replay
of 64 secondary autonomous paths. Current-only cutting improves refitted MBON
delayed decoding while persistent cutting impairs it, so the preregistered
current-path-dominance criterion fails in both cohorts. A descriptive
current/history interaction is visible; it does not locate stored memory.

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
| ACT III-D | Five preregistered pathways, matched removal controls, sensitivity map and fresh confirmation;273 cases; subsequent four-arm temporal-pathway study completed with a failed current-path-dominance criterion | Minimal/unique subnetwork, motifs, whole-brain localization and storage-site identification remain open |
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
80 exact replays and 8 independent refits. Both graphs pass trained-prefix gates
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
30 archived clean heads and 18 fresh heads were evaluated without perturbation
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
