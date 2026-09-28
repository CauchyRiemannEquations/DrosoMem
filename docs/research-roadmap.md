# Flying research roadmap

Flying studies computational models using actual Drosophila connectome structure.
It does not demonstrate a living fly memorizing pi. Always distinguish
**representation** (past inputs affect state), **decoding** (a head extracts useful
information), and **learning** (experience modifies recurrent connections).

## Current position — 2026-09-28

**ACT II blocked-state and frozen cross-arm diagnostics complete within scope.** Completion of a scoped experiment
does not mean every question in its ACT is solved.

| Axis | Completed scope | Remaining scope |
|---|---|---|
| ACT I | Matched partial/expanded/whole graphs, thresholds5/1; no established whole-brain recall advantage | Broader tasks or dynamics would require separate protocols |
| ACT II | Four-family pilot, random N32–512 scaling, prefix-weight tradeoff, independent-stream delay diagnostic, N128 K2/4/10/16 comparison,120 finite-context controls, fixed-K intervention and both state-decoding diagnostics | Next selected step: scoped ACT III structural control; additional families and a full length-by-K design remain open |
| ACT III | Earlier limited controls/ablations exist | Next: current-protocol real vs role/degree-rewired partial graph on independent K4 streams; broader grouped ablations and critical-subnetwork analysis remain open |
| ACT IV | Earlier local-rule implementation with negative functional results | Useful internal learning without external decoder dependence |
| ACT V | Scoped robustness curves exposed strong fragility | Broader perturbation/physiological validation, kept separate from clean benchmarks |

현재는 **ACT II의 저장 상태 해독과 고정 출력층 교차 수열 전이 진단까지 완료**했다.
두 그래프 모두 과거 기호 접근성과 제한적 전이를 보였으나 다음 기호 개선이나
자율 회상 개선을 새로 확인한 것은 아니다. 다음 한 실험은 **독립 K=4 입력에서 실제 부분망과 역할·입출력 차수를 보존한 재배선 망을 비교하는 구조 대조**이다.
현행 조건에서의 ACT III 대조를 제안한 단계이며 아직 실행하지 않았다.
ACT IV 내부 학습을 앞당기지 않는다. 추가 수열군과 전체 길이×K 설계는 여전히 열린다.

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
[fixed-K intervention](context-intervention-results.md), now complete. Confirmed graphs: none. The selected [blocked-state diagnostic](frozen-state-probe-results.md) and [cross-arm transfer](cross-arm-probe-results.md) are now complete. Next is the scoped ACT III structural control listed above, not yet executed.
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

## ACT V — Robustness (independent axis)

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
