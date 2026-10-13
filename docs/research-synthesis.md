# DrosoMem: historical-input access, structural limits and rule-continuation failure

Date: 2026-10-13. Evidence cutoff:
`c48e0c43836e640ec1a15b1ac1152f7e3ec4e2a4`.
This synthesis covers P1/P2 and M1–M11, with the original v1 closeout as an
anchor. It generates no new neural trajectory, fitted model or scientific
performance endpoint. Earlier phases, local-learning negatives and every
archived failure remain unchanged.

[Claim/evidence matrix](research-evidence-matrix.md) ·
[Pinned reproduction index](research-reproduction-index.md) ·
[Internal manuscript outline](manuscript-outline.md) ·
[Synthesis inventory](../results/research_synthesis_v1/inventory/checks.json) ·
[Final document audit](../results/research_synthesis_v1/final_audit_retry2/checks.json).

## 한국어 핵심 정리

현재 근거는 **고정된 연결망 상태에서 과거 입력의 흔적을 읽을 수 있다**는
계산 결과를 지지합니다. [P1](temporal-memory-curve-results.md)의 새 무작위
입력에서는 지정한1–5단계 지연 기준을 모두 통과했고,
[M6](counterfactual-tracking-results.md)의 고정 출력기는 과거 숫자를 실제로
바꾼 경우에도 지정한2단계 지연에서 그 변화를 따라갔습니다.

이 결과만으로 원래 초파리 배선의 특별한 우위가 확인되지는 않았습니다.
[P2](temporal-null-ensemble-results.md), [M9](structural-tracking-results.md),
[M11](rule-transfer-results.md)의 등록된 배선 우위 기준은 모두 실패했습니다.
특히 M9는 모든 망이100%여서 차이를 가리는 데 한계가 있었고, M11은
학습 수열의 습득부터 부족했습니다. 서로 다른 실패 원인과 범위를 구분합니다.

[M3](temporal-cycles-results.md)는 고리 없는 일부 구조에서도 시간차를 둔
전달만으로 과거 입력을 읽을 수 있음을 보였습니다. 고리를 없앨 때 다른 연결
특성도 변했으므로 고리만의 기여를 측정한 결과는 아닙니다.
[M4](cycle-attribution-feasibility-results.md)는 모든 조건을 엄격히 유지한
대조 설계가 불가능함을 확인했습니다. 구조적으로 전달 경로가 있다는 사실과
실제로 유용한 정보를 읽는 능력도 다릅니다.

[M10](rule-transfer-design-results.md)의 READY는 새로운 규칙 계수 시험을
준비했다는 뜻입니다. 공식이 주어진 계산 대조군은 이를 풀지만, 공식 없이
고정 연결망과 학습 출력층을 사용한 M11은 학습·전이 기준에 실패했습니다.
학습 정답 입력 정확도19.35%·18.86%, 새 규칙 첫 정답률13.23%·13.94%,
주기2를 제외한8기호 완주율0.51%·1.01%를 함께 보고합니다. 성공적으로
익힌 규칙의 전이만 실패했다고 분리해 해석할 수 없는 결과입니다.

실행·독립 검증의 완료와 가설의 성공은 다릅니다. M1의 미정 판정, M4의
설계 불가능, M8·M10의 준비 완료, 나머지 지정된 PASS/FAIL을 그대로
유지했습니다. 현재 단락은 종합 문서와 보관 점검까지 완료하며, 새로운
생물학적 학습·수학 학습 또는 외부 출판을 선언하지 않습니다.

## What is supported

The core supported statement is operational: a fixed source-derived
fly-connectome computational reservoir can retain information about previous
inputs that a fitted external decoder accesses. P1 tests independent iid
streams rather than recall of a memorized digit string: all30 predefined
partial lag1–5 cells exceed the strongest registered baseline by at least10pp.
Its partial lag2 mean is83.11%, with weaker long-lag access through that same
interface. This is not a formal capacity estimate or a physiological time axis.

M6 strengthens the history interpretation with an actual value intervention.
With the M5 decoder/scaler frozen, replacing one past symbol while retaining
the full prefix and future inputs gives384/384 partial and192/192 paired whole
joint transitions. Unchanged current/frequency/instantaneous predictions cannot
be correct for both distinct values; their joint0 is not a10% chance estimate.
The conclusion concerns that lag/substitution interface, not a biological
storage compartment or robustness at all perturbations.

M2 supports a specific decoder tradeoff under carry-only versus instantaneous
dynamics: partial short-history accuracy increases6.562pp and current accuracy
decreases15.025pp on fresh paired streams. Other arms retain strong current and
history access. The result does not imply a conserved information budget or a
universal cost of memory. M1's original90% all-arm current-access requirement
failed, so its mechanism endpoint remains undefined despite positive descriptive
contrasts. M2's narrower prospective question does not rescue M1.

M3 establishes bounded delayed-feedforward sufficiency: an imposed acyclic,
zero-direct-carry source subset decodes lag2 at11998/12000 (99.983%) in partial
analysis. Finite dependency certificates verify its scheduling. The mask also
changes degree, weight mass and paths; it cannot isolate a unique feedback-cycle
effect. M4's same-node exact-degree/all-lag matching obstruction is a design
result, not a functional performance comparison. Its whole degree-only local
tests remain UNRESOLVED even though exact all-lag support matching is infeasible.

M5 separates structural existence, local response amplitude and readout access.
Every partial lag5 relative local-gain gate passes; mean gain is0.661% of the
immediate derivative response. Nevertheless lag2 decoding is100%. Attenuation
of selected continuous encoder contrasts is neither categorical information
loss nor a measured fraction of memory. Long structural walks do not guarantee
large, useful functional influence.

## What the negative and feasibility results constrain

No registered material real-wiring advantage is established by P2, M9 or M11.
P2's partial real graph is below all20 nulls; whole's positive0.985pp direction
misses the2.7pp requirement. M8 constructs stricter controls preserving both
raw incoming/outgoing weight multisets, roles/degrees/normalization and current
pathways. Its13 reconstructed graphs are READY for comparison, not memory
success. Those finite conditioned banks share important structures and are not
uniform, fully mixed anatomical null populations.

M9's actual/null/engineered reference all reach100% joint historical tracking,
with valid current access. The0pp gap fails the3pp/strict-order hypothesis.
The ceiling limits discrimination; it neither proves equivalence nor justifies
automatically changing lag/noise/seed until the real graph wins. M7 separately
fails frozen tracking at one synthetic standardized observation-noise dose:
255/384 partial and109/192 paired whole joint successes. Clean M6 success and
M9's newly fitted clean heads do not repair that frozen-head robustness failure.

M10 supplies an identifiable mod-11 affine-family benchmark with disjoint
training/held-out multipliers and no shared held-out training prefix/cycle
templates. Its history-3 table completes seen-rule new starts but no held-out
coefficients. The supplied-form oracle infers the equation from the prefix and
completes all440 coefficient tests; that privileged rule knowledge is an assay
reference, not a learned or matched-knowledge model result. Period-2 rules admit
a trivial lag-copy shortcut, handled prospectively in M11's primary exclusion
and explicit references.

M11 compares all216 source/ten-null/shift by input-map/family fits with48
observations,539 nominal affine scalars and2200 training rows per head. Recurrent
weights and encoders remain fixed. All216 independent SVD fits and427680
teacher/autonomous trajectory digests match; all primary and secondary gates
fail. Training teacher accuracy19.35/18.86% already misses50% acquisition.
On held-out a7–9, completion5/990 and10/990 yields only0.505/1.010pp over shift,
below5pp; neither cohort is above every null. The declared learner budget fails
acquisition and continuation, so this does not isolate a generalization defect
after successful learning or establish a universal inability to learn rules.

## Learned parameters and evidence units

P1/P2 and M1/M2/M3/M5 fit external490-scalar affine heads per K10 lag, with
train-derived normalization. M6/M7 learn nothing new and reuse the exact old
M5 head/scaler. M4/M8 have no neural fits; M8's templates are analytical model
data. M10 fits frequency/suffix counts while its analytic oracle fits zero
parameters. M9 has174 lag heads (85260 nominal affine scalars plus8352 moments).
M11 has216 official539-scalar heads (116424 scalars plus20736 moments).
Independent verification refits are not extra independent experimental models.
These counts must not be pooled into an amount of biological synaptic learning.

Original full-model dynamics include direct carry; later zero-carry experiments
use their specified history/scheduling arms. Their results cannot be merged as
one unchanged neural condition. Whole analyses often reuse the firstthree
partial input blocks; anchors share streams/heads; M10/M11 reuse an exhaustive
finite task set; M11's shift interface repeats across six input maps. Report
paired computational conditions and raw denominators, not animal replication,
biological confidence intervals or a pooled cross-study success rate.

## Provenance and responsible release boundary

The synthesis authenticates13 canonical completed studies, their saved verdicts,
matching result/validation manifests, successful source records at recorded Git
revisions and statically parsed replay arguments. It inventories materialized
checksums, external cache/input availability and retained technical failures.
It does not freshly regenerate old trajectories, weaken source checks or edit
sealed results. Sparse-excluded evidence is protected by Git identity and
distinguished from fresh byte inspection. Producer and verifier pins can differ,
especially P1 and the memory-repaired M1 verifier.

M10's producer summary remains `READY_PENDING_INDEPENDENT_VERIFICATION`; its
successful independent checks and final audit establish final READY. M4's
complete preserved source-mismatch snapshot is scientifically invalid; archive
completeness does not make that captured source a successful experiment. The
new matrix records both distinctions without rewriting old hashes or prose.

The initial synthesis document audit checked its own result link before creating
that result file and stopped. Its first retry also detected Windows newline
bytes differing from the canonical Git document bytes. Both failed attempts and
manifests remain in the new synthesis namespace. The corrected audit validates
the planned own artifact, creates it and verifies its presence; all documents
were normalized and directly compared to committed Git bytes before the final
retry. No study verdict or old scientific archive changed.
[Retained synthesis attempts](../results/research_synthesis_v1/known-incomplete-attempts.json).

This completes the bounded synthesis/archive handoff after M11. The manuscript
outline is an internal draft with a mapped evidence base. Literature context,
novelty assessment, full manuscript authoring, target venue/format, author review
and any external submission remain separate work. No DOI, journal publication,
GitHub Release or external submission is created or claimed here. P3 and broader
learner/rule-family/physiological extensions remain unregistered separate studies.
