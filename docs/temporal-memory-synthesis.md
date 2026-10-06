# DrosoMem: v1 seal, TDC and matched null ensembles — synthesis

P0/P1/P2 are complete within their fixed scopes. Numerical/structural validation
passes; scientific confirmation and execution completion are separate.
[v1 closeout](drosomem-v1-closeout.md), [P1](temporal-memory-curve-results.md),
[P2](temporal-null-ensemble-results.md), and
[final provenance audit](../results/tdc_v2/final_audit/audit.json).

## 1. What did we ask?

How does access to past input decline as temporal lag increases in a fixed
source-derived fly-connectome reservoir? Does its real wiring preserve more
history than multiple degree/role/weight-matched rewired controls?
Independent iid K10 streams separate historical decoding from predictable
sequence replay. The TDC is `state(t) → symbol(t−lag)` accuracy at fixed48-MBON/
490-parameter ridge budgets. It is not formal Shannon/reservoir memory capacity.
Protocols were committed before new results, and every planned seed is retained.

## 2. What was confirmed?

P1 passes the prospective conjunction: all six partial-graph blocks exceed
chance/frequency/current-symbol-only controls by≥10 pp at every lag 1–5. Minimum
excess is29.85 pp. Mean partial accuracy is99.96% at lag 1,83.11% at lag 2,43.60%
at lag 5 and 26.73% at lag 8; lag 20 is9.62% near the 10% baselines. Independent test
streams and fixed recurrent weights rule out interpreting this as learning a
particular sequence's transitions. The curve documents operational decoding
through this interface; long-lag weak decoding is not proof of zero latent info.

P2 construction/validation confirms30 independent seeded matched controls:
20 partial and 10 whole-brain nulls, exact per-neuron degrees, role blocks and
raw presynaptic weight multisets. P1's 24 and P2's 318 complete state trajectories
match independent replay hashes exactly. All252+3,339 lag heads are independently
refitted and predictions checked. These are342 replayed streams, not342 animals.

## 3. What failed?

The registered real-wiring advantage fails in both P2 analyses. Partial real
mean accuracy is29.664% versus 31.949% null mean, a−2.285 pp difference, below
every one of 20 controls. Whole-brain real is28.070% versus 27.085% null mean:
above all 10 controls, but+0.985 pp is below the predeclared2.7 pp magnitude.
The100th percentile/large standardized effect in that small conditional ensemble
does not replace the failed criterion. The opposite directions and all per-lag/
paired values are preserved; no extra seed, reduced threshold or decoder search.

Earlier v1 whole-brain, intact-wiring, robustness/calibration and functional
local-learning failures remain unchanged. DAN→MBON affects temporal information
but does not identify storage. Local dopamine-like changes remain implemented
without established functional memory improvement.

The first P2 generation had a console-only CP949 logging failure. Its incomplete
outputs and interruptions are retained. The same-config retry reproduces24
completed graphs byte-for-byte. No neural outcome motivated that repair.

## 4. What can DrosoMem now claim?

> A fixed computational reservoir built from source-derived Drosophila connectome structure retains decodable information about past inputs across time.

Matched generic recurrence/roles/weights also support substantial temporal
decoding. The primary real partial wiring gives no advantage; the secondary
whole-brain result permits a small task/model-conditional structural difference,
without meeting the registered material gate or establishing a general rule.
The study does not show that all wiring is interchangeable or uniformly sample
all possible nulls. It does not identify an anatomical storage location, prove
flies memorize pi, validate physiological dopamine learning, establish general
whole-brain superiority or measure formal memory capacity. Only external ridge
parameters are learned in P1/P2. P3 physiology/spiking/plasticity is outside scope.

## Preservation and next boundary

Baseline `0cf0867736ca598ae233cdf3e185031667dd52be` is sealed by the full Git
inventory of 39,608 historical result files. P0 fresh byte/array audit covers
the 4,719 materialized archives,8,788 matching checksum entries and 28,512 finite
arrays;34,889 sparse-excluded files retain Git blob IDs but are not claimed
to have been freshly replayed. All new evidence uses `results/tdc_v2/`.
The final audit verifies historical blobs and new recorded-source/checksum links.
P1/P2 execution and verification have ended. A new mechanism/task/dynamics
question needs a distinct prospective protocol; none is automatically added
to make a failed result succeed. [Current handoff](next-work.md).

## 비전공자를 위한 한국어 요약

초파리 뇌의 연결 자료를 바탕으로 만든 **컴퓨터 모델**에 무작위 숫자를
계속 넣고, 지금 상태에서 예전 숫자를 다시 맞혀 보았습니다. 두 단계 전
숫자는 평균83%, 다섯 단계 전 숫자는44% 정도 맞혔습니다. 열 가지 숫자
중 무작위로 고를 때의10%보다 높아서, 최근 입력의 흔적이 모델에 남는다는
근거가 생겼습니다. 시간이 더 지나면 복원이 어려워졌습니다.

그런데 원래 초파리 배선이 특별히 유리하다는 사전 기준은 통과하지
못했습니다. 부분 모델에서는 조건을 맞춰 배선을 바꾼20개 모델이 모두
더 좋았습니다. 전체 뇌 모델의 원래 배선은10개 대조보다 조금 좋았지만,
미리 정한 충분한 차이에는 못 미쳤습니다. 이 결과는 그대로 실패로 남깁니다.

따라서 현재 말할 수 있는 것은 **이 계산 모델이 과거 입력 정보를 잠시
남긴다**는 것입니다. 실제 초파리가 원주율을 외운다거나, 특정 세포가
기억 저장소라는 결론은 아닙니다. 실험이 끝났다는 것과 가설이 성공했다는
것을 구분했고, 예전의 부정 결과와 이번 실패 기록도 보존했습니다.
