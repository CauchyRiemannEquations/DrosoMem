# DAN→MBON 시간 경로 개입 — 결과

**사전 등록한 ‘현재 단계의 DAN→MBON 경로가 손상을 지배한다’는 기준은 두 집단 모두 실패했다.**
현재 단계만 절단하면 같은 MBON 48개에서 출력층을 다시 학습한 과거 기호 해독이
발견 집단 **2.894%p**, 확인 집단 **2.225%p 좋아졌다**. 절단 이력과 현재 절단을
함께 적용하면 각각 **8.408%p**, **8.186%p 나빠졌다**. 이 차이를 단순한
저장 위치나 관측 경로 한쪽으로 귀속할 수 없다.

## 1. 사전 등록과 기존 근거

[설계](temporal-pathway-protocol.md)와 [설정](../configs/temporal_pathway.json)을
새 결과를 보기 전 commit `7d41e8f`로 고정했고, 실행기와 독립 검증기는
`4af58ad`에서 고정했다. 앞선 [ACT III-D 경로 연구](pathway-memory-results.md)와
[관측 위치 연구](observation-location-results.md)에서 DAN→MBON 제거 뒤
MBON 지연 해독 저하가 관찰됐지만, 출력 관측 경로의 즉시 효과와 반복 절단으로
달라진 신경 상태를 분리하지 못했다. 이번 연구는 그 구분을 한 가지 고정된
4조건 반사실 패널에서 시험했다. 앞선 결과와 체크섬은 변경하지 않았다.

## 2. 기준 모델과 새 개입

동일한 source-derived 686-neuron 부분 회로 2개(701/702), K4 독립 균등 입력,
warmup 100, 학습 2,000, 독립 시험 1,000, 48 MBON 관측, alpha 1 ridge를
사용했다. 회로의 반복 연결 가중치는 학습하지 않았다. 각 회로에서 정규화된
DAN→MBON 연결 189개만 제거하고 나머지는 재정규화하지 않았다.

| 조건 | 현재 입력 직전 상태 | 현재 입력의 연결 |
| --- | --- | --- |
| intact | intact 이력 | intact |
| current_only | intact 이력 | DAN→MBON 절단 |
| history_only | 계속 절단된 이력 | intact |
| persistent | 계속 절단된 이력 | DAN→MBON 절단 |

네 조건은 같은 외부 입력 기호를 받는다. `current_only`와 `history_only`는
각 시점에 기반 상태에서 계산한 **한 단계 probe**이며, 각자 독립적으로
이어가는 네트워크가 아니다. 각 조건의 학습 상태에서 별도 ridge를 학습해
시험 상태에 적용했다. 주 평가는 lag 1, 2, 3, 4, 5, 8을 같은 가중치로
평균하고 회로 2개를 먼저 seed 묶음 안에서 평균했다. 손실은
`intact 정확도 − 해당 조건 정확도`로 정의한다.

## 3. 실행과 독립 검증

발견 seed 720001–720003, 확인 seed 721001–721003에서 6묶음 × 2회로 =
**12개 상태 패널**, 네 조건 × MBON/KC 두 관측 지점 = **96개 새 ridge fit**을
실행했다. 사전 지정한 짧은 smoke 1패널/8 fit은 본 통계에서 제외했다.
[실행기](../scripts/temporal_pathway.py)는 원 입력, 네 상태, 원 lag 결과,
출력층, frozen 전이, 보조 자율 경로와 manifest를 저장한다.
[검증기](../scripts/verify_temporal_pathway.py)는 고정된 `TimedReservoir.step`으로
네 조건의 상태를 별도 재생하고, **96개 출력층을 모두 재학습**하며, frozen
예측과 요약 판정을 다시 만든다. 보조 자율 경로 **64개도 모두 독립 재생**했다.
[main 검증](../results/temporal_pathway_main_validation/checks.json)의 모든 검사가
통과했고 최대 자율 확률 오차는 0이었다. 위치를 옮긴 뒤에도 main 341개,
smoke 25개 및 각 검증 manifest의 1개 artifact가 전부 기록된 SHA256과 일치했다.

## 4. 사전 등록한 주 결과

각 값은 세 seed 묶음의 평균이며 단위는 **정확도 % 또는 차이 %p**다.
`current_only` 손실의 음수는 해당 조건의 성능 향상을 뜻한다.

| 집단 | intact 정확도 | persistent 손실 | current_only 손실 | history_only 손실 | 등록 판정 |
| --- | ---: | ---: | ---: | ---: | --- |
| 발견 | 77.725% | +8.408%p | −2.894%p | +1.319%p | 실패 |
| 확인 | 77.878% | +8.186%p | −2.225%p | +1.714%p | 실패 |

등록 기준은 각 집단에서 persistent 손실 ≥5%p, current_only 손실 ≥5%p와
모든 seed에서 양수, current_only/persistent 손실 ≥75%, history_only 손실
≤2%p를 **동시에** 요구했다. persistent와 history_only의 집단 평균 조건은
통과했으나 current_only 방향과 크기, 지배율 조건이 모두 실패했다.
current_only는 6개 seed 묶음 모두에서 오히려 향상됐다. [전체 통계](../results/temporal_pathway_main/summary.json)에는
중앙값, 표본분산, paired dz, 부호 수와 10,000회 seed 묶음 bootstrap
기술 구간을 저장했다. persistent 손실의 95% 기술 구간은 발견
[7.925, 8.708]%p, 확인 [7.800, 8.492]%p이고, current_only는 각각
[−3.500, −2.583]%p, [−2.575, −1.958]%p다. 세 계산 seed의 bootstrap
구간은 생물학적 표본 추론이 아니다.

| 집단 | seed | persistent 손실 | current_only 손실 | history_only 손실 |
| --- | ---: | ---: | ---: | ---: |
| 발견 | 720001 | +8.592 | −2.600 | +1.217 |
| 발견 | 720002 | +7.925 | −3.500 | +0.200 |
| 발견 | 720003 | +8.708 | −2.583 | +2.542 |
| 확인 | 721001 | +7.800 | −2.142 | +1.675 |
| 확인 | 721002 | +8.492 | −2.575 | +1.808 |
| 확인 | 721003 | +8.267 | −1.958 | +1.658 |

정확한 seed 값은 [짝 차이 CSV](../results/temporal_pathway_main/paired-differences.csv)에
있다. 발견 seed 720003의 history_only 손실은 2%p를 넘지만 등록 조건은
**집단 평균**에 관한 것이었다. 이 행을 새로운 실패 기준으로 사용하지 않는다.

## 5. 탐색적 상호작용과 lag별 양상

사전 판정에 없던 기술적 대비
`persistent 손실 − current_only 손실 − history_only 손실`은 발견
**+9.983%p**, 확인 **+8.697%p**였다. 여섯 seed 모두 양수다. 따라서
이 고정 모델에서는 계속 절단된 이력과 현재 절단의 조합이 각 단독 probe의
손실을 단순히 더한 값보다 크다. 이는 **사후 탐색적 상호작용**이며 새로운
확인 판정이나 독립 표본의 유의성 주장은 아니다.

사전 선택한 lag 가운데 current_only의 개선은 특히 lag 3에서 보였다
(발견 intact 78.52% → current_only 92.48%; 확인 77.60% → 90.48%).
persistent는 lag 2에서 크게 악화됐다(발견 99.47% → 76.77%; 확인
98.95% → 76.03%). 이 lag별 표는 주 평균을 대신하지 않는다.
[모든 lag 원자료](../results/temporal_pathway_main/raw-lag-metrics.csv)를 보존했다.

## 6. 보조 진단: frozen 출력층, KC, 자율 회상

Intact에서 학습한 MBON 출력층을 **고정**한 채 다른 조건에 적용하면
persistent 손실은 발견/확인 **50.233/49.217%p**, current_only는
**48.781/46.086%p**, history_only는 **44.683/43.153%p**였다. 재학습 결과와
크게 달라 해독 좌표의 변화가 중요함을 보여준다. 하지만 frozen 전이 실패만으로
과거 정보가 회로에서 사라졌다고 결론 낼 수 없다.

사전 지정된 비자극 KC 48개의 refit 기준 정확도는 발견/확인
**50.903/50.928%**이고 persistent 손실은 **+0.069/−0.044%p**였다.
current_only는 이 KC 상태에 직접 변화가 없어 두 집단 모두 차이가 0이다.
KC의 낮은 기준 해독률과 단일 부분망 범위 때문에 이는 ‘모든 저장은 KC에
남는다’는 증거가 아니다.

기존 fresh random-digit intact head 16개를 재학습하지 않고 네 조건에서
각각 자기 출력을 피드백한 보조 자율 경로 64개는 모두 독립 검증했다.
평균 첫 오류 전 정답 기호 수는 발견에서 intact/current_only/history_only/
persistent **32.000/1.000/2.800/1.000**, 확인에서
**28.000/1.000/1.833/0.833**이었다. 이 frozen head의 심한 전이 손실은
주 refit 해독 기준을 대체하지 않으며, 자율 회상의 회복을 보여주지 않는다.
[경로별 값](../results/temporal_pathway_main/autonomous-rollouts.csv)에 보존했다.

## 7. 해석과 한계

이번 개입은 *현재 DAN→MBON 입력만 제거해도 refit된 MBON의 지연 정보가
대부분 사라진다*는 단순 설명을 지지하지 않는다. 지속 절단의 손실은 확인됐고,
현재와 이력의 조합에 큰 비가산 효과가 있다. 절단 이력은 MBON뿐 아니라
반복망 전체의 상태를 바꿀 수 있으므로 이 결과만으로 특정 세포군의
**저장 위치**를 분리할 수 없다. MBON에서 읽을 수 있는 정보, 고정 출력층의
전이성, 닫힌 고리 회상은 서로 다른 측정값이다.

두 회로는 같은 connectome release의 부분망이며 seed는 새로운 동물이
아니다. 189개 연결의 제거는 모델상 개입으로, 도파민의 생리적 작용을
재현한 것이 아니다. 반복 연결의 경험 기반 학습, 살아 있는 초파리의 기억,
전체 뇌의 저장 위치, formal 기억 용량을 입증하지 않는다. 실패한 등록 판정을
lag 3 개선이나 탐색적 상호작용으로 바꾸지 않는다.

## 8. 재현

Main [manifest](../results/temporal_pathway_main/manifest.json) SHA256:
`670805f59bec2c8e55c34e4bd96e514a794fc261f4dd73868016e1d472430a25`.
실행은 48.40초/최대 sampled RSS 184,225,792 byte, 독립 검증은
29.35초/203,952,128 byte였다. 환경과 실행 예산은 각각
[실행 기록](../results/temporal_pathway_main/verification.json)과
[검증 기록](../results/temporal_pathway_main_validation/checks.json)에 있다.
Python 3.12, [잠긴 의존성](../requirements-act1-lock.txt), 저장소 root에서
기존 `results/fresh_coordinate_sd` parent를 보존한 채 새 출력 경로를 사용한다.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-act1-lock.txt
$env:PYTHONPATH = 'src;scripts'
$env:OPENBLAS_NUM_THREADS = '1'
.\.venv\Scripts\python.exe scripts/temporal_pathway.py --smoke --out outputs/temporal_smoke_new
.\.venv\Scripts\python.exe scripts/verify_temporal_pathway.py outputs/temporal_smoke_new --out outputs/temporal_smoke_new_validation
.\.venv\Scripts\python.exe scripts/temporal_pathway.py --out outputs/temporal_main_new
.\.venv\Scripts\python.exe scripts/verify_temporal_pathway.py outputs/temporal_main_new --out outputs/temporal_main_new_validation
```

## 9. 다음 연구 경계

이 한 가지 시간 경로 분해 연구는 실패 결과까지 포함해 완료했다. 비가산
상호작용의 위치와 시간 폭, 실제 도파민 가소성, 전체 뇌 일반화는 별도
질문·예산·사전 기준이 필요하다. 이번 결과에 맞춰 seed, lag, 출력층 또는
절단 강도를 추가 탐색하지 않는다.
