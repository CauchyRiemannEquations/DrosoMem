# 4번: 전체 뇌 규모의 구조·DAN→MBON 경로 — 결과

**전체 뇌망의 일반적인 지연 해독 우위는 실패했고, DAN→MBON 절단의 별도
경로 특이성 기준은 통과했다.** 동일한 입력 root ID와 48개 관측 MBON을
유지한 K4 독립 입력에서 intact 정확도는 부분망 `legacy5` **74.522%**,
138,639-neuron `brain5` **71.161%**였다. `brain5−legacy5`는 seed별
**−3.317/−2.917/−3.850%p**로 등록한 +5%p 기준의 반대 방향이다.
`brain5`에서 DAN→MBON 제거와 연결 수·정규화 가중치 총량을 맞춘 대조
제거의 정확도 차이는 seed별 **+4.083/+4.683/+5.017%p**로 지정한
+3%p 기준을 넘었다. 두 판정은 서로 다른 질문이다.

## 설계와 기준값

[사전 설계](followup-4-protocol.md)를 새 신경 결과 전에 `d542aaa`로,
[설정](../configs/followup4.json)과 구현을 `b0f1866`에서 고정했다.
저장소의 고정 [v783 원본](../src/flying/data/whole_brain.py)을 SHA256으로
확인하고 전체 노드 138,639개, threshold-5 연결 2,700,513개인 `brain5`를
구축했다. `legacy5`는 686개 뉴런과 3,309/3,241개 연결이다.
원본/관측 root ID의 교집합, KC 입력 패턴, gain .9/leak .6,
incoming-L1 정규화, `mbon_after_kc` 동역학과 48 MBON/alpha-1 ridge를
같게 유지했다. 연결 가중치는 학습하지 않았다. 3 seed × 2 입력 맵 ×
2 규모 × 3 절단 조건 = **36개 새 상태·출력층 조건**이다.

각 규모에서 intact, DAN→MBON 전체 제거, 비DAN→MBON 제거 대조를
비교했다. 대조는 활동·정답을 보지 않고 같은 **연결 수**와 가까운
**정규화 가중치 절댓값 총량**을 갖는 연결을 선택했다. 부분망에서
189개, `brain5`에서 735개를 제거했다. 대조 총량의 DAN 목표 대비
오차는 `brain5` **0.0028%**, 부분망 **최대 0.334%**로 허용한 10%보다
작았다. 연결의 presynaptic 역할과 각 MBON별 분포까지 맞춘 대조는
아니며, 경로 특이성을 생물학적 유일성으로 해석하지 않는다.

## 주 결과

각 값은 3 seed와 각 seed 안의 2입력 맵에서 lag 1,2,3,4,5,8을 같은
가중치로 평균한 held-out 정확도다. 학습 입력 1,000개와 시험 입력
500개는 독립 생성했고 warmup100을 제외했다.

| 규모 | intact | DAN→MBON 제거 | 맞춤 대조 제거 | 빈도 대조 |
| --- | ---: | ---: | ---: | ---: |
| 부분망 `legacy5` | 74.522% | 66.622% | 75.561% | 24.611% |
| 전체 뇌망 `brain5` | 71.161% | 66.811% | 71.406% | 24.611% |

전뇌 우위 gate는 intact `brain5−legacy5` 평균 ≥5%p, 세 seed 모두 양수,
양쪽 intact가 빈도 기준보다 ≥5%p 높은 것을 동시에 요구했다. 마지막
접근성 조건은 통과했지만 규모 우위 조건은 실패했다.
전뇌 경로 gate는 대조 질량이 유효하고 `맞춤 대조−DAN 제거` 평균 ≥3%p,
세 seed 모두 양수를 요구했다. 평균 차이는 **+4.594%p**이고 세 seed
모두 양수여서 통과했다. [Seed별 표](../results/followup4_main/seed-blocks.csv)와
[원자료](../results/followup4_main/raw-cases.csv)에 모든 정확도·맞춤 정보를
보존했다.

| seed | intact `brain5−legacy5` | `brain5` 대조−DAN 제거 |
| ---: | ---: | ---: |
| 820001 | −3.317%p | +4.083%p |
| 820002 | −2.917%p | +4.683%p |
| 820003 | −3.850%p | +5.017%p |

DAN 절단의 큰 부분은 lag2에서 나왔다. `brain5`의 lag2 intact/DAN 제거/
맞춤 대조는 **92.80/72.27/93.33%**, 부분망은
**97.10/73.63/98.43%**였다. 전체 [lag별 표](../results/followup4_main/raw-lags.csv)를
남겼다. Intact 출력층을 다른 절단 조건에 그대로 적용하면 해독이 더
크게 떨어졌고, 별도 출력층 재학습 후 위 표만큼 회복됐다. 따라서 고정
출력층 전이 손실만으로 내부 정보 소실을 주장할 수 없다.

## 독립 검증과 한계

제외된 [smoke](../results/followup4_smoke/manifest.json)는 1 seed·1입력 맵의
6조건이다. 본실험의 [독립 감사](../results/followup4_main_validation/checks.json)는
36개 그래프 절단·대조를 다시 구성하고 **36개 시험 신경 상태와 출력층
36개를 독립 계산**했다. Source/cache hash, 절단 연결과 시험 예측도
대조했다. Main manifest SHA256은
`baddd616ec8e33474588165983e463aa6fabb9939f657ee757c6776e6f876737`.
본실험 실행 시간은 **501.56초**, sampled peak RSS는
**283,303,936 byte**였다.

이 연구의 whole brain은 **전체 뉴런 범위의 threshold-5 계산망**이다.
약한 연결을 포함한 threshold-1망, 무작위화한 전체 배선, 실제 생리
도파민 효과, 자율 회상과 경험에 따른 연결 학습은 이번 설계에 없다.
두 입력 맵은 같은 원본 연결망에서 왔고 seed는 새 동물이 아니다.
새로운 정보가 특정 세포에 저장됐다는 해부학적 결론도 내릴 수 없다.
기존 ACT I의 pi 자율 회상 실패와 이번 K4 지연 해독 결과는 다른 과제와
측정값으로 구분한다.

## 재현

Python 3.12와 [rate lock](../requirements-act1-lock.txt)에서 먼저 고정
원본을 `phase6 prepare`로 확인·캐시한다. 기존 결과를 덮어쓰지 않고
새 출력 경로를 지정한다.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-act1-lock.txt
$env:PYTHONPATH = 'src;scripts'
$env:OPENBLAS_NUM_THREADS = '1'
.\.venv\Scripts\python.exe -m flying.training.phase6 prepare --raw outputs/followup_raw --cache outputs/followup_whole_cache
.\.venv\Scripts\python.exe scripts/followup4.py --smoke --out outputs/followup4_smoke_new
.\.venv\Scripts\python.exe scripts/verify_followup4.py outputs/followup4_smoke_new --out outputs/followup4_smoke_new_validation
.\.venv\Scripts\python.exe scripts/followup4.py --out outputs/followup4_main_new
.\.venv\Scripts\python.exe scripts/verify_followup4.py outputs/followup4_main_new --out outputs/followup4_main_new_validation
```
