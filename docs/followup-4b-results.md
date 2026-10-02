# 4B: 전체 뇌 degree·역할 보존 재배선 대조 — 결과

**고정한 전체 뇌 배선 우위 기준은 실패했다.** 실제 `brain5`와 뉴런별
입출력 degree, 역할별 연결 수, 각 presynaptic 뉴런의 출력 가중치 집합을
보존한 재배선망을 비교했다. 동일한 K4 입력·48 MBON·ridge 예산에서
실제−재배선의 지연 해독 차이는 seed 평균 **+0.178%p**이고,
세 seed 중 하나는 **−0.567%p**였다. 사전 지정한 평균 ≥3%p 및
전 seed 양수 조건을 함께 충족하지 못했다. 이는 이번 한 개 재배선
그래프와 과제에서의 부정 결과이며 구조의 보편적 동등성 증명은 아니다.

## 질문과 구조 감사

앞선 [4번 규모·경로 연구](followup-4-results.md)는 부분망과 원래
전체 뇌망을 비교했다. 이 [별도 사전 설계](followup-4b-protocol.md)는
그 결과를 본 뒤, **새 재배선망의 신경 결과를 보기 전** `809dc09`에서
고정했다. 실제 배선의 정확도는 이미 알려져 있었으므로 pair 전체를
결과 눈가림으로 등록한 연구라고 표현하지 않는다.

원본 threshold-5 `brain5`는 **138,639뉴런·2,700,513연결**이다.
고정 graph seed 825001로 역할별 연결 block 안에서 double-edge swap
**2,700,513회**를 모두 수락했다(시도 2,740,750회). 재배선 후 원래
edge와의 겹침은 **13.986%**였다. [구조 감사](../results/followup4b_graph/audit.json)는
뉴런별 in/out degree, 역할 block별 연결 수, 전체 raw 가중치 집합,
중복/자기 연결 부재를 확인했다. 별도 [독립 대조](../results/followup4b-structural-independent-check.json)는
138,639개 출발 뉴런 각각의 출력 가중치 multiset이 전부 동일하고
edge 겹침도 swap 로그와 같음을 재계산했다. 원래·재배선 raw 그래프는
각각 gain .9 incoming-L1로 정규화했다. 따라서 결과는 배선 변경과
정규화의 결합 효과다.

## 짝 지은 지연 해독 결과

앞선 4번의 실제 `brain5` intact 학습/시험 기호·입력 지도를 정확히
재사용했다. 재배선망에서 3 seed×2 입력 지도 = **6개 새 상태·ridge
fit**을 했다. 학습 1,000/시험 500기호, warmup100, lag 1,2,3,4,5,8,
48 MBON과 alpha-1 ridge를 유지했다. 표의 단위는 시험 정확도와 %p다.

| seed | 실제 배선 | 재배선 | 실제−재배선 |
| ---: | ---: | ---: | ---: |
| 820001 | 72.383% | 72.950% | −0.567%p |
| 820002 | 71.633% | 71.100% | +0.533%p |
| 820003 | 69.467% | 68.900% | +0.567%p |
| 평균 | 71.161% | 70.983% | +0.178%p |

회로 701/702는 같은 연결망의 입력 맵 두 개로 먼저 seed별 평균을
냈다. [개별 case](../results/followup4b_main/raw-cases.csv),
[seed별 차이](../results/followup4b_main/seed-blocks.csv)와
[lag별 값](../results/followup4b_main/raw-lags.csv)을 모두 보존했다.
입력 맵별 차이는 부호가 섞인다. 이 한 graph-null의 작은 평균을
실제 연결망이 좋다거나 똑같다고 해석하지 않는다.

## 실행·검증·실패 기록

[그래프 생성 기록](../results/followup4b_graph/verification.json)은
87.70초/최대 sampled RSS 1,191,862,272 byte를 기록한다.
본실험은 6개 case를 실행했다. [독립 감사](../results/followup4b_main_validation/checks.json)는
구조 보존, 원본 parent manifest, 여섯 시험 신경 상태, 출력층 여섯
개의 독립 최소제곱 재학습을 확인했다. 모든 검증을 통과한 본실험
manifest SHA256은
`f065d3caaea97d532fb6a9ea7c5a2ab95e7d17d28f783976d1d9dc74d0824647`이다.
본실험은 150.15초, 최대 sampled RSS 394,993,664 byte였다.

처음 그래프·smoke·main [시도](../results/followup4b_main_attempt1/manifest.json)는
같은 seed에서 완주했지만 독립 검증기가 예측이 같은 두 평균값의
`1.1e-16` 부동소수점 순서 차이를 정확한 동등성 실패로 처리했다.
[실패 기록](../results/followup4b_main_attempt1-verification-failure.json)을
보존하고 `6bd8ea9`에서 검증기의 비교 허용 오차만 고쳤다. 배선 생성,
모델, seed, 기준은 바꾸지 않았다. 새 그래프를 **같은 seed로 재생성한
파일 SHA256이 이전과 같음**을 확인한 뒤 새 smoke·본실험을 실행했다.
처음 시도의 결과를 새 검증 결과로 대체 계산하지 않았다.

## 한계와 재현

이것은 whole-brain **node coverage의 threshold-5 rate 모델**이다.
한 재배선 realization과 세 계산 seed만 평가했다. threshold-1망,
다른 null ensemble, 생물학적 가소성, 자율 수열 회상은 별개의 시험이다.
외부 ridge는 학습됐고 반복 연결은 고정했다. 과거 기호를 해독할 수
있다는 사실을 기억 저장 위치나 살아 있는 초파리의 학습으로 연결하지
않는다. 본 4B 부정 결과와 4번의 DAN→MBON 경로 특이성 통과는 서로
다른 대조의 결과로 함께 유지한다.

Python 3.12와 [rate lock](../requirements-act1-lock.txt), 앞선 4번의
완료된 `results/followup4_main`, 고정 원본 cache가 필요하다. 새 출력
경로에서 그래프를 먼저 만들고 smoke를 거쳐 본실험을 실행한다.

```powershell
$env:PYTHONPATH = 'src;scripts'
$env:OPENBLAS_NUM_THREADS = '1'
.\.venv\Scripts\python.exe -m flying.training.phase6 prepare --raw outputs/followup_raw --cache outputs/followup_whole_cache
.\.venv\Scripts\python.exe scripts/followup4b.py prepare
.\.venv\Scripts\python.exe scripts/followup4b.py run --smoke --out outputs/followup4b_smoke_new
.\.venv\Scripts\python.exe scripts/verify_followup4b.py outputs/followup4b_smoke_new --out outputs/followup4b_smoke_new_validation
.\.venv\Scripts\python.exe scripts/followup4b.py run --out outputs/followup4b_main_new
.\.venv\Scripts\python.exe scripts/verify_followup4b.py outputs/followup4b_main_new --out outputs/followup4b_main_new_validation
```
