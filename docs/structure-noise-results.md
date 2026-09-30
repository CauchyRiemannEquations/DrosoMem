# 추가 연구 — 구조 대조

**주요 사전 기준: 실패.** 완료와 가설 성공을 구분한다.

## 1. Repository audit

[사전 완료 계획](additional-research-completion-plan.md)의 고정 순서로 실행했다.
ACT V와 이전 피드백 연구의 보고서·manifest·원자료를 변경하지 않았다.
저장 artifact와 코드 identity를 먼저 확인했고, 이번 결과를 과거 실험에 소급하지 않는다.

## 2. Reproduced baseline

32개 intact 모델의 clean 출력·확률·관측값이 수열 연구와 정확히 일치했다. 아래는 재배선 대조를 포함한 96개 모델의 실제 clean 경로 평균이다.
새 refit 모델에는 과거 expected 성능이 없으며 독립 head refit과 신경 재계산으로 검증했다.
기존 Python 격리 환경을 재사용했다.

| cohort | 수열 | graph | topology | clean prefix |
| --- | --- | --- | --- | --- |
| confirmation | pi | legacy5 | degree | 30.667 |
| confirmation | pi | legacy5 | intact | 34.333 |
| confirmation | pi | legacy5 | role | 41.167 |
| confirmation | random | legacy5 | degree | 24.833 |
| confirmation | random | legacy5 | intact | 31.167 |
| confirmation | random | legacy5 | role | 40.000 |
| discovery | pi | legacy5 | degree | 23.800 |
| discovery | pi | legacy5 | intact | 34.500 |
| discovery | pi | legacy5 | role | 33.200 |
| discovery | random | legacy5 | degree | 31.100 |
| discovery | random | legacy5 | intact | 26.000 |
| discovery | random | legacy5 | role | 34.600 |

## 3. New implementation

[실행기](../scripts/research_suite.py), [독립 검증기](../scripts/verify_research_suite.py).
기존 target-free 자율 함수와 graph sampler를 재사용하고 수열별 checkpoint·graph별 재학습·
상관 잡음·정의 불가 유지율 전파를 추가했다. 관련 28개 테스트 통과. 지표는 exact_prefix_symbols로
저장하며 π일 때만 PMS와 같다. 내부 연결은 경험으로 학습하지 않는다.

## 4. Experiments executed

| 조건 | 보존한 특성 | 보존하지 않은 특성 |
| --- | --- | --- |
| intact | 기존 graph 전체 | 해당 없음 |
| degree | 뉴런별 signed in/out degree, incoming weight multiset, 전체 노드·연결 수 | 연결 상대, motif, 역할별 연결 수, outgoing strength |
| role | degree 조건 및 뉴런별 역할별 signed in/out counts | 구체적인 연결 상대, motif, outgoing strength |

32개 graph draw를 π/random에서 공유했다. 10E swaps(32,410/33,090개)를 모두 완료했고
독립 보존량 검사를 통과했다. 기존 연결 위치와의 겹침 비율은 degree 41.795–43.598%,
role 48.685–51.003%였다. 보존 제약 때문에 상당한 연결이 남으며, 독립 균일 무작위 graph가 아니다.

두 686뉴런 부분망에서 pi/random, intact/degree/role을 비교했다. 32개 intact head를 재사용하고
64개 control head를 같은 예산으로 새로 학습했다. Graph-specific refit 후 frozen 평가이며,
기존 head를 재배선 망에 옮기는 frozen transfer와 다르다. 모든 topology의 q는 paired intact에서
계산해 기대 raw 잡음 에너지를 맞췄다. 주 조건은 allocated에서 intact−role 이득이 두 수열·두 cohort에서
모두 기준을 통과하는 것이다. Degree 대조와 배분 이득의 차이는 보조 결과다.

공통 조건: 길이200 / prompt314 / 평가197, 48 MBON, 482 head 파라미터,
2000 Adam updates / LR .03 / L2 1e-5 / fixed32 ×4, gain.9 / leak.6 / mbon_after_kc.
입력 mapping·관측 위치를 paired 조건에서 맞췄다. Seed9142–9146/data9242–9246,
새 seed440142–440144/data440242–440244, circuits701/702, 잡음448001–448003,
강도.0003/.03/.3. 두 cohort는 합치지 않는다.

본 실험 96모델, 1728개 certificate, **672개 실제 평가 궤적**,
64개 새 head fit, 64개 새 teacher training 궤적.
Certificate는 첫 오류 이후의 실제 출력이 아니다. Smoke는 본 통계에서 제외했다.
Intact의 576개 certificate 노출은 수열 연구의 같은 조건을 반복한 것이다.
원자료의 reused_exposure는 rho0에만 쓰는 flag라 여기서는 false지만, 새 독립 노출로 세지 않는다.
구조 대조의 서로 다른 새 노출 설정은 control의 1,152개이며 종료 감사에서 intact 배열의
부모와의 정확한 일치를 별도로 검사한다.

## 5. Results

주 조건의 seed별 결과. 강도·두 circuit·세 잡음을 먼저 평균했다. 개별 시행은
[원자료](../results/research_suite_structure/raw-certificates.csv)에 모두 저장했다.

| cohort | 수열 | seed | intact−role prefix | intact−role 유지율 %p |
| --- | --- | --- | --- | --- |
| confirmation | pi | 440142 | 9.944 | 15.754 |
| confirmation | pi | 440143 | 0.444 | 7.886 |
| confirmation | pi | 440144 | -6.389 | 13.939 |
| confirmation | random | 440142 | 1.444 | 4.957 |
| confirmation | random | 440143 | -11.167 | -3.850 |
| confirmation | random | 440144 | -5.556 | 3.284 |
| discovery | pi | 9142 | 4.444 | -1.616 |
| discovery | pi | 9143 | -1.056 | 7.955 |
| discovery | pi | 9144 | -2.333 | 0.250 |
| discovery | pi | 9145 | -0.833 | 0.289 |
| discovery | pi | 9146 | 3.111 | -13.947 |
| discovery | random | 9142 | 6.333 | 13.721 |
| discovery | random | 9143 | -10.722 | 7.555 |
| discovery | random | 9144 | -4.056 | 2.233 |
| discovery | random | 9145 | -4.111 | 6.130 |
| discovery | random | 9146 | -2.722 | 15.198 |

조건별 평균 및 배분 차이. 마지막 gate는 **배분 이득** 기준이며 구조 연구의 intact−role 판정과 다르다.

| cohort/family/graph/topology/rho | flat | allocated | prefix 차이 | 유지율 차이 %p | 배분 gate |
| --- | --- | --- | --- | --- | --- |
| confirmation/pi/legacy5/degree/0.0 | 18.352 | 23.019 | 4.667 | 14.011 | True |
| confirmation/pi/legacy5/intact/0.0 | 12.574 | 21.241 | 8.667 | 24.290 | True |
| confirmation/pi/legacy5/role/0.0 | 16.870 | 19.907 | 3.037 | 7.881 | False |
| confirmation/random/legacy5/degree/0.0 | 14.000 | 18.333 | 4.333 | 15.796 | True |
| confirmation/random/legacy5/intact/0.0 | 11.759 | 17.537 | 5.778 | 18.882 | True |
| confirmation/random/legacy5/role/0.0 | 16.481 | 22.630 | 6.148 | 13.135 | True |
| discovery/pi/legacy5/degree/0.0 | 13.389 | 18.467 | 5.078 | 정의 불가 | None |
| discovery/pi/legacy5/intact/0.0 | 12.967 | 21.622 | 8.656 | 25.114 | True |
| discovery/pi/legacy5/role/0.0 | 15.367 | 20.956 | 5.589 | 15.784 | True |
| discovery/random/legacy5/degree/0.0 | 17.900 | 23.711 | 5.811 | 18.612 | True |
| discovery/random/legacy5/intact/0.0 | 10.711 | 16.022 | 5.311 | 19.081 | True |
| discovery/random/legacy5/role/0.0 | 14.567 | 19.078 | 4.511 | 11.880 | True |

주 분석 통계. 유지율 평균·중앙값·구간은 %p, 분산은 비율 제곱 단위다.
작은 n5/n3의 seed-block bootstrap이며 잡음 반복을 독립 모델로 세지 않는다. 큰 dz는 작은 분산에서
나올 수 있어 일반화의 확실성으로 해석하지 않는다. [전체 통계](../results/research_suite_structure/summary.json).

| 조건 | metric | mean | median | variance | bootstrap95 | dz | win/tie/loss |
| --- | --- | --- | --- | --- | --- | --- | --- |
| discovery/pi/role | exact_prefix_symbols | 0.667 | -0.833 | 8.61574 | -1.522 / 3.078 | 0.227 | 2/0/3 |
| discovery/pi/role | retention | -1.414 | 0.250 | 0.00626377 | -8.268 / 4.500 | -0.179 | 3/0/2 |
| discovery/random/role | exact_prefix_symbols | -3.056 | -4.056 | 37.2886 | -7.789 / 2.178 | -0.500 | 1/0/4 |
| discovery/random/role | retention | 8.968 | 7.555 | 0.00292056 | 4.856 / 13.089 | 1.659 | 5/0/0 |
| confirmation/pi/role | exact_prefix_symbols | 1.333 | 0.444 | 67.287 | -6.389 / 9.944 | 0.163 | 2/0/1 |
| confirmation/pi/role | retention | 12.526 | 13.939 | 0.00169745 | 7.886 / 15.754 | 3.040 | 3/0/0 |
| confirmation/random/role | exact_prefix_symbols | -5.093 | -5.556 | 39.9208 | -11.167 / 1.444 | -0.806 | 1/0/2 |
| confirmation/random/role | retention | 1.464 | 3.284 | 0.00218734 | -3.850 / 4.957 | 0.313 | 2/0/1 |

![Prefix curves](../results/research_suite_structure_analysis/prefix-curves.png)

강도는 범주 축이며 간격이 선형인 것처럼 해석하지 않는다. 그림의 평균은 원자료를 대신하지 않는다.

## 6. Interpretation

Intact가 role 대조보다 우수하다는 주 기준은 네 조건 모두 실패했다. π의 prefix 차이는
발견 +0.667/확인 +1.333기호, 유지율 차이는 −1.414/+12.526%p다.
Random에서는 prefix −3.056/−5.093기호, 유지율 +8.968/+1.464%p다.
특히 random은 원래 망이 자신의 clean에 비해 조금 더 유지하더라도 절대 prefix가
대조보다 낮을 수 있음을 보여 준다. 유지율의 분모는 각 graph의 clean이며 공통 분모가 아니다.
따라서 두 지표의 joint 기준을 그대로 사용했다.

잡음 배분 이득은 재배선 망에서도 여러 조건에서 유지됐다. 원래 연결 상대만이 이득에
필수라는 설명은 이 패널에서 지지되지 않는다. 다만 degree/role가 보존한 구조가 여전히
중요할 수 있으므로 모든 topology가 동등하거나 해부학이 무의미하다는 결론은 아니다.
Role 대조의 관측 effective rank는 원래 망보다 높았지만, 표현 다양성과 성능의 인과관계를
이 비교만으로 분리하지 못한다. Graph별 head refit과 상태 분포 변화가 함께 포함된다.

저장된 own_q/used_q의 **사후 표현 진단**: degree는 최소1.775/중앙값2.596/최대3.703,
role은 .727/.891/1.085였다(각 32개 graph/task cell, 두 cohort를 기술적으로 요약).
Own_q는 해당 graph의 관측 SD 중앙값이고 used_q는 paired intact 기준이다.
Degree의 신호 규모가 커져 같은 raw 잡음이 더 작은 상대 교란이 될 수 있다.
이는 기존 artifact의 기술적 요약으로, primary metric·gate를 변경하거나 새 표본을 만든 분석이 아니다.
다음 연구에서는 공통 raw 에너지와 graph별 상대 강도라는 두 보정의 차이를 분리할 정보가치가 있다.

사전 joint gate는 평균 prefix 이득≥2자리와 유지율 이득≥10%p,
각 지표의 양의 차이가 발견 집단 4/5 이상, 확인 집단 3/3이어야 한다.
Clean prefix=0이면 유지율은 정의 불가이며 누락·epsilon 대체 없이 판별 불가로 전파했다.

학습 손실은 고정 prefix 가중치와 정규화 항을 포함한다. 아래 정확도는 정답을 계속 입력한
학습 수열의 next-symbol 정확도다. 자율 회상의 정확도가 아니다. Effective rank는 관측한
48개 좌표의 표현 다양성 진단이며, 전체망 정보량이나 기억 용량으로 부르지 않는다.
상관 연구에서는 같은 부모의 학습 상태를 재사용한 진단이다.

| cohort | 수열 | graph | topology | train loss | teacher 정확도 % | effective rank | 잡음 기준 q |
| --- | --- | --- | --- | --- | --- | --- | --- |
| confirmation | pi | legacy5 | degree | 0.570 | 76.047 | 12.001 | 0.00397154 |
| confirmation | pi | legacy5 | intact | 0.382 | 84.506 | 15.252 | 0.00397154 |
| confirmation | pi | legacy5 | role | 0.321 | 87.270 | 18.126 | 0.00397154 |
| confirmation | random | legacy5 | degree | 0.582 | 75.377 | 12.157 | 0.00409658 |
| confirmation | random | legacy5 | intact | 0.441 | 83.082 | 15.244 | 0.00409658 |
| confirmation | random | legacy5 | role | 0.311 | 87.856 | 18.250 | 0.00409658 |
| discovery | pi | legacy5 | degree | 0.578 | 75.377 | 11.994 | 0.00379373 |
| discovery | pi | legacy5 | intact | 0.404 | 83.015 | 15.695 | 0.00379373 |
| discovery | pi | legacy5 | role | 0.398 | 83.367 | 17.550 | 0.00379373 |
| discovery | random | legacy5 | degree | 0.582 | 75.678 | 11.879 | 0.00384823 |
| discovery | random | legacy5 | intact | 0.412 | 83.166 | 15.604 | 0.00384823 |
| discovery | random | legacy5 | role | 0.329 | 86.131 | 17.300 | 0.00384823 |

Clean prefix가 0인 모델은 1/96개다.

실제 random 경로의 전체 정확도와 오류 이후 정확도다. 이 표의 clean도 noisy 경로가
실제 실행된 **동일 모델 subset**만 포함한다. 상관 연구는 지정 subset만,
수열 연구의 전체망은 각 cohort 첫 block/c701만 해당한다. 전체 모델의 clean 평균은
2절의 별도 표다. 이 표를 전체 seed의 post-error 결과로 확대하지 않는다.

| cohort | graph | topology | rho | arm | prefix | 전체 정확도 % | 첫 오류 다음부터 % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| confirmation | legacy5 | degree | 0.0 | clean | 24.833 | 22.589 | 11.376 |
| confirmation | legacy5 | degree | 0.0 | flat | 13.000 | 16.469 | 10.591 |
| confirmation | legacy5 | degree | 0.0 | training_sd | 19.111 | 19.656 | 10.974 |
| confirmation | legacy5 | intact | 0.0 | clean | 31.167 | 24.450 | 10.292 |
| confirmation | legacy5 | intact | 0.0 | flat | 11.611 | 15.510 | 10.278 |
| confirmation | legacy5 | intact | 0.0 | training_sd | 17.722 | 17.738 | 9.633 |
| confirmation | legacy5 | role | 0.0 | clean | 40.000 | 28.426 | 10.265 |
| confirmation | legacy5 | role | 0.0 | flat | 15.556 | 17.992 | 10.958 |
| confirmation | legacy5 | role | 0.0 | training_sd | 23.778 | 20.953 | 10.152 |
| discovery | legacy5 | degree | 0.0 | clean | 31.100 | 24.721 | 10.614 |
| discovery | legacy5 | degree | 0.0 | flat | 17.933 | 18.596 | 10.491 |
| discovery | legacy5 | degree | 0.0 | training_sd | 24.233 | 21.218 | 10.230 |
| discovery | legacy5 | intact | 0.0 | clean | 26.000 | 22.030 | 10.202 |
| discovery | legacy5 | intact | 0.0 | flat | 11.033 | 15.990 | 11.030 |
| discovery | legacy5 | intact | 0.0 | training_sd | 15.633 | 17.665 | 10.582 |
| discovery | legacy5 | role | 0.0 | clean | 34.600 | 25.838 | 10.116 |
| discovery | legacy5 | role | 0.0 | flat | 15.400 | 17.208 | 10.238 |
| discovery | legacy5 | role | 0.0 | training_sd | 19.733 | 19.577 | 10.672 |

## 7. Negative findings

원래 배선의 역할 보존 대조 대비 우위는 확인되지 않았다. π 확인 집단의 prefix도
3/3 양수가 아니었고, random의 평균 prefix는 두 집단 모두 음수였다.
Degree 대조 대비 우위도 어느 집단·수열에서든 확인된 joint 성공이 없다.
Discovery π/degree, seed9143/c702는 clean prefix=0이었다. 유지율을 생략해 평균내지 않아
해당 degree 구조 비교와 배분 gate는 판별 불가다. Role/π 확인 집단의 배분 개선 gate도 실패했다.

실패·판별 불가 조건과 강한 잡음의 회상 손실을 보존했다. 낮은 clean prefix나 오류 이후 붕괴,
배분 변경에 따른 악화도 삭제하지 않았다. 결과를 본 뒤 hyperparameter 탐색은 하지 않았다.

## 8. What we can claim

이 두 부분망과 지정 수열·예산에서는 실제 연결 상대를 보존하는 것이 degree/role 대조보다
일관된 자율 회상 이득을 준다는 증거를 얻지 못했다. 잡음 배분 효과의 일부는 이 재배선 대조에서도
나타나므로 그 효과를 실제 connectome만의 계산적 특성으로 부를 수 없다.

실제 초파리 connectome 구조를 사용한 계산 모델에서 위 고정 조건을 평가했다.
주 판정은 실패이며 그 범위를 다른 생물학적 주장으로 확대하지 않는다.

## 9. What we cannot claim

실제 동물의 기억·지능, 내부 연결의 경험 의존 학습, formal/Shannon capacity, unseen 예측을
주장할 수 없다. 관측 잡음은 내부 상태 잡음과 다르다. 구조 대조는 부분망이고 finite swap sampler다.
상관 잡음은 합성 공통 성분이며 실측 생리적 동역학이 아니다. 모델·수열 seed가 paired이므로
각각의 분산 기여를 분리하지 못한다.

## 10. Reproducibility

Source commit `f1c667cea5370dba501da2e35b8f3d42f5278466`.
[Config](../results/research_suite_structure/config.json), [manifest](../results/research_suite_structure/manifest.json),
[검증](../results/research_suite_structure_validation/checks.json),
[실제 경로 원자료](../results/research_suite_structure/raw-rollouts.csv), [테스트](../results/research_suite_tests/manifest.json).
각 case는 checkpoint 또는 부모 checkpoint 경로, 수열 identity, graph hash와 환경을 기록한다.
독립 검증: 1728개 certificate, 672개 저장 경로 지표,
**672개 전체 자율 재실행**, 96개 teacher 경로,
64개 정확 head refit. 수치 오차 최대 4.19664e-14, 예측 기호 정확 일치.
기존 파일 27,011개 보존. 본 실험 347.52초, 검증 327.49초.
RSS는 resources.json의 주기적 process-tree 측정이며 절대 순간 peak는 아니다.
본 실행의 worker당 최대 sampled RSS는 157.82MiB였다. 최대 두 worker를 동시에 사용했으므로
이 값을 전체 작업의 합산 peak로 해석하지 않는다. 환경: Python 3.12.10,
Windows-11-10.0.26200-SP0, NumPy 2.3.5, SciPy 1.17.0,
pandas 2.2.3, 수치 연산 thread 1개/worker.

실행 명령(기존 폴더를 덮어쓰지 않으며 재현할 때는 새 output 경로 사용):

```powershell
$env:PYTHONPATH='src'
../venv-act1/Scripts/python.exe scripts/research_suite.py run --stage structure --out results/research_suite_structure
../venv-act1/Scripts/python.exe scripts/verify_research_suite.py results/research_suite_structure --out results/research_suite_structure_validation
../venv-act1/Scripts/python.exe scripts/research_suite.py plot --source results/research_suite_structure --out results/research_suite_structure_analysis
```

환경 설치는 [README](../README.md)의 dependency lock을 따른다. 이후 연구를 새 부모로 재현하려면
--parent에 새 sequence 경로를 지정하고 correlation은 --structural-validation에 새 structure 검증
경로를 지정한다. Parent 검증 폴더는 부모 경로에 _validation을 붙인 이름이다.

## 11. Next highest-information experiment

**공통 성분을 갖는 상관 관측 잡음**이 다음 지정 실험이다. 같은 96개 intact head를
재사용해 독립 잡음 가정을 바꾼다. 그 결과는 [상관 잡음 보고서](correlated-noise-results.md),
전체 프로그램 이후의 한 가지 제안은 [종합 보고서](additional-research-results.md)를 따른다.
