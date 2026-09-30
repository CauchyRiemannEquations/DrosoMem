# 추가 연구 — 상관 잡음

**주요 사전 기준: 통과.** 완료와 가설 성공을 구분한다.

## 1. Repository audit

[사전 완료 계획](additional-research-completion-plan.md)의 고정 순서로 실행했다.
ACT V와 이전 피드백 연구의 보고서·manifest·원자료를 변경하지 않았다.
저장 artifact와 코드 identity를 먼저 확인했고, 이번 결과를 과거 실험에 소급하지 않는다.

## 2. Reproduced baseline

96개 head와 clean 기준을 수열 연구에서 재사용했다. 아래는 지정 random subset의 새 clean 경로 평균이며, 출력·확률·관측값이 부모와 정확히 일치했다. 나머지 모델의 clean 기준은 부모 artifact를 사용했다.
새 refit 모델에는 과거 expected 성능이 없으며 독립 head refit과 신경 재계산으로 검증했다.
기존 Python 격리 환경을 재사용했다.

| cohort | 수열 | graph | topology | clean prefix |
| --- | --- | --- | --- | --- |
| confirmation | random | brain1 | intact | 1.000 |
| confirmation | random | legacy5 | intact | 34.000 |
| discovery | random | brain1 | intact | 32.000 |
| discovery | random | legacy5 | intact | 33.000 |

## 3. New implementation

[실행기](../scripts/research_suite.py), [독립 검증기](../scripts/verify_research_suite.py).
기존 target-free 자율 함수와 graph sampler를 재사용하고 수열별 checkpoint·graph별 재학습·
상관 잡음·정의 불가 유지율 전파를 추가했다. 관련 28개 테스트 통과. 지표는 exact_prefix_symbols로
저장하며 π일 때만 PMS와 같다. 내부 연결은 경험으로 학습하지 않는다.

## 4. Experiments executed

수열 연구의 96개 intact head를 재사용했다. 새 학습은 없다.
독립 잡음 z에 공통 성분 g를 섞어 sqrt(1−rho)z+sqrt(rho)g, rho=0/.25/.75를 비교했다.
각 관측 좌표의 주변 분산과 총 기대 잡음 에너지는 그대로다. Rho0의 1728개 노출은 수열 연구와 정확히
동일하며 새 독립 증거로 세지 않는다. 새 상관 노출은 3456개다. 주 조건은 random/brain1의 배분 이득이
두 비영점 rho 및 두 cohort 모두에서 통과하는 것이다. 이는 시간별 합성 공통 관측 잡음이며,
뉴런 내부 상태의 잡음이나 생리학적으로 측정한 상관 구조가 아니다.

공통 조건: 길이200 / prompt314 / 평가197, 48 MBON, 482 head 파라미터,
2000 Adam updates / LR .03 / L2 1e-5 / fixed32 ×4, gain.9 / leak.6 / mbon_after_kc.
입력 mapping·관측 위치를 paired 조건에서 맞췄다. Seed9142–9146/data9242–9246,
새 seed440142–440144/data440242–440244, circuits701/702, 잡음448001–448003,
강도.0003/.03/.3. 두 cohort는 합치지 않는다.

본 실험 96모델, 5184개 certificate, **52개 실제 평가 궤적**,
0개 새 head fit, 0개 새 teacher training 궤적.
Certificate는 첫 오류 이후의 실제 출력이 아니다. Smoke는 본 통계에서 제외했다.

## 5. Results

주 조건의 seed별 결과. 강도·두 circuit·세 잡음을 먼저 평균했다. 개별 시행은
[원자료](../results/research_suite_correlation/raw-certificates.csv)에 모두 저장했다.

| cohort | seed | rho | flat prefix | allocated prefix | 차이 | 유지율 차이 %p |
| --- | --- | --- | --- | --- | --- | --- |
| confirmation | 440142 | 0.25 | 6.056 | 11.333 | 5.278 | 32.118 |
| confirmation | 440143 | 0.25 | 13.611 | 17.000 | 3.389 | 8.793 |
| confirmation | 440144 | 0.25 | 12.222 | 21.500 | 9.278 | 27.103 |
| confirmation | 440142 | 0.75 | 5.722 | 11.611 | 5.889 | 34.375 |
| confirmation | 440143 | 0.75 | 13.611 | 17.111 | 3.500 | 9.005 |
| confirmation | 440144 | 0.75 | 12.278 | 23.111 | 10.833 | 31.829 |
| discovery | 9142 | 0.25 | 11.778 | 17.667 | 5.889 | 18.254 |
| discovery | 9143 | 0.25 | 9.222 | 16.500 | 7.278 | 25.983 |
| discovery | 9144 | 0.25 | 6.111 | 10.389 | 4.278 | 29.125 |
| discovery | 9145 | 0.25 | 11.778 | 17.667 | 5.889 | 17.320 |
| discovery | 9146 | 0.25 | 11.278 | 15.056 | 3.778 | 11.448 |
| discovery | 9142 | 0.75 | 12.000 | 17.667 | 5.667 | 17.604 |
| discovery | 9143 | 0.75 | 9.222 | 15.833 | 6.611 | 28.355 |
| discovery | 9144 | 0.75 | 6.111 | 10.611 | 4.500 | 29.798 |
| discovery | 9145 | 0.75 | 11.667 | 17.222 | 5.556 | 16.340 |
| discovery | 9146 | 0.75 | 11.389 | 15.278 | 3.889 | 11.785 |

조건별 평균 및 배분 차이. 마지막 gate는 **배분 이득** 기준이며 구조 연구의 intact−role 판정과 다르다.

| cohort/family/graph/topology/rho | flat | allocated | prefix 차이 | 유지율 차이 %p | 배분 gate |
| --- | --- | --- | --- | --- | --- |
| confirmation/pi/brain1/intact/0.0 | 10.426 | 15.944 | 5.519 | 20.822 | True |
| confirmation/pi/brain1/intact/0.25 | 10.315 | 15.185 | 4.870 | 20.162 | True |
| confirmation/pi/brain1/intact/0.75 | 10.389 | 15.852 | 5.463 | 22.742 | True |
| confirmation/pi/legacy5/intact/0.0 | 12.574 | 21.241 | 8.667 | 24.290 | True |
| confirmation/pi/legacy5/intact/0.25 | 12.722 | 21.056 | 8.333 | 23.697 | True |
| confirmation/pi/legacy5/intact/0.75 | 12.537 | 19.741 | 7.204 | 21.061 | True |
| confirmation/random/brain1/intact/0.0 | 10.593 | 16.722 | 6.130 | 21.241 | True |
| confirmation/random/brain1/intact/0.25 | 10.630 | 16.611 | 5.981 | 22.671 | True |
| confirmation/random/brain1/intact/0.75 | 10.537 | 17.278 | 6.741 | 25.070 | True |
| confirmation/random/legacy5/intact/0.0 | 11.759 | 17.537 | 5.778 | 18.882 | True |
| confirmation/random/legacy5/intact/0.25 | 11.407 | 17.963 | 6.556 | 21.626 | True |
| confirmation/random/legacy5/intact/0.75 | 12.352 | 18.481 | 6.130 | 19.811 | True |
| confirmation/shuffled_pi/brain1/intact/0.0 | 13.130 | 21.130 | 8.000 | 22.422 | True |
| confirmation/shuffled_pi/brain1/intact/0.25 | 13.278 | 22.019 | 8.741 | 24.553 | True |
| confirmation/shuffled_pi/brain1/intact/0.75 | 13.074 | 21.000 | 7.926 | 21.381 | True |
| confirmation/shuffled_pi/legacy5/intact/0.0 | 14.981 | 22.130 | 7.148 | 18.515 | True |
| confirmation/shuffled_pi/legacy5/intact/0.25 | 15.241 | 22.426 | 7.185 | 18.647 | True |
| confirmation/shuffled_pi/legacy5/intact/0.75 | 15.870 | 23.593 | 7.722 | 19.344 | True |
| discovery/pi/brain1/intact/0.0 | 11.444 | 16.667 | 5.222 | 17.643 | True |
| discovery/pi/brain1/intact/0.25 | 11.222 | 18.100 | 6.878 | 21.893 | True |
| discovery/pi/brain1/intact/0.75 | 11.033 | 18.633 | 7.600 | 24.419 | True |
| discovery/pi/legacy5/intact/0.0 | 12.967 | 21.622 | 8.656 | 25.114 | True |
| discovery/pi/legacy5/intact/0.25 | 12.878 | 20.600 | 7.722 | 22.278 | True |
| discovery/pi/legacy5/intact/0.75 | 13.122 | 20.867 | 7.744 | 22.494 | True |
| discovery/random/brain1/intact/0.0 | 10.000 | 15.533 | 5.533 | 20.226 | True |
| discovery/random/brain1/intact/0.25 | 10.033 | 15.456 | 5.422 | 20.426 | True |
| discovery/random/brain1/intact/0.75 | 10.078 | 15.322 | 5.244 | 20.776 | True |
| discovery/random/legacy5/intact/0.0 | 10.711 | 16.022 | 5.311 | 19.081 | True |
| discovery/random/legacy5/intact/0.25 | 10.711 | 15.667 | 4.956 | 20.140 | True |
| discovery/random/legacy5/intact/0.75 | 11.056 | 16.011 | 4.956 | 19.556 | True |
| discovery/shuffled_pi/brain1/intact/0.0 | 10.467 | 14.856 | 4.389 | 16.571 | True |
| discovery/shuffled_pi/brain1/intact/0.25 | 10.600 | 15.133 | 4.533 | 16.309 | True |
| discovery/shuffled_pi/brain1/intact/0.75 | 10.589 | 16.644 | 6.056 | 20.593 | True |
| discovery/shuffled_pi/legacy5/intact/0.0 | 11.522 | 17.211 | 5.689 | 19.271 | True |
| discovery/shuffled_pi/legacy5/intact/0.25 | 11.978 | 17.244 | 5.267 | 18.378 | True |
| discovery/shuffled_pi/legacy5/intact/0.75 | 12.256 | 17.100 | 4.844 | 16.750 | True |

주 분석 통계. 유지율 평균·중앙값·구간은 %p, 분산은 비율 제곱 단위다.
작은 n5/n3의 seed-block bootstrap이며 잡음 반복을 독립 모델로 세지 않는다. 큰 dz는 작은 분산에서
나올 수 있어 일반화의 확실성으로 해석하지 않는다. [전체 통계](../results/research_suite_correlation/summary.json).

| 조건 | metric | mean | median | variance | bootstrap95 | dz | win/tie/loss |
| --- | --- | --- | --- | --- | --- | --- | --- |
| discovery/random/brain1/intact/0.25 | exact_prefix_symbols | 5.422 | 5.889 | 1.97315 | 4.300 / 6.444 | 3.860 | 5/0/0 |
| discovery/random/brain1/intact/0.25 | retention | 20.426 | 18.254 | 0.00503789 | 15.158 / 26.135 | 2.878 | 5/0/0 |
| discovery/random/brain1/intact/0.75 | exact_prefix_symbols | 5.244 | 5.556 | 1.13364 | 4.367 / 6.022 | 4.926 | 5/0/0 |
| discovery/random/brain1/intact/0.75 | retention | 20.776 | 17.604 | 0.00623549 | 14.771 / 26.818 | 2.631 | 5/0/0 |
| confirmation/random/brain1/intact/0.25 | exact_prefix_symbols | 5.981 | 5.278 | 9.04115 | 3.389 / 9.278 | 1.989 | 3/0/0 |
| confirmation/random/brain1/intact/0.25 | retention | 22.671 | 27.103 | 0.0150743 | 8.793 / 32.118 | 1.847 | 3/0/0 |
| confirmation/random/brain1/intact/0.75 | exact_prefix_symbols | 6.741 | 5.889 | 13.9887 | 3.500 / 10.833 | 1.802 | 3/0/0 |
| confirmation/random/brain1/intact/0.75 | retention | 25.070 | 31.829 | 0.0195168 | 9.005 / 34.375 | 1.795 | 3/0/0 |

![Prefix curves](../results/research_suite_correlation_rho_analysis/prefix-curves.png)

이 그림의 강도는 로그 축이다. 원래 범주 축 그림도
[초기 plot artifact](../results/research_suite_correlation_analysis/prefix-curves.png)에 보존했다.
그림의 평균은 seed별 원자료를 대신하지 않는다.

## 6. Interpretation

Random/brain1의 배분 개선은 rho .25와 .75 모두 두 집단에서 사전 joint 기준을 통과했다.
.25의 평균 prefix 차이는 +5.422/+5.981기호, 유지율은 +20.426/+22.671%p였다.
.75는 +5.244/+6.741기호, +20.776/+25.070%p였다. 각 조건의 발견5/5·확인3/3
seed-block에서 두 지표가 모두 양수였다. 두 그래프·세 수열의 비영점 상관 배분 gate 24개도
모두 통과했다. Rho0의 12개 gate는 이미 수열 연구에서 확인한 반복이며 새 확인으로 세지 않는다.

아래는 별개의 사전 지정 기술 분석이다. 같은 모델·배분에서 rho−0을 비교하며,
음수면 상관 추가에 따른 악화다. 상관 자체의 효과가 항상 음수이지는 않았다.
Allocated random/brain1의 prefix 변화는 .25에서 −.078/−.111기호, .75에서 −.211/+.556기호였다.
작은 표본과 두 합성 상관값의 결과이며, 무상관과 동등하다거나 모든 상관에 강건하다는 검정은 아니다.

| cohort | rho | 배분 | rho−0 prefix | bootstrap95 | rho−0 유지율 %p |
| --- | --- | --- | --- | --- | --- |
| confirmation | 0.25 | flat | 0.037 | -0.056 / 0.222 | -1.670 |
| confirmation | 0.75 | flat | -0.056 | -0.111 / 0.000 | -1.958 |
| confirmation | 0.25 | training_sd | -0.111 | -0.667 / 0.444 | -0.241 |
| confirmation | 0.75 | training_sd | 0.556 | -0.556 / 1.500 | 1.870 |
| discovery | 0.25 | flat | 0.033 | -0.067 / 0.167 | 0.187 |
| discovery | 0.75 | flat | 0.078 | -0.044 / 0.211 | 0.146 |
| discovery | 0.25 | training_sd | -0.078 | -1.000 / 0.867 | 0.387 |
| discovery | 0.75 | training_sd | -0.211 | -1.122 / 0.867 | 0.696 |

[192개 paired seed-block 차이](../results/research_suite_correlation_rho_analysis/rho-minus-zero-seed-blocks.csv)와
[48개 전체 조건 통계](../results/research_suite_correlation_rho_analysis/statistics.json)를 별도 저장했다.
[분석 코드](../scripts/analyze_research_suite_correlation.py)는 원 certificate의 18개 circuit/잡음/강도
행으로 각 block 차이를 다시 계산하고 평균·분산·bootstrap 통계를 검사했다. 새 학습·궤적은 없다.

사전 joint gate는 평균 prefix 이득≥2자리와 유지율 이득≥10%p,
각 지표의 양의 차이가 발견 집단 4/5 이상, 확인 집단 3/3이어야 한다.
Clean prefix=0이면 유지율은 정의 불가이며 누락·epsilon 대체 없이 판별 불가로 전파했다.

학습 손실은 고정 prefix 가중치와 정규화 항을 포함한다. 아래 정확도는 정답을 계속 입력한
학습 수열의 next-symbol 정확도다. 자율 회상의 정확도가 아니다. Effective rank는 관측한
48개 좌표의 표현 다양성 진단이며, 전체망 정보량이나 기억 용량으로 부르지 않는다.
상관 연구에서는 같은 부모의 학습 상태를 재사용한 진단이다.

| cohort | 수열 | graph | topology | train loss | teacher 정확도 % | effective rank | 잡음 기준 q |
| --- | --- | --- | --- | --- | --- | --- | --- |
| confirmation | pi | brain1 | intact | 0.556 | 76.047 | 11.050 | 0.000763286 |
| confirmation | pi | legacy5 | intact | 0.382 | 84.506 | 15.252 | 0.00397154 |
| confirmation | random | brain1 | intact | 0.566 | 76.214 | 10.928 | 0.000786607 |
| confirmation | random | legacy5 | intact | 0.441 | 83.082 | 15.244 | 0.00409658 |
| confirmation | shuffled_pi | brain1 | intact | 0.537 | 76.633 | 10.918 | 0.000787791 |
| confirmation | shuffled_pi | legacy5 | intact | 0.363 | 86.097 | 15.222 | 0.00410256 |
| discovery | pi | brain1 | intact | 0.532 | 78.543 | 11.097 | 0.000785654 |
| discovery | pi | legacy5 | intact | 0.404 | 83.015 | 15.695 | 0.00379373 |
| discovery | random | brain1 | intact | 0.570 | 75.829 | 11.152 | 0.000791835 |
| discovery | random | legacy5 | intact | 0.412 | 83.166 | 15.604 | 0.00384823 |
| discovery | shuffled_pi | brain1 | intact | 0.518 | 78.040 | 11.144 | 0.000788558 |
| discovery | shuffled_pi | legacy5 | intact | 0.499 | 79.799 | 15.611 | 0.00386955 |

Clean prefix가 0인 모델은 0/96개다.

실제 random 경로의 전체 정확도와 오류 이후 정확도다. 이 표의 clean도 noisy 경로가
실제 실행된 **동일 모델 subset**만 포함한다. 상관 연구는 지정 subset만,
수열 연구의 전체망은 각 cohort 첫 block/c701만 해당한다. 전체 모델의 clean 평균은
2절의 별도 표다. 이 표를 전체 seed의 post-error 결과로 확대하지 않는다.

| cohort | graph | topology | rho | arm | prefix | 전체 정확도 % | 첫 오류 다음부터 % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| confirmation | brain1 | intact | 0.0 | clean | 1.000 | 9.645 | 9.231 |
| confirmation | brain1 | intact | 0.25 | flat | 0.667 | 8.799 | 8.532 |
| confirmation | brain1 | intact | 0.25 | training_sd | 1.000 | 10.660 | 10.256 |
| confirmation | brain1 | intact | 0.75 | flat | 0.667 | 10.829 | 10.580 |
| confirmation | brain1 | intact | 0.75 | training_sd | 1.000 | 11.844 | 11.453 |
| confirmation | legacy5 | intact | 0.0 | clean | 34.000 | 24.365 | 8.642 |
| confirmation | legacy5 | intact | 0.25 | flat | 12.000 | 16.751 | 11.292 |
| confirmation | legacy5 | intact | 0.25 | training_sd | 16.667 | 19.966 | 12.464 |
| confirmation | legacy5 | intact | 0.75 | flat | 12.000 | 15.905 | 10.437 |
| confirmation | legacy5 | intact | 0.75 | training_sd | 14.000 | 15.228 | 8.814 |
| discovery | brain1 | intact | 0.0 | clean | 32.000 | 27.411 | 13.415 |
| discovery | brain1 | intact | 0.25 | flat | 10.667 | 14.721 | 9.984 |
| discovery | brain1 | intact | 0.25 | training_sd | 21.333 | 21.827 | 12.282 |
| discovery | brain1 | intact | 0.75 | flat | 10.667 | 17.936 | 13.249 |
| discovery | brain1 | intact | 0.75 | training_sd | 21.333 | 20.643 | 11.258 |
| discovery | legacy5 | intact | 0.0 | clean | 33.000 | 22.843 | 7.362 |
| discovery | legacy5 | intact | 0.25 | flat | 17.000 | 17.428 | 9.628 |
| discovery | legacy5 | intact | 0.25 | training_sd | 23.000 | 21.997 | 11.521 |
| discovery | legacy5 | intact | 0.75 | flat | 17.000 | 17.936 | 10.064 |
| discovery | legacy5 | intact | 0.75 | training_sd | 22.333 | 21.320 | 11.059 |

## 7. Negative findings

가장 강한 .3 잡음의 allocated random/brain1 prefix는 rho .25에서 1.333/1.833기호,
rho .75에서 1.600/2.222기호였다. .75에서 세 강도 평균 clean 대비 유지율은
58.717/65.488%였다. 배분 이득이 있어도 긴 수열의 안정적 회상은 복원하지 못했다.
상관을 넣어도 원래 clean prefix1인 확인 subset 모델의 근본적 회상 한계가 사라지지 않았다.

실패·판별 불가 조건과 강한 잡음의 회상 손실을 보존했다. 낮은 clean prefix나 오류 이후 붕괴,
배분 변경에 따른 악화도 삭제하지 않았다. 결과를 본 뒤 hyperparameter 탐색은 하지 않았다.

## 8. What we can claim

이번 K10·길이200 과제와 두 rate graph에서는 좌표별 SD에 따른 배분의 이득이
독립 잡음에만 한정되지 않았고, 지정한 공통 성분 공간 상관에서도 유지됐다.
같은 모델을 재사용한 paired perturbation 증거이며 새로운 독립 모델 확인은 아니다.

실제 초파리 connectome 구조를 사용한 계산 모델에서 위 고정 조건을 평가했다.
주 판정은 통과이며 그 범위를 다른 생물학적 주장으로 확대하지 않는다.

## 9. What we cannot claim

실제 동물의 기억·지능, 내부 연결의 경험 의존 학습, formal/Shannon capacity, unseen 예측을
주장할 수 없다. 관측 잡음은 내부 상태 잡음과 다르다. 구조 대조는 부분망이고 finite swap sampler다.
상관 잡음은 합성 공통 성분이며 실측 생리적 동역학이 아니다. 모델·수열 seed가 paired이므로
각각의 분산 기여를 분리하지 못한다.

## 10. Reproducibility

Source commit `f1c667cea5370dba501da2e35b8f3d42f5278466`.
[Config](../results/research_suite_correlation/config.json), [manifest](../results/research_suite_correlation/manifest.json),
[검증](../results/research_suite_correlation_validation/checks.json),
[실제 경로 원자료](../results/research_suite_correlation/raw-rollouts.csv), [테스트](../results/research_suite_tests/manifest.json).
각 case는 checkpoint 또는 부모 checkpoint 경로, 수열 identity, graph hash와 환경을 기록한다.
독립 검증: 5184개 certificate, 52개 저장 경로 지표,
**52개 전체 자율 재실행**, 4개 teacher 경로,
0개 정확 head refit. 수치 오차 최대 4.68514e-14, 예측 기호 정확 일치.
기존 파일 28,614개 보존. 본 실험 292.16초, 검증 220.20초.
RSS는 resources.json의 주기적 process-tree 측정이며 절대 순간 peak는 아니다.
본 실행의 worker당 최대 sampled RSS는 984.44MiB였다. 최대 두 worker를 동시에 사용했으므로
이 값을 전체 작업의 합산 peak로 해석하지 않는다. 환경: Python 3.12.10,
Windows-11-10.0.26200-SP0, NumPy 2.3.5, SciPy 1.17.0,
pandas 2.2.3, 수치 연산 thread 1개/worker.

실행 명령(기존 폴더를 덮어쓰지 않으며 재현할 때는 새 output 경로 사용):

```powershell
$env:PYTHONPATH='src'
../venv-act1/Scripts/python.exe scripts/research_suite.py run --stage correlation --out results/research_suite_correlation
../venv-act1/Scripts/python.exe scripts/verify_research_suite.py results/research_suite_correlation --out results/research_suite_correlation_validation
../venv-act1/Scripts/python.exe scripts/research_suite.py plot --source results/research_suite_correlation --out results/research_suite_correlation_analysis
```

상관 자체의 변화량과 위 로그 축 그림을 재현하는 명령:

```powershell
../venv-act1/Scripts/python.exe scripts/analyze_research_suite_correlation.py --source results/research_suite_correlation --out results/research_suite_correlation_rho_analysis
```

환경 설치는 [README](../README.md)의 dependency lock을 따른다. 이후 연구를 새 부모로 재현하려면
--parent에 새 sequence 경로를 지정하고 correlation은 --structural-validation에 새 structure 검증
경로를 지정한다. Parent 검증 폴더는 부모 경로에 _validation을 붙인 이름이다.

## 11. Next highest-information experiment

지정한 세 추가 축은 종료했다. 다음 하나는 **구조 대조의 graph별 상대 잡음 보정 비교**다.
관측 신호 규모 차이가 배선 대조의 강건성 차이에 얼마나 기여하는지, 고정 head와 새 잡음으로
분리하는 제안이다. 아직 등록·실행하지 않았으며 [종합 보고서](additional-research-results.md)를 따른다.
