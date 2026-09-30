# 추가 연구 — 수열 일반화

**주요 사전 기준: 통과.** 완료와 가설 성공을 구분한다.

## 1. Repository audit

[사전 완료 계획](additional-research-completion-plan.md)의 고정 순서로 실행했다.
ACT V와 이전 피드백 연구의 보고서·manifest·원자료를 변경하지 않았다.
저장 artifact와 코드 identity를 먼저 확인했고, 이번 결과를 과거 실험에 소급하지 않는다.

## 2. Reproduced baseline

기존 ACT II의 60개 head에서 출력·확률·관측값이 부모 artifact와 정확히 일치했다.
아래는 96개 모델의 실제 clean 경로 평균이다.
새 refit 모델에는 과거 expected 성능이 없으며 독립 head refit과 신경 재계산으로 검증했다.
기존 Python 격리 환경을 재사용했다.

| cohort | 수열 | graph | topology | clean prefix |
| --- | --- | --- | --- | --- |
| confirmation | pi | brain1 | intact | 29.167 |
| confirmation | pi | legacy5 | intact | 34.333 |
| confirmation | random | brain1 | intact | 29.833 |
| confirmation | random | legacy5 | intact | 31.167 |
| confirmation | shuffled_pi | brain1 | intact | 38.167 |
| confirmation | shuffled_pi | legacy5 | intact | 40.333 |
| discovery | pi | brain1 | intact | 33.700 |
| discovery | pi | legacy5 | intact | 34.500 |
| discovery | random | brain1 | intact | 28.800 |
| discovery | random | legacy5 | intact | 26.000 |
| discovery | shuffled_pi | brain1 | intact | 30.200 |
| discovery | shuffled_pi | legacy5 | intact | 31.200 |

## 3. New implementation

[실행기](../scripts/research_suite.py), [독립 검증기](../scripts/verify_research_suite.py).
기존 target-free 자율 함수와 graph sampler를 재사용하고 수열별 checkpoint·graph별 재학습·
상관 잡음·정의 불가 유지율 전파를 추가했다. 관련 28개 테스트 통과. 지표는 exact_prefix_symbols로
저장하며 π일 때만 PMS와 같다. 내부 연결은 경험으로 학습하지 않는다.

## 4. Experiments executed

기존 ACT II의 60개 head를 재사용하고, 새로운 모델·수열 seed의 36개 head를 같은 예산으로 학습했다.
무작위 숫자/섞은 π는 새로운 수열도 확인했지만 π는 같은 구간에서 모델 seed만 바꿨다.
주 조건은 random/brain1의 allocated−flat 이득이다. 다른 수열의 성공으로 실패를 대체하지 않는다.

공통 조건: 길이200 / prompt314 / 평가197, 48 MBON, 482 head 파라미터,
2000 Adam updates / LR .03 / L2 1e-5 / fixed32 ×4, gain.9 / leak.6 / mbon_after_kc.
입력 mapping·관측 위치를 paired 조건에서 맞췄다. Seed9142–9146/data9242–9246,
새 seed440142–440144/data440242–440244, circuits701/702, 잡음448001–448003,
강도.0003/.03/.3. 두 cohort는 합치지 않는다.

본 실험 96모델, 1728개 certificate, **420개 실제 평가 궤적**,
36개 새 head fit, 36개 새 teacher training 궤적.
Certificate는 첫 오류 이후의 실제 출력이 아니다. Smoke는 본 통계에서 제외했다.

## 5. Results

주 조건의 seed별 결과. 강도·두 circuit·세 잡음을 먼저 평균했다. 개별 시행은
[원자료](../results/research_suite_sequence/raw-certificates.csv)에 모두 저장했다.

| cohort | seed | rho | flat prefix | allocated prefix | 차이 | 유지율 차이 %p |
| --- | --- | --- | --- | --- | --- | --- |
| confirmation | 440142 | 0.0 | 5.833 | 10.889 | 5.056 | 26.042 |
| confirmation | 440143 | 0.0 | 13.667 | 17.667 | 4.000 | 10.445 |
| confirmation | 440144 | 0.0 | 12.278 | 21.611 | 9.333 | 27.238 |
| discovery | 9142 | 0.0 | 11.778 | 19.389 | 7.611 | 23.175 |
| discovery | 9143 | 0.0 | 8.944 | 15.778 | 6.833 | 22.276 |
| discovery | 9144 | 0.0 | 6.111 | 8.944 | 2.833 | 24.747 |
| discovery | 9145 | 0.0 | 11.778 | 17.944 | 6.167 | 18.137 |
| discovery | 9146 | 0.0 | 11.389 | 15.611 | 4.222 | 12.795 |

조건별 평균 및 배분 차이. 마지막 gate는 **배분 이득** 기준이며 구조 연구의 intact−role 판정과 다르다.

| cohort/family/graph/topology/rho | flat | allocated | prefix 차이 | 유지율 차이 %p | 배분 gate |
| --- | --- | --- | --- | --- | --- |
| confirmation/pi/brain1/intact/0.0 | 10.426 | 15.944 | 5.519 | 20.822 | True |
| confirmation/pi/legacy5/intact/0.0 | 12.574 | 21.241 | 8.667 | 24.290 | True |
| confirmation/random/brain1/intact/0.0 | 10.593 | 16.722 | 6.130 | 21.241 | True |
| confirmation/random/legacy5/intact/0.0 | 11.759 | 17.537 | 5.778 | 18.882 | True |
| confirmation/shuffled_pi/brain1/intact/0.0 | 13.130 | 21.130 | 8.000 | 22.422 | True |
| confirmation/shuffled_pi/legacy5/intact/0.0 | 14.981 | 22.130 | 7.148 | 18.515 | True |
| discovery/pi/brain1/intact/0.0 | 11.444 | 16.667 | 5.222 | 17.643 | True |
| discovery/pi/legacy5/intact/0.0 | 12.967 | 21.622 | 8.656 | 25.114 | True |
| discovery/random/brain1/intact/0.0 | 10.000 | 15.533 | 5.533 | 20.226 | True |
| discovery/random/legacy5/intact/0.0 | 10.711 | 16.022 | 5.311 | 19.081 | True |
| discovery/shuffled_pi/brain1/intact/0.0 | 10.467 | 14.856 | 4.389 | 16.571 | True |
| discovery/shuffled_pi/legacy5/intact/0.0 | 11.522 | 17.211 | 5.689 | 19.271 | True |

주 분석 통계. 유지율 평균·중앙값·구간은 %p, 분산은 비율 제곱 단위다.
작은 n5/n3의 seed-block bootstrap이며 잡음 반복을 독립 모델로 세지 않는다. 큰 dz는 작은 분산에서
나올 수 있어 일반화의 확실성으로 해석하지 않는다. [전체 통계](../results/research_suite_sequence/summary.json).

| 조건 | metric | mean | median | variance | bootstrap95 | dz | win/tie/loss |
| --- | --- | --- | --- | --- | --- | --- | --- |
| discovery/random/brain1/intact/0.0 | exact_prefix_symbols | 5.533 | 6.167 | 3.85432 | 3.911 / 7.011 | 2.818 | 5/0/0 |
| discovery/random/brain1/intact/0.0 | retention | 20.226 | 22.276 | 0.00232318 | 16.000 / 23.624 | 4.196 | 5/0/0 |
| confirmation/random/brain1/intact/0.0 | exact_prefix_symbols | 6.130 | 5.056 | 7.97634 | 4.000 / 9.333 | 2.170 | 3/0/0 |
| confirmation/random/brain1/intact/0.0 | retention | 21.241 | 26.042 | 0.00877844 | 10.445 / 27.238 | 2.267 | 3/0/0 |

![Prefix curves](../results/research_suite_sequence_analysis/prefix-curves.png)

강도는 범주 축이며 간격이 선형인 것처럼 해석하지 않는다. 그림의 평균은 원자료를 대신하지 않는다.

## 6. Interpretation

Primary random/brain1은 두 집단 모두 통과했다. 평균 prefix 차이는 +5.533/+6.130기호,
유지율 차이는 +20.226/+21.241%p였다. 발견 5/5, 새 확인 3/3 seed-block에서 두 지표가
양수였다. 섞은 π도 두 집단에서 통과해, 이 고정 길이·alphabet·동역학의 잡음 배분 이득이
π에만 한정된 현상이라는 설명은 지지되지 않는다. 두 그래프·세 수열의 12개 배분 gate 모두 통과했다.
이는 이미 학습한 수열의 제한된 계산적 확인이며 unseen 수열 예측 성능은 측정하지 않았다.

주된 차이는 중간 강도 .03에서 나타났다. Random/brain1은 flat 1.033/1.556기호에서
allocated 16.633/18.500기호로 높아졌다. 가장 약한 .0003에서는 대부분 clean을 유지해
배분의 추가 이득이 작았다. 이 결과만으로 특정 배선 구조의 필요성은 판단할 수 없으므로
다음 사전 지정 연구에서 부분망의 degree/role 보존 대조를 평가한다.

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
| confirmation | brain1 | intact | 0.0 | flat | 0.667 | 8.460 | 8.192 |
| confirmation | brain1 | intact | 0.0 | training_sd | 1.000 | 9.814 | 9.402 |
| confirmation | legacy5 | intact | 0.0 | clean | 31.167 | 24.450 | 10.292 |
| confirmation | legacy5 | intact | 0.0 | flat | 11.611 | 15.510 | 10.278 |
| confirmation | legacy5 | intact | 0.0 | training_sd | 17.722 | 17.738 | 9.633 |
| discovery | brain1 | intact | 0.0 | clean | 32.000 | 27.411 | 13.415 |
| discovery | brain1 | intact | 0.0 | flat | 10.667 | 15.905 | 11.175 |
| discovery | brain1 | intact | 0.0 | training_sd | 21.333 | 21.827 | 12.282 |
| discovery | legacy5 | intact | 0.0 | clean | 26.000 | 22.030 | 10.202 |
| discovery | legacy5 | intact | 0.0 | flat | 11.033 | 15.990 | 11.030 |
| discovery | legacy5 | intact | 0.0 | training_sd | 15.633 | 17.665 | 10.582 |

## 7. Negative findings

Random/brain1의 allocated 강도 .3 회상은 1.167/1.833기호에 그쳤다. 세 강도 평균의
clean 대비 capped 유지율도 58.021/63.618%다. 개선이 안정적 장기 회상의 복원을 뜻하지 않는다.
Confirmation random/brain1의 사전 지정 실제 경로 모델(seed440142,c701)은 clean부터
prefix=1이었다. 이를 다른 seed로 바꾸거나 post-error 표에서 제외하지 않았다.
전체 seed의 certificate 개선과 이 지정 subset의 약한 실제 회상은 함께 보고한다.

실패·판별 불가 조건과 강한 잡음의 회상 손실을 보존했다. 낮은 clean prefix나 오류 이후 붕괴,
배분 변경에 따른 악화도 삭제하지 않았다. 결과를 본 뒤 hyperparameter 탐색은 하지 않았다.

## 8. What we can claim

이 200기호 K10 과제와 고정 readout에서, 학습 관측값의 좌표별 SD에 비례한 잡음 배분은
동일 기대 에너지의 균일 배분보다 무작위 숫자·섞은 π의 정확한 자율 prefix를 개선했다.
새 모델·수열 seed에서도 사전 기준을 충족했다. 이 비교는 같은 그래프 안의 배분 효과이며
전체망이 부분망보다 우수하다는 ACT I 주장을 새로 확립하지 않는다.

실제 초파리 connectome 구조를 사용한 계산 모델에서 위 고정 조건을 평가했다.
주 판정은 통과이며 그 범위를 다른 생물학적 주장으로 확대하지 않는다.

## 9. What we cannot claim

실제 동물의 기억·지능, 내부 연결의 경험 의존 학습, formal/Shannon capacity, unseen 예측을
주장할 수 없다. 관측 잡음은 내부 상태 잡음과 다르다. 구조 대조는 부분망이고 finite swap sampler다.
상관 잡음은 합성 공통 성분이며 실측 생리적 동역학이 아니다. 모델·수열 seed가 paired이므로
각각의 분산 기여를 분리하지 못한다.

## 10. Reproducibility

Source commit `f1c667cea5370dba501da2e35b8f3d42f5278466`.
[Config](../results/research_suite_sequence/config.json), [manifest](../results/research_suite_sequence/manifest.json),
[검증](../results/research_suite_sequence_validation/checks.json),
[실제 경로 원자료](../results/research_suite_sequence/raw-rollouts.csv), [테스트](../results/research_suite_tests/manifest.json).
각 case는 checkpoint 또는 부모 checkpoint 경로, 수열 identity, graph hash와 환경을 기록한다.
독립 검증: 1728개 certificate, 420개 저장 경로 지표,
**350개 전체 자율 재실행**, 50개 teacher 경로,
38개 정확 head refit. 수치 오차 최대 3.55271e-14, 예측 기호 정확 일치.
기존 파일 25,807개 보존. 본 실험 829.66초, 검증 268.51초.
RSS는 resources.json의 주기적 process-tree 측정이며 절대 순간 peak는 아니다.
본 실행의 worker당 최대 sampled RSS는 1012.26MiB였다. 최대 두 worker를 동시에 사용했으므로
이 값을 전체 작업의 합산 peak로 해석하지 않는다. 환경: Python 3.12.10,
Windows-11-10.0.26200-SP0, NumPy 2.3.5, SciPy 1.17.0,
pandas 2.2.3, 수치 연산 thread 1개/worker.

실행 명령(기존 폴더를 덮어쓰지 않으며 재현할 때는 새 output 경로 사용):

```powershell
$env:PYTHONPATH='src'
../venv-act1/Scripts/python.exe scripts/research_suite.py run --stage sequence --out results/research_suite_sequence
../venv-act1/Scripts/python.exe scripts/verify_research_suite.py results/research_suite_sequence --out results/research_suite_sequence_validation
../venv-act1/Scripts/python.exe scripts/research_suite.py plot --source results/research_suite_sequence --out results/research_suite_sequence_analysis
```

환경 설치는 [README](../README.md)의 dependency lock을 따른다. 이후 연구를 새 부모로 재현하려면
--parent에 새 sequence 경로를 지정하고 correlation은 --structural-validation에 새 structure 검증
경로를 지정한다. Parent 검증 폴더는 부모 경로에 _validation을 붙인 이름이다.

## 11. Next highest-information experiment

**부분망의 role-preserving 구조 대조**가 다음 지정 질문이다. 같은 관측·head·잡음 예산과
graph별 refit으로 실제 배선의 추가 기여를 검사한다. 이 후속 단계의 완료 여부와 결과는
[구조 대조 보고서](structure-noise-results.md)를 따른다. 전체 프로그램 이후의 다음 제안은
[종합 보고서](additional-research-results.md)에 하나만 남긴다.
