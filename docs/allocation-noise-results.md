# ACT V final diagnostic — 같은 기대 잡음 에너지의 뉴런별 배분

**Brain1의 사전 지정 해독 개선 기준을 통과했다.** 확인된 그래프는 `['legacy5', 'brain5', 'brain1']`다.
이는 정답 입력에서 관측 잡음 배분을 바꾼 결과이며 자율 회상이나 내부 학습의 성공 판정이 아니다.

## 1. Repository audit

기존 ACT V 다섯 교란 곡선, 상대 잡음 및 현재 관측 잡음 결과를 확인했다.
앞선 보고서·checkpoint·실패 기록은 그대로 보존했다. 현재 모델의 추가 실험을
하나로 한정한 [종료 계획](act5-completion-plan.md)을 결과 전에 고정했다.
과거의 Windows RSS 측정 오류와 CSV 파서 실패는 수정된 결과와 구분해 남긴다.

## 2. Reproduced baseline

부모48개 frozen head의 clean 예측·확률을 재현했다. 기존 잡음의 flat 조건
144개도 이전 관측 대조와 일치한다. 부모의 모델 seed와 학습 checkpoint를 모두
재사용했으며 이번에 새 모델을 학습하지 않았다. Python3.12 격리 환경도 재사용했다.

## 3. New implementation

[실행기](../scripts/allocation_noise.py), [독립 검증기](../scripts/verify_allocation_noise.py),
[테스트](../tests/test_allocation_noise.py). Clean 학습199개 행의 뉴런별 SD를 s_j,
그 중앙값을 q라 하고 a_j=s_j/sqrt(mean(s²))로 둔다. Flat 잡음 표준편차는 rq,
배분 조건은 rq a_j다. 두 조건의 사전 clip 기대 제곱합은 모두48(rq)²다.
평가 상태를 이용해 배분을 맞추지 않았고, 출력층의 평균·scale floor도 그대로다.
개별 SD0은 배분0, 중앙값/RMS가 유효하지 않으면 중단한다.

## 4. Experiments executed

[Protocol](allocation-noise-protocol.md), [config](../configs/allocation_noise.json).
등록 source commit `f3b45be629354d78fc25cb1767a0e2d8f38e4d0b`. 모델seed7142–7146와381142–381144,
circuits701/702,legacy5/brain5/brain1. 같은48 MBON·482개 head 파라미터,
π200/prompt314,정답 입력197위치. r=[.0003,.03,.3].
기존372001 잡음과 새408001/408002/408003 잡음을 모두 평가했다.
새 잡음은 관측 root ID를 정렬한48좌표에서 생성하고 head 순서로 복원한다.
같은 모델 seed/circuit의 세 그래프,두 배분,세 강도는 같은 Gaussian draw를 쓴다.
새 draw는 새로운 모델 seed나 생물학적 표본이 아니다.

**비영 평가1152개,별도 영점 검사96개.** Smoke72개는 본 결과에서 제외했다.
새 신경 동역학 궤적·자율 회상·head refit은 모두0개다. 출력층에 보이는 값만
오염하며 다음 상태나 다음 입력으로 되먹임하지 않는다.

## 5. Results

Seed별 정답 입력 정확도 이득(배분−flat,%p). 각 칸은 `기존 draw / 새3draw 평균`이다.
두 circuit과 세 비영 강도를 먼저 평균했다. 기존 n5와 부모의 확인 n3을 분리한다.

| cohort | seed | legacy5 | brain5 | brain1 |
| --- | --- | --- | --- | --- |
| discovery | 7142 | +11.421 / +10.660 | +30.541 / +31.246 | +19.628 / +21.348 |
| discovery | 7143 | +12.267 / +13.959 | +28.173 / +28.708 | +20.389 / +20.135 |
| discovery | 7144 | +13.198 / +13.170 | +30.288 / +30.993 | +22.420 / +23.548 |
| discovery | 7145 | +14.975 / +15.454 | +29.695 / +30.570 | +22.166 / +20.784 |
| discovery | 7146 | +10.237 / +10.321 | +32.657 / +31.359 | +18.951 / +19.205 |
| confirmation | 381142 | +14.382 / +13.508 | +31.557 / +29.695 | +21.320 / +20.135 |
| confirmation | 381143 | +13.283 / +13.536 | +28.765 / +28.849 | +21.066 / +20.276 |
| confirmation | 381144 | +10.491 / +10.434 | +32.910 / +30.767 | +19.712 / +20.164 |

정답 입력 정확도와 남은 clean 대비 손실(% 또는%p):

| cohort | noise stage | graph | clean % | flat % | allocated % | remaining loss pp |
| --- | --- | --- | --- | --- | --- | --- |
| discovery | archived | legacy5 | 83.655 | 55.465 | 67.885 | 15.770 |
| discovery | archived | brain5 | 80.406 | 38.206 | 68.477 | 11.929 |
| discovery | archived | brain1 | 77.310 | 38.105 | 58.816 | 18.494 |
| discovery | fresh | legacy5 | 83.655 | 55.076 | 67.789 | 15.866 |
| discovery | fresh | brain5 | 80.406 | 38.381 | 68.957 | 11.450 |
| discovery | fresh | brain1 | 77.310 | 37.851 | 58.855 | 18.455 |
| confirmation | archived | legacy5 | 83.249 | 55.048 | 67.766 | 15.482 |
| confirmation | archived | brain5 | 79.865 | 36.125 | 67.202 | 12.662 |
| confirmation | archived | brain1 | 75.635 | 37.394 | 58.094 | 17.541 |
| confirmation | fresh | legacy5 | 83.249 | 55.621 | 68.114 | 15.134 |
| confirmation | fresh | brain5 | 79.865 | 37.733 | 67.503 | 12.361 |
| confirmation | fresh | brain1 | 75.635 | 37.441 | 57.633 | 18.002 |

Paired 기술 통계. 평균·중앙값·구간은%p,분산은 정확도 비율의 제곱이다.

| cohort | stage | graph | mean | median | variance | bootstrap95 | dz | wins/ties/losses | gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| discovery | archived | legacy5 | 12.420 | 12.267 | 0.000322949 | 11.066 / 13.892 | 6.911 | 5/0/0 | True |
| discovery | archived | brain5 | 30.271 | 30.288 | 0.000262468 | 29.017 / 31.591 | 18.685 | 5/0/0 | True |
| discovery | archived | brain1 | 20.711 | 20.389 | 0.000235269 | 19.509 / 21.912 | 13.502 | 5/0/0 | True |
| discovery | fresh | legacy5 | 12.713 | 13.170 | 0.000480248 | 11.027 / 14.399 | 5.801 | 5/0/0 | True |
| discovery | fresh | brain5 | 30.575 | 30.993 | 0.000118116 | 29.588 / 31.241 | 28.133 | 5/0/0 | True |
| discovery | fresh | brain1 | 21.004 | 20.784 | 0.00026572 | 19.820 / 22.312 | 12.885 | 5/0/0 | True |
| confirmation | archived | legacy5 | 12.719 | 13.283 | 0.000402494 | 10.491 / 14.382 | 6.340 | 3/0/0 | True |
| confirmation | archived | brain5 | 31.077 | 31.557 | 0.00044687 | 28.765 / 32.910 | 14.701 | 3/0/0 | True |
| confirmation | archived | brain1 | 20.699 | 21.066 | 7.46772e-05 | 19.712 / 21.320 | 23.953 | 3/0/0 | True |
| confirmation | fresh | legacy5 | 12.493 | 13.508 | 0.000317875 | 10.434 / 13.536 | 7.007 | 3/0/0 | True |
| confirmation | fresh | brain5 | 29.771 | 29.695 | 9.2359e-05 | 28.849 / 30.767 | 30.978 | 3/0/0 | True |
| confirmation | fresh | brain1 | 20.192 | 20.164 | 5.56699e-07 | 20.135 / 20.276 | 270.623 | 3/0/0 | True |

[개별 실행 원자료](../results/allocation_noise_main/raw-metrics.csv),
[모든 paired differences](../results/allocation_noise_main/paired-differences.csv),
[dose/seed](../results/allocation_noise_main/dose-seed-table.csv),
[통계와 판정](../results/allocation_noise_main/summary.json).
작은 모델 seed 표본의 bootstrap은 기술 통계다. 새 잡음3개를 독립 모델3배로 세지 않았다.
특히 confirmation/fresh/brain1의 큰 dz는 세 seed의 이득이 매우 비슷해 분모가
작기 때문이다. 생물학적 효과의 확실성이나 일반화 범위를 뜻하지 않는다.

![Allocation curves](../results/allocation_noise_analysis/allocation-noise-curves.png)

## 6. Interpretation

새 잡음에서 brain1의 grid 평균 해독 이득은 기존 모델 집단
21.004%p, 부모 확인 집단
20.192%p다. 남은 clean 대비 평균 손실은
18.455%p와
18.002%p다.
Primary는 기존/새 잡음×두 모델 집단 네 cell 모두 평균>=5%p이고,
각각4/5 또는3/3seed의 차이가 양수여야 한다. 이 판정을 사후 변경하지 않았다.
Secondary 두 그래프도 같은 기준이며 실패한 primary의 대체 결과로 사용하지 않는다.

두 배분의 **기대** raw 잡음 에너지가 같아도 실제 draw의 제곱합과 clip 후 에너지는
다를 수 있다. 아래는 각 집단/그래프/배분의 세 강도·네 stream·두 circuit·모델seed
평균이다. Energy ratio는 평균 실현 에너지/평균 기대 에너지다.

| cohort | graph | arm | injected/expected | clipped/expected | standardized MSE | clipped % |
| --- | --- | --- | --- | --- | --- | --- |
| confirmation | brain1 | flat | 0.9961 | 0.9961 | 3.55696 | 0 |
| confirmation | brain1 | training_sd | 0.9962 | 0.9962 | 0.0201574 | 0 |
| confirmation | brain5 | flat | 0.9958 | 0.9958 | 7.08105 | 0 |
| confirmation | brain5 | training_sd | 0.9973 | 0.9973 | 0.00869667 | 0 |
| confirmation | legacy5 | flat | 0.9958 | 0.9958 | 0.174883 | 0 |
| confirmation | legacy5 | training_sd | 0.9937 | 0.9937 | 0.0129906 | 0 |
| discovery | brain1 | flat | 1.0044 | 1.0044 | 4.67147 | 0 |
| discovery | brain1 | training_sd | 1.0044 | 1.0044 | 0.0214485 | 0 |
| discovery | brain5 | flat | 1.0043 | 1.0043 | 5.55341 | 0 |
| discovery | brain5 | training_sd | 1.0011 | 1.0011 | 0.00701551 | 0 |
| discovery | legacy5 | flat | 1.0045 | 1.0045 | 0.214974 | 0 |
| discovery | legacy5 | training_sd | 1.0012 | 1.0012 | 0.0150398 | 0 |

같은 모델 안에서 배분을 비교한 결과다. 그래프마다 q가 달라 전체망과 부분망의
raw 총 에너지가 같다는 뜻은 아니다. 작은 변동 좌표의 상대적 오염,출력층의
좌표별 scale과 민감도가 함께 작용한다. 하나의 원인 비율이나 특정 기억 회로를
식별한 실험이 아니다. 기존 지속 상태 잡음의 부정 결과는 이 대조로 교체하지 않는다.

## 7. Negative findings

등록 기준을 통과하지 못한 그래프는 `[]`다.
각 강도·seed의 음수/0차이도 원자료에 보존했다. 어떤 해독 개선도 잡음에 강한
자율 회상이나 전체망의 부분망 대비 우위로 바꾸어 말하지 않는다.
기존 ACT V의 상대/절대 강건성 실패와 ACT I clean 우위 실패는 유지된다.
정답 입력·한 수열·기존 모델 재사용이라는 범위도 남는다.

## 8. What we can claim

실제 초파리 connectome 구조를 사용한 계산 모델에서 고정 출력층의 관측 잡음
민감도를 같은 기대 잡음 에너지의 두 배분으로 검사했고,새 잡음 draw에서도
사전 기준을 확인했다. 이것은 해독 민감도에 대한 계산 대조다.

## 9. What we cannot claim

Shannon/formal memory capacity,실제 동물의 학습,생리학적 잡음,내부 연결 학습,
현재 범위 밖의 자율 회상 회복을 주장할 수 없다. 관측 잡음에 대한 개입이지
모든 뉴런의 내부 잡음에 대한 대책이 아니다. 두 모델 집단은 이미 본 자료이며,
새 잡음 반복이 새로운 독립 모델 확인 집단을 대신하지 않는다.

## 10. Reproducibility

[Main manifest](../results/allocation_noise_main/manifest.json),
[독립 검증](../results/allocation_noise_validation/checks.json),
[smoke 검증](../results/allocation_noise_smoke_validation/checks.json),
[22개 관련 테스트](../results/allocation_noise_tests/manifest.json).
Manifest의 config hash `799736d77d1e7c5cc62be1fbadf188837c4adc9591b7dd4d3d082a302e10cdf7`.
Main manifest SHA256 `dbc4fc01565d783d4664135a5d55cd8b085b84cb8e41040cdcc83baffd4042d5`.
1152개 신규 평가를 모두 독립 재계산했다. 최대 수치 차이4.77396e-14,
영점96개와 clean48개,기존 flat144개 일치를 확인했다.
이전 파일24,830개가 변경되지 않았다.
Main 96.94s,peak sampled RSS 0.158GiB;
검증 59.66s,peak 0.160GiB.
새 신경 궤적이나 학습 checkpoint는 없으며 부모 checkpoint와 graph identity를
각 case의 manifest로 고정했다. 환경 버전은 main/environment.json에 보존했다.

```powershell
$env:PYTHONPATH='src'
../venv-act1/Scripts/python.exe scripts/allocation_noise.py run --smoke --out outputs/allocation_smoke_new
../venv-act1/Scripts/python.exe scripts/verify_allocation_noise.py outputs/allocation_smoke_new --out outputs/allocation_smoke_check_new
../venv-act1/Scripts/python.exe scripts/allocation_noise.py run --out outputs/allocation_new
../venv-act1/Scripts/python.exe scripts/verify_allocation_noise.py outputs/allocation_new --out outputs/allocation_check_new
../venv-act1/Scripts/python.exe scripts/allocation_noise.py plot --source outputs/allocation_new --out outputs/allocation_plot_new
```

## 11. Next highest-information experiment

향후 별도 확장 하나는 **같은 두 관측 잡음 배분을 자기 출력 피드백의 자율 회상에
적용하는 대조**다. 정답 입력 해독의 변화가 연속 정확 회상으로 이어지는지 검사한다.
새 protocol·seed·종료 기준이 필요하며 아직 실행하지 않았다. 현재 ACT V는
[사전 종료 계획](act5-completion-plan.md)에 따라 이 진단과 종합 감사로 닫는다.
