# 새 모델 coordinate-SD 구조 확인 — 결과

**사전 등록한 원래 배선 우위 기준은 실패했다.** 새 모델·수열·대조 그래프 seed에서
coordinate-SD 관측 잡음을 적용했을 때, 무작위 숫자 수열의 intact−degree
평균 자율 회상 길이는 발견 집단 **−0.967기호**, 확인 집단 **−0.241기호**였다.
두 집단 모두 요구한 +2기호 및 +10%p 유지율과 seed 방향 기준을 함께 충족하지
못했다. 이는 등록된 모델 비교의 부정 결과이며, 동등성 또는 생물학적 일반성을
입증하는 결과가 아니다.

## 1. Repository audit

[사전 등록 설계](fresh-coordinate-sd-protocol.md)는 실험 전에 commit
`319a454`로 고정했다. 구현은 `bc78fdb`에서 고정했고, 역사적 결과·설정·체크섬은
수정하지 않았다. 이전 [coordinate-SD 연구](coordinate-sd-noise-results.md)는
96개 기존 head를 재사용했지만 이번 연구는 96개 head를 새로 학습한다.
이전 결과의 실패를 현재 결과로 소급 변경하지 않는다.

## 2. Reproduced baseline

두 source-derived 686-neuron 회로와 48 MBON 관측, 482-parameter readout,
고정 recurrent weights, 2,000-step 학습, K10/길이200/3기호 prompt/197기호
평가를 유지했다. `legacy5`와 같은 연결·정규화·timing을 데이터 파일에서 직접
구성했다. 새 seed에서의 clean 회상 평균은 다음과 같다. 이는 이번 연구에서
새로 측정한 기준값이며 과거 head의 clean 재생 수치가 아니다.

| 수열 | 집단 | intact | degree | role |
| --- | --- | ---: | ---: | ---: |
| random | 발견 | 32.000 | 33.100 | 40.200 |
| random | 확인 | 28.000 | 30.167 | 34.000 |
| pi | 발견 | 31.800 | 30.100 | 31.400 |
| pi | 확인 | 31.167 | 33.000 | 35.667 |

## 3. New implementation

[실행기](../scripts/fresh_coordinate_sd.py), [독립 검증기](../scripts/verify_fresh_coordinate_sd.py),
[설정](../configs/fresh_coordinate_sd.json)을 추가했다. 각 seed 묶음에서
intact/degree/role을 같은 모델·데이터·잡음 draw로 비교하고, 대조 구조의
재배선 seed만 새로 사용한다. 출력층은 각 graph별로 새로 학습한다. 독립 검증은
그래프, 정답 입력 상태, 출력층, 잡음, 전체 자율 경로 및 판정표를 재구성한다.

## 4. Experiments executed

발견 모델 seed 680001–680005와 확인 681001–681003, 서로 다른 데이터 seed
690001–690005/691001–691003, 새 graph seed 698000+, 새 잡음
699001–699003을 사용했다. 8묶음 × random/pi × 2회로 × 3 graph =
**96개 새 head fit**이다. 각 모델에서 clean 1경로와 own/coordinate ×
3강도 ×3잡음의 18경로를 실행했다. 총 **1,824개 실제 자율 경로**와 별도의
**1,728개 teacher first-error certificate**를 저장했다. 독립 검증은
1,824개 경로 전부와 96개 정답 입력 상태 경로를 재실행하고, 두 head를
결정론적으로 다시 학습했다. 경로 수는 독립 모델이나 동물 수가 아니다.

본 실험 전에 명령에서 `--smoke`를 빠뜨려 13개 case log가 완료된 미완성
main 시도가 발생했다. 즉시 중단해
[미완성 시도](../results/fresh_coordinate_sd_aborted_pre_smoke/interruption.json)로
보존했고 그 지표를 분석하거나 설계를 바꾸지 않았다. 이후 별도
[smoke](../results/fresh_coordinate_sd_smoke/manifest.json) 3개 fit/57개
실제 경로와 [독립 검증](../results/fresh_coordinate_sd_smoke_validation/checks.json)을
통과한 다음 전체 main을 새 경로에서 실행했다. 미완성 시도는 본 통계에 없다.

## 5. Results

주 차이는 각 seed에서 회로2개 × 잡음3개 × 강도3개를 먼저 평균한
`intact−degree`이다. 유지율은 **각 graph 자신의 clean prefix**로 나눈다.
두 지표는 분모가 달라 방향이 같을 필요가 없다.

| random / coordinate-SD | 평균 prefix 우위 | 평균 유지율 우위 | 양의 seed (prefix/유지율) | 등록 기준 |
| --- | ---: | ---: | ---: | --- |
| 발견 n=5 | −0.967기호 | +2.570%p | 2/5 · 5/5 | 실패 |
| 확인 n=3 | −0.241기호 | +8.516%p | 2/3 · 2/3 | 실패 |

Prefix 우위의 10,000회 seed-block bootstrap 95% 기술 구간은 발견
[−4.011,+2.844], 확인 [−3.000,+1.500]기호다. 작은 계산 seed 집단의
기술 구간이며 생물학적 표본 신뢰구간이 아니다. [seed별 짝 차이](../results/fresh_coordinate_sd/paired-differences.csv)와
[전체 통계](../results/fresh_coordinate_sd/summary.json)에 중앙값·표본분산·
paired dz·부호 수를 포함했다.

| 집단 | 모델 seed | intact−degree prefix | intact−degree 유지율 |
| --- | ---: | ---: | ---: |
| 발견 | 680001 | +6.389 | +6.054%p |
| 발견 | 680002 | −3.778 | +3.107%p |
| 발견 | 680003 | −5.000 | +0.295%p |
| 발견 | 680004 | +0.056 | +1.177%p |
| 발견 | 680005 | −2.500 | +2.219%p |
| 확인 | 681001 | +1.500 | +6.672%p |
| 확인 | 681002 | −3.000 | −10.201%p |
| 확인 | 681003 | +0.778 | +29.078%p |

보조 비교인 **own→coordinate의 degree−intact gap 감소**는 발견
+4.089기호/+12.979%p, 확인 +2.426기호/+9.106%p이며 prefix 방향은
각각 5/5·3/3 양수였다. 이는 보정 방식에 따라 구조 간 **차이**가 달라진다는
기술 결과다. 사전 주 질문인 coordinate 조건에서 원래 배선의 절대 우위로
바꿔 판정하지 않는다. Pi 및 role의 모든 보조 비교도
[원자료](../results/fresh_coordinate_sd/raw-rollouts.csv)와 summary에 남겼다.

강도 .3의 random coordinate 조건에서 실제 prefix 평균은 발견
intact/degree/role = 1.167/1.167/0.867, 확인 = 1.722/1.278/1.333기호다.
잡음이 강할 때의 자율 회상은 여전히 약하다. [강도별 표](../results/fresh_coordinate_sd/dose-seed-table.csv).

## 6. Interpretation

새로 학습한 모델에서도 좌표별 training SD에 맞춰 잡음 단위를 정하면
own 보정과 비교한 구조 간 gap이 바뀐다. 그러나 직접 SD 보정 상태의
intact graph가 degree 대조보다 더 긴 자율 회상을 한다는 주장은 이 seed
집단에서 확인되지 않았다. 이전의 재사용-head 좌표 보정 주 기준 실패와
이번의 신학습 구조 우위 실패는 서로 다른 질문과 표본의 결과다.

## 7. Negative and undefined findings

두 새 집단의 primary conjunction이 모두 실패했다. Role 대조의 random
coordinate prefix 우위도 발견 −3.811, 확인 −2.296기호로 원래 망 우위가
없다. 이번 96모델은 clean prefix 0이 없어 유지율 정의 불가 셀은 0개다.
새 seed, 강도 또는 head를 결과에 맞춰 추가하거나 선정하지 않았다.

## 8. What we can claim

고정 recurrent connectivity와 새 학습 외부 readout을 사용하는 지정 계산
모델에서, 새 모델·데이터·재배선 seed를 사용한 구조 비교와 synthetic
coordinate-SD 잡음 조건의 자율 회상을 재현했다. 등록 기준에서는 intact
우위를 확인하지 못했고, calibration 간 gap 변화는 보조 결과로 관찰했다.

## 9. What we cannot claim

실제 초파리 기억, 전체 뇌 우위, 생리학적으로 측정된 잡음, 내부 연결의
경험 기반 학습, unseen 수열 예측 또는 formal 기억 용량을 입증하지 않는다.
두 회로는 같은 connectome release의 부분망이다. 새 seed는 새 동물이
아니며 graph별 head가 다르다. 좌표 SD는 clipping 이전 단변량 진폭을
맞출 뿐 공분산·실현 에너지·결정 경계까지 같게 하지 않는다.

## 10. Reproducibility

실험 manifest SHA256:
`da33018a8a946359919ed62d3b6201fa9278b01450dd51752cdefe82b2c617fd`.
[검증 결과](../results/fresh_coordinate_sd_validation/checks.json)는
96 graph 재구성, 96 teacher replay, 1,824 전체 자율 replay,
1,728 certificate 확인, 정확한 head refit 2개를 기록한다. 최대 독립
확률 차이는 2.70e-14였다. 실행 228.94초/최대 sampled RSS 174.3MB,
검증 86.43초/178.7MB였으며 실행 환경은 각 verification.json/checks.json에
기록돼 있다. `results`의 이전 파일을 덮어쓰지 않는다.

Python 3.12와 [잠긴 의존성](../requirements-act1-lock.txt), repository root에서
새 출력 경로를 사용한다.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-act1-lock.txt
$env:PYTHONPATH = 'src;scripts'
$env:OPENBLAS_NUM_THREADS = '1'
.\.venv\Scripts\python.exe scripts/fresh_coordinate_sd.py --out outputs/fresh_sd_smoke_new --smoke
.\.venv\Scripts\python.exe scripts/verify_fresh_coordinate_sd.py outputs/fresh_sd_smoke_new --out outputs/fresh_sd_smoke_new_validation
.\.venv\Scripts\python.exe scripts/fresh_coordinate_sd.py --out outputs/fresh_sd_main_new
.\.venv\Scripts\python.exe scripts/verify_fresh_coordinate_sd.py outputs/fresh_sd_main_new --out outputs/fresh_sd_main_new_validation
```

## 11. Next research boundary

이번 한 가지 새 모델 확인 연구는 결과 방향과 무관하게 종료한다. 전뇌
재배선, 생리 측정값을 이용한 잡음 보정, 다른 동역학, 내부 학습은 각각 새
질문·예산·사전 기준이 필요한 별도 연구다. 이번 결과만으로 추가 강도 또는
seed 탐색을 이어가지 않는다.
