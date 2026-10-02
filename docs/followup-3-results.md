# 3번: 길이 × 기호 수 × 수열 계열 — 결과

**사전 등록한 독립 입력의 과거 기호 접근 기준은 통과했다.** K10·길이128에서
현재 기호만 사용하는 빈도/역방향 Markov-1 예측보다 lag 2 MBON 해독이
iid에서 평균 **72.72%p**, Markov 수열에서 **44.20%p** 높았다. 발견
2 seed와 확인 1 seed의 각 조건이 지정한 +5%p 기준을 모두 넘었다.
이는 고정 연결망 상태에 과거 입력 정보가 남는다는 계산 근거다. 학습한
특정 수열의 자율 회상 능력이나 새로운 수열의 자율 예측과 혼동하지 않는다.

## 설계와 기존 근거

[사전 설계](followup-3-protocol.md)를 실험 전에 `d542aaa`로 고정했고,
[설정](../configs/followup3.json)과 [실행·검증 코드](../scripts/followup3.py)는
`2af9632`에서 고정했다. 앞선 [길이](length-scaling-results.md)와
[기호 수](alphabet-memory-results.md) 실험은 서로 다른 단일 축 비교였다.
이번에는 동일한 부분 연결망 2개와 48 MBON 관측을 유지하고 64/128/256
기호 × K4/10/16 × iid/Markov(.7 반복)/period-8 motif의 전 격자를
채웠다. 길이가 짧은 수열은 같은 seed의 256기호 수열 앞부분이다.
두 발견 seed 810001–810002와 하나의 확인 seed 810003을 미리 고정했다.

162개의 개별 신경망 출력층을 300 epoch 학습하고 각자의 3기호 prompt에서
남은 기호를 끝까지 생성했다. Majority/Markov-1 대조는 같은 학습 수열만
보고 자체 출력을 피드백한다. 별도로 N128에서 K×계열×seed×2회로의
54개 독립 입력 lag-2 probe를 실행했다. 학습 입력 2,000개와 시험 입력
1,000개를 분리하고 warmup100 뒤 alpha-1 ridge를 새로 학습했다. 대조는
현재 기호 하나에서 과거 기호를 예측하는 역방향 Markov-1과 빈도 예측 중
높은 값이다. Motif phase-8 oracle은 시험 warmup에서 주기를 알 때의
예측 가능성 진단이며 신경망 입력에 주지 않았다.

## 주 결과와 예측 대조

| K10/N128 계열 | lag-2 MBON 정확도 | 현재 기호 역 Markov 정확도 | 빈도 정확도 | 더 강한 대조 대비 |
| --- | ---: | ---: | ---: | ---: |
| iid | 83.48% | 10.70% | 9.90% | +72.72%p |
| Markov | 95.33% | 51.13% | 9.70% | +44.20%p |

[Seed별 결과](../results/followup3_main/seed-probes.csv)에서 iid의 excess는
71.50/71.25/75.40%p, Markov는 43.25/45.75/43.60%p다. 두 회로를 먼저
seed 안에서 평균했다. 모두 +5%p를 넘으므로 등록한 네 조건의 conjunction이
통과한다. 작은 계산 seed 집단은 독립 동물 표본이나 형식적 용량 추정이 아니다.
Motif의 별도 시험 수열은 seed마다 motif가 달라 훈련한 주기를 그대로
전이할 수 없었다. K10의 lag-2 신경망 정확도는 평균 14.6%이고, 시험
warmup에서 period-8 motif 자체를 아는 oracle은 100%다. Motif 자율
회상 성적을 미지 motif 일반화의 근거로 사용하지 않는다.

## 전체 격자와 자율 회상

아래 값은 첫 오류 전 정답 기호 수의 6개 경로 평균이다. 최대값은 길이별
61/125/253이며, ceiling 조건은 상대적인 기억 용량 순위를 판별하지 못한다.

| 계열 / 길이 | K4 | K10 | K16 |
| --- | ---: | ---: | ---: |
| iid / 64 | 47.33 | 61.00 | 61.00 |
| iid / 128 | 25.83 | 41.67 | 56.83 |
| iid / 256 | 22.33 | 24.17 | 25.33 |
| Markov / 64 | 46.83 | 48.50 | 56.00 |
| Markov / 128 | 35.00 | 41.00 | 67.50 |
| Markov / 256 | 20.00 | 35.50 | 28.33 |
| Motif / 64 | 61.00 | 61.00 | 61.00 |
| Motif / 128 | 125.00 | 125.00 | 125.00 |
| Motif / 256 | 253.00 | 253.00 | 253.00 |

K10/N128에서 자율 prefix는 iid **41.67**, Markov **41.00**,
motif **125.00**기호였고 단순 Markov-1 대조는 각각 **0.33/4.33/5.00**이다.
특정 학습 수열을 재생할 때의 차이이며, motif의 완주는 반복 규칙의
쉬운 예측 가능성에서도 생긴다. N256에서는 iid/Markov의 평균 prefix가
**24.17/35.50**으로 줄었다. K의 증가에 따른 일부 prefix 증가는 문맥
중복 변화도 동반하므로 정식 저장 용량 증가라고 부르지 않는다.
[전 격자 표](../results/followup3_main/grid-summary.csv),
[경로별 지표](../results/followup3_main/raw-cases.csv),
[독립 probe](../results/followup3_main/raw-probes.csv)를 모두 보존했다.

## 검증·한계·재현

제외된 [smoke](../results/followup3_smoke/manifest.json)는 1경로/1 probe였다.
[독립 감사](../results/followup3_main_validation/checks.json)는 본실험의
**162개 신경 상태·전체 자율 경로**, **54개 시험 상태·ridge 재학습**을
전부 재구성했다. 산출물 manifest SHA256은
`264439f61f452d70588b62d676fc4d71a328d9cf3c4e6f772856944def652269`.
본실험은 28.78초, sampled peak RSS 142,135,296 byte였다. 기존 결과 파일은
수정하지 않았다.

반복망 가중치는 고정이며 외부 비선형 head와 ridge만 학습했다. 합성 수열과
두 겹치는 부분망에서 얻은 결과이므로 살아 있는 초파리, 전체 뇌,
생리학적 학습 또는 보지 못한 수열의 자율 회상 우위를 입증하지 않는다.
Markov·motif의 예측 용이성과 과거 정보 해독은 서로 다른 시험으로 보고했다.

Python 3.12, [rate lock](../requirements-act1-lock.txt), 저장소 root에서
새 출력 경로를 사용한다.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-act1-lock.txt
$env:PYTHONPATH = 'src;scripts'
$env:OPENBLAS_NUM_THREADS = '1'
.\.venv\Scripts\python.exe scripts/followup3.py --smoke --out outputs/followup3_smoke_new
.\.venv\Scripts\python.exe scripts/verify_followup3.py outputs/followup3_smoke_new --out outputs/followup3_smoke_new_validation
.\.venv\Scripts\python.exe scripts/followup3.py --out outputs/followup3_main_new
.\.venv\Scripts\python.exe scripts/verify_followup3.py outputs/followup3_main_new --out outputs/followup3_main_new_validation
```
