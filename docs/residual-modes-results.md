# Residual source-mode decomposition — ACT III

**하위24개 저분산 방향이 출력 왜곡을 주도한다는 가설은 실패했다.** 분할을 바꾸지
않았다. 같은 cutoff에서 높은 분산의 상위24방향이 대부분의 부호 있는 기여를 가진다.
Protocol `8e62dc5`, implementation `9e5f70f`. [Protocol](residual-modes-protocol.md).

## 1. Repository audit

기준 `dae7a5e`의 handoff·연구 방향·phase scope·moment-alignment 결과와 실제
source/target checkpoint를 읽었다. 기존 결과 7,283파일 해시가 그대로다.
기존 정확도 개선 및 R²/retention 실패를 뒤집는 연구가 아니다.

## 2. Reproduced baseline

34방향에서 source within과 정렬 후 actual/null 점수를 정확 재현했다.
본 집단 정렬 정확도 R→F51.007%,F→R51.607%; archived 확인49.447%,52.886%.
이미 관찰한 기존 두 집단을 재사용했으며 새로운 확인 seed는 없다.

## 3. New implementation

`residual_modes.py`, 별도 projector verifier, cancellation/sign/train-only basis
검증 tests를 추가했다. 학습 source 표준화 상태에서만 SVD 방향을 정했다.
기준을 저장한 뒤 paired target−source test 상태 차이와 고정 head를 분해했다.
새 출력층 또는 상태 정렬을 학습하지 않았다.

## 4. Experiments executed

K4,48MBON,686-neuron rewired controls R/F, threshold5; 원시 그래프·입력·관측·
stream 동일. Train2000/test1000,warmup100; smoke200/100. Main34142–34146,
archived confirmation41142–41144, circuit701/702, 양방향. Smoke34001/c701 제외.
20+12+2방향,11lags,374행. Primary1,2,3,4,5,8을 평균하고 circuit을 평균했다.

출력 왜곡은 **정렬된 target score−source within score**다. 정답 대비 오차와 다르다.
각 방향의 제곱 에너지는 단순 합이 실제 출력 에너지가 아니므로, 실제 total과의
내적을 부호 있는 attribution으로 계산했다. Negative/>1 share는 상쇄로 가능하며
확률이 아니다. Low/high group 에너지와 cross-term 합도 독립 검증했다.

## 5. Results

| 집단 | 방향 | low 상태 에너지 % | low signed 출력 기여 % | low gain | high gain | 등록 가설 |
| --- | --- | --- | --- | --- | --- | --- |
| confirmation | F→R | 9.8554 | 0.5843 | 0.0790 | 0.8336 | False |
| confirmation | R→F | 15.0242 | 0.6373 | 0.0357 | 1.2063 | False |
| main | F→R | 14.6104 | 1.0473 | 0.0565 | 1.2962 | False |
| main | R→F | 17.8167 | 0.2814 | 0.0303 | 1.4656 | False |


| 집단 | 방향 | seed | low 상태 % | low signed 출력 % |
| --- | --- | --- | --- | --- |
| confirmation | F→R | 41142 | 13.4188 | 0.2410 |
| confirmation | F→R | 41143 | 8.0248 | 1.0933 |
| confirmation | F→R | 41144 | 8.1226 | 0.4187 |
| confirmation | R→F | 41142 | 16.7503 | 1.0173 |
| confirmation | R→F | 41143 | 13.9930 | 1.9941 |
| confirmation | R→F | 41144 | 14.3293 | -1.0994 |
| main | F→R | 34142 | 11.9577 | 1.7425 |
| main | F→R | 34143 | 10.8800 | 0.0995 |
| main | F→R | 34144 | 24.8336 | -0.6870 |
| main | F→R | 34145 | 11.8643 | 1.9393 |
| main | F→R | 34146 | 13.5164 | 2.1423 |
| main | R→F | 34142 | 18.8172 | 0.4988 |
| main | R→F | 34143 | 11.5236 | 0.4759 |
| main | R→F | 34144 | 27.5103 | -0.7290 |
| main | R→F | 34145 | 12.9346 | 1.2021 |
| main | R→F | 34146 | 18.2976 | -0.0409 |


Low는 train singular rank25–48로 결과 전에 고정했다. 기대 기준은 signed 기여≥75%,
상태 share≤50%, block별 과잉기여4/5및3/3였다. 모든 방향·집단이 실패했다.
전체 분할 경계와 분모는 유효했고, 부호 있는 low 기여가 음수인 block도 보존했다.
[모든 lag raw](../results/residual_modes/raw-lag-table.csv),
[circuit별 raw](../results/residual_modes/raw-past-table.csv),
[mean/median/variance/bootstrap95/paired effect](../results/residual_modes/summary.json).
Bootstrap10000,seed50399,n5/n3 별도. 작은 표본의 기술통계다.

## 6. Interpretation

이번 source-SVD 분할에서 하위 절반의 증폭 설명은 맞지 않았다. 상위 절반 안에서도
작은 분산의 방향이 중요할 가능성을 배제하지는 않지만, cutoff를 옮겨 긍정 결과를
찾지 않는다. 다음에는 현재 분할을 유지한 방향 대조로 별도 질문을 검사한다.

## 7. Negative findings

하위 절반에서 상태 차이의 약10–18%가 있으나 signed score 기여는 약0–1%다.
사전 low-variance-dominance 가설 실패. 이 결과로 뉴런 그룹/생물학적 경로를
특정할 수 없으며, 회상 능력 향상을 얻은 것도 아니다.

## 8. What we can claim

이 계산 모델의 R/F 전이 왜곡을 고정 readout과 source 학습 상태의 방향으로
정확히 분해할 수 있고, 등록된 분할에서 주된 signed 기여는 상위 절반이다.

## 9. What we cannot claim

생물학적 기억 회로, 실제 배선 우위, 전체망 우위, 내부 학습, 자율 회상 개선 또는
과거 정보 소실을 입증하지 않았다. SVD 방향은 여러 뉴런을 섞는 수학적 축이다.

## 10. Reproducibility

34 exact replays,374행 별도 projector/algebra/statistics 검증.
전체 테스트189 passed,8 optional skips. Runtime 47.89초,
50ms sampled peak RSS 193.12MiB.
Locked Python3.12.10 environment, NumPy2.3.5/SciPy1.17.0, numerical BLAS1thread.
[Manifest](../results/residual_modes/manifest.json), [checks](../results/residual_modes_validation/checks.json).

```powershell
$env:PYTHONPATH='src'
.venv/Scripts/python.exe scripts/residual_modes.py --out outputs/residual_modes_new
.venv/Scripts/python.exe scripts/verify_residual_modes.py outputs/residual_modes_new --out outputs/residual_modes_check_new
```

## 11. Next highest-information experiment

**Low/high 상태 에너지를 보존한 방향 대조** 하나를 선택했다. 고정 head에 대한
실제 distortion을 같은 에너지를 가진 무작위 방향의 기대 distortion과 비교한다.
현재 분할을 유지하고 가중치/정답/동역학을 바꾸지 않는다. 별도 protocol을 먼저
고정하고 시행한다. 이 문서는 첫 진단 완료 시점의 기록이다.
