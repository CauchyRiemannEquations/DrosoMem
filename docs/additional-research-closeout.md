# 관측 잡음 추가 연구 — 지정 계산 연구 종료

**사용자가 지정한 수열·구조 대조·상관 잡음과 이후 두 잡음 보정까지 실행·독립 검증을 마쳤다.**
완료는 모든 가설의 성공이나 생리학 검증의 완료를 뜻하지 않는다. 이 문서는 과거 봉인 보고서의
당시 ‘다음 제안/미실행’ 표시를 현재 상태와 연결한다. 원자료와 과거 실패는 보존했다.

## 1. Repository audit

ACT I–V의 지정 모델 연구, 자율 피드백 확장, 세 추가 축, 중앙값 보정, 마지막 좌표 SD 보정을
연결해 확인했다. 각 연구의 checkpoint·config·source commit·manifest·검증 결과는 그대로 유지한다.
[연구 상태](research-status.md), [마지막 종료 감사](../results/coordinate_sd_noise_closeout/audit.json).
최초 전체 저장소 audit와 당시 문서/결과 불일치는 기존 상태 문서의 이력에 남아 있다.

## 2. Reproduced baseline

마지막 두 보정 연구는 구조 연구의96개 head를 재사용했다. 각 연구에서 모든 clean 경로가
정확히 재현됐고 head·graph는 frozen이었다. 새 학습이나 독립 모델 모집단 확인으로 세지 않는다.

## 3. New implementation

Target-free 자율 회상, 수열 통일, graph 대조, 합성 공간 상관, 세 잡음 보정, seed-block 분석,
독립 recurrence/head/metric 검증과 manifest 보존을 구현했다. 마지막 관련 테스트32개 통과.
실험별 고정 source를 남기고 과거 numerical source를 소급 변경하지 않았다.

## 4. Experiments executed

| 연구 | 실제 평가 경로 | 독립 전체 재실행 | 사전 주 결과 |
| --- | ---: | ---: | --- |
| [자율 피드백](feedback-noise-results.md) | 336 | 140 | 배분 이득 기준 통과 |
| [수열 일반화](sequence-noise-results.md) | 420 | 350 | random/brain1 기준 통과 |
| [구조 대조](structure-noise-results.md) | 672 | 672 | intact-over-role 기준 실패 |
| [합성 상관 잡음](correlated-noise-results.md) | 52 | 52 | 등록한 상관 조건 기준 통과 |
| [망별 중앙값 보정](graph-relative-noise-results.md) | 1824 | 1824 | random/degree joint 기준 실패 |
| [좌표 SD 직접 보정](coordinate-sd-noise-results.md) | 2688 | 2688 | 추가 감소 joint 기준 실패 |

합계5992개 실제 평가 경로,5726개 독립 전체 재실행이다. 이것은 **물리적 실행 횟수**이며,
서로 다른 모델·노출·동물 수가 아니다. Clean 반복과 조건 중복을 포함하고 smoke와 학습 경로는 제외한다.
초기 피드백/수열 연구의 독립 전체 재실행은 등록한 subset이며 전부 재실행했다고 주장하지 않는다.
세 추가 축에서 새 head100개를 학습했지만 두 보정에서는 새 fit0이다. 각 연구의 head는 서로 재사용된다.

## 5. Results

수열 연구의 random/brain1 배분 prefix 이득은5.533/6.130기호, 유지율 이득20.226/21.241%p였다.
구조 연구의 intact−role random prefix 차이는−3.056/−5.093기호로 intact 우위를 지지하지 않았다.
합성 상관 rho.75에서도 배분 이득 기준은 통과했으나 강도.3의 실제 prefix는1.600/2.222기호였다.
중앙값 보정의 gap 감소는2.522/1.093기호와7.822/5.748%p여서 joint 기준에 부족했다.
서로 다른 연구의 수치를 하나의 paired 통계로 합치지 않는다. 앞선 수열 집단은 당시 fresh였지만
후속 보정에서는 같은 모델을 재사용했다. [마지막 전체 seed 결과](coordinate-sd-noise-results.md#5-results).

마지막 직접 SD 보정의 **추가** gap 감소는1.500/3.407기호와4.302/13.801%p였다.
Prefix 감소 seed는3/5와2/3으로 등록한4/5와3/3에 부족했고 primary는 실패했다.
공통 raw 크기부터의 총 감소4.256/5.148기호와13.419/21.570%p는 별도 보조 비교로 유지했다.
보정 후 degree−intact prefix gap은+3.656/−4.074기호로 두 기존 집단의 평균 방향도 달랐다.
마지막 단계가 모든 seed를 같은 결론으로 정리하지는 못했다.

## 6. Interpretation

학습 관측값의 변동 크기에 따른 잡음 배분은 지정 모델의 해독과 자율 회상에 영향을 줬다.
그러나 실제 연결 상대가 반드시 더 좋은 기억 장치를 만든다는 결론은 지지하지 못했다.
잡음 보정은 비교의 단위를 정하는 통제이며 graph별 readout과 상태 분포 차이를 제거하지 않는다.
연구를 원래 망이 이길 때까지 이어가지 않고 등록한 끝점에서 종료한다.

## 7. Negative findings

전체망 우위, intact-over-role 우위, 강한 잡음에서 긴 회상, 내부 plasticity의 확실한 개선은
여전히 확인되지 않았다. 중앙값 보정의 joint 실패와 clean0 사례도 보존했다.
마지막 결과가 앞선 실패를 소급해서 성공으로 바꾸지는 않는다.

## 8. What we can claim

**실제 초파리 connectome 구조를 사용한 계산 모델**에서 고정 reservoir state를 외부 readout이
해독하는 sequential recall을 재현하고, 지정 수열·구조·잡음 조건에서 성공과 실패를 측정했다.
현재 확장의 주된 범위는 representation과 decoding이다. 내부 연결의 학습과 구분한다.

## 9. What we cannot claim

살아 있는 초파리의 원주율 기억, connectome의 일반적 우월성, unseen prediction,
생리학적 noise 모델, formal/Shannon memory capacity 또는 내부 학습 성공을 주장할 수 없다.
Coordinate SD는 clipping 이전 좌표 진폭만 통제한다. 작은 n5/n3, 같은 connectome,
유한 재배선, graph별 head, 한 동역학의 범위를 넘어 일반화하지 않는다.

## 10. Reproducibility

마지막 [protocol](coordinate-sd-noise-protocol.md), [config](../configs/coordinate_sd_noise.json),
[manifest](../results/coordinate_sd_noise/manifest.json), [독립 검증](../results/coordinate_sd_noise_validation/checks.json),
[실행 명령과 checkpoint 참조](coordinate-sd-noise-results.md#10-reproducibility)를 따른다.
종료 감사는 여섯 연구의 원본/검증 manifest와 실제 경로 수를 확인하고 이전 봉인 문서의 hash를 검사한다.
모든 결과는 새 경로에 저장했고 기존 결과를 덮어쓰지 않았다.

## 11. Next highest-information experiment

**새 model/data/graph seed에서 coordinate-SD 조건의 intact 대조를 확인하는 한 가지 실험**을
미래 제안으로 남긴다. 이번 종료 범위에는 넣지 않았으며 등록·실행하지 않았다.
실측 생리 자료·다른 dynamics·전체망 재배선도 미해결이다. 이는 현재 연구의 미완료 실행 항목이 아니라
별도 질문과 예산이 필요한 다음 연구다. 현재 지정된 관측 잡음 추가 연구에는 남은 실행·검증 작업이 없다.
