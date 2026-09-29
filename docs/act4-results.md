# ACT IV — 현재 부분 모델 연구 단락 종료 감사

**현재 부분 모델에 대해 사전에 정한 ACT IV 실험·진단은 완료했다.**
**전체 생물학적 학습 문제가 해결됐다는 뜻은 아니다.**
[결과 전 완료 계획](act4-completion-plan.md)과 [마지막 진단](reward-margin-results.md)을
구분하며,성공할 때까지 규칙을 바꾸는 방식으로 연구를 연장하지 않는다.

## 1. Repository audit

README,로드맵,연구 상태,프로토콜,실행기,tests,원자료·checkpoint·manifest와
독립 verification 기록을 질문별로 다시 대조했다. 이번 감사는 과거 모든 실험의
대규모 재실행이 아니라 보존된 결과 해시와 검증 기록의 연결을 확인하는 감사다.
[기계 판독 감사](../results/act4_closeout/audit.json)에 각 source hash를 남긴다.

발견한 문서 차이: 과거 `phase5b-results.md`의 다음 단계 문장은 Phase6가
미완료라고 쓰지만,그 뒤 Phase6와 ACT I의 실제 artifact는 지정 범위 실행 완료를
보여 준다. 역사적 문장은 보존하고 현재 상태는 최신 roadmap/status를 따른다.
같은 이유로 'ACT IV 생물학적 질문 open'과 '지정 실험 완료'를 혼동하지 않는다.

## 2. Reproduced baseline

마지막 [방향 미분 진단](reward-margin-results.md)은 기존50checkpoint의 점수·clean
상태를 정확히 재현했다. 직전 연구에서 원래13학습의 weights/logs도 정확히 재현했다.
새3seed×2회로 학습은 중간 연결과 제안 방향까지 독립 계산으로 확인했다.
기존 인공 교사/국소 보상/도파민 실험의 검증 기록은 아래 근거에 연결한다.

## 3. New implementation

학습기의 규칙·learning rate를 새로 최적화하지 않았다. 마지막 추가는 순방향
state tangent와 독립 역방향 adjoint,합성 중앙차분 검증 및
[ACT IV 감사 도구](../scripts/audit_act4_closeout.py)다.
모든 과거 결과와 실패를 보존한다.

## 4. Experiments executed

| Research question | Executed evidence | Finding | Important boundary |
| --- | --- | --- | --- |
| A. Frozen real + trained readout | [IV-A117조건](readout-dependency-results.md) | 과거 lag2 정보 접근은 약99.93% | 외부 출력층이 학습됨 |
| B. Matched/random + trained readout | 같은 matched factorial | Random/role도 near ceiling;실제 배선 우위 기준 실패 | 한 ceiling 과제에서 구조 동등성을 증명하지 않음 |
| C. Plastic connectivity + simple readout | 인공 벡터 교사,shifted control | Real fixed-code49.58/51.52%,내부 recoding 기준 통과 | 생물학적 보상 규칙이 아닌 계산 양성 대조 |
| D. Direct-code proxy | 0 fitted parameter nearest-code decoder | 고정 코드로 읽을 수 있는 내부 변화가 있음 | 외부 코드 해석이 여전히 있으며 literal readout-free가 아님 |
| Scalar reward + local eligibility | [39조건](local-reward-results.md),frozen/yoked 대조 | Coding·representation improvement 모두 full gate 실패 | 한 국소 rate rule의 결과 |
| Dopamine/compartment gating | [기존 Phase5B12조건](phase5b-results.md) | Causal/locality controls 통과;response 실패;π recall3→2 | gamma1/pedc의 제한된 assay;생리학적 검증 아님 |
| 초기 방향 진단 | [169graphs](reward-direction-results.md) | 일부 양의 반응,full criterion 실패 | 다른 task seed의 initial-only 연구 |
| 학습·평가 잡음 일치 | [39최종checkpoint](reward-noise-results.md) | 기존 coding 실패를 해소하지 못함 | Noise curve 전체가 아님 |
| 같은 학습의 유한 변화 비교 | [50checkpoint](reward-trajectory-results.md) | 반경 제한으로 판별 불가 | 감소·유지를 반증하지 않음 |
| 점수 차이 방향 미분 | [74checkpoint·새seed포함](reward-margin-results.md) | `{"attenuation": true, "persistent_utility": false}` | Margin sensitivity이며 정확도·자율 회상이 아님 |

STDP-like와 다른 compartment rule은 원래 청사진의 **후보**이며 이번에 실행했다고
표시하지 않는다. 한 번에 여러 규칙을 섞거나 끝없는 sweep으로 확장하지 않았다.

## 5. Results

각 연구의 모든 seed raw table,paired differences,통계,그래프,checkpoint는 위
결과 문서에 연결돼 있다. 이번 새 진단의 [seed 원자료](../results/reward_margin/seed-blocks.csv)와
[판정](../results/reward_margin/summary.json)을 포함한다. 서로 다른 과제·코호트의 평균을
합쳐 하나의 성공 지표로 만들지 않는다. 마지막 진단의 확인 결과는
`{"attenuation": true, "persistent_utility": false}`이며,과거의 판정은 그대로다.

## 6. Interpretation

A. **Representation**: 과거 입력 정보가 상태에서 해독되는 조건은 있다.
B. **Decoding**: 출력층 학습,고정 코드,내부 recoding은 서로 다른 접근 방식이다.
C. **Learning**: 내부 연결이 경험에 따라 바뀌는 구현은 검증했지만,현재 국소
보상 규칙의 안정적인 full coding 개선이나 새로운 기억 용량은 확인하지 못했다.

인공 교사로 내부 출력을 재배치할 수 있다는 양성 대조와,생물학적 동기 규칙으로
기억을 안정적으로 형성했다는 주장을 구분해야 한다. ACT IV의 완료 기준은 이
차이를 분리한 실험과 검증의 종료이며,양성 결과를 얻는 것이 아니다.

## 7. Negative findings

과거 dopamine response/π improvement,scalar coding/representation improvement,
초기 full-direction gate,noise-match rescue는 지지되지 않았다. Finite temporal
probe는 판별 불가였다. 미분 진단은 별도 결과이며 이러한 실패를 덮어쓰지 않는다.
현재 데이터를 성공처럼 보이도록 seed·metric·learning rate를 변경하지 않았다.

## 8. What we can claim

**실제 초파리 connectome 구조를 사용한 계산 모델**에서 내부 연결 변화,외부
해독,고정 코드 접근 및 국소 규칙의 제한을 분리하는 지정 연구 단락을 마쳤다.
현재 단락의 코드·실험·독립 검증·문서화는 완료했다.

## 9. What we cannot claim

모든 생물학적 ACT IV 질문이 해결됐거나,실제 초파리가 원주율을 배웠거나,
국소 규칙만으로 전체 뇌의 자율 순서 기억을 확립했다고 주장하지 않는다.
다음은 명시적으로 미해결 확장이다:생리학적 검증,literal decoder-free biology,
whole-brain 내부 학습,local-only autonomous recall 향상,다른 plasticity rule의
별도 가설 검증. 종료 감사는 이 항목을 완료로 바꾸지 않는다.

## 10. Reproducibility

[감사 manifest](../results/act4_closeout/manifest.json),
[마지막 독립 검증](../results/reward_margin_validation/checks.json),
[프로토콜](reward-margin-protocol.md),[config](../configs/reward_margin.json).
240 tests pass, 8 optional skips. 마지막 분석은2,382scalar trajectories와14,292방향별
평균 민감도를 검증했고,새6학습을 독립 재현했다. 정규화된 방향 민감도의 감소 가설은 새 seed까지 포함해 통과했고 후기 유지 기준은 실패했다. 이전 검증과 합산해서 독립 표본
수가 커진 것처럼 표시하지 않는다. 각 연구의 pinned dependencies/환경·command·hash와
원자료는 개별 manifest/report를 따른다.

```powershell
$env:PYTHONPATH='src'
../venv-act1/Scripts/python.exe scripts/audit_act4_closeout.py --out outputs/act4_closeout_new
```

이 감사는 기존 원자료와 검증 기록이 모두 있는 checkout에서 실행한다.
새 output 폴더를 사용하고 기존 결과를 덮어쓰지 않는다.

## 11. Next highest-information experiment

다음 하나는 **ACT V의 동일 관측·고정 출력층 조건에서 부분망과 전체망의
ongoing-noise 강건성 곡선 비교**다. 기존 checkpoint를 재사용하고 별도 protocol에
noise grid·paired seeds·예산·판정을 먼저 고정한다. 아직 실행하지 않았다.
ACT IV의 현재 연구 단락 종료와 미해결 생물학적 확장을 그대로 보존한다.
