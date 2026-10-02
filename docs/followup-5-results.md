# 5번: 생리 근거와 내부 학습 규칙 — 결과

**지정한 γ1/pedc 국소 도파민 학습 규칙은 연결 변화의 인과·국소성 대조를
통과했지만, 기능적 발화 반응과 유용한 고정 해독 기준은 실패했다.**
새 KC 자극 지도 3개 × 부분 회로 2개 × 결합/대조 5조건 = **30조건**을
실행했다. 결합 조건은 각 case에서 13–16개 KC→MBON11 연결을 약화시켰고,
역순 자극·도파민 없음·다른 구획 자극·학습 차단의 24대조는 모두 연결
변화가 0이었다. 그러나 6개 결합 case 모두 **학습 전 CS+의 MBON11 발화가
0**이어서 발화 억제율은 정의되지 않았다. 이 0→0을 학습 성공이나
0% 억제로 처리하지 않았다.

## 근거·사전 설계·구현

[사전 설계](followup-5-protocol.md)는 새 결과 전에 `d542aaa`로,
[설정](../configs/followup5.json)과 [실행기](../scripts/followup5.py)는
`6a76fdf`에서 고정했다. [Hige 등 2015 원논문](https://pmc.ncbi.nlm.nih.gov/articles/PMC4674068/)은
γ1pedc에서 냄새와 DAN 활성의 순서·구획에 의존하는 시냅스 약화를
보고하고, 조건화된 냄새의 유발 **시냅스 전류가 대략 90% 감소**했다고
서술한다. 이 전류는 본 모델의 연결 가중치, KC 접촉 수 또는 발화량과
동일한 단위가 아니다. 한 수치에 학습률을 맞추거나 그림에서 임의
좌표를 추출하지 않았다. 실제 생리 원자료에 맞춘 파라미터 추정은
이번에 성립하지 않는다.

기존 [Phase 5B 결과](phase5b-results.md)의 보수적인 PPL101/MBON11
매핑과 Brian2 스파이킹 모델을 유지했다. 규칙 상수는 학습률 .1,
eligibility 1,000ms, 최초 접촉 수의 10% 하한으로 고정했다.
정답 수열, 출력층 gradient, 기호별 보상은 연결 학습에 주지 않았다.
결합과 네 대조는 같은 출발 연결·KC 지도에서 각각 독립 실행했다.
CS+는 digit3, CS−는 digit1이며 새 지도 seed 830001–830003을
결과 선택 없이 사용했다.

## 연결·발화·해독 결과

| 측정 | 결합 조건 | 네 대조 조건 |
| --- | ---: | ---: |
| γ1/pedc에서 변한 KC→MBON11 연결 | case별 13–16개, 평균 14.33개 | 모두 0개 |
| 해당 DAN 발화 | case별 4회 | 조건에 따라 0 또는 4회 |
| CS+ 활성 KC의 MBON11 입력 접촉 합 감소 | 6 case 모두 28.124% | 모두 0% |
| CS+ MBON11 발화, 전→후 | 6 case 모두 0→0 | 0→0 |
| 고정 MBON11 이진 판별 정확도 | 평균 58.33% | 평균 58.33% |
| 별도 lag-2 ridge 시험 정확도 | 평균 20.67% | no-dopamine 평균 21.00% |

모델의 28.124%는 선택한 활성 KC의 MBON11 입력 **접촉 가중치 합** 감소다.
원논문의 약 90% **실측 전류**와 나란히 보고할 수는 있지만, 두 값을
같은 생리 변수로 보거나 그 차이를 하나의 오차율로 계산할 수 없다.
CS− 입력 접촉도 일부 겹치는 활성 KC 때문에 감소했으며, 남은 MBON11
발화가 대부분 0이라 선택적 발화 효과를 평가할 수 없다.
[조건별 원자료](../results/followup5_main/raw-cases.csv)에는 6개 결합 case의
접촉·발화·해독값과 모든 대조를 남겼다.

사전 **기능 기준**은 모든 seed·회로에서 학습 전 CS+ 발화가 양수이고,
결합 뒤 ≥20% 선택적 CS+ 억제와 CS− 변화 ≤10%, 네 대조의 연결 변화
0을 동시에 요구했다. 대조의 0은 통과했지만 CS+ 기준값이 전부 0이어서
전체 기준은 실패했다. 별도 **고정 해독 이득**은 no-dopamine 대비
≥5%p를 모든 seed·회로에서 요구했으나 차이가 전부 0이었다.
학습 후 alpha-1 ridge는 별도의 관측 진단으로만 사용했고 두 gate를
대체하지 않았다. [Seed별 판정](../results/followup5_main/seed-blocks.csv)과
[요약](../results/followup5_main/summary.json)에 실패를 명시했다.

## 재현·검증 기록

처음 제외된 [smoke 시도](../results/followup5_smoke_attempt1/manifest.json)는
독립 검증기가 닫힌형 계산의 부동소수 오차에도 바이트 단위 hash 일치를
요구해 [기술적으로 거절](../results/followup5_smoke_attempt1-verification-failure.json)됐다.
`7d0e4ce`에서 검증기만 고쳐 재구성값의 수치 일치와 실제 저장된 가중치
해시를 분리해 검사했다. 학습 상수, 자극, seed, 판정 기준은 바꾸지 않았다.
새 [smoke](../results/followup5_smoke/manifest.json)의 5조건과
[독립 검증](../results/followup5_smoke_validation/checks.json)이 통과한 뒤
본실험을 시작했다. 처음 smoke는 본 통계에 없다.

본실험 [독립 감사](../results/followup5_main_validation/checks.json)는 30조건의
도파민 발화에서 **30개 연결 갱신을 닫힌형으로 재구성**하고 30개 전후
LIF 반응을 다시 실행했다. 결합/no-dopamine의 12개 lag-2 ridge도
독립 최소제곱으로 재학습했다. Main manifest SHA256은
`1668b098b4178b038d0f86c2733fd04f308dec25b9f70a0e93d0198e35144846`.
실행 시간은 **479.26초**, case 경계에서
측정한 최대 RSS는 **194,781,184 byte**였다. 이전 Phase 5B 결과를
수정하지 않았다.

Python 3.12, [LIF lock](../requirements-lif-lock.txt), 추가 `psutil==7.2.2`에서
저장소 root의 새 출력 경로로 재현한다. 고정 출처의 부분 연결망 자료는
`data/`에 포함된다.

```powershell
py -3.12 -m venv .venv-lif
.\.venv-lif\Scripts\python.exe -m pip install -r requirements-lif-lock.txt
.\.venv-lif\Scripts\python.exe -m pip install psutil==7.2.2
$env:PYTHONPATH = 'src;scripts'
$env:OPENBLAS_NUM_THREADS = '1'
.\.venv-lif\Scripts\python.exe scripts/followup5.py --smoke --out outputs/followup5_smoke_new
.\.venv-lif\Scripts\python.exe scripts/verify_followup5.py outputs/followup5_smoke_new --out outputs/followup5_smoke_new_validation
.\.venv-lif\Scripts\python.exe scripts/followup5.py --out outputs/followup5_main_new
.\.venv-lif\Scripts\python.exe scripts/verify_followup5.py outputs/followup5_main_new --out outputs/followup5_main_new_validation
```

## 해석 경계

이 연구는 **한 가지 고정 상수의 국소 규칙**을 신선한 KC 지도에서
평가했다. 연결 약화의 구현과 시간·구획 대조가 통과해도, 실제 MBON11
출력이 자극에 반응하지 않으면 행동 또는 수열 기억의 개선으로 이어졌다고
말할 수 없다. 출력층을 새로 학습한 진단에서도 lag-2 이득은 없었다.
실측 전류·칼슘·발화의 같은 조건 raw data가 없는 상태에서 생리학적
파라미터 보정을 완료했다고 주장하지 않는다. 더 강한 자극이나 새로운
학습률을 이번 실패 뒤에 선택하지 않았다.
