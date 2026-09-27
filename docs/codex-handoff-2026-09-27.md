# Flying — 최신 Codex 인계 (2026-09-27 KST)

## 최신 범위 결정: 순수 연구 저장소

- 이 저장소는 연결망 기반 학습·기억 연구에 집중한다. `AGENTS.md`와
  [연구 방향](research-direction.md)을 따른다. 다음은 전체 뇌/부분망의 원주율
  학습·자율 회상 비교 설계를 고정하는 연구다. 아직 새 실험을 실행하지 않았다.
- 과거 연구 결과와 체크섬은 그대로 보존한다. 현재 문서의 용어 정리로 프로토콜
  바이트가 바뀌었으므로 과거 재현은 해당 Git revision을 사용한다. Phase 6는
  `503aa9e`에서 재현한다. [재현 안내](historical-reproduction.md).
- 범위 정리 후 현재 테스트는 LIF 환경 164개 통과, 기존 환경 125개 통과·9개 skip.
  아래 과거 보고의 테스트 개수는 당시 실행 기록이며 소급해서 바꾸지 않는다.

## Phase 6 전체 뇌 실행 완료 범위

- [Phase 6 결과](phase6-results.md), [고정 설계](phase6-protocol.md)부터 읽는다.
  원본 138,639개 뉴런·15,091,983개 연결 전체를 포함한다. 그래프 구축·원본 ID
  대조·부분망 일치 검사·입력 손실 분석과 자원 제한 실행을 구현했다.
- 4개 그래프 수준 × 기존 입력 지도 2개 × 자극 3개를 각각 두 번 실행했다.
  24개 반복에서 발화·시간·카운트·최종 v/g 배열이 정확히 같다. 두 부분망의 6개
  1초 자극은 Brian2와 발화가 같고 상태 오차는 1e-8 mV 기준 안이다.
- 전체 연결 조건은 약 522~541 MiB RSS, 모의 1초당 실제 약 23~25초.
  테스트 193개 통과. 기존 환경은 154개 통과·선택 의존성 관련 9개 skip.
- 기존 부분망은 왼쪽 MBON의 입력 접촉 수 중 10.34~10.45%만 보존했다.
  약한 연결까지 복원하자 MBON11의 digit3/digit1 반응이 s701에서 34/49회,
  s702에서 13/25회로 회복됐다. 학습하지 않은 기본 반응이며 Phase 5B 학습
  실패를 통과로 바꾸지 않는다. 전체 뇌 pi 학습·회상은 측정하지 않았다.
- `results/phase6`, `results/phase6_analysis`, `results/phase6_validation` 보존.
  설계 커밋 `59b6eea`, 실행 소스 `bdfc228`; 기존 LIF lock에 pyarrow/psutil만
  더한 `requirements-phase6-lock.txt` 사용. 원본과 큰 그래프는 ignored 경로.
  `python -m flying.training.phase6 prepare`, `run --out 새폴더`, `verify`로 재현.
  Phase 6는 reset 반복을 했으며 전체 CLI를 별도 두 번째 실행한 것으로 쓰지 말 것.
- 원래 phase는 문서에 명시된 범위에서 실행·검증됐다. [phase audit](phase-status.md)의
  실패와 한계를 유지한다. fixed-32 모델은 기존 연구 기준선으로 유지한다.
  전체 뇌 원주율 학습·확장 그래프 도파민·생리학적 보정은 별도 후속 연구다.

아래는 이전 단계의 역사적 인계이며 당시의 “다음” 항목은 위 결과로 갱신됐다.

## 앞선 연구 우선 / Phase 5B 완료 범위

- [Phase 5B 결과](phase5b-results.md): PPL101/MBON11 gamma1/pedc 매핑과 실제
  DAN 발화로 작동하는 국소 LTD 구현. alpha3는 다른 구획 대조군으로만 매핑했다.
- 두 실제 부분망에서 12개 조건을 실행했다. 올바른 KC→DAN 순서에서만 14개·11개
  연결이 약 28.12% 약화했고, 나머지 다섯 대조군은 연결 변화가 없었다.
  전체 재실행에서 12개 조건 및 8개 회상열의 모든 저장 배열·payload가 정확히
  일치했다. LIF 환경 181개 테스트 통과, 기존 환경 148개 통과·LIF 모듈 3개 skip.
- 기능 기준은 실패했다. 두 회로의 고립 CS+·CS- 시험에서는 학습 전부터 MBON11
  발화가 없었다. s701에서는 다양한 원주율 입력 중 발화하므로 “항상 침묵”이라고
  확대 해석하지 말 것. 원주율 평균 회상은 3→2자리로 개선되지 않았다.
- 다음은 **Phase 6 입력 보존량·희소 그래프 성능·규모 확장 검증**이다.
  512/2,580 KC 샘플링이 얼마의 실제 입력을 제거했는지 측정해야 한다. 전체 뇌를
  이미 실행했다거나 Phase 5B의 생물학적 타당성이 검증됐다고 쓰지 말 것.
- 설계 고정 `5f4f2ae`, 실행 소스 `ea4e837`. 첫 실행의 SI→mV 반올림 비교 오류를
  수정했으며 실험 조건은 바꾸지 않았다. 이후 검사에서 동시에 쓰이던 테스트 로그
  체크섬 차이를 발견해 로그 완료 후 manifest를 다시 확정했다. 결과 checkpoint는
  변하지 않았다. `results/phase5b/`와 별도 LIF 환경을 사용한다.

## 앞선 Stage B/C 인계

- 사용자는 검증된 변경을 `main`에 직접 푸시하도록 승인했다. 연구 작업은 [phase 현황](phase-status.md)과 [다음 작업](next-work.md)을 따른다.
- 이후 anchored 확인과 Phase 2 강건성 실험은 완료했지만 기준에 실패했다.
  새로 완료한 [Stage B/C LIF](stage-bc-results.md)는 수치 검증을 통과했고,
  실제 연결망 평균 회상은 3.0자리 대 기존 rate 33.5자리였다. 모델 승격은 하지 않는다.
- 당시 다음 단계는 Phase 5B였으며 위의 새 결과로 갱신됐다.
  Phase 6 전체 뇌는 아직 미완료다. Stage B/C에서 DAN 노드는 발화하지 않았으며,
  그래프에 DAN이 있다는 사실만으로 학습 규칙이 구현된 것이 아니다.
- LIF는 별도 환경과 `requirements-lif-lock.txt`를 사용한다. Brian2/SymPy 때문에
  mpmath 1.3.0이 필요하므로 기존 rate 환경과 lock을 바꾸지 말 것.
- `results/stage_bc/`에는 네 그래프 조건의 LIF/rate 상태, 가중치, 발화 기록,
  회상열과 수치 진단이 있다. 재현은 새 출력 폴더에서 수행한다.

아래는 앞선 구간별 확인 단계의 역사적 인계다. 아래의 당시 “다음 작업”을
현재 우선순위로 다시 실행하지 말 것.

이 문서는 9/25 인계를 갱신한다. [구간별 확인 결과](phase5-prefix-confirmation-results.md),
[고정 설계](phase5-prefix-confirmation-protocol.md), `results/phase5_prefix_confirmation/`부터 읽는다.

## 이번에 추가한 것

- 32자리·4배 가중 규칙을 유지하고 offsets 0/1000/2000, 새 seed 1142–1144에서 비교.
  각 200자리 구간은 별도 학습한다. 다른 구간 결과는 미학습 π의 즉석 예측을 뜻하지 않는다.
- 36개 신경 상태 조건, 총 216개 모델. 동일 그래프/입력/초기화의 균등 학습과 paired 비교.
- 조건별 원자적 NPZ 저장, 체크섬 ledger, 설정/코드/원자료/프로토콜/패키지 버전 검사,
  중단 후 `--resume`. 기존 결과를 지우고 다시 시작할 필요가 없다.
- 전체 테스트 71개. 중단→재개가 연속 실행과 같은 결과인지, 손상/누락된 저장 모델과
  변경한 설정/코드를 거부하는지 검사한다.
- 실제 본 실험도 첫 조건을 저장하고 프로세스를 종료한 다음 `--resume`으로 이어 실행했다.

## 재현·재개 명령

인증된 저장소 clone 또는 최신 pull을 먼저 수행한다. Windows PowerShell 예:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\python.exe -m pytest -q
$env:OPENBLAS_NUM_THREADS = "1"
.\.venv\Scripts\python.exe -m flying.training.phase5_prefix_confirmation --output outputs/confirmation_codex
.\.venv\Scripts\python.exe -m flying.training.phase5_prefix_confirmation --verify --output outputs/confirmation_codex
```

중단됐으면 마지막으로 완료된 조건 다음부터 재개:

```powershell
.\.venv\Scripts\python.exe -m flying.training.phase5_prefix_confirmation --output outputs/confirmation_codex --resume
```

짧게 점검하려면 첫 실행에 `--max-conditions 1`을 붙인다. 설정을 바꿀 때는
별도의 `--config`와 **새 출력 디렉터리**를 쓴다. 같은 출력 폴더에 두 프로세스를
동시에 쓰지 않는다. 현재 기능은 조건 경계에서의 재개이며 epoch 중간 재개는 아니다.
누락/손상으로 기록된 checkpoint는 자동 무시하지 않고 오류를 낸다.

ledger 갱신 직전 중단되어 ledger에 없는 파일은 완료된 조건으로 인정하지 않고 다시 계산한다.
`.part` 파일 역시 완료 기록으로 인정하지 않는다. 이미 완료된 전체 실행을 다시
`--resume`하면 기존 체크포인트를 읽어 집계하며 manifest의 마지막 호출 시간은 바뀔 수 있다.
기존의 커밋된 결과는 수정하지 말고 새 출력 폴더에서 재현할 것.

## Work 제약의 현재 상태

지난번에 설치했던 pytest/mpmath는 새 세션 시작 때 다시 없었다. 이번에도 같은 버전을
재설치해 해결했다. 따라서 재현 환경은 패키지 설치를 명시해야 하며, 한 번 설치했다고
다음 세션까지 유지된다고 가정하지 않는다. 소스·데이터는 이전 manifest와 대조해 그대로임을 확인했다.

비공개 GitHub 저장소는 이전 Work 터미널에서 clone 인증이 없었고 연결된 GitHub 기능을
통해 반영했다. Codex에서는 정식 인증으로 clone/pull한다. 현재 연구는 번들 부분망만
사용하므로 전체 원자료 재다운로드는 필요 없다. GPU·전체 뇌·LIF·생물학적 도파민은 이번에도
실행하지 않았으며, 확인되지 않은 하드웨어/네트워크 한계를 차단 사유로 꾸미지 않는다.

## Codex에 줄 요청

> 최신 main과 docs/phase5-prefix-confirmation-results.md, 이 인계 문서를 읽어라.
> 전체 테스트를 실행한 뒤 새 출력 폴더에서 동일 확인 실험을 재현하고 verifier를 실행해라.
> 보고서에 있는 초반 회상 이득과 후반 정확도 손실을 모두 비교해라. 최고 시드만 선택하지 마라.
> 다음 실험은 보고서의 제안에 따라 하나의 개입과 고정 계산 예산으로 설계하고,
> 전체 뇌 확장보다 기존 부분망에서 더 긴 연속 회상을 확보할 수 있는지 먼저 확인해라.
> 오래 걸리면 조건별 checkpoint와 resume을 활용하고, 실행/검증/미완성을 분리해 보고해라.
