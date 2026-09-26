# Flying — 최신 Codex 인계 (2026-09-27 KST)

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
