# Flying — Codex 인계 (2026-09-25)

## 현재 기준과 이번 작업

검토 시작 시 main은 `9c9f9148ec4fc5c7a0e2c45ded05da9e526d92d8`였다.
그 커밋의 마지막 완료 작업은 파라미터 수를 맞춘 비선형 출력층 실험이다.
이 문서와 함께 추가한 후속 작업은 [초반 오차 가중 학습 실험](phase5-prefix-results.md)이다.
실험 설계, 측정값, 해석, 검증을 해당 보고서와 `results/phase5_prefix/`에서 확인한다.

| 단계 | 저장소에서 확인한 성과 | 아직 주장할 수 없는 것 |
|---|---|---|
| Phase 2 | 고정 연결의 정규화에 따라 200자리 학습 회상 성능이 크게 달라짐. 원래 300뉴런 표본의 incoming-L1은 197 이상 / 41 / 84자리 | 실제 뇌 전체의 용량, 생물학적 배선 우위, 안정적인 200자리 암기 |
| Phase 3–3b | 해부학적으로 주석된 686뉴런 부분망 두 개, KC 입력·MBON 출력, 시간 간격과 무작위 입력 기억 진단 | 더 많은 뉴런 자체의 효용, 독립된 초파리 개체 사이의 재현 |
| Phase 4–5 | 제한된 기존 연결 학습과 보상·활동흔적의 대조 실험 완료 | 고정 연결보다 안정적으로 더 긴 회상 |
| Phase 5 진단·BPTT | 작은 변화량 후보가 새 시드에서 확인되지 않음. 정확한 시간 역전파로 학습 손실 감소 | 보상 학습 또는 BPTT의 안정적인 회상 이득 |
| 입력 시차 | 현재 입력 복원 약 10%→100% | 연속 회상 향상: 실제 연결에서는 오히려 감소 |
| 비선형 출력층 | 같은 고정 상태에서 평균 학습 정확도 47.80%→83.70% | 연속 회상 향상: 6.92→5.75자리 |

점수는 제공한 `314`를 **제외한 새 숫자**의 첫 오류까지 길이이다.
학습에 사용한 유한 π 접두사를 재생하는 과제이며, 미학습 π 예측이 아니다.
두 부분망은 한 개체에서 추출했고 서로 겹친다. 시드를 생물학적 표본 수로 세지 않는다.
이전 대화의 수치라도 현재 저장소에 대응하는 실행 기록이 없으면 확정 성과에 합치지 않는다.

## Work 제약: 확인된 사실과 해결 여부

| 항목 | 근거·현재 상태 | Codex에서 할 일 |
|---|---|---|
| 이전 전체 pytest 실행 불가 | timing 보고서는 pytest 미설치·다운로드 시간 초과, readout 보고서는 미설치를 기록. 당시 일부 검사만 수행 | **이번 세션에서는 설치 성공, 전체 65 tests 통과.** 설치 문제를 영구적인 Work 제약으로 취급하지 말 것. Codex에서도 전체 테스트부터 실행 |
| mpmath 부재 | 이전 두 실험은 검증된 Decimal 대체 계산 사용 | 이번에는 mpmath 1.4.1 설치 성공. 기존 숫자 생성 교차 검증 테스트 통과. 해결됨 |
| 비공개 저장소 직접 git clone 인증 | 이번 터미널에서 `fatal: could not read Username for 'https://github.com': No such device or address` | Codex 환경의 정식 GitHub 인증으로 clone. Work에서는 연결된 GitHub 도구로 필요한 소스·데이터를 가져와 작업함 |
| 로컬 복원 시 파일 바이트 보존 | 이번 텍스트 복원에서 추가된 끝 개행을 데이터 체크섬이 감지. 원래 Git blob SHA와 일치하도록 복구 후 실행 | 일반 git clone 사용. 해시 검사를 제거하거나 provenance를 바꿔 오류를 숨기지 말 것. 최종 실험은 원본 데이터 체크섬 통과 후 실행 |
| 과거 NPZ 동기화 손상 | diagnostic 보고서에 실제 절단 기록 있음. 해시가 동일한 완전한 원본으로 복원했고 atomic 저장 적용 | 저장 후 SHA256·재생 검사 유지. 이 사건은 당시 복구됐으며 미완성 실험으로 세지 않음 |
| 과거 실험 전부의 재학습 | 이번에는 최신 코드 전체 테스트와 새 실험을 검증. 과거 모든 대규모 실험을 다시 학습하지 않음 | 아래 별도 worktree에서 기존 검증 수행. 재생과 독립 재학습을 구분해 보고 |

Work에서 Python 계산 자체가 불가능했던 것은 아니다. 이번 120개 조건 학습도 CPU에서 수행했다.
GPU 미지원, RAM 부족, 특정 외부 데이터 다운로드 차단은 이번에 실제로 확인하지 않았으므로
확정된 차단 사유로 쓰지 않는다. 새 환경에서도 권한·패키지·네트워크 가능 여부를 먼저 확인한다.

## 아직 안 한 일: 환경 차단과 구분

1. **Whole-brain:** 실험적으로 보류한 연구 단계다. 현재 추출기는 10–3000뉴런을 허용하며,
   숫자 옵션만 바꿔 전체 뇌 모델이 되지 않는다. 전체 그래프 로딩·희소 연산·기억/시간 측정이 필요하다.
2. **Brian2/LIF와 생물학적 도파민:** 아직 구현·검증하지 않았다. 현재는 leaky-tanh rate model이고
   DAN도 단순 부호 노드다. 이것은 Work가 막아서 완성하지 못한 구현이라고 단정할 수 없다.
3. **견고한 기억 용량:** 더 긴 학습열, perturbation/noise, 별도 표본·조건 확인이 남았다.
   단일 최고점 또는 평균 분류 정확도로 기억 용량을 주장하지 않는다.
4. **원자료 전체 재다운로드:** 이번에는 저장소에 포함된 검증된 부분망을 사용했다.
   약 101 MB의 고정 버전 parquet와 주석 재구축은 이번 작업 범위에서 재실행하지 않았다.
5. **장기 작업 재개:** 현재 새 runner는 결과를 마지막에 기록하고 출력 디렉터리가 존재하면 거부한다.
   수시간 이상의 sweep 전에 조건별 원자적 체크포인트·진행률·안전한 재개를 구현할 것.

## Codex에서 바로 실행

Windows PowerShell, 인증된 GitHub 접근 및 Python 3.12 기준:

```powershell
git clone https://github.com/CauchyRiemannEquations/Flying.git
cd Flying
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\python.exe -m pytest -q
$env:OPENBLAS_NUM_THREADS = "1"
.\.venv\Scripts\python.exe -m flying.training.phase5_prefix --output outputs/phase5_prefix_codex
.\.venv\Scripts\python.exe -m flying.training.phase5_prefix --verify --output outputs/phase5_prefix_codex
```

출력 디렉터리는 새 이름을 사용한다. 각 패키지 버전은 새 결과의 manifest에 기록된다.
기록된 실행과 같은 직접 의존성 버전이 필요하면 `requirements-lock.txt`를 설치한다.
OS/BLAS가 다르면 실수 연산 및 이후 학습 결과가 달라질 수 있다. 기존 결과를 덮어쓰지 말고
원인·새 버전·차이를 기록한다. 재현 불일치 때 검증 조건을 임의로 완화하지 않는다.

과거 비선형 readout 전체 재생/부분 재학습은 **당시 소스 버전**에서 실행:

```powershell
git worktree add ../Flying-readout-replay 9c9f9148ec4fc5c7a0e2c45ded05da9e526d92d8
cd ../Flying-readout-replay
$env:PYTHONPATH = "src"
..\Flying\.venv\Scripts\python.exe scripts/verify_phase5_readout.py results/phase5_readout
```

기존 verifier는 source hash까지 검사하므로 최신 코드에서 과거 manifest를 검증하면
코드 변경 때문에 실패할 수 있다. 이를 결과 손상과 혼동하지 않는다.
전체 재학습은 같은 worktree에서 `scripts/run_phase5_readout.py --output outputs/readout_codex`를 실행하고
그 새 디렉터리를 verifier에 전달한다. 구버전의 source hash 검증 정책은 그대로 유지한다.

## Codex용 작업 요청문

> 이 저장소의 README, docs/phase5-prefix-results.md, docs/phase5-prefix-protocol.md,
> docs/codex-handoff-2026-09-25.md부터 읽고 현재 main 커밋과 변경 파일을 확인해라.
> 전체 pytest를 실행하고, 새 출력 디렉터리에 prefix weighting 실험을 재학습·재생해 보고서와 비교해라.
> 차이가 있으면 패키지/BLAS/원자료/코드 해시를 확인하고 기존 기록은 보존해라.
> 다음 실험은 결과 보고서의 해석에 따라 사전 고정된 독립 확인으로 설계하며,
> 시드·초기화·길이 중 최고 결과를 선택해서 성공으로 보고하지 마라.
> Whole-brain은 자동으로 시작하지 말고 먼저 부분망에서 이득·대조군·용량/견고성을 확인해라.
> 장기 실행이 필요하면 조건별 체크포인트와 재개를 먼저 구현하고,
> 실제 실행한 것, 코드만 구현한 것, 환경 때문에 막힌 것을 구분해 보고해라.
