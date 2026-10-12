# M10 rule-transfer task design — READY within the registered checks

Date: 2026-10-13. **Design preflight READY; no neural transfer experiment.**
[Protocol](rule-transfer-design-protocol.md), [prior audit](rule-transfer-prior-audit.md),
[config](../configs/rule_transfer_design.json),
[independent checks](../results/rule_transfer_design_v1/independent_validation/checks.json),
[final artifact audit](../results/rule_transfer_design_v1/final_audit/audit.json).

## 1. Question and completed scope

Can a bounded mod-11 affine sequence task separate trained-instance recall,
new phases of seen rules and held-out rule coefficients, without shared training
prefixes/cycles solving the coefficient test? All registered task checks and
independent reconstruction pass. This prepares a benchmark; it does not show
that DrosoMem learns rules, internally changes synapses or outperforms a delay.

Baseline main `81ad81f25c445c8964599d35f52e0bef51ce4755` ended M9 with a valid
negative wiring result. Protocol/config `61b6139028720b708463e442c149cd74f8d2bd2b`
preceded new control outcomes. Producer source `7805fda8652dee3a54bc372b35c891e1613afbe3`;
independent source `11f52f314719346291603d493f126fe959e7de5a` is recorded in its
source artifact.
No prior experiment is rerun or reclassified. New artifacts use only
`results/rule_transfer_design_v1/`.

The task is `x[t+1]=(a*x[t]+b) mod 11`, with 3 prefix inputs and 8 continuation
targets. All b=0..10; nonfixed starts are exhaustive. Train multipliers 2..6
and offsets y=1..5 give 275 episodes. New starts y=6..10 under those same rules
give 275 instance tests. Multipliers 7..10 and all y=1..10 give 440 coefficient
tests. This is one affine family/modulus, not transfer to arbitrary rule families.

Learner NPZs expose only prefixes and scorer/training targets. Equation parameters,
fixed points and start assignments are in separate audit metadata. Each family
supplies 2200 within-episode training transitions. Tables learn frequency/suffix
counts; no neural head, recurrent weight, encoder or template is fitted.
All 7 controls and all 3 families/splits are retained: 63 predictor evaluations.

## 2. Confirmed task properties and raw control outcomes

All 990 structured initial prefixes are unique and identify the intended rule.
Training versus coefficient test overlaps are zero for initial 3-symbol prefixes,
every teacher prediction's 3-symbol window and directed cycles up to rotation.
There are 88 distinct training cycles and 99 coefficient-test cycles.
Training and instance tests share all 88 cycles; that overlap is expected and
prevents their success from being interpreted as unseen-coefficient transfer.

| Predictor on held-out coefficients | First next symbol | All-8 teacher accuracy | Autonomous full completion |
| --- | ---: | ---: | ---: |
| Frequency | 40/440 (9.09%) | 320/3520 (9.09%) | 0/440 |
| Current copy | 0/440 | 0/3520 | 0/440 |
| Repeated modular difference | 0/440 | 0/3520 | 0/440 |
| History-1 table | 44/440 (10%) | 352/3520 (10%) | 0/440 |
| History-2 table | 0/440 | 0/3520 | 0/440 |
| History-3 finite-state table | 0/440 | 0/3520 | 0/440 |
| Known-form analytic oracle | 440/440 (100%) | 3520/3520 (100%) | 440/440 (100%) |

History-3 completes every training AND seen-rule instance episode (275/275 each),
but none of the held-out coefficient episodes. This directly illustrates why
new starts of the same cycle family are insufficient evidence for rule transfer.
The supplied-form oracle completes all 990 structured episodes, mean prefix 8.
Its algebraic solvability was known before outcomes, not learned from training.

| Negative coefficient-test family, same 440 episodes | Oracle first accuracy | Teacher accuracy | Full completion | Unidentifiable initial prefixes |
| --- | ---: | ---: | ---: | ---: |
| Independent iid inputs | 43/440 (9.77%) | 326/3520 (9.26%) | 0/440 | 38 |
| Within-episode shuffled symbols | 98/440 (22.27%) | 707/3520 (20.09%) | 0/440 | 59 |

Both oracle negative first scores are below the preregistered 25% ceiling.
The shuffled task preserves multisets and short-cycle repetition; it is not
iid and remains more predictable. No example/seed was removed for accidental
local rule fits or unidentifiable prefixes. All other negative-family predictors,
exact rational rates and autonomous prefix lengths are in
[every metric](../results/rule_transfer_design_v1/main/metrics.json).

Storage differs across controls. Affine history tables have 11/110/550 suffix
cells and 110/550/550 nonzero symbol-count entries for orders 1/2/3, plus 11
frequency counts. Iid/shuffled larger tables and their budgets are retained in
[storage accounting](../results/rule_transfer_design_v1/main/storage-budgets.json).
They are not equal-parameter trained neural comparators. Current/difference/
formula controls fit zero parameters; the oracle is privileged with the family
formula and modulus. It infers a,b only from the initial prefix and keeps them
fixed in teacher and target-free rollout scoring.

## 3. Failures, limits and meaning

No registered task check, technical execution or independent check failed;
the [incomplete-attempt registry](../results/rule_transfer_design_v1/known-incomplete-attempts.json)
is empty. There was no success-metric adjustment after outcomes. M9's ceiling
and failed real-wiring advantage, P2/M7 FAIL, M1 invalid, M4 INFEASIBLE and
M8 READY remain unchanged. No new neural model was run.

READY applies to these finite checks. It does not exclude every possible trivial
predictor. In particular a=10 yields period-2 episodes: copying the input two
sequence positions back solves those 110/440 coefficient tests mechanically.
That algebraic limitation follows from the selected rule set. Such a fixed lag
copy was not a registered executed control and is not added post hoc to the
63 evaluations. A future neural protocol must include explicit lag-copy/delay
references and rule-stratified outcomes; an effect driven only by a=10 cannot
support a broad rule-transfer claim. More generally cycle lengths need not be
novel and the finite coefficient split changes conditional next-symbol support.

Oracle 100% demonstrates a usable family-specific reference, not learning,
biological mathematics or human reasoning. Table failure reflects this split
and predictor class, not an impossibility result for finite-state systems.
Episodes are correlated rules/phases, not biological replicates. No p-value or
formal memory-capacity interpretation follows. Training recall, teacher prediction,
target-free continuation and held-out coefficients remain distinct endpoints.

## 4. Independent verification, preservation and next work

The verifier rebuilds structured symbols with fixed-point enumeration and powers
rather than the producer's recurrence; it finds oracle rules by brute-force
coefficient enumeration rather than modular inversion. Training counts, all
predictions, every exact metric and split/cycle gate are reconstructed without
importing producer task/predictor/scoring algorithms. All 2970 structured/iid/
shuffled episodes, 144 data/prediction arrays and 9 identifiability arrays agree.
All 63 evaluations and training tables/storage counts agree exactly.
Seven synthetic safeguards cover target-free rollout, identifiable/nonidentifiable
prefixes, cycle rotation, episode boundaries, ties, independent replay and
import separation. Complete checksum coverage is retained for both stages.

| Stage | Wall seconds | OS process peak bytes |
| --- | ---: | ---: |
| Synthetic guards | 1.456 | 41525248 (parent process; test child excluded) |
| Producer | 34.021 | 75976704 |
| Independent reconstruction | 17.689 | 75108352 |

Producer/verifier are below the fixed 300-second/1-GiB per-process caps. RSS is
sampled every 50 ms and OS peaks/environment are archived. Their times include
the materialized-history checks. All 47,001 prior result Git identities and
47,689 protected baseline paths remain unchanged; 8,077 materialized files,
including 7,393 old results, also pass Git blob rehashing. The 39,608 sparse-
excluded historical results are protected by Git identity, not freshly replayed.

Main manifest SHA256:
`40b857dc78b20a8e6965903e309cbbadf16bdea19b4df4b7343c5b17d897772d`.
Use recorded source/environment and new output directories for reproduction:

```powershell
$env:PYTHONPATH='src;scripts'
python scripts/run_rule_transfer_design_guards.py --out outputs/m10_replay_guards
python scripts/rule_transfer_design.py --out outputs/m10_replay_main
python scripts/verify_rule_transfer_design.py --result outputs/m10_replay_main --out outputs/m10_replay_validation
```

This design phase is closed. Next is a **separate performance preregistration**
for fitted then frozen held-out coefficient continuation, provisionally M11.
Freeze the K11 input interface, source/control circuits and seeds before fits;
use the proposed 48-coordinate/539-affine-parameter/2200-row budget, matched
3-slot shift reference and explicit lag-copy controls. Specify exact validity/
advantage gates and rule-stratified reporting before outcomes. Validate target-
free rollout and independently replay complete states/refits. It remains
unregistered/unexecuted; no neural transfer success or wiring advantage is claimed.

## 비전공자를 위한 요약

기존 M9에서는 실제 배선과 대조 배선이 모두 과거 숫자 추적에서 만점을
받아 실제 배선의 우위를 확인하지 못했습니다. 이번에는 그 결과를 보존하고,
숫자열 암기와 새로운 규칙 적용을 구별하는 시험 과제를 만들었습니다.

같은 규칙의 시작점만 바꾸면 최근 3기호를 외운 표도 전부 맞췄습니다.
학습에 없던 규칙 계수로 바꾸면 그 표는 완주하지 못했고, 수열의 공식이
주어진 계산 대조군은 440개 모두 다음 8기호를 맞췄습니다. 독립 검증도
통과해 과제 설계는 READY입니다. 반복 주기 2인 일부 문제는 단순 지연
복사로 풀 수 있으므로 다음 모델 비교에는 그 대조도 포함해야 합니다.

초파리 연결망 모델이 새로운 규칙을 학습했다는 결과는 아직 없습니다.
다음 작업은 같은 학습 예산의 모델 비교와 판정 기준을 사전 등록하는 것입니다.
