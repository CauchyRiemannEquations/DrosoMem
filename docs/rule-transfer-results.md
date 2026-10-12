# M11 matched rule-coefficient continuation — verified primary/secondary FAIL

Date: 2026-10-13. **All 216 fits and independent validations complete.**
The registered rule-continuation advantage and secondary fine-wiring hypotheses
both **FAIL**. Technical/task checks pass; low acquisition is a scientific
failure under this learner budget, not an invalid assay or unfinished execution.
[Protocol](rule-transfer-protocol.md), [prior audit](rule-transfer-prior-audit-m11.md),
[config](../configs/rule_transfer.json),
[independent checks](../results/rule_transfer_v1/main_validation/checks.json),
[final audit](../results/rule_transfer_v1/final_audit/audit.json).

## 1. Question, registered scope and actual learning

Does a fixed source-derived circuit plus an equally fitted affine head show
material held-out affine-rule continuation over a simple three-symbol delay?
If so, does that effect exceed the complete matched partial rewiring bank?
Both discovery and confirmation must satisfy all prospective gates. We did not
change them or select another family/head after failure.

Baseline `5467811412cb36a0a56f6137e98f93b9e31f8206` completed M10 READY.
Protocol/config commit `68b49d37e79a9a00c95f00baaf44c73d61147c6f` preceded
new M11 neural trajectories/fits. Nine guards and smoke used
`c6406401b6ae07c9301d827e5d0d1310c0a6399f`. Main source:
`87b00311bc30d5044d6ea65dc58a1ac5a3fb1b81`; independent main source:
`e106de6a9d1b0af4f09cbe2675b856fd0c5c9a90`.
All new evidence is in `results/rule_transfer_v1/`.

Reuse the full mod-11 M10 splits (275 train, 275 seen-rule new starts, 440
held-out coefficients), 3 input symbols and 8 target outputs. Models are the
real partial s701 circuit, all ten M8 conditional partial rewires and a canonical
three-slot one-hot shift reference. Every model runs all affine/iid/shuffled
families and all six fresh K11 input maps 2710001..2710006: **216 heads**.
First three maps are discovery, last three confirmation; the exhaustive task
arrays are shared, so these are interface variations, not independent datasets.

All source/null weights remain fixed: 686 scalar states, 512 KC input pool,
51 active cells per symbol at .5, incoming-L1 gain .9, drive .6, zero direct carry,
same-step KC→MBON scheduling, 48 original MBON observations. The matched bank
preserves degrees/roles, both signed raw weight multisets, normalization factors
and current pathway coefficients. No new graphs or whole-brain run.
The engineered shift has 686 padded scalar slots, 33 active one-hot coordinates,
48 observations, a canonical seed-independent interface and different dynamics.
Its six fits per family are repeated paired references, not six independent
delay models or an anatomical null population.

Only an affine head is fitted: 528 coefficients +11 intercepts =539 nominal
learned scalars, plus 96 train-only mean/SD moments. Alpha1, SD floor1e-5,
2200 unweighted next-symbol training rows, then freeze before EVERY evaluation.
Official fits comprise116424 affine scalars and20736 scaler moments; independent
SVD refits are verification, not another set of independent experimental models.
Recurrent weights, encoders, rules and templates learn zero parameters.

## 2. Confirmed computations and failed performance gates

The primary excludes a=10's known period-2 shortcut and uses all330 held-out
episodes for multipliers7/8/9. The full440 and every rule/intercept remain
reported. Scores below pool counts within each cohort's three paired maps;
the denominators do not imply990 distinct biological or task instances.

| Affine endpoint | Discovery real | Confirmation real | Registered real requirement |
| --- | ---: | ---: | ---: |
| Training teacher accuracy | 1277/6600 =19.35% | 1245/6600 =18.86% | >=50% |
| First next, a=7..9 | 131/990 =13.23% | 138/990 =13.94% | >=25% |
| All-8 teacher accuracy, a=7..9 | 1319/7920 =16.65% | 1217/7920 =15.37% | Diagnostic; cannot replace rollout |
| Full autonomous completion, a=7..9 | 5/990 =0.51% | 10/990 =1.01% | >=10% |
| Mean exact autonomous prefix, a=7..9 | 79/330 =0.239 symbols | 257/990 =0.260 symbols | Diagnostic; cannot replace completion |
| Real−shift completion | +0.505pp | +1.010pp | >=5pp |

Every primary gate fails in BOTH cohorts, including adequate training acquisition.
The shift's first/teacher accuracy is13.33% for a=7..9, mean autonomous prefix
1/3, and full completion0. A tiny positive real completion difference against
that zero baseline does not satisfy a material advantage or acquired-rule test.

| Real completion by held-out multiplier | Discovery | Confirmation | Prospective interpretation |
| --- | ---: | ---: | --- |
| a=7 | 0/330 | 0/330 | Fails >=5% per-rule gate |
| a=8 | 0/330 | 0/330 | Fails >=5% per-rule gate |
| a=9 | 5/330 (1.52%) | 10/330 (3.03%) | Both fail >=5% gate |
| a=10, excluded period2 | 0/330 | 0/330 | Retained diagnostic; never rescues primary |

On ALL440 coefficients, real completion is5/1320 (0.38%)/10/1320 (0.76%).
Seen-rule new-start completion is1/825 (0.12%)/2/825 (0.24%). Training completion
is9/825 (1.09%)/8/825 (0.97%); the shift completes16% training and4% seen-rule
instances. Training teacher accuracy and teacher-forced performance are therefore
distinct from stable target-free continuation. No generated symbol is replaced
with a true target after an error.

| Secondary conditional fine-wiring comparison, a=7..9 | Discovery | Confirmation |
| --- | ---: | ---: |
| Ten-null mean full completion | 31/2475 =1.253% | 43/4950 =0.869% |
| Null minimum–maximum | 0.505–1.818% | 0.404–1.616% |
| Real−null mean | −37/4950 =−0.747pp | 7/4950 =+0.141pp |
| Real strictly above EVERY null | false | false |
| Local >=5pp/strict-order gate | FAIL | FAIL |

The secondary requires primary support as well as both local wiring gates;
all fail. Do not select the few nulls that real exceeds or reinterpret a small
confirmation-only mean gap as established fine-wiring benefit.
[Exact summary](../results/rule_transfer_v1/main/summary.json),
[all model/map/family/split/rule counts](../results/rule_transfer_v1/main/metrics.json),
[paired means/medians/SD/effects](../results/rule_transfer_v1/main/paired-descriptions.json).
The latter are finite computational descriptions, not p-values or biological
uncertainty; shift repeats and shared cycles do not supply independent replication.

![M11 equal-budget training, first-next prediction and autonomous completion](../results/rule_transfer_v1/report/rule-transfer.png)

Bars average paired conditions; null bars include all ten graphs/all maps.
Dots are individual computational conditions. Dashed lines show only the named
real absolute gates; primary also needs a5pp completion difference and all
nonperiod2 multiplier gates. Full axes prevent a tiny positive gap from looking
like acquired rule learning. PNG and SVG are archived with the exact plotted
Fractions, input manifest and source version.

## 3. Fixed-copy and negative-family references

As predicted before outcomes, fixed lag2 completes every a=10 episode110/110,
and hence110/440 (25%) of the full coefficient task, with zero completion on
a=7..9. Fixed lag3 completes none. The simple rule-copy success is not inferred
mathematical learning; it confirms why period2 was excluded prospectively.
[All fixed-copy counts](../results/rule_transfer_v1/main/fixed-lag-reference-metrics.json).

The M10 history-3 table and formula oracle are authenticated PRIOR references:
history-3 completes all seen-rule starts but no held-out coefficients; the
privileged supplied-form oracle completes all440. They were not newly fitted
or treated as equal-knowledge/parameter neural models in M11.
[All prior controls](../results/rule_transfer_v1/main/m10-prior-control-metrics.json).

| Real source, full440 negative-family tests | First next discovery/confirmation | Teacher discovery/confirmation | Full completion |
| --- | ---: | ---: | ---: |
| Iid | 119/1320 (9.02%) /130/1320 (9.85%) | 9.32% /9.48% | 0 in both |
| Within-episode shuffle | 137/1320 (10.38%) /146/1320 (11.06%) | 8.14% /8.23% | 0 in both |

They retain the same split sizes/exposure and are diagnostic, not alternate
successful primary families. Shuffled symbols preserve multisets/short-cycle
repetition and are not iid. On negative families, the a/b grouping labels the
originating affine episode, not a true equation governing the shuffled/iid inputs.
Every graph/map/negative-family outcome remains in the complete metrics.

## 4. What the failure means, and what it does not

This fixed reservoir plus539-scalar affine readout does not meet the declared
trained-rule acquisition or held-out continuation requirements. Low training
accuracy shows that the selected representation/decoder budget already struggles
to acquire the supervised task. The experiment cannot cleanly attribute failure
only to generalization after successful acquisition. It is a valid negative
result for this finite learner/interface/task budget, not a general impossibility
of rule learning in all fly-derived models or finite-state systems.

Past inputs being decodable is insufficient for predicting a new affine rule
and feeding generated outputs back stably. M9's100% historical tracking and
failed fine-wiring advantage remain unchanged. M10 READY only established usable
task checks and an analytic reference. Neither implied that these trained heads
would succeed. No nonlinear head, extra epochs/data, new multiplier split,
successful seed selection or post-outcome threshold changes were attempted.

The ten-null bank shares substantial wiring/current paths and is not fully mixed
uniform anatomical sampling. Shift uses a different input representation and
dynamics despite matched scalar/observation/head budgets; only the real/null
comparison conditions the detailed source pathway. The model is a partial fixed
rate circuit, not a calibrated whole brain. New coefficients are within one
known affine family/modulus, not unseen mathematical families. Internal synaptic
learning, living-fly arithmetic, human mathematical learning, storage location
and formal capacity are not established. P3 was not performed.

## 5. Independent verification, failures and reproducibility

All216 cases independently replay and refit. The verifier reconstructs the
partial source/roles from pinned edge/annotation files, regenerates ALL10 M8
partial graphs from their swap seeds and checks exact CSR arrays/invariants.
It independently generates K11 codes and M10 inputs/labels, manually implements
episode-reset source and shift dynamics, and fits ridge using augmented SVD,
without importing M11 model/trajectory/fitter/scorer/summary algorithms.

Every427680 complete-state teacher/autonomous trajectory digest matches;
observed feature digests, all predictions and per-rule counts match exactly.
Mean/scale/coefficients and all score arrays meet the fixed1e-9 checks; stable
1e-10 ties were registered before outcomes. Autonomous replay generates from
the independent SVD head and checks its predictions against the authenticated
recorded head at every state; inputs never come from targets. All exact gates,
paired descriptions, fixed-copy controls and complete case coverage match.
The main archive contains213840 teacher and213840 autonomous episodes (same
conditions), comprising4,490,640 state updates; these are execution counts.

Nine synthetic safeguards and3-case smoke/SVD replay also pass; smoke198
trajectories are excluded from main and are not a confirmation cohort.
No technical execution/verification failed. The
[incomplete-attempt registry](../results/rule_transfer_v1/known-incomplete-attempts.json)
is empty. Scientific primary/secondary FAIL is retained as the completed outcome.

| Stage | Attempt seconds | Sampled aggregate process-tree RSS |
| --- | ---: | ---: |
| Smoke | 4.182 | See recorded tree resources |
| Independent smoke | 5.201 | See recorded tree resources |
| Main216 fits | 157.387 | 624312320 bytes |
| Independent main216 refits | 250.063 | 599678976 bytes |

Four workers/one numeric thread; all stages meet7200s/4GiB and1GiB artifact caps.
Tree sampling every50ms sums live RSS, can double-count shared pages and is not
an exact OS aggregate peak; per-process OS peaks/environment also remain.
Pool timings omit source capture/bank reconstruction; attempt timings include
those after the recorded attempt starts, with source capture before that start.
Main artifacts before final seal are324938013 bytes; validation21929462 bytes.

Baseline protects47049 prior result Git identities and47747 old scientific paths.
All8135 materialized files, including7441 old results, rehash to their original
Git blobs. The39608 sparse-excluded old results are protected by Git identities,
not claimed freshly replayed. All old source/protocol/data/test/results/failures
remain unchanged, including P2/M7 FAIL, M1 invalid, M4 INFEASIBLE, M8/M10 READY
and M9 FAIL. Only four overview documents receive the dated current handoff.

Main manifest SHA256:
`7d15d388d1ebe5b667e1f0304bd3f3e12afe214283311c51676e372c6905edf1`.
For reproduction, use its recorded source/environment and new directories:

```powershell
$env:PYTHONPATH='src;scripts'
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
python scripts/run_rule_transfer_guards.py --out outputs/m11_replay_guards
python scripts/rule_transfer.py --out outputs/m11_replay_smoke --smoke
python scripts/verify_rule_transfer.py --result outputs/m11_replay_smoke --out outputs/m11_replay_smoke_validation --smoke
python scripts/rule_transfer.py --out outputs/m11_replay_main
python scripts/verify_rule_transfer.py --result outputs/m11_replay_main --out outputs/m11_replay_validation
```

The bounded M10/M11 task-design/performance comparison is closed; no specified
experiment or verification remains. Next roadmap priority is responsible
synthesis/archive and a claim/evidence/limitation matrix for M1–M11. No DOI,
publication, new performance preregistration or automatic successful-head search
is claimed. Broader rule-family/internal-learning studies remain separate,
unregistered extensions, and cannot retrospectively repair this failure.

## 비전공자를 위한 요약

새로운 수열 규칙을 적용하는지 실제 배선망·재배선 대조망10개·단순 지연
모델을 같은 출력층 학습 예산으로 비교했습니다. 사전등록한216개 조건을
모두 실행하고, 별도 구현이 출력층과 약42.8만개 상태 궤적을 다시 계산했습니다.
예측과 판정이 일치했지만, 규칙 전이와 실제 배선 우위 기준은 모두 실패했습니다.

쉬운 두 칸 반복 문제를 제외하면 실제 배선망의 첫 정답률은13.23%·13.94%,
다음8기호 완주율은0.51%·1.01%였습니다. 학습한 수열의 정답 입력 예측도
약19%로 충분하지 않았습니다. 따라서 성공적으로 익힌 규칙을 새로운
사례에 옮기지 못했다고만 해석하기보다, 이번 표현·출력층 예산의 학습
단계부터 부족했다는 한계를 함께 기록합니다.

과거 숫자의 흔적을 읽을 수 있다는 기존 결과와 새로운 규칙을 학습·적용하는
능력은 다릅니다. 이번 실패를 보존했고, 성공할 때까지 모델·시드·기준을
바꾸는 추가 탐색은 하지 않았습니다.
