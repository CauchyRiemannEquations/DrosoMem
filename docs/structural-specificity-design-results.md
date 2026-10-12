# M8: structural-specificity control construction — results and synthesis

Date: 2026-10-12. **Registered primary READY; secondary READY.** This closes
the finite construction-feasibility preflight. It does not close the roadmap's
remaining structural-performance question or establish a memory advantage.
[Protocol](structural-specificity-design-protocol.md),
[prior audit](structural-specificity-design-prior-audit.md),
[config](../configs/structural_specificity_design.json).

## 1. What was asked?

Can we build genuinely different recurrent controls while keeping the source
node degrees/roles, both directions' raw weight multisets, row-normalization
factors and the exact current KC→MBON block? Can a separate engineered delay
reference retain current and lag2 channels within the scalar state/observation/
affine-head budget? We audited this before another neural comparison.

Baseline main `4185b214ae831499b556764fa288295b1dbd917b`. Protocol/config commit
`8f36a891f3b572aabc5e6865e8683cc5cbcdf1a4` precedes new graph/prototype outcomes.
Main and independent verification source is
`d9dd96dcb8be10bd2c31d7a3213edc08ba6b1800`. Namespace
`results/structural_specificity_design_v1/`; existing results are not overwritten.

P2 allowed changes in incoming strength and current coefficients under a common
normalization rule. M8 uses a stricter conditional construction: exchange only
delayed edges of equal signed RAW weight and equal pre/post roles, locking ALL
KC→MBON edges. Both incoming and outgoing raw weight multisets remain exact,
so incoming-L1 gain0.9 factors and protected coefficients also remain exact.
This is a different bounded control question, not a repair of failed P2.

Ten partial graph seeds1840001–1840010, three whole seeds1841001–1841003.
Start each independently from the original raw graph; request5 accepted swaps
per delayed edge partial,1 per delayed edge whole;200-proposal-per-request cap.
The fixed proposal pool excludes groups without at least two sources/targets.
Partial maps1810001–1810006; whole firstthree are paired counterparts.
No input training/test stream, source-state trajectory or learned head is made.

## 2. What was confirmed?

Every one of13 graphs satisfies all invariants and the full swap budget. All
ten/three raw graph hashes are distinct within their level. Each keeps at most
90% of original delayed edge locations, using the predeclared exact integer
gate. The independently regenerated CSR data/indices/pointers match exactly.

| Structural quantity | Partial legacy5 | Whole brain5 secondary |
| --- | ---: | ---: |
| Source neurons / edges | 686 /3309 | 138639 /2700513 |
| Protected current edges | 1474 | 21438 |
| Eligible delayed edges | 1835 | 2679075 |
| Proposal-pool edges | 602 | 2678113 |
| Fixed graph members | 10 | 3 |
| Accepted swaps per member | 9175 | 2679075 |
| Attempts per member | 14389–14647 | 2717603–2717839 |
| Common delayed edge locations | 1399–1425 | 375317–376613 |
| Delayed overlap | 76.23978–77.65668% | 14.00920–14.05758% |
| Changed delayed locations | **22.34332–23.76022%** | **85.94242–85.99080%** |
| Registered construction/mapping rule | READY | READY |

Exact checks cover labeled node in/out degrees, directed role-block counts,
per-neuron incoming/outgoing signed raw weight multisets, source coordinate IDs,
no self/duplicate/zero edges, incoming absolute strengths, and protected raw
AND normalized coefficients. Nonempty normalized rows retain gain0.9 within
the predeclared1e-14 check. [Every member audit](../results/structural_specificity_design_v1/main/structural-audits.json),
[CSV summary](../results/structural_specificity_design_v1/main/graph-summary.csv),
[registered summary](../results/structural_specificity_design_v1/main/summary.json).

The partial proposal pool is only602 of1835 delayed edges. Forced/single-group
structure remains; ten constructions are not ten uniformly sampled graphs
from the entire degree-constrained space. The finite chain is not claimed fully
mixed. Normalized outgoing weight multisets and delayed paths/cycle organization
are not constrained. No pure cycle effect is isolated.

### Separate engineered current-and-lag2 reference

Each ofnine source input maps has ten analytically computed zero-history
current response codes on the same48 MBON labels. Ascending numeric root IDs
fix the first24 current and last24 past channels before rank outcomes. No
favorable column selection follows the result. Both halves, after appending
the intercept column, have numerical row rank10 in all18 checks.

Smallest/largest singular ratios are.00062802–.00182374 partial and
.00011634–.00031725 whole, far above the fixed1e-12 cutoff. The independent
verifier uses a different SVD driver and agrees on every rank and prototype.
Full row rank is a sufficient numerical affine-code certificate; it is not
held-out alpha1 ridge accuracy. **No affine map or decoder was fitted.**

The reference uses512 instantaneous stimulus slots,24 current slots, and
three24-coordinate past-code banks:608 active coordinates. Pad78/138031 idle
coordinates to the original686/138639 scalar state budget. Observe current24
plus the two-step-delayed24, restored to the original48 coordinate order.
The48 unit shift edges form an acyclic transition D with24 edges in D² and
zero in D³. This algebra verifies lag2 routing and finite history without a
source-neural input stream. Current channels are separately reserved.

It inherits the fixed source instantaneous KC→MBON front end and original
input codes. Its abstract slots, unit shifts, degrees, roles, normalization
and dynamics are different from source neurons. Padding/observation labels
do not turn it into a matched anatomical graph or M4's impossible all-lag
same-node DAG. It is a transparent generic-delay candidate conditional on
that source front end, not a biology result or proof that all fly wiring is
irrelevant. [Partial layout](../results/structural_specificity_design_v1/main/definitions/legacy5/definition.json),
[whole layout](../results/structural_specificity_design_v1/main/definitions/brain5/definition.json).

## 3. What failed or remains unanswered?

The finite design rule passes; **no real/null memory performance was tested**.
Matched zero-history current coefficients do not guarantee equal current-symbol
accuracy under history. Finite-sample head fitting, lag2 tracking, wider-lag
decoding, observation-noise behavior and material wiring advantage remain
unmeasured here. A later performance protocol must specify matched graph-specific
fitting then frozen held-out evaluation, input/current access checks, finite
ensemble/seeds/pairing, metrics/thresholds and independent replay before outcomes.

The first smoke stopped at its source loader: current IDs were converted to
integers but indexed with original string IDs. No new rewired graph or prototype
was generated. Exact int64 root conversion fixes representation only, including
18-digit IDs without float conversion. No source data, role, mapping value,
seed, budget, coordinate split or criterion changed. The interrupted parent
and nested definition manifests/failures/resources are unchanged in
[first smoke](../results/structural_specificity_design_v1/smoke/manifest.json)
and the [registry](../results/structural_specificity_design_v1/known-incomplete-attempts.json).
These are two records of ONE interrupted smoke, not two independent experiments.

Initial8 synthetic safeguards passed on their recorded source. Add the exact
root-ID integration safeguard; all9 pass in a fresh source-authenticated guard
retry, then smoke retry and independent smoke pass before main. No main or
independent verification failed. Smoke reuses the first fixed graph seed/full
budget; its identical realization is not an extra ensemble member and its
scientific READY endpoint is null.

## 4. What can now be said about DrosoMem?

A useful bounded structural comparison is constructible without changing
current input coefficients or row-normalization factors. An explicit simpler
history system also fits the chosen scalar state/48-observation/490-head budget.
This permits designing a fairer conditional performance comparison; it does
not establish that real wiring helps memory, that the controls perform equally,
or that biological memory/learning/storage has been identified.

M4's strict degree/all-lag DAG obstruction stays INFEASIBLE. P2's material
real-wiring gates, M7's artificial-noise gate and local biological-learning
negatives stay FAIL; M1 stays assay-invalid. Existing source-derived reservoir
past-input decodability and clean frozen-readout tracking claims stay as sealed.
No physiological/spiking/plasticity P3, formal capacity or whole-brain superiority
claim is added. **M8 preflight is closed; the roadmap's structural-performance
priority remains open.** Working M9 is only the next proposed protocol, not
registered/executed. Mathematical rule transfer remains later.

## Verification, provenance and reproduction

[Independent main checks](../results/structural_specificity_design_v1/main_validation/checks.json)
rebuild pinned raw source/annotation data, regenerate every13 swap sequence and
raw graph exactly, independently check both multisets/degrees/roles/current
coefficients/normalization, rebuild nine root-based input maps and all18 rank
checks with gesvd, and verify the padded delay layouts/nilpotency and exact
READY conjunction. It imports no main M8 generator, normalizer, prototype,
rank/layout or summary helper. New learned heads0; source neural trajectories0.

| Stage | Attempt wall seconds | Resource peak |
| --- | ---: | ---: |
| Baseline audit | 38.489 | Per-process resource record |
| Initial8 safeguards | 21.355 | Per-process resource record |
| Interrupted first smoke | 14.130 | Per-process resource record; retained |
| Valid9 safeguard retry | 24.687 | Per-process resource record |
| Smoke retry | 20.880 | Per-process143933440 bytes; aggregate archived |
| Independent smoke | 28.813 | Per-process578633728 bytes; aggregate archived |
| Main | 208.574 | Sampled aggregate962433024 bytes |
| Independent main | 267.251 | Sampled aggregate1290555392 bytes |

One graph worker/one numerical thread,7200s/4GiB fixed stage budget.50ms tree
sampling sums RSS and may count shared pages repeatedly. Initial source capture
before main/verifier attempts is outside their reported attempt wall time;
OS lifetime process peaks and environment are separately recorded.

[Baseline audit](../results/structural_specificity_design_v1/baseline_audit/audit.json)
protects45245 prior result Git identities/45913 baseline paths. Materialized
files authenticate;39608 sparse-excluded old results are protected by identity,
not claimed freshly replayed. Old science and failures remain intact; only
four overview documents are updated while dated snapshots stay.
[Final artifact audit](../results/structural_specificity_design_v1/final_audit/audit.json)
checks manifest coverage/checksums, recorded Git sources, finite arrays, all
CSV values, exact overlap/readiness, documents and retained failure registry.

Main manifest SHA256:
`528d420d16b55ccc99d8c180fdd07886cab1235d14b6ce1043bb381f2a0780eb`.
Use the captured environment and pinned partial/raw/whole cache hashes from
the source records. From repo root with new output paths and source unchanged
during each stage:

```powershell
$env:PYTHONPATH='src;scripts'
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
python scripts/run_specificity_design_guards.py --out outputs/m8_replay_guards
python scripts/structural_specificity_design.py --out outputs/m8_replay_smoke --smoke
python scripts/verify_structural_specificity_design.py --result outputs/m8_replay_smoke --out outputs/m8_replay_smoke_validation
python scripts/structural_specificity_design.py --out outputs/m8_replay_main
python scripts/verify_structural_specificity_design.py --result outputs/m8_replay_main --out outputs/m8_replay_main_validation
```

## 비전공자를 위한 한국어 요약

이번에는 기억 점수를 겨루기 전에 공정한 비교 상대를 만들 수 있는지
확인했습니다. 현재 숫자가 들어오는 길과 뉴런별 연결 수·가중치를
보존하면서 과거 정보를 전달하는 연결 위치를 바꾼 대조망13개를
만들었고, 별도 검증기가 모두 똑같이 재구성했습니다.

현재 숫자와 두 단계 전 숫자용 칸을 따로 둔 단순 지연 장치도 같은
상태 수·관측 수 예산 안에 구성했습니다. 이 장치는 인공 설계이며
초파리 뉴런 자체를 그대로 구현한 것은 아닙니다. 설계가 준비됐다는
뜻이지 실제 초파리 연결망이 더 잘 기억한다는 결과는 아닙니다.
다음에는 별도의 사전 설계 후 이 대조들에서 실제 해독·교체 추적
성능을 비교합니다. 기존 P2·M7 실패는 그대로 유지합니다.
