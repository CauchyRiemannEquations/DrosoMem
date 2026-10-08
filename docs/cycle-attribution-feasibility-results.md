# M4: exact cycle-attribution control feasibility — results and synthesis

Date: 2026-10-08. **Registered feasibility outcome: INFEASIBLE**, with independent
validation complete. This is a negative design-feasibility result, not a test
that cycles improve or impair decoding. Nine fixed mapping cases and90 symbol
cells are closed. No new source-graph neural trajectories or readout fits were
executed. [Protocol](cycle-attribution-feasibility-protocol.md),
[config](../configs/cycle_attribution_feasibility.json).

## 1. What was asked?

Can an acyclic control retain the same labeled neurons, per-neuron degrees,
role-block totals, weight constraints, current KC→MBON block and **all-lag
structural input-to-observed dependency-walk support**? M3 showed feedforward
sufficiency but its mask also removed degrees, weights and paths. We audited
the strict proposed control before running another decoding comparison.

Baseline main `5afe8f28cc1d820973b9156c20501a0d31c88a26`; namespace
`results/cycle_attribution_v1/`. Initial protocol/config commit `a1def337`,
universal same-node witness clarification `2c7a5f593c22909652a1374c9338571656411855`,
both before new structural outcomes. Main source
`c9e96e936280c5e622ac45432bd2557150d9f8ff`. Six fixed mapping seeds1410001–1410006,
blocks1400001–1400006; whole secondary firstthree, paired by input source IDs.
Each of10 symbols stimulates51 of the same512 KC input neurons; same48 observed
MBON IDs. These are computational mapping probes, not biological replicates.
No iid train/test stream or autonomous sequence was generated in M4.

Use the original gain0.9 incoming-L1 normalized signed graph and the exact
sealed M3 rank DAG. Drive0.6, direct carry0, mbon_after_kc; current KC→MBON edges
have delay0 and all other edges delay1. No new mask, search, renormalization,
seed, decoder or acceptance-criterion adjustment was made after outcomes.

## 2. What was confirmed?

Two separate obstruction classes were independently validated:

| Registered structural question | Partial legacy5 | Whole brain5 secondary |
| --- | --- | --- |
| Exact degree/role-total DAG under five sufficient count tests | INFEASIBLE | UNRESOLVED |
| Exact all-lag dependency-walk support in a same-node DAG | INFEASIBLE | INFEASIBLE |
| Symbol input sets with relevant positive-delay-cycle witness | 60/60 | 30/30 |
| Overall strict-control feasibility | INFEASIBLE | INFEASIBLE |

UNRESOLVED means no tested local count certificate triggered, not that a
degree/strength-matched whole-brain DAG was constructed or is known feasible.
The obstruction proofs need no memory-performance threshold.

### Partial degree and role constraints already force cycles

The partial source graph has686 neurons,683 nonisolated. Its APL neuron has
in-degree566 and out-degree574: total1140 incident edges. A DAG on those683
active neurons permits at most682 incident edges at any neuron, since a pair
cannot be connected in both directions. Exact labeled degrees therefore force
at least458 reciprocal neighbors for this neuron. Reassigning weights or
relaxing current input access cannot remove this count obstruction.

Independently, the role totals force reciprocal pairs:

| Role pair | First direction | Reverse direction | One-direction pair capacity | Consequence |
| --- | ---: | ---: | ---: | --- |
| APL1 ↔ KC512 | 512 | 512 | 512 | All512 pairs must be bidirectional |
| APL1 ↔ MBON48 | 27 | 23 | 48 | At least2 pairs must be bidirectional |

These use fixed role totals rather than the particular reciprocal edges in
one graph. A same-role-total DAG is impossible for this partial baseline even
if its fine wiring changes. This does not identify an APL biological storage
site or rule out approximate matching, an observed-coordinate-only comparison,
or other cycle interventions. [Raw count certificates](../results/cycle_attribution_v1/main/legacy5/degree-certificates.json).
Whole's corresponding five local tests find no obstruction;
[its result remains UNRESOLVED](../results/cycle_attribution_v1/main/brain5/degree-certificates.json).

### Relevant cycles prevent complete all-lag path matching

Every one of90 symbol input sets has an authenticated path from at least one
stimulated KC to a directed cycle, then to at least one observed MBON. Save
source-ID paths once plus repeat counts. If access-path delays are A,B and
cycle delay P≥1, repeated traversal gives total delay A+kP+B. Any same-node
DAG has at most N−1 edges on a walk, so at most N−1 total delay under this fixed
0/1 schedule. The source graph's positive-delay cycle permits walks beyond that
bound, contradicting complete all-lag support equality for **every** same-N DAG.
This is not limited to M3's particular rank mask.

| Witness arithmetic | Partial | Whole secondary |
| --- | ---: | ---: |
| Same-node DAG universal bound N−1 | 685 | 138,638 |
| Authenticated cycle delay P | 3 | 3 |
| Compressed witness total delay beyond N−1 | 686–688 | 138,639–138,641 |
| Reference M3 DAG dependency bound H | 11 | 153 |
| Separate witness total delay beyond H | 13–14 | 154–156 |

All90 paths, endpoints, every source edge, scheduled delay and repeat calculation
are independently checked. Each witness establishes a mismatch for at least
one input/observed pair in that symbol set; it does not claim unbounded
dependence for all51 stimulated neurons or all48 observed neurons.
[Primary witnesses example](../results/cycle_attribution_v1/main/legacy5/witnesses-s1400001.json),
[secondary example](../results/cycle_attribution_v1/main/brain5/witnesses-s1400001.json),
[complete cells](../results/cycle_attribution_v1/main/summary.json).

### What remains matched and what M3's mask changes

The M3 reference is a source-coefficient-preserving mask, not a matched-degree
control. Its per-neuron incoming/outgoing degree mismatch counts are608/89 in
partial and110,834/109,746 in whole; absolute incoming/outgoing weight-mass
mismatch counts are the same. Role-block totals also change. All original
current KC→MBON edges1474/21,438 and coefficients remain exact **within each
graph's original-versus-DAG pair**. Zero-state responses for every fresh symbol
mapping are analytically equal within each pair. Partial and whole normalization
and response amplitudes are not asserted equal, and history-bearing current
decoding was not measured in M4.

Raw per-neuron arrays, matrices' hashes, root IDs, roles, mapping indices and
boolean lag0–20 support are archived in each graph's main folder. Boolean
support means a structural walk exists; it discards signs, magnitudes, path
multiplicity, nonlinear suppression and decoder access. For example the whole
DAG has supported paths through lag20, which does not establish useful long
memory. These are not Temporal Decodability Curves or formal capacities.
[Partial mismatch](../results/cycle_attribution_v1/main/legacy5/mismatch.json),
[whole mismatch](../results/cycle_attribution_v1/main/brain5/mismatch.json).

## 3. What failed or remains unanswered?

The prospectively registered **strict exact-control design fails feasibility**.
No replacement performance experiment was added after seeing this result.
The all-lag requirement is intentionally stronger than what every useful causal
study needs. Changing delayed paths may mediate a cycle intervention's effect
rather than merely confound it. Thus M4 does **not** prove that all cycle
attribution is impossible, that cycles have no functional effect, or that M3's
intact-minus-DAG gap is explained by a uniquely identified mechanism.

Signed-path cancellation and nonlinear/decoder effects can make a structural
walk have no useful functional influence. No cycle-related decoding gain,
biological learning, storage location or Shannon/reservoir capacity was tested.
Standard finite-time unrolling creates extra coordinate copies and changes the
fixed-neuron state budget; it is not silently treated as the same reservoir.

One technical provenance validation failed before smoke. The first guard
execution passed29 tests, but an auditor update during source capture meant
its recorded source commit did not authenticate the captured auditor hash.
The original complete execution/manifest was not overwritten. Its exact
captured code snapshot, failed source comparison and registry are preserved:
[validation failure](../results/cycle_attribution_v1/validation_failure_001/failure.json),
[registry](../results/cycle_attribution_v1/validation-failures.json).
Stable HEAD and start/end clean-tree checks now prevent this capture race.
The same29 tests were rerun without changing scientific settings in
[guard_tests_retry1](../results/cycle_attribution_v1/guard_tests_retry1/checks.json)
and passed with valid source provenance. First guard remains validation-invalid;
it is not used as the valid prerequisite. No structural smoke/main execution
or independent validation failed, and no new neural run was performed.

## Independent verification and reproducibility

[Main verification](../results/cycle_attribution_v1/main_validation/checks.json)
rebuilds pinned partial CSV and whole parquet/annotation sources, normalization,
rank-mask CSR arrays, degree/role integer arithmetic, finite dependency bounds,
all9 fresh mappings, boolean lag support, analytic zero-state responses,
canonical source-ID witnesses and final strict endpoint. It imports no runner
or M3 structural helper. Integer/boolean/certificate decisions and CSR hashes
are exact. Descriptive floating reductions use pre-outcome
atol1e-12+rtol1e-12, independently recorded; this is not a feasibility threshold.
Smoke has2 mapping cases/20 witnesses and scientific endpoint null.

| Stage | Seconds | OS process peak working set (MiB) |
| --- | ---: | ---: |
| Baseline audit | 11.671 | 65.25 |
| First guard, provenance-invalid and retained | 82.060 | 117.60 |
| Same29 guards, valid retry | 22.314 | 116.13 |
| Structural smoke | 87.467 | 411.23 |
| Independent smoke validation | 59.439 | 573.36 |
| Structural main | 165.655 | 413.04 |
| Independent main validation | 69.389 | 557.46 |

One numerical thread; unchanged7200-second/4GiB process budget. Resources include
50ms sampled RSS and OS-reported process peak; these are not exact system-wide
memory use. Stage manifests contain source/config/environment/data/graph/array
hashes and runtime. Main manifest SHA256:
`2e24e3f10867dc5945100908ec77c50fc8e7dce043144251fad468b114b49898`.
[Manifest](../results/cycle_attribution_v1/main/manifest.json),
[source/environment](../results/cycle_attribution_v1/main/source.json),
[runner](../scripts/cycle_attribution_feasibility.py),
[verifier](../scripts/verify_cycle_attribution_feasibility.py),
[final artifact/provenance audit](../results/cycle_attribution_v1/final_audit/audit.json).

The baseline seal protects43,983 prior result-file Git identities and44,610
prior paths excluding the four authorized overview documents. Fresh sealed
byte checks cover4,375 materialized result files;39,608 sparse-excluded result
identities are protected without claiming a fresh replay. M3's74 manifests,
810 artifact checksums and174 recorded-source entries authenticate. Existing
scientific source, protocols/configs, data, tests and results remain unchanged.
[Baseline audit](../results/cycle_attribution_v1/baseline_audit/audit.json).

## 4. What can now be said about DrosoMem?

The fixed source-derived reservoir retains decodable past-input information
(P1), and delayed acyclic propagation can suffice for short-lag access (M3).
M4 identifies concrete limits on an exact cycle-removal comparison: preserving
partial degree/role totals forces some cycles, and complete all-lag support
cannot survive removal of relevant positive-delay cycles in either graph.
This methodological result adds no new memory-performance or biological claim.

P2's failed material wiring gates, M1 assay-invalid, M2's specified joint result,
M3's limited sufficiency finding and earlier whole-brain/local-learning negatives
remain sealed. Physiology/spiking/plasticity P3 stays outside this study.

One next candidate is a prospective **finite-lag functional-sensitivity study**:
measure actual signed/nonlinear responses to past input on the fixed graph,
check against independent derivatives/finite differences, and distinguish
structural support from useful decoding. This is a proposal only, not a
registered experiment or a claim of unique cycle causality. Its streams,
finite lags, budgets and endpoints must be fixed before outcomes.

## 비전공자를 위한 한국어 요약

이번에는 기억 성능을 더 높이는 실험보다, 되먹임 고리만의 효과를 공정하게
비교할 수 있는지 먼저 확인했습니다. 부분망에서는 연결 수를 그대로 유지하면
고리가 반드시 남습니다. 또 고리를 반복해 지나는 모든 시간 경로까지 그대로
유지하면서 고리를 없애는 것은 두 그래프 모두 불가능했습니다.

그래서 등록한 엄격한 비교 설계는 불가능하다고 종료했습니다. 이는 고리가
기억에 도움이 없다는 뜻이 아닙니다. 신호가 지나갈 길이 있다는 것과, 그 신호가
실제로 과거 숫자를 읽어내는 데 도움이 된다는 것도 다릅니다. 기존 기억 결과와
실패 결과는 보존했고, 다음 후보는 이 실제 영향이 시간에 따라 얼마나 줄어드는지
검증하는 연구입니다.
