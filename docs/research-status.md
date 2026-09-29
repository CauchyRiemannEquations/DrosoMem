# Research status — audit 2026-09-28

Audit base: bd438e3d4b69b782e23cec41a157871fd1868e07. Evidence has priority
over summaries. Inventory: 822 tracked files, 23 experiment/analysis manifests,
223 result NPZ archives (25,315 arrays inspected; no nonfinite numerical arrays). All 374 artifact checksum entries recognized by the
audit matched, with zero missing files. Some early manifests contain provenance
only, so this is not a claim that every archived result has a checksum.
See [inventory](../results/act1_audit/inventory.csv),
[manifest audit](../results/act1_audit/artifact-checks.json) and
[recent commits](../results/act1_audit/recent-commits.txt).
The review covered README, docs, configurations, runners, dynamics/readout/data/
evaluation modules, test coverage and archived result/checkpoint schemas.
This is not a new execution of every historical experiment.

## Research questions and evidence

| Question | Status | Evidence and interpretation |
|---|---|---|
| Can a fixed source-derived reservoir support trained-prefix recall? | Completed, configuration-dependent | MVP 300-neuron baseline; normalization/size/length study: 450 runs in results/phase2. Incoming-L1 original 300-node scores 197 (censored),41,84 at length 200; not a typical capacity estimate |
| Does anatomical restriction improve memory? | Completed, no general superiority | results/phase3: two 686-node left MB samples; KC stimulation, 48 MBON observation; no advantage established over controls |
| Does temporal scheduling alter accessible past input? | Completed | results/phase3b and phase5_timing: short-delay decoding changes strongly; better current-digit access does not imply better autonomous recall |
| Does more flexible decoding help? | Completed, mixed | results/phase5_readout: 384 evaluations, stronger training fit but unreliable prefix recall |
| Does fixed prefix weighting help recall? | Established scoped baseline | results/phase5_prefix (120 fits), phase5_prefix_confirmation (216 fits), separate Windows replication. Fixed 32-target/4x improves early recall across offsets, sacrifices later teacher-forced accuracy; shuffled graphs show similar effect |
| Does expanding/anchoring the weighted prefix help? | Discovery superseded by failed confirmation | results/phase5_curriculum: 27.19 vs 33.87; retention discovery 38.89 vs 31.67, but retention confirmation fails. Do not promote anchored weighting |
| Do constrained recurrent learning rules improve recall? | Executed, negative | results/phase4 (432 runs), phase5 (576 evaluations); diagnostic discovery/confirmation (4,384), BPTT (192): no reliable recall advantage |
| Are clean fixed-prefix models robust? | Executed, criteria failed | results/phase2_robustness: 72 heads, 16,128 replays; small state noise and 1% edge removal sharply reduce recall |
| Do sourced spiking dynamics improve recall? | Small-circuit implementation/validation complete; negative recall | results/stage_bc: rate 33.5 vs LIF 3.0, scoped one-seed comparison, not calibrated physiology |
| Does local dopamine-gated plasticity work? | Rule controls pass; functional criterion fails | results/phase5b: 12 cases, silent isolated baseline probes, pi recall 3→2. Not evidence of useful internal memory learning |
| Can the complete source graph execute? | Engineering scope completed | results/phase6, phase6_analysis, phase6_validation: 138,639 nodes, 15,091,983 edges, 24 exact repeated probes, peak sampled worker RSS 541 MiB |
| Does whole-brain connectivity improve matched-budget recall? | Completed; main improvement criteria failed | 50 new fits: legacy5 35.0, brain5 31.8, brain1 34.1. [Locked protocol](whole-brain-memory-protocol.md), [current results](whole-brain-memory-results.md). Before this audit, untested |
| Does trained-prefix recall extend beyond pi? | ACT II four-family pilot complete, limited criterion passed | 80 fits: legacy5/brain1 means random26.0/28.8, shuffled-pi31.2/30.2. Both pass the registered limited beyond-pi rule, neither establishes whole-brain superiority. Periodic197/197 is also solved by a first-order control; [results](sequence-memory-results.md) |
| How does random-sequence recall scale with length? | Completed within fixed training procedure | 100 fits at N32/64/128/256/512; both graphs complete N32, N512 means legacy5=14.6 / brain1=9.5. Relative-collapse criterion first met at N256 / N128. No whole-brain superiority; [results](length-scaling-results.md) |
| Does restoring early-prefix loss mass explain part of the N512 drop? | Completed, prefix benefit confirmed with later cost | 44 new fits plus 20 reused baseline fits; fresh-seed means legacy5 10.83→33.00, brain1 11.17→23.33. Both pass prefix confirmation but fail later teacher-forced cost tolerance. Same reservoir states; external decoder tradeoff, not increased storage. [Results](prefix-mass-results.md) |
| Is past iid input linearly accessible under the current matched interface? | Completed ACT II diagnostic | 20 conditions /20 exact full repeats; mean past-lag accuracy legacy5=60.42% / brain1=51.10%. Whole-brain criterion fails; [results](delayed-symbol-results.md). Not autonomous recall or formal memory capacity |
| Does trained-prefix recall extend across alphabet sizes? | ACT II-A fixed-N128 comparison completed | 80 fits at K2/4/10/16; both graphs pass scoped trained-prefix gates at all K, no whole-brain superiority. Mean prefixes partial25.0/43.4/82.2/103.1; whole27.3/26.3/55.1/88.0. [Results](alphabet-memory-results.md). Full length-by-K design and additional families remain open |
| Can finite-context distinctiveness help explain the K curve? | ACT II diagnostic completed; both scoped endpoints pass | 120 context tables on20 unique tasks; order2 ambiguity K16−K2=−0.2272; order3 prefix +59.0; all independently checked. Context5 completes K10/K16. A task-level explanation candidate, not neural causality; [results](context-memory-results.md) |
| Does a fixed-K low/high context-conflict intervention change recall? | Completed within registered cohort scope | Main40 fits; confirmed graphs: none. [Results](context-intervention-results.md). Matched unigram counts and interface; ordering covariates remain |
| Are past symbols accessible on held-out positions and portable across paired orderings? | Two ACT II diagnostics completed | All four blocked graph/arm and transfer graph/direction cells pass past-access criteria in both cohorts; no confirmed next-symbol benefit. Frozen-head transfer within5pp tolerance; [blocked](frozen-state-probe-results.md), [transfer](cross-arm-probe-results.md) |
| Which populations affect decoding in the current model? | Bounded ACT III-A panel completed with confirmation | DAN refit and APL frozen sensitivity confirmed under count/input matching; degree/edge-strength confounds remain. Whole-brain/downstream and critical-subnetwork localization open. [Latest results](neuron-panel-results.md) |
| Internal plasticity without external decoder dependence? | Unestablished | ACT IV open; do not turn A+B representation/decoding into C learning |

## Latest ACT III-A current-model population closeout — 2026-09-29

[Report](neuron-panel-results.md), [preregistered protocol](neuron-panel-protocol.md).
Input-only28 + population299 conditions;327 exact full replays/refits,314 frozen
evaluations,654 independent train/test trajectories,7,051 audited lag rows.
206 tests passed,8 optional skips;9,046 preceding result files unchanged.
Confirmed under the fixed two-cohort criterion: **refit/DAN** and **frozen/APL**.
DAN refit control-minus-target6.798/6.417pp; APL frozen48.618/50.499pp.
APL refit extra loss1.975/2.242pp remains below5pp. KCab/KCapbp fail specificity;
hub passes different modes in different cohorts and is NOT confirmed.
Count/input controls for DAN/APL/hubs do not match edge count or strength;
APL removes1,140 edges versus3.94 for its one-neuron random controls.
This closes the bounded two-partial-circuit III-A panel, not whole-brain/
downstream localization or autonomous-recall mechanism. Next single experiment:
DAN-associated edge removal with matched edge count/weight bins, ACT III-B.

Source provenance follow-up: six recorded Python files had execution CRLF/Git LF
hash differences only. Their exact execution bytes are preserved in
[source audit](../results/neuron_source_provenance/audit.json); all Python ASTs
match and two representative canonical-LF full replays match every array/metric.
Historical manifests were not rewritten.

## Preceding ACT III-A fresh-seed KC confirmation — 2026-09-29

[Fresh cohort](kc-confirmation-results.md): KCγ26.047% frozen/79.367% refit,
matched26.962%/77.323%. Primary refit benefit confirms across new3blocks without
pooling discovery. Frozen access/retention and5pp specificity fail again.
Frozen specificity+0.915pp,all3positive,is not evidence of equivalence or zero.
35new condition replays,28frozen transfers,693metric rows checked;202tests pass,
8optional skips;8,446prior result files unchanged. Same biological circuit strata.
Next: input-only silencing versus full lesions to separate the added recurrent
edge-removal effect. Useful internal plasticity and circuit localization remain open.

## Preceding ACT III-A frozen/refit lesions — 2026-09-29

[Frozen intact-head comparison](kc-frozen-results.md): full KCγ26.250% frozen
versus79.219% refit; matched26.377%/77.156%. Primary refit-benefit criterion passes
all3 paired blocks. Subset frozen30.788%/32.181% versus refit76.957%/77.072%.
All4 groups fail frozen access and5pp retention. Frozen KCγ-specific impairment
fails both full and subset criteria. Decoder mismatch is not information loss.
70transfers/score replays,7source baselines,770metric rows independently checked;
200tests pass,8optional skips;8,222prior result files unchanged.
This was the same cohort with no new target fits/trajectories; fresh confirmation is now reported above.

## Preceding ACT III-A KCγ lesions — 2026-09-29

[Population and disjoint subset results](kc-ablation-results.md): intact77.303%,
KCγ lesion79.219%, exact degree/input-matched KC lesions77.156%. Specificity
criterion fails; opposite-direction gain is exploratory, not a revised success.
Controls overlap61.9–72.0%. A separately registered same-cohort follow-up removes
disjoint KCγ/non-KCγ subsets71–93neurons:76.957%/77.072%,criterion fails again.
Dose/background changed too; do not attribute the difference only to overlap.
77conditions/full replays,847including-smoke metric rows independently checked.
198tests pass,8optional skips;7,908 earlier result files preserved by follow-up.
No identified memory-critical subnetwork. The frozen/refit comparison is now complete above.

## Preceding ACT III fresh-seed alignment confirmation — 2026-09-29

[Fresh cohort](fresh-alignment-results.md) uses new mapping55142–55144 and separate
new train/test/rewire seeds on existing biological strata701/702. R→F45.142%,
F→R47.256%, versus unaligned25.033%/26.061% and refit80.108%/77.606%.
H1 improvement passes both directions/all3blocks; H2 access and H3 retention fail.
12 new conditions/full replays,12 transfers,132 independently checked lag rows;
194 tests pass/8 optional skips.7,511 historical result files unchanged.
New pseudorandom realizations are not new biological connectome samples.
The matched KCγ refit study and disjoint follow-up are now complete above.

## Preceding ACT III residual diagnostics — 2026-09-29

[Two completed diagnostics](residual-orientation-results.md) reject low-half
source-mode dominance and preferential decoder-sensitive residual orientation.
Fixed ranks25–48:10–18% state energy but about0–1% mean signed score attribution.
Actual/reference geometric score-distortion ratios are below1 in both directions
and both archived cohorts. Each study34 exact replays/374 rows; second5984
signed-permutation control energies.192 tests pass/8 optional skips. These are
geometric decoding diagnostics, not neuron-group ablations or memory improvements.
Prior moment-alignment improvements and access/retention failures stay intact.
The subsequent fresh-seed replication is complete as reported above.

## Preceding moment-alignment update

[Train-only mean/std alignment](moment-alignment-results.md) improves both
directions in both archived cohorts: main51.007% /51.607%, confirmation49.447%
/52.886%, versus unaligned25–27% and target refit79–80%. H1 improvement passes,
H2 access fails because mean R² stays negative; H3 5pp retention also fails.
The label-free adapter estimates96 moments; source heads stay fixed.34 directions
and374 lag rows verified,186 tests pass/8 optional skips.7,168 old result files
unchanged. These are previously observed cohorts, not new confirmation seeds.
The subsequent source-mode and orientation diagnostics are complete above.
This remains decoding analysis, not recurrent learning.

## Preceding frozen-transfer update

[Frozen normalization transfer](normalization-transfer-results.md) completed both
directions in both archived cohorts: main25.728% /25.468%; confirmation24.958%
/27.172%, versus target refits about79–80%. All access/retention gates fail.
32 analyzed plus2 smoke directions verify; zero new target fits.374 lag rows
checked,181 tests pass,8 optional skips;7,053 historical result files unchanged.
Train-feature means shift despite high centered temporal correlations. The subsequent
train-only moment alignment is now complete as reported above.
These results concern frozen decoder portability, not absent past information.

## Preceding normalization intervention

[Normalization control](normalization-control-results.md) completed30 main and18
fresh-seed confirmation runs, plus3 separate smoke runs. Same raw rewired graph,
original versus graph-specific incoming factors. Fixed-original improvement
1.245pp main,0.894pp confirmation; registered1pp confirmation fails despite all
paired blocks having the same sign. Do not pool cohorts or promote this as an
established >=1pp improvement. All51 runs and561 lag rows verified;178 tests pass,
8 optional skips. Historical6,653 result files unchanged. Fixed factors lose the
sufficient contraction certificate; observed finite states/one small perturbation
do not establish global stability. The subsequent frozen-head transfer is now complete as described above;
no gain search or internal-plasticity expansion was performed.

## Preceding K4 structural control

[Current K4 structural control](structural-k4-results.md) completed20 graph runs,
20 exact replays and refits. Real76.965% versus role-/degree-rewired78.603%; paired
delta−1.638pp (five blocks, all negative). Both pass past-access controls, but
real-wiring superiority fails. Observed effective rank6.68 versus7.22; association
only. Renormalization changes644–645 outgoing weight multisets per control, so
this is not a pure topology effect. Historical6,472 result files unchanged.
175 tests pass,8 optional skips. The standalone verifier's initial BLAS-thread
mismatch is preserved with its corrected exact verification. Broad ACT III
ablations, critical subnetworks, and ACT IV useful internal learning remain open.

## Baseline reproducibility

Clean new Python 3.12.10 virtualenv, original rate dependency versions, one BLAS
thread. First archived real s701/seed1142/offset0 condition, **all three weighted
initializations**, no score-based choice: expected/replayed/refitted scores
**4/35/34**. Saved generation strings and independent refit parameter hashes all
match exactly. [Full report](../results/act1_baseline/report.json).
Train length 200, prompt 3, horizon 197, 2,000 Adam steps, LR .03, 482 parameters.
This reproduces one condition of the established baseline, not every archived
model or the strict full historical source-context verifier.
The original offline test pass was 126 passed, 8 skipped: optional Brian2 tests
were unavailable in this rate-only environment. Historical 193-test LIF reports
refer to a different optional-dependency environment.

## Best established, unconfirmed, negative

- Best established *controlled research reference*: frozen rate connectivity plus
  fixed-32 readout training. It is selected for replicated design and matched
  anatomical interface, not the largest digit score.
- A high historical individual score is not a general memory capacity claim:
  Phase 2's 197/197 is censored at the horizon and uses a different observation
  design from the 48-MBON baseline.
- Unconfirmed: anchored-prefix discovery improvement; fresh confirmation failed.
- Known negatives: recurrent-plasticity recall advantage, anchored retention,
  noise robustness, improved LIF recall and Phase 5B functional response.
- Whole-brain activity feasibility alone does not establish whole-brain recall.

## Documentation/artifact discrepancies

1. README contained an older “candidate pending separate confirmation” paragraph
   after reporting that confirmation failed. The latter artifact takes precedence.
2. Older result-page “next” suggestions (curriculum, robustness, deferred whole
   brain) are historical proposals; later commits executed those studies.
   Preserve those pages as historical records and use this index for current status.
3. “Phase 6 completed” means engineering feasibility; its pi20 traces are supplied
   input, not generated digits or a trained recall result.
4. The established baseline is a leaky-tanh rate model; Phase 6 is sparse LIF.
   Direct score comparison would confound graph expansion with dynamics.
5. Readout is a trained nonlinear 48→8→10 head, not an untrained output or simply
   a linear decoder. Recurrent connectivity is frozen in the established result.
6. Source/protocol context hashes from older revisions can reject current source
   despite intact artifacts. Preserve old hashes and use historical revisions
   for strict archived verification; never rewrite provenance.
7. requirements-lock uses mpmath 1.4.1 while the LIF lock uses 1.3.0 because of
   Brian2's environment constraints. ACT I explicitly uses the rate environment.

## ACT I completion

Protocol 186a5f6 preceded outcomes. Five smoke conditions passed exact replay and
independent refit. All 50 main heads replayed exactly; five first-seed/first-stratum
heads independently refitted with identical parameters. Main comparison and
weak-edge success criteria failed. Representation diagnostics were reported;
no post-outcome tuning or fresh-seed positive confirmation. ACT II followed
as separate studies described below.
The extended rate test suite passed 132 tests with eight optional-dependency skips.

## ACT II pilot completion

Protocol4038f8f and implementation2528c9e preceded outcomes. Eight smoke runs
replayed/refitted exactly. All80 main runs replayed exactly; eight first-block/
first-stratum heads refitted exactly. The full rate suite passed141 tests with
eight optional-dependency skips. [Protocol](sequence-memory-protocol.md) and
[results](sequence-memory-results.md) preserve raw seeds, controls and limits.
The pilot's selected follow-up was random-sequence length scaling, now completed
below. The pilot does not establish universal arbitrary
memory, connectome-specific superiority, unseen prediction or recurrent learning.

## ACT II length-scaling completion

Protocol8885aca preceded outcomes; main execution070b4bf. All100 fits replayed
exactly and10 independently refitted. Both networks meet the registered endpoint
length-drop criterion; no graph-superiority confirmation was triggered. Raw
teacher-forced features for shared prefixes match exactly across task lengths.
Fixed prefix loss mass changes with N, so the curve is not intrinsic capacity.
The rate suite passed144 tests with8 optional-dependency skips. All1,932 prior
result artifacts remained unchanged. [Protocol](length-scaling-protocol.md),
[results](length-scaling-results.md). Its selected loss-mass diagnostic is now
complete below, with frozen data/states and no multiplier sweep.

## N512 prefix-loss-mass completion

Protocol a7badad preceded outcomes; execution and analysis code 2dfc1d9.
Two smoke fits passed replay/refit and were excluded from estimates. Main reused
20 verified baseline fits and added20 treatment fits. Both graphs qualified for
the registered fresh-seed confirmation, which added12 baseline and12 treatment
fits. All44 new scientific heads replayed exactly; six independently refitted.
Two selected historical baseline heads were also replayed/refitted exactly.

Both graphs pass prefix confirmation, but neither passes the predefined later
teacher-forced accuracy tolerance. Confirmation later accuracy changes are
legacy5 −5.21 percentage points and brain1 −3.46 points. Brain1's individual
seed13143/stratum701 worsens from7 to0 even though its paired-block mean improves.
All low scores are retained; no509-target completion or new default promotion.
The rate suite passed146 tests with8 optional-dependency skips. All3,054 prior
result files remain unchanged. [Protocol](prefix-mass-protocol.md),
[results and verification](prefix-mass-results.md).

The selected independent-stream delayed-symbol comparison is now complete below.

## Matched delayed-symbol decoding completion

Protocol28a407c preceded outcomes; runner332e340 and analyzer correction30ecb6b.
Five paired seed blocks, two input strata, two graphs:20 scientific conditions,
each with11 real and11 misaligned-target probes. Every trajectory and every
head independently regenerated/refitted exactly. Same48 MBON observation,
input IDs and train/test streams. Mean primary past-lag accuracy: legacy5
60.420% / brain1=51.100%; paired difference −9.320pp. H2 whole-brain
superiority fails, so no fresh confirmation or tuning is triggered.
[Results](delayed-symbol-results.md), [protocol](delayed-symbol-protocol.md).

The smoke analyzer initially lacked the registered single-thread setting;
exact reconstruction failed at4.996e-16, then passed with that setting restored
and no relaxed tolerance. A premature initial main launch was stopped and
preserved; the full scientific cohort uses delay_main_v2. Its completed partial
condition matches the restart exactly. Execution deviations are documented,
not hidden. All3,565 prior result files and numerical modules remain unchanged.
Tests:148 passed,8 optional-dependency skips.

Its selected N128 alphabet-size comparison is now complete below.

## ACT II-A alphabet comparison completion

Protocol91c3032 preceded outcomes; adapters and runner e84f330. Existing
numerical modules are unchanged, and the K10 adapter exactly reproduces the
old N128314-prompt baseline (legacy5=34, brain1=38), including independent fits.
The new study uses010 for every K. Eight smoke runs pass replay/refit and are
excluded. All80 main runs replay exactly; eight designated heads refit exactly.
Tests154 passed,8 optional-dependency skips; all3,792 prior result files unchanged.

Both graphs pass H1 at all four K. Whole-brain H2 fails everywhere; no fresh
confirmation or tuning. The K16-minus-K2 prefix fraction increases in every
block, so the registered alphabet-load decrease fails too. AtK16 the evaluation
horizon is reached in5/10 partial and4/10 whole runs; this is censored performance.
Low whole-brain K4 runs of0 and1 remain in the raw table. [Results](alphabet-memory-results.md).

The selected saved-task context diagnostic is now completed below. Do not
attribute the K curve to memory capacity or topology from these data alone. Comprehensive ACT III structural
controls, additional ACT II families and ACT IV useful internal learning remain
open; ACT V robustness remains separate.

Recent history puts research-only scope after Phase 6 execution/analysis,
preceded by Phase 5B failed response and Stage B/C validation. No historical
result files were overwritten.

## Finite-context diagnostic completion

Protocol382501b preceded new diagnostic outcomes; implementation bffc0cc.
All120 table fits on20 unique datasets match independent recounts, including
probabilities, teacher outputs and ambiguity. All20 old order1 controls match;
80 saved neural prefixes were rescored. Tests165 passed,8 optional skips;
all4,574 prior result files and old numerical source are unchanged.

Both endpoints pass: order2 ambiguity falls0.2272 fromK2 toK16 (all five seeds),
order3 prefix rises59.0 (all five). Yet order1 ambiguity rises, K2 context8
recalls only7.2, and K4/data22143/context8 recalls0 despite zero long-context
ambiguity because the short initial prompt is ambiguous. Every failure is kept.
[Full results](context-memory-results.md). No neural training was repeated.

Position at that study's completion: ACT II-A plus task/decoding diagnostic.
Its selected fixed-K intervention is now complete below. Full length
by K, additional families, comprehensive ACT III and useful ACT IV remain open.

## Fixed-K context intervention completion

Protocol3b9e0f1 preceded bounded sequence construction and neural outcomes;
implementation212e6b9. Main40 fits: legacy5:low43.2/high30.9; brain1:low30.6/high30.4.
Discovery-qualified graphs: ['legacy5']; confirmation executed:
True; confirmed graphs: none.
All replay checks and registered independent refits passed. Four smoke runs
are excluded. [Results](context-intervention-results.md), including every seed,
all control orders, regional conflict diagnostics and negative paired differences.

K4, N128, symbol multiset, prompt010, input IDs,48 MBONs and428 head parameters
are matched. Other ordering statistics can differ, so this does not isolate
context conflict as a unique causal variable. Tests167 passed,8 optional skips.
The selected frozen-state diagnostic and its authorized cross-arm follow-up are completed below.

## Frozen-state and cross-arm diagnostics completed

Two scoped ACT II diagnostics completed, each with its own protocol committed
before outcomes:0b86a86 and5080e94. All64 archived trajectories remain unchanged.
Blocked probe:192 folds,1536 real+1536 null task heads; all four graph/arm cells
pass past-access criteria in both seed cohorts, but no confirmed low-high next
advantage. [Results](frozen-state-probe-results.md).

Frozen transfer:192 folds with ZERO target-arm fits; all four graph/direction
cells pass past access, both graphs pass the registered5pp retention tolerance.
Main transfer-minus-within means−0.97pp partial/−0.83pp whole; confirmation
+0.06pp/+0.93pp. Next-symbol prediction remains weak. Related source/target pairs
share about59–60% same-position symbols; changed-target diagnostics are retained.
[Results](cross-arm-probe-results.md). This is not arbitrary-stream generalization,
autonomous recall improvement, equivalence or formal memory capacity.

All384 folds across the two studies pass exact checks and independent augmented
least-squares verification. Tests172 passed,8 optional skips. The earlier
fixed-K recall confirmation failure remains unchanged. The selected scoped
ACT III real-versus-role/degree-rewired comparison under current K4 dynamics
has subsequently completed: real76.965% versus rewired78.603%, no established
real-wiring advantage. See the latest update above and its linked report.
Earlier Phase3/3b controls remain negative within their original scopes.
