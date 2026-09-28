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
| Does recall generalize across alphabet sizes? | Open | Generator-level integer alphabets tested; network experiments remain K=10. Length1024 and additional sequence families remain unexecuted |
| Memory-critical subnetwork? | Partial prior ablations only | No comprehensive matched ablation/control study or causal circuit localization; ACT III open |
| Internal plasticity without external decoder dependence? | Unestablished | ACT IV open; do not turn A+B representation/decoding into C learning |

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
[results](length-scaling-results.md). Next: one controlled loss-mass diagnostic
at N512, with frozen data/states and no multiplier sweep.

Recent history puts research-only scope after Phase 6 execution/analysis,
preceded by Phase 5B failed response and Stage B/C validation. No historical
result files were overwritten.
