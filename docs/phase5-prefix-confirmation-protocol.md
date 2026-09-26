# Locked cross-segment confirmation — 2026-09-27 KST

Base commit cc7d68fc874959e604df157f038c27bd226cf4f8. Written before observing outcomes.

Keep the previous 32-target window, weight 4 versus 1, incoming-L1 normalization,
mbon_after_kc schedule, 482-parameter nonlinear decoder, unweighted training
statistics, 2,000 Adam epochs, learning rate .03 and L2=1e-5 unchanged.
Use existing subsets s701/s702, real/role-degree-shuffled topology, fresh seeds
1142/1143/1144, all initializations 0/1/2, and offsets 0/1000/2000.
Offsets count the leading 3 as index zero and omit the decimal point. Each segment
contains 200 digits and receives its own first three digits as the prompt.

Two subsets × three seeds × two topologies × three segments = 36 state conditions;
three initializations × two objectives = 216 fits. Reset both reservoir and head
for EVERY segment. The same encoder/graph/head initialization seed is paired
across objectives and reused across segments; no warm start, pretrained head,
position feature, or preceding π history. Every segment is explicitly trained:
this is cross-task replication, NOT zero-shot prediction of unseen π.

Primary confirmation: offsets 1000 and 2000, reported separately. Offset zero is
a secondary new-seed replication. For each new offset, call the prespecified
pattern confirmed only if real graphs show a positive mean paired recall delta,
at least four of six state conditions improve in initialization-averaged recall,
and at least four of six improve for ALL three initializations. These thresholds
are descriptive replication criteria, not statistical significance tests.
Report failed criteria and every initialization, without changing the rule,
window, offsets, seeds or epochs based on results. Shuffled controls and the
later-position accuracy cost must be reported even if confirmation succeeds.

Primary outcome: consecutive correct generated digits, excluding the supplied
three-digit prompt, horizon 197. Also report early 32 / later 165 target accuracy,
full 199-target accuracy and unweighted cross entropy. Average initializations
within a graph/input condition; do not count segments, initializations or two
overlapping subsets of one animal as independent biological replicates.

Save complete heads, predictions, generated strings, training histories, source,
data and protocol hashes per state condition atomically. Resume only matching
config/code/data/protocol and validated checkpoint checksums. A checkpoint not
listed in the ledger is uncommitted and must be recomputed. The ledger is not a
concurrency lock: only ONE writer may use a given output directory.

Verification: reconstruct all 36 states, replay all 216 heads and full strings;
independently refit six heads (both objectives × three initializations) in the
first circuit, first seed, real graph at EACH offset: 18 total. Cross-check all
2,200 π digits against the Decimal generator. Tests must cover segment indexing,
reset equivalence, interruption/resume equivalence, changed config/code rejection
and corrupt/missing checkpoint rejection. Full pytest suite must pass.

Do not launch whole-brain scaling or tune another intervention in this run.
