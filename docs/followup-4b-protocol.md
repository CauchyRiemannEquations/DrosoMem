# Follow-up 4B: degree- and role-preserving whole-brain structural control

Prospectively fixed before any rewired whole-brain neural outcome. The preceding
follow-up 4 kept real whole-brain wiring while expanding coverage and cutting
pathways. This separate question tests whether that whole-brain wiring itself
beats a degree- and role-preserving control on the same independent K4
delayed-symbol task. The preceding intact results are held fixed as a paired
reference. The intact outcomes are already known; this is a prospective
rewiring comparison but not an outcome-blind registration of the whole pair.

Take the pinned threshold-5 `brain5` raw graph (138,639 nodes, 2,700,513
edges). Use one predetermined shuffled graph seed 825001 and the existing
`role_shuffled` double-edge-swap implementation with one requested accepted
swap per edge. Require all 2,700,513 accepted swaps, exact per-node in/out
degree, source weight multiset and source/target role-block edge counts,
unchanged node/root IDs, no autapses/parallel edges, and record final edge
overlap. Normalize both real and shuffled raw graphs independently using the
same incoming-L1/gain .9 rule. This makes the comparison a graph plus
normalization intervention; it is not a pure synapse-identity effect.

Reuse exactly follow-up 4's three model/data seed blocks 820001–820003,
circuits 701/702 as input-map strata, same 48 MBONs, K4 symbols, warmup100,
train1000/test500 and lags 1,2,3,4,5,8. Refit alpha-1 ridge separately on
each shuffled graph trajectory; no recurrent learning. Paired real intact
accuracy is read from the sealed follow-up-4 manifest and raw case files, not
recomputed or selected after seeing the shuffled outcomes. Six new graph/head
cases; one smoke input-map case uses the same predeclared shuffled graph.

**Criterion:** mean `real intact − shuffled` held-out lag accuracy at least
3 percentage points and positive in every one of the three seed blocks
after averaging the two circuits. Report signed per-seed results and all
individual lag values regardless. This tests one null graph and small seed
cohort, not all possible rewires or a biological optimality theorem.

Execution and verification each have 3,600 s and 4 GiB sampled RSS, one
process/numerical thread. Preserve an incomplete run if the swap budget,
runtime or memory bound fails; do not weaken the swap requirement after
observing results. Independent audit checks source/parent hashes, every
structural invariant, all test states and fresh ridge coefficients.
