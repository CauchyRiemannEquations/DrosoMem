# Follow-up 4: whole-brain structure and pathway scope

Prospectively fixed before new neural outcomes. One question: with the same
source KC input IDs and same 48 left MBON readout, does expansion to a
138,639-neuron threshold-5 whole-brain graph improve independent-stream
lagged decoding, and is DAN→MBON sensitivity specific against an MBON-input
removal control? This is a bounded whole-brain-node-coverage study; it does
not claim arbitrary global rewiring or threshold-1 generalization.

Use the pinned v783 full source and SHA256 checks. Compare `legacy5` (686 nodes)
and `brain5` (138,639 nodes), circuits 701/702 as input-map strata, three new
seed blocks 820001–820003. Share exact K4 input patterns, 100 warmup, 1,000
train and 500 independent test symbols in each paired case. Freeze recurrent
weights; same incoming-L1/gain .9/leak .6/MBON-after-KC dynamics, 48 MBONs and
alpha-1 ridge at both scales. Primary lags 1,2,3,4,5,8. The 2-circuit
average is the seed unit.

At each scale compare intact, all DAN→MBON normalized edges removed without
renormalization, and a seeded, prespecified MBON-input control that removes
the same number of non-DAN incoming MBON edges, chosen to minimize absolute
normalized-weight-mass difference without consulting activity or labels.
Record edge count, removed mass and unmatched residual. A matched control
that misses >10% removed mass is invalid for specificity and reported as such,
not silently treated as matched. Refit ridge per graph/arm; frozen intact
transfer is secondary. Structural expansion is brain5−legacy5 within arm.

**Joint criterion:** whole-brain improvement is confirmed only if brain5−
legacy5 intact lag accuracy ≥5 percentage points, positive in all three seed
blocks, and both scales' intact accuracies exceed frequency baseline by 5
points. DAN pathway specificity is a separate gate: in brain5 the DAN cut
loss exceeds valid matched-control loss by ≥3 points and is positive in all
blocks. Report both gates independently. Neither identifies stored memory.
No autonomous or physiological claim is inferred from teacher-driven probes.

One smoke pair then 3 seeds × 2 circuits × 2 scales × 3 arms = 36 cases and
36 ridge refits. Budget 7,200 seconds/4 GiB sampled RSS for execution and
verification each, one process/numerical thread. Save train/test symbols,
states, graph/source hashes, fitted heads, all lag rows and paired differences.
Independent verification rebuilds graph cuts and controls, replays every test
trajectory and refits every ridge head. If any physical resource bound is
exceeded, preserve partial work and report the uncompleted scope.

