# Phase 6 — locked coverage and whole-brain feasibility protocol

Editorial update: research terminology only; the registered numerical design is
unchanged. Original protocol bytes are preserved in Git history; see
[historical reproduction](historical-reproduction.md).

Locked before activity outcomes, 2026-09-27. This closes the original Phase 6
engineering acceptance scope: whole-brain graph, sparse resource measurements,
validated dynamics and a reproducible whole-brain execution. It does not promise
whole-brain pi learning, physiological calibration or rescue of Phase 5B.

## Data and comparisons

Use the existing pinned Shiu v783 parquet and annotation hashes. Node universe is
the author-distributed `Completeness_783.csv` at the same source commit, SHA-256
`bbb847a4cc2caaa7a16349722d220c087317b946d148d4d592d94d250617a311`.
It contains 138,639 unique completed cells, including cells with no retained
edges. Validate every parquet index/ID against this exact universe. Fourteen
source cells have no annotation: retain them as OTHER, list their IDs. Do not
infer cell type from a numerical ID. Source signs remain model assumptions.

Four levels for each of the existing s701/s702 input maps:

1. Existing 686-cell circuit, threshold 5, no autapses.
2. All annotated left KC/MBON/DAN/APL (2,795 cells), threshold 5, no autapses.
3. Full 138,639-cell universe, threshold 5, no autapses.
4. Full universe, threshold 1, no autapses.

Level 4 retains every positive-count inter-neuron source edge, not necessarily
source autapses; report their excluded count. No weight normalization or fitting.
Confirm each bundled graph equals the source-induced threshold-5 graph exactly.
Report coverage of every left MBON: incoming absolute contacts, excitatory and
inhibitory contacts separately, left-KC contribution, retained subset contacts.
Partition missing contacts into autapses, below-threshold edges, and outside-node
edges (disjoint in that order). Never treat signed cancellation as input coverage.

## Stimulus, dynamics, measurements

Reuse KCEncoder seed 6142, fraction .1 on each original 512-KC sample: exactly 51
stimulated KCs per digit. Map these exact root IDs into larger graphs. No extra
input on added cells. All annotated KCs have zero refractory time as in Stage B/C;
other cells retain 2.2 ms. Uniform LIFParameters and dt .1 ms are unchanged.
Source-inspired deterministic 100 Hz voltage pulses remain an engineering digit
code, not the author's Poisson input or physiological sensory stimulation.

Three independent-from-rest 1-second probes: twenty digit-3 windows, twenty
digit-1 windows, and the first twenty pi digits including 3. These are supplied
stimuli, not autonomous recall. Repeat every probe from reset and require exact
spike arrays, per-window counts and final v/g states. Save these arrays, mapped
input IDs, node IDs, parameter/source hashes and environment/resource metadata.
Summarize total/active-cell/MBON/MBON11 counts; publish silent and negative cases.

Use a memory-bounded CSC delayed-event LIF backend, independently checked against
Brian2 2.10.1 on signed/refractory/delay cases and both legacy circuits. Require
identical spike arrays/counts and final states within 1e-8 mV for the three probes
on legacy circuits. Failure blocks larger runs; do not relax tolerance after
outcomes. New backend arithmetic is in mV/ms; Brian2 stores SI, so validation is
bounded, not a claim of bit-identical Brian2 trajectories on the full brain.

## Resources, replay and interpretation

One CPU numerical thread, sequential isolated workers. Per condition (six
one-second simulations including repetitions): 1,800 seconds and 3 GiB sampled
RSS limit, with a parent watchdog. Sample at least every .2 s and record actual
peak RSS, graph-load and simulation wall times, sparse matrix bytes, host RAM.
Resource-limit failures are preserved and mean that condition is incomplete;
do not weaken the protocol or fabricate completion. A resource limit is a
watchdog, not a guarantee against instantaneous allocation peaks.

Graph construction streams batches and is separately measured; no dense N×N
matrix or full time×neuron trace. Store raw downloads/caches under ignored paths.
Archive compact results and checksums, plus a CLI to regenerate and compare
all conditions. Do not include changing logs in an already hashed manifest.
Source topology and exact-repeat checks are necessary acceptance criteria.
Activity improvement is exploratory, with no seed selection or tuning. No
readout is fitted; rate-model baseline and Phase 5B conclusions remain unchanged.

Sources: [pinned author model](https://github.com/philshiu/Drosophila_brain_model/blob/91bdd1e7dcf193f3e7ca5a8933497fcef63b7960/model.py),
[pinned data](https://github.com/philshiu/Drosophila_brain_model/tree/91bdd1e7dcf193f3e7ca5a8933497fcef63b7960),
[Stage B/C specification](stage-bc-protocol.md),
[Phase 5B failure](phase5b-results.md).
