# Phase 2 — structure, normalization and sequence-length sensitivity

This is an **exploratory computational experiment**, not a biological or
confirmatory claim. The original Phase 1 files and results are unchanged.

## Protocol fixed before this run

450 CPU runs: **5 real subsets × 2 normalization methods × 3 training lengths ×
3 fresh model seeds × 5 models**. Configuration: `configs/phase2.json`.

- Subsets: existing top-strength 300; three randomized breadth-first neighborhoods
  of 300 neurons (selection seeds 101, 202, 303); top-strength 1000.
- Same verified FlyWire FAFB v783 source and min-5-synapse filter throughout.
- Models: real fly; exact-degree shuffled; matched-N/E random;
  `leaky_only` (W=0, leak=.6); `memoryless` (W=0, leak=1).
- Model seeds 142, 143, 144 are new, distinct from Phase 1's 42, 43, 44.
- Training prefixes: 50, 200, 400 digits including the leading 3. A fresh readout
  is fitted for each length; longer-length experiments are **not** continuation
  of the same trained model. Lower recall after longer training is possible.
- Prompt `314`; recall horizon L−3. A horizon-reaching score is right-censored.
- Same heldout targets for all runs: digit indices **1000–1099** (leading 3=index 0).
  Teacher forcing consumes all preceding true digits; the untrained gap supplies
  context only. No heldout gradients, feature fitting, epoch selection or tuning.
- Fixed readout: 400 epochs, learning rate .03, L2 1e-5; same sparse stimulation
  fraction/amplitude and leak as Phase 1.

`leaky_only` still has within-neuron memory. `memoryless` removes that memory too.
These two controls are identical across normalization labels and are intentionally
repeated for paired comparisons; do not treat the duplicates as independent data.

## Normalization is an explicit modeling intervention

| Method | Operation | Interpretation |
|---|---|---|
| `spectral` | Scale all original signed synapse counts by one scalar to radius .9 | Original Phase 1 method; preserves global relative count ratios |
| `incoming_l1` | Scale each postsynaptic row so the sum of absolute incoming weights is .9 (zero rows stay zero) | Keeps edges and signs; changes relative strengths across target neurons; bounded recurrent input |

For `incoming_l1`, the infinity norm of W is ≤.9. Together with the 1-Lipschitz
`tanh` and leak α, the state update contracts by at most
`(1−α)+α*.9` for identical inputs. This is a mathematical stability guarantee for
this simplified model, not evidence about biological currents. The two conditions
do **not** match actual spectral radius or temporal memory scale, so their
comparison does not isolate one pure causal mechanism.

## Actual graph structure

| Subset | Neurons | Edges | Largest strongly connected component |
|---|---:|---:|---:|
| hubs300 | 300 | 3,303 | 277 |
| connected101 | 300 | 5,477 | 233 |
| connected202 | 300 | 2,438 | 162 |
| connected303 | 300 | 1,682 | 291 |
| hubs1000 | 1,000 | 24,604 | 972 |

No subset has isolated neurons. The three neighborhood subsets are weakly
connected by construction, not necessarily strongly connected. hubs1000 has two
weak components (largest 998). The original hubs300 was already weakly connected:
its poor score cannot be attributed simply to isolated nodes.

## Measured Pi Memory Scores, 200-digit training

Every cell lists seeds **142 / 143 / 144**, not selected best runs.

| Real Fly subset | Global spectral normalization | Incoming L1 normalization |
|---|---|---|
| hubs300 | 2 / 2 / 2 | **≥197 / 41 / 84** |
| connected101 | 3 / 3 / 0 | 3 / 3 / 2 |
| connected202 | 2 / 3 / 2 | 2 / 3 / 3 |
| connected303 | 1 / 0 / 0 | 1 / 1 / 0 |
| hubs1000 | 9 / 8 / 9 | 77 / 41 / 106 |

The original 300-node structure can recall the entire requested training prefix
for one fresh seed under a different normalization. This is **not** a robust
197-digit capacity estimate: the other two seeds fail at 41 and 84.
Increasing the neuron count to 1000 does not consistently beat the 300-node
condition. Merely selecting connected neighborhoods does not improve long-prefix
recall in this protocol.

For hubs300 at L=200, all paired controls were:

| Model | Spectral | Incoming L1 |
|---|---|---|
| Real fly | 2 / 2 / 2 | ≥197 / 41 / 84 |
| Shuffled | ≥197 / ≥197 / ≥197 | ≥197 / 9 / 84 |
| Random | ≥197 / ≥197 / ≥197 | ≥197 / ≥197 / 66 |
| Leaky only | 1 / 1 / 1 | 1 / 1 / 1 |
| Memoryless | 0 / 0 / 0 | 0 / 0 / 0 |

The experiments still do not establish a general advantage of real wiring.

## Length dependence and neural states

All 30 real-fly runs at **50 training digits** recalled the entire 47-digit
continuation. At **400 training digits**, real-fly scores were only 0–4 across all
subsets and both normalizations. Thus the data support successful short-prefix
memorization and substantial sensitivity to task length and modeling choices.
They do not support a reliable 400-digit memory claim.

For hubs300 at L=200, the mean raw centered-state effective rank rose from
**16.58 to 28.94** with incoming normalization. Effective rank here means the
exponential Shannon entropy of normalized singular values, computed before
readout standardization. It is a descriptive measure, not a memory-capacity
estimate or proof of the mechanism causing recall improvement. States remain
sensitive to numerical precision, gain and stimulation placement.

On the common 100-digit heldout block, pooled longest-prefix real-fly accuracy
was 12.53% (spectral) and 12.87% (incoming L1). The memoryless model was 12.73%.
These values do not demonstrate novel π prediction. The fixed block is short,
reused across runs, and its finite transition distribution differs from a
uniform-guess reference. No statistical superiority claim is made.

## Evidence and reproducibility

- Full 450-row metrics: `results/phase2/results.csv` and finalized `metrics.jsonl`.
- Every recall target/prediction/first-error position: `recall.jsonl`.
- Epoch 1, each 25th epoch and final epoch: `training.csv`.
- All summary ranges, graph diagnostics, config, package versions, code hashes
  and source/subset checksums are included.
- Run time here: **89.43 seconds**, CPU numerical kernels restricted to one
  thread; excludes rebuilding the data subsets. Personal hardware may differ.
- JSONL metrics/recall records are published as complete atomic snapshots, avoiding
  partially visible append logs during shared-filesystem synchronization.
- **20 tests passed**, including original MVP regressions, normalization
  invariants, seeded connected sampling, ablation semantics, deterministic
  repeated small runs and heldout-suffix perturbation tests.
- Full rerun reproduced all 450 metric rows exactly (excluding elapsed time).
  All 450 stored recall sequences independently agree with scores and censoring.
- Phase 2 logs predictions and parameters; it does not save 450 model checkpoints.
  Re-run a selected configuration to reproduce results. Phase 1's checkpoint
  replay feature remains available.

```bash
# Bundled real subsets; no live API or new source download required.
python scripts/run_phase2.py --config configs/phase2_quick.json
python scripts/run_phase2.py --config configs/phase2.json
python -m pytest -q
```

## What remains uncertain / next experiment

This study varies graph selection and normalization together with many model
conditions, reports them all and selects no winner for a final test. Neighborhoods
may overlap, hubs300 is nested within hubs1000, and all originate from one animal.
They are not independent biological replicates. A constant stimulation fraction
activates more cells in the larger network, so a size comparison also changes
absolute input-population size.

The next useful step is an **anatomically annotated mushroom-body circuit** with
KC/MBON/DAN roles and a stated input/output mapping, alongside delayed-digit
memory and perturbation/recall-stability tests. Confirm annotations against the
pinned snapshot before extracting it. Plasticity, spiking dynamics, dopamine and
whole-brain execution remain unimplemented. More neurons or a prettier UI alone
would not resolve the scientific uncertainties identified here.
