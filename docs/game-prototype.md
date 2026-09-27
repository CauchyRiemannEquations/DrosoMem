# First playable console prototype — 2026-09-27

The game connects a trained FlyWire opponent to a timed study phase and a shared
first-error judge. This is a local console prototype; graphical screens and live
training mode remain unfinished and are now deferred by the user's research-first
priority. See the [original phase audit](phase-status.md).

## Play

From the repository root, with Python and project dependencies installed:

```bash
python -m pip install -e ".[test]"
python -m flying.game --record outputs/my_first_match.json
```

Default study time is **600 seconds**. The human sees the 200 training digits,
then recalls the 197 digits after the supplied prompt `314`. Enter one digit at
a time, without a decimal point. Each player's first incorrect digit ends their
recall; the other continues. Scores exclude the prompt. Equal scores are a draw.
Completing the horizon is recorded as a lower bound (`censored=true`), not evidence
that the player could recall no further.

The model is labelled **pretrained**: the countdown is human study time. Its state
resets before recall; its learned parameters stay frozen. It receives neither
human digits nor judge feedback. A model move appears only after the corresponding
human submission.

The API stops exposing study material at the deadline. Terminal scrollback remains,
so this prototype relies on honest play. Ctrl+C/EOF cancels without saving a result.
JSON records include opponent ID, selection seed, catalog hash, rounds and scores.
Existing record files are never overwritten. Quick smoke game:

```bash
python -m flying.game --seed 0 --study-seconds 0 --horizon 3
```

## Opponent artifacts and verification

The catalog includes **all 18** real-graph, fixed-32-weighted heads at 6,000 updates
from the retention study: two circuits × three seeds × three initializations.
All were trained on 200 digits; “fixed-32” means increased loss weight on the
first 32 continuation targets, not a 32-digit training dataset. A seed chooses
uniformly among entries before play. No best-score selection or adjustment to
human performance occurs. Anchored weighting subsequently failed its
[fresh-seed confirmation](phase5-retention-confirmation-results.md); fixed-32
remains the shipped baseline.

Each NPZ contains sparse connectivity, digit encoder patterns, neuron roles,
observation indices, normalization and readout weights. It contains **no target
sequence or prerecorded continuation**. The runtime `flying.game.opponent` imports
neither a pi generator nor a training runner. It generates each move from reservoir
state, then feeds back its own digit. The prompt `314` is public; reference digits
belong to the study screen and judge.

Offline export checks archived artifact/data hashes, reconstructed state/weight
hashes and saved-head predictions. It then loads each exported model through the
game adapter and compares its entire **197-digit rollout** with the archived
record: all 18 match. This verifies inference equivalence, not a refit of the
whole study. [Export evidence](../results/game_adapter/export.json).

Re-export to new paths without overwriting the shipped bundle:

```bash
python -m flying.game.export --output outputs/opponents_reexport --report outputs/opponents_reexport.json
```

## Can it actually learn during ten minutes?

Yes, at the current scale on this host. A prespecified fresh fit of the first
configured condition (s701, seed 3142, initialization 0) took **2.254 seconds for
6,000 updates**, or **2.619 seconds including preparation**. All three stage-head
hashes exactly reproduced the archived run. This trained 482 readout parameters
with recurrent connectivity fixed, using one BLAS thread. Imports, UI startup,
verification and file writing are excluded. This is one CPU measurement, not a
browser/mobile benchmark or a guarantee for larger models.
[Timing and environment](../results/game_adapter/training_benchmark.json).

```bash
python scripts/benchmark_game_training.py --output outputs/my_training_timing.json
```

Recommended progression:

1. Standard mode: ten-minute human preparation and an explicitly pretrained,
   reproducible opponent. Implemented in this prototype.
2. Live-training mode: initialize a fresh head when study begins, train a fixed
   declared budget, show actual progress/completion and freeze before recall.
   Do not stretch a two-second fit into a fake ten-minute training animation.
   Handle slow-device failure before beginning recall.
3. A literal ten-minute compute-budget mode is a separate experiment: devices
   perform different numbers of updates and extra training is not yet shown to
   improve free recall.

Ten minutes is the user's initial product setting, not calibrated human difficulty.
Playtesting should determine preparation time and difficulty. Research is sufficient
for a first game; anchored cross-segment confirmation and portability remain open.

## Validation

125 tests pass, including all-population rollout parity, independent/resettable
state, checksum rejection, the exact 600-second deadline, invalid-input atomicity,
both directions of early failure, draws, horizon bounds and console subprocesses
using Windows CP949 encoding. Manual smoke games also produced a 3–3 draw and
a 0–3 model win.
