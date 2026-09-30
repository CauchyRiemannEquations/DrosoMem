# Data sources and attribution

DrosoMem combines source-derived Drosophila connectivity with generated
computational tasks and derived experimental outputs. This file is the short,
release-facing attribution summary. Exact hashes, extraction rules and rebuild
instructions are maintained in [`data/README.md`](data/README.md) and the
provenance files inside each data directory.

## 1. FlyWire FAFB v783 connectivity

The bundled connectivity subsets are derived from the public **FlyWire FAFB
version 783** release.

- Public data license: **CC BY-NC 4.0**.
- Official FlyWire archive: <https://zenodo.org/records/10676866>
- FlyWire citation and publication guidance: <https://home.flywire.ai/guidelines>
- The project uses transformed subsets, not a claim of a complete biological
  simulation.
- DrosoMem's MIT code license does **not** relicense FlyWire-derived data.

The main small subset contains selected root IDs and induced directed
connections after the documented thresholding and selection procedure. Other
bundled subsets use deterministic or seeded selection procedures described in
[`data/README.md`](data/README.md). A selected subset is not a separate fly,
a representative biological sample, or necessarily a named anatomical circuit.

## 2. Author-distributed processed connectivity

The reproducible download path uses processed FAFB v783 connectivity distributed
with the Drosophila brain-model work by Philip Shiu / Nico Spiller.

- Upstream repository: <https://github.com/philshiu/Drosophila_brain_model>
- Pinned source commit used by DrosoMem:
  `91bdd1e7dcf193f3e7ca5a8933497fcef63b7960`
- The retained upstream software notice is stored in
  `data/UPSTREAM_LICENSE.txt` where applicable.
- Raw source hashes and derived-file hashes are recorded in provenance JSON
  files and checked by the rebuild scripts.

The upstream software license and the FlyWire public-data license cover
different things. The presence of an MIT notice for upstream software does not
replace the attribution and non-commercial terms that apply to FlyWire public
data.

## 3. Cell annotations

Anatomically restricted experiments use public FlyWire annotations and preserve
the source/annotation revision information in their provenance records.

When publishing analyses based on these annotations, consult the current
citation guidance of the annotation project and FlyWire rather than treating
DrosoMem as the original source of those labels.

## 4. Generated sequences and experimental outputs

Tasks involving π digits, random symbol sequences, shuffled sequences,
periodic controls, synthetic noise and rewired graph controls are computational
benchmarks generated or transformed by the project. They are not recordings
from animals.

Experimental outputs under `results/` are derived computational artifacts.
Multiple random seeds, replays, trajectories or graph controls must not be
described as independent animals or biological replicates.

## 5. What to cite

If you use DrosoMem itself, cite the software using
[`CITATION.cff`](CITATION.cff) and, after archival, the Zenodo DOI for the
specific release.

If you use or redistribute FlyWire-derived connectivity, also follow FlyWire's
current citation guidance and the CC BY-NC 4.0 terms. Relevant primary-source
links and the project's source review are collected in
[`docs/research.md`](docs/research.md).

## 6. No new biological data collection

DrosoMem does not report a new experiment on living flies and does not collect
human-subject data. It is a computational research-software project built on
public source data and generated simulations.
