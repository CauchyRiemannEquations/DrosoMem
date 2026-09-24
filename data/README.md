# Data

The repository includes an actual **FlyWire FAFB v783 induced subset**, not a
synthetic fly-like network: 300 original root IDs and 3,303 directed edges.

Origin: Philip Shiu / Nico Spiller's author-distributed processed v783 connectivity
at the pinned upstream commit recorded in `flywire_783_subset/provenance.json`.
The source repository's MIT notice is retained in `UPSTREAM_LICENSE.txt`.
For original FlyWire data terms and attribution also consult the
[official archive](https://zenodo.org/records/10676866),
[FlyWire](https://flywire.ai/) and the papers cited in the main README.
Our code's license does not relicense upstream data or waive citation obligations.

Extraction: remove self connections; keep pairs with at least 5 synapses;
rank neurons by total retained incoming + outgoing synapse count (ties by root ID);
select the top 300 and retain all remaining edges within them. No connection is
invented. This intentionally small, hub-biased subset is not a named anatomical
circuit or a representative random sample of the whole brain.

`edges.csv`: decimal-string-safe `pre`, `post`, positive `count`, `sign` (+1/-1).
`neurons.json`: all selected IDs, including any isolated nodes.
`provenance.json`: source revision, checksums, selection and dimensions.
Count × sign is rescaled globally for reservoir dynamics. Synapse count is not a
measured physiological efficacy. Signs are upstream modeling assumptions.

The full ~101 MB parquet is excluded from Git. Regenerate the subset with:

```bash
python -m pip install -e '.[data]'
python scripts/download_connectome.py
```

For a larger subset (up to 3000), give `--neurons 1000 --output data/flywire_783_1000`
and update the config path and `max_neurons`. CSV/IDs are validated by checksums
on every run. Missing/corrupt real data causes an error, never a synthetic fallback.

## Phase 2 bundled subsets

The original subset remains unchanged. Three additional 300-neuron subsets use
randomized breadth-first growth from seeded starting cells in the largest weakly
connected component of the same thresholded source graph (seeds 101, 202, 303).
Every saved edge is an original induced connection; weak connectivity is guaranteed,
strong connectivity is not. Seeded neighborhoods are not independent biological
samples or named anatomical regions, and may overlap. A fourth new subset contains
the top 1,000 neurons by retained incident synapse count. All source hashes,
starting root IDs, selected IDs and edge hashes are bundled.

To rebuild these four new subsets into a fresh destination:

```bash
python -m pip install -e ".[data]"
python scripts/build_phase2_subsets.py --output data/rebuilt_phase2
```

The raw pinned parquet from `download_connectome.py` must already exist. The
builder streams its rows, then holds the thresholded sparse graph in RAM for
selection (more memory than the small simulations). It never runs whole-brain
neural dynamics. Connected-subset folders must not already exist.

## Phase 3 anatomical subsets

`flywire_783_mb_left_kc512_s701` and `s702` contain 686 neurons each: 512 KC, 48 MBON, 125 DAN and one APL. Their 3,309/3,241 edges are real, thresholded FlyWire v783 connections. Each includes `annotations.csv`, exact root IDs, edge counts/signs and source/annotation hashes. See [Phase 3 protocol](../docs/phase3-results.md) for deterministic selection and limitations.

## Data licensing and attribution

Flying's MIT code license does **not** relicense the underlying FlyWire data. The [official FlyWire public-release guidelines](https://flywire.ai/guidelines) specify **CC BY-NC 4.0** for the public data; see [license terms](https://creativecommons.org/licenses/by-nc/4.0/). Bundled data are transformed subsets (selection, thresholding, removal of autapses) of that release. The separately retained upstream MIT notice applies to the author software repository; it is not a replacement for FlyWire data terms.

Credit the FlyWire Consortium, Dorkenwald et al. (2024), Shiu et al. (2024) for the processed model data, and the annotation authors. The [annotation repository](https://github.com/flyconnectome/flywire_annotations#how-to-cite) requests Berg et al. (2025), Schlegel et al. (2024), Matsliah et al. (2024) and Dorkenwald et al. (2024) for current annotations. Consult its pinned source and official citation guide when publishing.
