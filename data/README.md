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
