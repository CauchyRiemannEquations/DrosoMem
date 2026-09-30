# Phase 0 — source and implementation review

Checked 2026-09-24. Links below are primary sources. “Current” applies to the
named dataset and documentation, not all fruit-fly reconstructions.

Update 2026-09-27: Brian2 2.10.1 has now been installed and used for the
[small-circuit Stage B/C experiment](stage-bc-results.md). Its [locked parameter
and input protocol](stage-bc-protocol.md) cites the pinned author model and
versioned Brian2 semantics. The table below records the earlier Phase 0 review.

| Resource | Verified finding | Decision / verification status |
|---|---|---|
| [FlyWire/Codex FAQ](https://codex.flywire.ai/faq) | FAFB snapshot 783; FAQ also lists BANC 888 and other datasets | Pin FAFB 783; do not call it the newest of all fly datasets |
| [FlyWire Consortium Zenodo](https://zenodo.org/records/10676866) | `proofread_connections_783.feather` (852 MB), proofread root IDs, full synapse table (9.5 GB); pre/post root IDs, syn_count, neuropil and transmitter probabilities | Official reference archive; not downloaded for this small MVP |
| [Shiu original repository](https://github.com/philshiu/Drosophila_brain_model) | Author-distributed `Connectivity_783.parquet`, `Completeness_783.csv`; paper default originally v630, v783 configurable | Downloaded ~101 MB table from pinned commit; verified hash, schema, extracted and executed |
| [fafbseg get_adjacency](https://fafbseg-py.readthedocs.io/en/latest/source/generated/fafbseg.flywire.synapses.get_adjacency.html) | `sources`, `targets`, `materialization`, `dataset`, `batch_size`; rows are sources, columns targets | Documented alternative; NOT executed here, no CAVE token supplied |
| [fafbseg setup](https://fafbseg-py.readthedocs.io/en/latest/source/tutorials/flywire_setup.html) | CAVE/FlyWire authentication setup | Follow current official setup rather than committing tokens |
| [navis](https://navis-org.github.io/navis/stable/) | Neuron morphology analysis and visualization, meshes/skeletons | Optional for morphology later; not a simulation engine and not required for MVP |
| [Brian2 tutorial](https://brian2.readthedocs.io/en/stable/resources/tutorials/1-intro-to-brian-neurons.html) | NeuronGroup differential equations, units, thresholds and reset | Appropriate for LIF next stage; not installed/run in this experiment |
| [Shiu et al., Nature 2024](https://www.nature.com/articles/s41586-024-07763-9) | Published connectome-constrained computational sensorimotor model | Biological modeling reference; does not establish π memory |
| [Eon fly-brain](https://github.com/eonsystemspbc/fly-brain) | Public whole-brain LIF implementations and multiple compute backends based on Shiu | Repository inspected through current public README; not benchmarked or validated here |

## Verified data path

The author-distributed parquet has 15,091,983 rows and fields
`Presynaptic_ID`, `Postsynaptic_ID`, `Presynaptic_Index`, `Postsynaptic_Index`,
`Connectivity`, `Excitatory`, `Excitatory x Connectivity` and an index field.
The last weight column is also used in the authors' `model.py` (lines 182–183
at the pinned commit). We use count × upstream sign without pretending that
these simplified signs, our activation function, or normalization recover true
biophysical currents. All signs in this source are +1 or -1.

The source commit is `91bdd1e7dcf193f3e7ca5a8933497fcef63b7960`.
Raw SHA-256 is in `data/flywire_783_subset/provenance.json`; download and
extraction both reject an unexpected hash. IDs are never converted through float.

A documented alternative after following official authentication setup is:

```python
from fafbseg import flywire
adj = flywire.get_adjacency(
    sources=root_ids, targets=root_ids,
    materialization=783, dataset="public", batch_size=100,
)
# adj rows = pre, cols = post. DrosoMem's recurrent W must be adj.T.
```

This is a documentation-verified example, not a claim that authenticated API
access was tested. Large queries can be truncated; heed batch warnings.

## Technology choice

NumPy + SciPy sparse CSR and a 10-class affine softmax readout are sufficient.
Only readout weights/bias are optimized; no recurrent or encoder gradients.
CPU, one numerical thread, no GPU, no server, no morphology downloads.
The heavy parquet dependency is optional for regenerating the subset only.
Static data makes the default quick start independent of live API changes.

## Interpretation and future tests

The first experiment is a plumbing and measurement result. Its real subset is
hub-biased, some neurons may become isolated when outside connections are cut,
and count thresholding discards weak connections. A shared spectral radius
controls one scalar, not all dynamics: strong hubs can determine the scaling,
leaving many connections very small. Network mixing, effective rank, signs,
normalization and stimulation placement remain confounds. Three encoder seeds
on one selected subset do not characterize biological variability.

Before whole-brain or plasticity work: compare multiple independently selected
connected subsets, characterize SCCs/isolates and state effective rank, sweep
sequence length and stability parameters using a separate development protocol,
then freeze settings and repeat on fresh seeds. Add random-digit and shuffled-π
sequence controls, recurrence-removal ablations, multiple recall starting
positions and delayed-input memory capacity tasks. Do not tune on the heldout
suffix and later present that suffix as a clean test. π normality is not proven;
“10% chance” is a uniform-guess reference, not a theorem about this finite sample.
