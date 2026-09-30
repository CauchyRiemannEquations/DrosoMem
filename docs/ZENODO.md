# Zenodo release guide for DrosoMem

This file is the release checklist and ready-to-paste metadata for archiving
DrosoMem as **research software**. It is not a paper template.

## Recommended first record

- **Resource type:** Software
- **Title:** DrosoMem: Sequence Memory in a Fly Connectome
- **Version:** 0.1.0
- **Repository:** <https://github.com/CauchyRiemannEquations/DrosoMem>
- **Code license:** MIT
- **Creator for the current pseudonymous release metadata:**
  CauchyRiemannEquations
- **Keywords:** Drosophila; connectome; sequence memory; reservoir computing;
  neural networks; computational neuroscience; reproducible research

If you want the Zenodo record to show a personal name and/or ORCID instead of
the GitHub identity, edit `CITATION.cff` before creating the GitHub release.

## Suggested description

DrosoMem is exploratory research software for studying sequential recall in
computational models constrained by Drosophila connectome structure. The
repository contains source code, reproducible configurations, provenance
records, evaluation pipelines, structural controls, robustness analyses and
preserved positive and negative results.

The project distinguishes representation, decoding and internal learning.
Experiments include trained-sequence recall, random and shuffled sequence
controls, anatomically restricted and whole-graph models, rewired structural
controls, and synthetic observation-noise studies.

DrosoMem does **not** claim that living flies memorize mathematical constants,
that the model predicts unseen digits, that connectome structure alone
reconstructs biological memory, or that the implemented noise and neural
dynamics are physiologically validated.

Some bundled connectivity files are transformed subsets of the public FlyWire
FAFB v783 release. DrosoMem code is MIT-licensed; FlyWire-derived public data
remain subject to their upstream CC BY-NC 4.0 terms and attribution
requirements. See `DATA_SOURCES.md`, `data/README.md` and `LIMITATIONS.md`
for provenance and interpretation boundaries.

## Why there is no .zenodo.json

Zenodo supports both `CITATION.cff` and `.zenodo.json` for GitHub software
releases. When both are present, Zenodo uses `.zenodo.json` and ignores
`CITATION.cff` for the release metadata.

DrosoMem therefore keeps a single `CITATION.cff` source of citation metadata
unless Zenodo-specific fields such as a community, grant or special related
identifier are later needed.

## Before publishing

1. Read [`CITATION.cff`](../CITATION.cff) and decide whether the creator
   should remain the GitHub identity or be replaced by a preferred personal
   name and ORCID.
2. Confirm [`DATA_SOURCES.md`](../DATA_SOURCES.md) and
   [`LIMITATIONS.md`](../LIMITATIONS.md) still match the tagged commit.
3. Run the project's verification/test commands appropriate to the release.
4. Check the actual source-archive size before publishing. The GitHub repository
   currently contains a large amount of committed output, so the archive may be
   substantially larger than a typical software release.
5. Do not present the MIT license as covering FlyWire-derived data.

## GitHub → Zenodo workflow

1. Sign in to Zenodo and connect the GitHub account.
2. In Zenodo's GitHub integration, sync repositories and enable
   `CauchyRiemannEquations/DrosoMem`.
3. Create a GitHub release from the exact commit intended for archival.
4. For the current package version, use tag **v0.1.0** unless the project version
   is intentionally changed first.
5. Let Zenodo ingest the GitHub release and inspect the draft/record metadata.
6. Confirm title, creator, version, description, license and files.
7. Publish the Zenodo record and record the assigned DOI.
8. Add the DOI back to the README and, if desired, to a later version of
   `CITATION.cff`.

Zenodo's normal record quota is currently 50 GB. If GitHub release ingestion
fails because of the archive or metadata, inspect the Zenodo integration error
before making a new release. A manual software deposit is also possible; for
Software Heritage archival Zenodo recommends a single compressed source-code
file.

## Software record vs. dataset record

For the first public archival step, keep DrosoMem as one **Software** record.

If the result artifacts later become a separately curated research object, make
a second **Dataset** record containing only the documented analysis outputs,
manifests and checksums that are intended for reuse. Link the software and
dataset records rather than treating every generated file as part of the
software citation.

## After DOI assignment

Add a small README badge or citation line such as:

`Archived release: Zenodo DOI 10.5281/zenodo.XXXXXXX`

Use the DOI of the specific release when reproducibility depends on a particular
version. Use Zenodo's concept DOI when referring to the evolving software
project as a whole.
