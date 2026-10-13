# Manuscript literature context and claim-access notes

Reviewed 2026-10-13. This is a representative, claim-focused primary-literature
selection for [the reviewed draft](manuscript-draft.md), not a systematic review,
exhaustive priority search or novelty certification. No third-party experiment
was reproduced and no third-party figure is imported. The
[machine ledger](../configs/manuscript_literature.json) records access scope,
locators, metadata sources and bounded claim roles;
[BibTeX](manuscript-references.bib) supplies the bibliographic titles.

## Primary references and access boundaries

| ID | Record | Access checked | Use in the draft |
| --- | --- | --- | --- |
| L1 | [Dorkenwald et al., 2024](https://doi.org/10.1038/s41586-024-07558-y) | Publisher-indexed abstract/citation and Crossref metadata; direct publisher opening intermittently failed | Structural resource context |
| L2 | [Schlegel et al., 2024](https://doi.org/10.1038/s41586-024-07686-5) | Publisher abstract/main annotation context and Crossref metadata | Role/annotation context |
| L3 | [Shiu et al., 2024](https://doi.org/10.1038/s41586-024-07763-9) | Publisher abstract/model/sensorimotor-validation passages and Crossref metadata | Distinct spiking/empirical model context |
| L4 | [Maass et al., 2002](https://doi.org/10.1162/089976602760407955) | Primary abstract indexed by PubMed and Crossref metadata; direct PubMed body did not consistently load | Transient-state/readout framework |
| L5 | [Jaeger, 2001](https://www.ai.rug.nl/minds/uploads/STMEchoStatesTechRep.pdf) | Author-institution publication list and original report metadata; PDF text extraction incomplete and a later fetch timed out | Historical echo-state memory context only |
| L6 | [Dambre et al., 2012](https://doi.org/10.1038/srep00514) | [Author-institution published PDF](https://biblio.ugent.be/publication/2972574/file/2972575.pdf), Definition1/Theorem4 and bibliographic metadata | Formal reconstruction-capacity distinction |
| L7 | [Suárez et al., 2024](https://doi.org/10.1038/s41467-024-44900-4) | Publisher full-text framework/tutorial/data-availability passages and Crossref metadata | Connectome-reservoir precedent and scope differences |
| L8 | [Costi et al., 2025](https://doi.org/10.3390/biomimetics10050341) | Publisher-indexed abstract/conclusions, [authors' ESA project page](https://www.esa.int/gsp/ACT/projects/fly_connectome/) and Crossref metadata; direct MDPI/PMC full-text access unsuccessful | Forecasting/overfitting precedent at reported conditions |
| L9 | [Churchland et al., 2026 v1](https://arxiv.org/abs/2609.07355v1) | Primary arXiv abstract/HTML and version record | Normalization-confound context; preprint status explicit |

Seven records are journal articles, one is an original technical report and one
is a preprint. L9's primary record says submitted to *Neural Networks*; it is
treated here as a preprint rather than inferred to be a published journal article.
The original report L5 is not labelled a journal article. Detailed capacity
claims use the accessible L6 definition; the draft does not imply a full review
of the incompletely parsed L5 report. Access limitations are not hidden behind
an undifferentiated full-text-review claim.

## Comparison discipline

The literature supplies structural, modelling and measurement context. Our
numeric endpoints come from the frozen DrosoMem archives and their successful
independent checks, not those external sources. The preceding
[evidence matrix](research-evidence-matrix.md) remains authoritative for P1/P2
and M1–M11 outcomes and learning budgets.

Three comparisons particularly require scope matching:

- L3's sensorimotor validation is external experimental evidence for its own
  spiking model; it is not validation of our rate/symbolic-memory implementation.
- L7's comparative fruit-fly example uses a different structural resource;
  L8's forecasting results have different tasks and model selection/control
  conditions. Neither supplies direct numerical confirmation or contradiction
  of our categorical registered gates.
- L9 uses a different organism/normalization regime. It contextualizes joint
  structural and rescaling effects; it does not quantify our incoming-L1 effects.

No claim is made that DrosoMem is the first fly-connectome reservoir, that every
related study agrees with its negative endpoints, or that these nine papers
exhaust current work. Generalization beyond the chosen task/model budget remains
a question for separate prospective studies. This text is a literature-context
record, not an additional model study or a change to an old acceptance criterion.

## Bibliographic handling

Journal DOI metadata was checked against Crossref and primary publisher records.
The technical report uses its institutional URL; the preprint's identity and
status come from the primary arXiv v1 record, rather than a journal DOI record. BibTeX
uses the correctly verified first author with `and others` for long author
lists; this is abbreviated bibliography formatting, not removal of contributors
or a proposed authorship list for DrosoMem. Full author lists remain available
from the linked primary metadata.

The [scientific review](manuscript-scientific-review.md) corrects the conn2res
comparison's direction using rechecked publisher-indexed Figure5c text and
retains the access boundaries for the forecasting study and versioned preprint.

This is an internally reviewed draft. No external submission, project
DOI or journal publication is created by adding literature citations to the
repository.
