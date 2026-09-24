# FACET pretraining tasks

FACET was pretrained with supervised multitask learning on **29 prediction tasks** from
six public TCGA cohorts, covering **3,359 diagnostic H&E whole-slide images**.

## Provenance and curation

Candidate tasks were built from clinical and molecular attributes in
[cBioPortal](https://www.cbioportal.org) for TCGA-BRCA, TCGA-NSCLC (LUAD + LUSC),
TCGA-ESCA, TCGA-STAD, TCGA-COADREAD (COAD + READ) and TCGA-UCEC. Curation steps:

1. Label transformation: rare categories grouped where appropriate; ill-defined labels removed.
2. Filtering: candidates with too few samples or severe class imbalance removed.
3. Morphological-signal screening with a simple ABMIL baseline.
4. Manual selection of a diverse, clinically and biologically meaningful set.

Disease-specific survival (`DSS_SURVIVAL`) tasks are discretized into 4 time bins
crossed with the event indicator (8 classes), and were optimized with a survival
cross-entropy loss.

Task and class descriptions (`task_desc.json`) were generated with GPT-5.4 using the
fixed prompt in [`facet/text/prompts/description_generation.md`](../../facet/text/prompts/description_generation.md),
conditioned only on each task's `config.yaml`.

## Layout

```
<cohort>/<task>/
    config.yaml       # task metadata (Patho-Bench style)
    labels.tsv        # one row per slide
    task_desc.json    # task and class descriptions
```

- `config.yaml`: `task_col`, `task_type` (`classification` | `survival`), `label_dict`
  (class index to name), `sample_col`, `num_samples`, and for survival tasks the
  `extra_cols` holding event indicator and time (days).
- `labels.tsv`: `case_id` (TCGA patient barcode), `slide_id` (slide identifier, the stem of
  the TRIDENT feature file), the label column named by `task_col`, and for survival tasks
  `<task_col>_event` and `<task_col>_days`.

## Terms

TCGA data are provided by the NCI Genomic Data Commons and cBioPortal under their
respective terms of use. The label files here are derived annotations released under the
repository license (CC BY-NC-ND 4.0). Whole-slide images are not included; download them
from the [GDC portal](https://portal.gdc.cancer.gov).
