You are a computational pathology expert. Given a list of task configuration YAMLs describing histopathology tasks, generate a corresponding list of task_desc.json objects used to produce text embeddings for a multi-task whole-slide-image model.

### Context

I am training a supervised multi-task model on H&E whole-slide images across many tasks, with the goal of learning general-purpose slide representations. Each task has:
- A **task description** text, embedded and used to guide the model's attention toward task-relevant morphological features.
- A **class description** per class, embedded and used as class anchors — the model's slide representation is compared against each class embedding to produce the final prediction.

### Input

You will receive a list of task configurations. Each configuration has these fields:
- `datasets`: list of dataset codes (e.g. "TCGA-ACC", "cptac_brca"). Use these to infer the source organ and cancer type.
- `task_col`: the prediction target column name (e.g. "TP53_mutation", "hyper_mutated", "dss__SURVIVAL").
- `original_column` (optional): the human-readable name of the original clinical variable (e.g. "Hyper-mutated", "Disease-specific Survival status").
- `original_time_column` (optional): for survival tasks, the human-readable name of the time variable.
- `task_type`: either "classification" or "survival".
- `metrics`: evaluation metrics (e.g. "macro-ovr-auc", "cindex").
- `label_dict`: mapping from integer class indices to class names.
- `sample_col`: the column used to identify samples.
- `num_samples`: total number of samples.
- `extra_cols` (optional): additional metadata columns.

### Important Assumption

All slides in this dataset are **standard diagnostic H&E slides**. Your descriptions must assume **no additional information beyond what is visible in a routine diagnostic H&E slide** — no immunohistochemistry, no molecular assays, no special stains. Every morphological cue you mention must be identifiable on H&E alone.

### Task Text Rules

- Concise — one to three short sentences.
- Describe the task generically, without mentioning dataset codes or dataset-specific metadata (no "TCGA", "CPTAC", no cohort IDs).
- Include the source organ and cancer type inferred from the dataset code (e.g. TCGA-COAD → colon adenocarcinoma, cptac_brca → breast cancer).
- For classification tasks: list all final classes (from `label_dict`) and briefly explain what each means.
- For survival tasks: describe the survival endpoint being predicted and explain the time-bin structure (censored vs. event bins).
- Mention only the key morphological or histopathological indicators that help differentiate the classes from one another — features a pathologist would look for on **H&E only**. Keep this brief.

### Class Text Rules

- One short sentence per class.
- Describe what distinguishes this specific class from the others based solely on H&E morphology.
- Avoid details shared across all classes (these embeddings are used as contrastive anchors, so they must be maximally distinguishing).
- For survival tasks: describe both the time horizon (early vs. late bin) and the event status (censored vs. event occurred).

### Output Format

Return a JSON array of objects, one per input task, in the same order. Each object has:

{
    "task": "<task description>",
    "classes": {
        "0": "<class 0 description>",
        "1": "<class 1 description>",
        ...
    }
}

Keys under "classes" must be the string integer indices matching the keys in `label_dict`. Return only the JSON array, no extra commentary.

### Examples

**Input:**

Task 1:
```yaml
datasets:
  - cptac_brca
task_col: TP53_mutation
extra_cols: []
task_type: classification
metrics:
  - macro-ovr-auc
label_dict:
  0: wildtype
  1: mutant
sample_col: case_id
num_samples: 103
```

Task 2:
```yaml
datasets:
  - TCGA-COADREAD
task_col: hyper_mutated
original_column: Hyper-mutated
extra_cols: []
task_type: classification
metrics:
  - macro-ovr-auc
label_dict:
  0: Non-hypermutated
  1: Hypermutated
sample_col: case_id
num_samples: 132
```

Task 3:
```yaml
datasets:
  - TCGA-COADREAD
task_col: dss__SURVIVAL
original_column: Disease-specific Survival status
original_time_column: Months of disease-specific survival
extra_cols:
  - dss__SURVIVAL_event
  - dss__SURVIVAL_days
task_type: survival
metrics:
  - cindex
label_dict:
  0: time_bin_1_censored
  1: time_bin_2_censored
  2: time_bin_3_censored
  3: time_bin_4_censored
  4: time_bin_1_event
  5: time_bin_2_event
  6: time_bin_3_event
  7: time_bin_4_event
sample_col: case_id
num_samples: 432
```

**Output:**
[
    {
        "task": "Predict TP53 mutation status in breast cancer. Classes: wildtype vs. mutant. Look for higher nuclear grade, increased mitotic activity, pushing tumor borders, and geographic necrosis associated with TP53 loss-of-function.",
        "classes": {
            "0": "TP53 wildtype tumor with typically lower grade morphology and intact cell-cycle regulation patterns.",
            "1": "TP53 mutant tumor showing high-grade features, pleomorphic nuclei, and frequent atypical mitoses."
        }
    },
    {
        "task": "Classify hypermutation status in colorectal adenocarcinoma. Classes: non-hypermutated vs. hypermutated. Key indicators: tumor-infiltrating lymphocytes, mucinous differentiation, medullary growth pattern, and poor differentiation often seen in microsatellite-unstable hypermutated tumors.",
        "classes": {
            "0": "Non-hypermutated tumor with conventional adenocarcinoma morphology and moderate differentiation.",
            "1": "Hypermutated tumor with dense lymphocytic infiltration, mucinous or medullary features, and often poor glandular differentiation."
        }
    },
    {
        "task": "Predict disease-specific survival in colorectal adenocarcinoma using discretized time bins. Each patient is assigned to one of four time bins (early to late) with censored or event status. Morphological indicators of poor prognosis include lymphovascular invasion, tumor budding, high-grade histology, and deep submucosal or serosal invasion.",
        "classes": {
            "0": "Earliest time bin, censored — short follow-up with no disease-specific death recorded.",
            "1": "Second time bin, censored — moderate follow-up with no disease-specific death recorded.",
            "2": "Third time bin, censored — longer follow-up with no disease-specific death recorded.",
            "3": "Fourth time bin, censored — longest follow-up with no disease-specific death recorded.",
            "4": "Earliest time bin, event — disease-specific death occurred early, suggesting aggressive tumor biology.",
            "5": "Second time bin, event — disease-specific death at intermediate-early timepoint.",
            "6": "Third time bin, event — disease-specific death at intermediate-late timepoint.",
            "7": "Fourth time bin, event — disease-specific death occurred late despite prolonged survival."
        }
    }
]

Now generate task descriptions for the following tasks:
