"""Export task-conditioned FACET embeddings in a Patho-Bench-compatible layout.

For every task folder ``<tasks_root>/<dataset>/<task>/`` containing
``config.yaml``, a split/label file (``k=all.tsv`` or ``labels.tsv``) and
``task_desc.json``, writes::

    <output_dir>/<dataset>/<task>/<slide_id>.h5
    <output_dir>/<dataset>/<task>/<case_id>.h5     # if config sample_col == case_id

Pass ``<output_dir>/<dataset>/<task>`` to Patho-Bench as ``pooled_embeddings_dir``.

Example::

    python scripts/extract_embeddings.py \
        --model /path/to/facet \
        --tasks-root data/evaluation \
        --features-dir /path/to/trident/20x_512px_0px_overlap/features_conch_v15 \
        --output-dir embeddings/facet

Optional clinical context (appended to the task description per patient)::

    --clinical-csv clinical.csv --clinical-column smoking \
    --clinical-format-map '{"True": "The patient is a smoker.", "False": "The patient is not a smoker."}'
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
import yaml

from facet import FACETPipeline
from facet.io import (
    group_slides_by_case,
    index_feature_files,
    load_patch_features,
    write_case_embeddings,
    write_embedding,
)
from facet.text import TaskDescription, augment_task_description, parse_format_map

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("facet.extract")

SPLIT_FILES = ("k=all.tsv", "labels.tsv")


def discover_tasks(tasks_root: Path, selected: Optional[List[str]] = None) -> List[Path]:
    tasks = sorted(p.parent for p in tasks_root.glob("*/*/config.yaml"))
    if selected:
        wanted = set(selected)
        tasks = [t for t in tasks if f"{t.parent.name}/{t.name}" in wanted or t.name in wanted]
    return tasks


def read_samples(task_dir: Path) -> pd.DataFrame:
    for name in SPLIT_FILES:
        path = task_dir / name
        if path.exists():
            df = pd.read_csv(path, sep="\t", dtype={"case_id": str, "slide_id": str})
            return df[["case_id", "slide_id"]].drop_duplicates()
    raise FileNotFoundError(f"no {' / '.join(SPLIT_FILES)} in {task_dir}")


def load_clinical(csv_path: str, column: str) -> Dict[str, object]:
    df = pd.read_csv(csv_path, dtype={"case_id": str})
    if "case_id" not in df.columns or column not in df.columns:
        raise ValueError(f"{csv_path} must contain 'case_id' and '{column}' columns")
    return df.set_index("case_id")[column].to_dict()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HuggingFace repo id or local FACET directory")
    ap.add_argument("--tasks-root", type=Path, required=True, help="Root with <dataset>/<task>/ folders")
    ap.add_argument("--features-dir", type=Path, nargs="+", required=True,
                    help="Director(ies) searched recursively for <slide_id>.h5 TRIDENT feature files")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="Subset of tasks, as <dataset>/<task> or <task> (default: all)")
    ap.add_argument("--text-encoder-checkpoint", default=None,
                    help="Local CONCH pytorch_model.bin (default: hf_hub:MahmoodLab/conch)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--clinical-csv", default=None, help="CSV with case_id and a clinical column")
    ap.add_argument("--clinical-column", default=None)
    ap.add_argument("--clinical-format-map", default=None,
                    help='JSON mapping column values to sentences; optional "Unknown" key for missing values')
    args = ap.parse_args()

    clinical_args = [args.clinical_csv, args.clinical_column, args.clinical_format_map]
    if any(clinical_args) and not all(clinical_args):
        ap.error("--clinical-csv, --clinical-column and --clinical-format-map must be given together")
    clinical = load_clinical(args.clinical_csv, args.clinical_column) if args.clinical_csv else None
    format_map = parse_format_map(args.clinical_format_map) if args.clinical_format_map else None

    tasks = discover_tasks(args.tasks_root, args.tasks)
    if not tasks:
        logger.error("no task folders found under %s", args.tasks_root)
        return 1

    feature_index = index_feature_files(args.features_dir)
    logger.info("indexed %d feature files", len(feature_index))

    pipe = FACETPipeline.from_pretrained(
        args.model, text_encoder_checkpoint=args.text_encoder_checkpoint, device=args.device
    )

    for task_dir in tasks:
        dataset, task = task_dir.parent.name, task_dir.name
        with open(task_dir / "config.yaml", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        desc = TaskDescription.load(task_dir)
        samples = read_samples(task_dir)
        out_dir = args.output_dir / dataset / task

        missing = sorted(set(samples["slide_id"]) - set(feature_index))
        if missing:
            logger.warning("%s/%s: %d slides without features (skipped)", dataset, task, len(missing))
        samples = samples[samples["slide_id"].isin(feature_index)]

        n_written = 0
        for case_id, slide_id in samples.itertuples(index=False):
            out_path = out_dir / f"{slide_id}.h5"
            if out_path.exists() and not args.overwrite:
                continue
            task = desc
            if clinical is not None:
                if case_id not in clinical:
                    raise KeyError(f"case {case_id!r} missing from {args.clinical_csv}")
                task = augment_task_description(desc, clinical[case_id], format_map)
            features, coords = load_patch_features(feature_index[slide_id])
            with torch.no_grad():
                z = pipe.embed_slide((features, coords), task=task)
            write_embedding(out_path, z)
            n_written += 1

        n_cases = 0
        if config.get("sample_col") == "case_id":
            groups = group_slides_by_case(samples["case_id"].tolist(), samples["slide_id"].tolist())
            n_cases = write_case_embeddings(out_dir, groups, overwrite=args.overwrite)
        logger.info("%s/%s: %d slide and %d case embeddings written", dataset, task, n_written, n_cases)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
