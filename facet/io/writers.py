"""Writing slide / case embeddings in the layout Patho-Bench reads.

Patho-Bench's ``pooled_embeddings_dir`` expects one ``<sample_id>.h5`` per
sample with a ``features`` dataset. Because FACET embeddings are task
specific, each task gets its own directory::

    <output_dir>/<dataset>/<task>/<slide_id>.h5
    <output_dir>/<dataset>/<task>/<case_id>.h5   # mean over the case's slides
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Union

import h5py
import numpy as np
import torch

PathLike = Union[str, Path]


def write_embedding(path: PathLike, embedding: torch.Tensor) -> None:
    """Save a (D,) or (1, D) embedding as dataset ``features`` of shape (1, D)."""
    arr = embedding.detach().float().cpu().reshape(1, -1).numpy()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("features", data=arr)


def read_embedding(path: PathLike) -> np.ndarray:
    with h5py.File(path, "r") as f:
        return f["features"][()]


def write_case_embeddings(
    task_dir: PathLike,
    case_to_slides: Mapping[str, List[str]],
    overwrite: bool = False,
) -> int:
    """Mean-pool slide embeddings of each case into ``<case_id>.h5``.

    Returns the number of case files written.
    """
    task_dir = Path(task_dir)
    written = 0
    for case_id, slide_ids in case_to_slides.items():
        out = task_dir / f"{case_id}.h5"
        if out.exists() and not overwrite:
            continue
        embs = [read_embedding(task_dir / f"{s}.h5") for s in slide_ids if (task_dir / f"{s}.h5").exists()]
        if not embs:
            continue
        write_embedding(out, torch.from_numpy(np.stack(embs).mean(axis=0)))
        written += 1
    return written


def group_slides_by_case(case_ids: List[str], slide_ids: List[str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for c, s in zip(case_ids, slide_ids):
        groups.setdefault(str(c), [])
        if str(s) not in groups[str(c)]:
            groups[str(c)].append(str(s))
    return groups
