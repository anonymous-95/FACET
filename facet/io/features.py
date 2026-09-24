"""Reading patch features produced by TRIDENT."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple, Union

import h5py
import torch

PathLike = Union[str, Path]
DEFAULT_PATCH_SIZE_LEVEL0 = 512


def load_patch_features(h5_path: PathLike) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load a TRIDENT feature file.

    Returns:
        features: (N, D) float32 patch features.
        coords: (N, 2) int64 patch-grid coordinates, i.e. level-0 pixel
            coordinates divided by ``coords.attrs["patch_size_level0"]``.
    """
    with h5py.File(h5_path, "r") as f:
        features = torch.from_numpy(f["features"][:]).squeeze()
        coords = torch.from_numpy(f["coords"][:]).squeeze()
        patch_size = int(f["coords"].attrs.get("patch_size_level0", DEFAULT_PATCH_SIZE_LEVEL0))
    if features.dim() == 1:
        features, coords = features.unsqueeze(0), coords.reshape(1, 2)
    return features.float(), (coords // patch_size).long()


def index_feature_files(feature_dirs: Iterable[PathLike], recursive: bool = True) -> Dict[str, Path]:
    """Map ``slide_id`` (file stem) to its ``.h5`` path across one or more directories."""
    index: Dict[str, Path] = {}
    for d in feature_dirs:
        d = Path(d)
        for p in (d.rglob("*.h5") if recursive else d.glob("*.h5")):
            index.setdefault(p.stem, p)
    return index
