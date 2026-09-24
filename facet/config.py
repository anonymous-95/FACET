"""FACET model configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict


@dataclass
class FACETConfig:
    """Architecture hyperparameters of FACET, serialized as ``config.json``.

    Attributes:
        input_dim: dimension of the patch features (CONCH v1.5: 768).
        embed_dim: hidden size of the slide encoder and of ``z_t``.
        text_dim: dimension of the text-encoder output (CONCH v1: 512).
        num_heads: attention heads in the transformer and pooling layers.
        num_layers: number of transformer blocks.
        mlp_ratio: hidden-size multiplier of the feed-forward layers.
        dropout: dropout rate (inactive in ``eval()`` mode).
        patch_encoder: patch encoder the model expects features from.
        text_encoder: text encoder the model expects embeddings from.
    """

    input_dim: int = 768
    embed_dim: int = 256
    text_dim: int = 512
    num_heads: int = 4
    num_layers: int = 2
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    patch_encoder: str = "conch_v15"
    text_encoder: str = "conch_v1"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FACETConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown FACETConfig keys: {sorted(unknown)}")
        return cls(**d)
