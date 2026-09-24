"""FACET: task-conditioned whole-slide representation model."""

from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from ..config import FACETConfig
from .alibi import alibi_2d_bias
from .layers import ResidualMLPEmbedder, TransformerBlock
from .pooling import TaskConditionedPooling


class FACET(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="facet",
    license="cc-by-nc-nd-4.0",
    tags=["pathology", "whole-slide-image", "foundation-model", "multiple-instance-learning"],
):
    """Task-conditioned slide encoder.

    Given patch features ``X`` of a slide and the text embedding of a task
    description ``g(s_t)``, returns the task-conditioned slide representation
    ``z_t``.

    Load weights with ``FACET.from_pretrained(<hub id or local directory>)``.
    """

    def __init__(self, config: Optional[Union[FACETConfig, dict]] = None):
        super().__init__()
        if config is None:
            config = FACETConfig()
        elif isinstance(config, dict):
            config = FACETConfig.from_dict(config)
        self.config = config
        c = config

        self.text_projection = nn.Linear(c.text_dim, c.embed_dim)
        self.patch_embed = ResidualMLPEmbedder(c.input_dim, c.embed_dim, c.dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(c.embed_dim, c.num_heads, c.mlp_ratio, c.dropout)
            for _ in range(c.num_layers)
        )
        self.task_query = ResidualMLPEmbedder(c.embed_dim, c.embed_dim, c.dropout)
        self.pooling = TaskConditionedPooling(c.embed_dim, c.num_heads, c.mlp_ratio, c.dropout)
        self.norm = nn.LayerNorm(c.embed_dim)

    @property
    def embed_dim(self) -> int:
        return self.config.embed_dim

    def encode_prototypes(self, text_embedding: torch.Tensor) -> torch.Tensor:
        """Project text embeddings of class descriptions into the slide space.

        Args:
            text_embedding: (..., text_dim) text-encoder outputs.

        Returns:
            (..., embed_dim) class prototypes ``Proj(g(s_c))``.
        """
        return self.text_projection(text_embedding)

    def encode_task(self, text_embedding: torch.Tensor) -> torch.Tensor:
        """Map text embeddings of task descriptions to pooling queries.

        Args:
            text_embedding: (B, text_dim) text-encoder outputs.

        Returns:
            (B, embed_dim) task queries.
        """
        return self.task_query(self.text_projection(text_embedding))

    def encode_patches(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Contextualize patch features with the ALiBi transformer.

        Args:
            features: (B, N, input_dim) patch features.
            coords: (B, N, 2) integer patch-grid coordinates.
            attn_mask: (B, N), True for valid patches.

        Returns:
            (B, N, embed_dim) patch tokens.
        """
        x = self.patch_embed(features)
        num_heads = self.config.num_heads
        bias = alibi_2d_bias(coords, num_heads, attn_mask)
        bias = bias.reshape(-1, bias.shape[-2], bias.shape[-1])  # (B * H, N, N)
        key_padding_mask = ~attn_mask if attn_mask is not None else None
        for block in self.blocks:
            x = block(x, attn_bias=bias, key_padding_mask=key_padding_mask)
        return x

    def forward(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        task_embedding: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute task-conditioned slide representations.

        Args:
            features: (B, N, input_dim) or (N, input_dim) patch features.
            coords: (B, N, 2) or (N, 2) integer patch-grid coordinates
                (level-0 pixel coordinates divided by the patch size).
            task_embedding: (B, text_dim) or (text_dim,) text-encoder embedding
                of the task description.
            attn_mask: optional (B, N) mask, True for valid (non-padded) patches.

        Returns:
            (B, embed_dim) task-conditioned slide representations ``z_t``
            (or (embed_dim,) for unbatched inputs).
        """
        unbatched = features.dim() == 2
        if unbatched:
            features, coords = features.unsqueeze(0), coords.unsqueeze(0)
            if attn_mask is not None:
                attn_mask = attn_mask.unsqueeze(0)
        if task_embedding.dim() == 1:
            task_embedding = task_embedding.unsqueeze(0)
        if task_embedding.shape[0] == 1 and features.shape[0] > 1:
            task_embedding = task_embedding.expand(features.shape[0], -1)

        tokens = self.encode_patches(features, coords, attn_mask)
        query = self.encode_task(task_embedding).unsqueeze(1)  # (B, 1, D)
        key_padding_mask = ~attn_mask if attn_mask is not None else None
        z = self.pooling(query, tokens, key_padding_mask=key_padding_mask).squeeze(1)
        z = self.norm(z)
        return z.squeeze(0) if unbatched else z
