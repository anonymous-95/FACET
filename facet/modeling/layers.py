"""Building blocks of the FACET slide encoder."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class ResidualMLPEmbedder(nn.Module):
    """Two-layer MLP with a residual connection and LayerNorm.

    Used to embed patch features into the encoder space and to map the
    projected task embedding to the pooling query.
    """

    def __init__(self, input_dim: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )
        self.residual = nn.Identity() if input_dim == embed_dim else nn.Linear(input_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x) + self.residual(x))


class FeedForward(nn.Sequential):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        hidden = int(dim * mlp_ratio)
        super().__init__(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )


class TransformerBlock(nn.Module):
    """Pre-norm transformer block; spatial information enters via the ALiBi bias."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, mlp_ratio, dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(
            h, h, h, attn_mask=attn_bias, key_padding_mask=key_padding_mask, need_weights=False
        )
        x = x + h
        return x + self.mlp(self.norm2(x))
