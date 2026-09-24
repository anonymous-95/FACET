"""Task-conditioned attention pooling."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .layers import FeedForward


class TaskConditionedPooling(nn.Module):
    """Cross-attention pooling with the task embedding as the single query.

    ``u = q_t + MHA(LN(q_t), LN(H), LN(H))`` followed by ``u + FFN(LN(u))``.
    """

    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_ff = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, mlp_ratio, dropout)

    def forward(
        self,
        query: torch.Tensor,
        tokens: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: (B, 1, D) task query.
            tokens: (B, N, D) contextualized patch tokens.
            key_padding_mask: (B, N), True for padded positions.

        Returns:
            (B, 1, D) pooled representation.
        """
        kv = self.norm_kv(tokens)
        out, _ = self.attn(
            self.norm_q(query), kv, kv, key_padding_mask=key_padding_mask, need_weights=False
        )
        out = out + query
        return out + self.mlp(self.norm_ff(out))
