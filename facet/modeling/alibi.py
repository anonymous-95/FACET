"""2D ALiBi attention bias over patch-grid coordinates."""

from __future__ import annotations

import math
from typing import Optional

import torch


def get_alibi_slopes(num_heads: int) -> torch.Tensor:
    """Per-head slopes following the geometric sequence of ALiBi (Press et al., 2022)."""

    def _power_of_2_slopes(n: int):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start * start**i for i in range(n)]

    if math.log2(num_heads).is_integer():
        return torch.tensor(_power_of_2_slopes(num_heads))

    closest = 2 ** math.floor(math.log2(num_heads))
    slopes = _power_of_2_slopes(closest)
    extra = _power_of_2_slopes(2 * closest)
    slopes.extend(extra[0::2][: num_heads - closest])
    return torch.tensor(slopes)


def alibi_2d_bias(
    coords: torch.Tensor,
    num_heads: int,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Additive attention bias from pairwise L1 distances on the patch grid.

    Args:
        coords: (B, N, 2) integer grid coordinates.
        num_heads: number of attention heads.
        attn_mask: (B, N), True for valid patches.

    Returns:
        (B, num_heads, N, N) bias; padded keys are set to ``-inf``.
    """
    dist = (coords.unsqueeze(2) - coords.unsqueeze(1)).abs().sum(dim=-1).float()
    slopes = get_alibi_slopes(num_heads).to(coords.device)
    bias = -dist.unsqueeze(1) * slopes.view(1, -1, 1, 1)
    if attn_mask is not None:
        bias = bias.masked_fill((~attn_mask)[:, None, None, :], float("-inf"))
    return bias
