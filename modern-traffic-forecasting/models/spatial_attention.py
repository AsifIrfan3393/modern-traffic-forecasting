from __future__ import annotations

import torch
import torch.nn as nn


class SpatialAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, nodes, hidden = x.shape
        flat = x.reshape(batch * steps, nodes, hidden)
        attended, weights = self.attention(flat, flat, flat, need_weights=True)
        self.last_attention = weights.detach()
        return self.norm(flat + attended).reshape(batch, steps, nodes, hidden)
