from __future__ import annotations

import torch
import torch.nn as nn


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, nodes, hidden = x.shape
        flat = x.permute(0, 2, 1, 3).reshape(batch * nodes, steps, hidden)
        attended, weights = self.attention(flat, flat, flat, need_weights=True)
        self.last_attention = weights.detach()
        out = self.norm(flat + attended)
        return out.reshape(batch, nodes, steps, hidden).permute(0, 2, 1, 3)