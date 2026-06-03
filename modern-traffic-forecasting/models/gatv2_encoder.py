"""Stacked GATv2 encoder with residual normalization."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv


class GATv2Block(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, dropout: float, edge_dim: Optional[int] = None) -> None:
        super().__init__()
        self.conv = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // heads,
            heads=heads,
            concat=True,
            dropout=dropout,
            edge_dim=edge_dim,
            add_self_loops=True,
            residual=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        x = self.conv(x, edge_index, edge_attr=edge_attr)
        x = self.norm(residual + self.dropout(x))
        x = self.norm(x + self.dropout(self.ffn(x)))
        return x


class GATv2Encoder(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, heads: int, dropout: float, edge_dim: Optional[int] = None) -> None:
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads for GATv2 concat output.")
        self.layers = nn.ModuleList([GATv2Block(hidden_dim, heads, dropout, edge_dim=edge_dim) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr=edge_attr)
        return x