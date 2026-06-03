"""Learnable fusion for graph, temporal, attention, and node embedding features."""

from __future__ import annotations

import torch
import torch.nn as nn


class FeatureFusion(nn.Module):
    def __init__(self, hidden_dim: int, node_embedding_dim: int, dropout: float) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 4 + node_embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 5),
            nn.Softmax(dim=-1),
        )
        self.node_projection = nn.Linear(node_embedding_dim, hidden_dim)
        self.output = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(hidden_dim))

    def forward(
        self,
        graph: torch.Tensor,
        temporal: torch.Tensor,
        spatial: torch.Tensor,
        temporal_attention: torch.Tensor,
        node_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        batch, steps, nodes, _ = graph.shape
        node = node_embeddings[None, None, :, :].expand(batch, steps, nodes, -1)
        weights = self.gate(torch.cat([graph, temporal, spatial, temporal_attention, node], dim=-1))
        node_hidden = self.node_projection(node)
        stacked = torch.stack([graph, temporal, spatial, temporal_attention, node_hidden], dim=-2)
        fused = (weights.unsqueeze(-1) * stacked).sum(dim=-2)
        return self.output(fused)