"""Adaptive, static, and hybrid graph learning."""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

GraphMode = Literal["static", "adaptive", "hybrid"]


class AdaptiveGraphLearner(nn.Module):
    """Learn A=softmax(E Eᵀ) and optionally fuse it with a static graph."""

    def __init__(self, num_nodes: int, embedding_dim: int, mode: GraphMode = "hybrid", top_k: Optional[int] = None) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.mode = mode
        self.top_k = top_k
        self.node_embeddings = nn.Parameter(torch.empty(num_nodes, embedding_dim))
        self.alpha_logit = nn.Parameter(torch.tensor(0.0))
        nn.init.xavier_uniform_(self.node_embeddings)

    def adaptive_adjacency(self) -> torch.Tensor:
        scores = torch.matmul(self.node_embeddings, self.node_embeddings.t())
        adjacency = F.softmax(F.relu(scores), dim=-1)
        if self.top_k and self.top_k < self.num_nodes:
            values, indices = torch.topk(adjacency, self.top_k, dim=-1)
            sparse = torch.zeros_like(adjacency)
            adjacency = sparse.scatter(-1, indices, values)
            adjacency = adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return adjacency

    def forward(self, static_adjacency: Optional[torch.Tensor] = None) -> torch.Tensor:
        adaptive = self.adaptive_adjacency()
        if self.mode == "adaptive" or static_adjacency is None:
            return adaptive
        static = static_adjacency.to(adaptive.device, adaptive.dtype)
        static = static / static.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        if self.mode == "static":
            return static
        alpha = torch.sigmoid(self.alpha_logit)
        return alpha * static + (1.0 - alpha) * adaptive

    @staticmethod
    def dense_to_edge_index(adjacency: torch.Tensor, threshold: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
        rows, cols = torch.nonzero(adjacency > threshold, as_tuple=True)
        edge_index = torch.stack([rows, cols], dim=0)
        edge_weight = adjacency[rows, cols]
        return edge_index, edge_weight