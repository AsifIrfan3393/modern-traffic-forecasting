"""Multi-scale temporal convolution branches with learnable fusion."""

from __future__ import annotations

from typing import Iterable, List

import torch
import torch.nn as nn


class MultiScaleTemporalEncoder(nn.Module):
    def __init__(self, hidden_dim: int, scales: Iterable[int], dropout: float) -> None:
        super().__init__()
        self.scales: List[int] = list(scales)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=scale, padding=scale // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
                )
                for scale in self.scales
            ]
        )
        self.scale_logits = nn.Parameter(torch.zeros(len(self.scales)))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch,time,nodes,hidden]
        batch, steps, nodes, hidden = x.shape
        flat = x.permute(0, 2, 3, 1).reshape(batch * nodes, hidden, steps)
        outputs = []
        for branch in self.branches:
            y = branch(flat)
            y = y[..., :steps]
            outputs.append(y)
        weights = torch.softmax(self.scale_logits, dim=0)
        fused = sum(weight * output for weight, output in zip(weights, outputs))
        fused = fused.reshape(batch, nodes, hidden, steps).permute(0, 3, 1, 2)
        return self.norm(x + fused)