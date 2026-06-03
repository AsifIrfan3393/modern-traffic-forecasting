from __future__ import annotations

import torch
import torch.nn as nn


class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.project_skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim))
        self.gate = nn.Sequential(nn.Linear(output_dim, output_dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        gated = self.gate(y) * y
        return self.norm(self.project_skip(x) + gated)


class VariableSelectionNetwork(nn.Module):
    """Learn feature-wise selection weights before temporal modeling."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.feature_grn = GatedResidualNetwork(input_dim, hidden_dim, input_dim, dropout)
        self.weight_layer = nn.Linear(input_dim, input_dim)
        self.projection = nn.Linear(input_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.weight_layer(self.feature_grn(x)), dim=-1)
        return self.projection(x * weights)


class TemporalFusionTransformer(nn.Module):
    """TFT-style encoder/decoder with learnable temporal positional embeddings."""

    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int, num_layers: int, dropout: float, max_window: int) -> None:
        super().__init__()
        self.variable_selection = VariableSelectionNetwork(input_dim, hidden_dim, dropout)
        self.position = nn.Parameter(torch.zeros(1, max_window, 1, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder_grn = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, time, nodes, features]
        batch, steps, nodes, _ = x.shape
        x = self.variable_selection(x) + self.position[:, :steps]
        flat = x.permute(0, 2, 1, 3).reshape(batch * nodes, steps, -1)
        encoded = self.encoder(flat)
        attended, _ = self.attention(encoded, encoded, encoded, need_weights=False)
        decoded = self.norm(encoded + attended)
        decoded = self.decoder_grn(decoded)
        return decoded.reshape(batch, nodes, steps, -1).permute(0, 2, 1, 3)