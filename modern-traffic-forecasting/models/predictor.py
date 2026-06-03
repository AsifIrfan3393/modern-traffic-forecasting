"""End-to-end modern spatio-temporal traffic forecasting model."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from models.adaptive_graph import AdaptiveGraphLearner
from models.feature_fusion import FeatureFusion
from models.gatv2_encoder import GATv2Encoder
from models.multiscale_encoder import MultiScaleTemporalEncoder
from models.node_embeddings import NodeEmbeddings
from models.spatial_attention import SpatialAttention
from models.temporal_attention import TemporalAttention
from models.temporal_fusion_transformer import TemporalFusionTransformer


class TrafficForecastingModel(nn.Module):
    """Adaptive graph + GATv2 + multi-scale TFT forecasting network."""

    def __init__(self, num_nodes: int, input_dim: int, config: Dict) -> None:
        super().__init__()
        model_cfg = config["model"]
        hidden_dim = int(model_cfg["hidden_dim"])
        heads = int(model_cfg["num_heads"])
        dropout = float(model_cfg["dropout"])
        node_dim = int(model_cfg["node_embedding_dim"])
        self.num_nodes = num_nodes
        self.horizon = int(model_cfg["prediction_horizon"])
        self.hidden_dim = hidden_dim

        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.node_embeddings = NodeEmbeddings(num_nodes, node_dim)
        self.graph_learner = AdaptiveGraphLearner(
            num_nodes=num_nodes,
            embedding_dim=node_dim,
            mode=model_cfg.get("graph_mode", "hybrid"),
            top_k=model_cfg.get("adaptive_top_k"),
        )
        self.gatv2 = GATv2Encoder(
            hidden_dim=hidden_dim,
            num_layers=int(model_cfg["num_layers"]),
            heads=heads,
            dropout=float(model_cfg["attention_dropout"]),
            edge_dim=model_cfg.get("edge_dim"),
        )
        self.multiscale = MultiScaleTemporalEncoder(hidden_dim, model_cfg.get("temporal_scales", [3, 6, 12]), dropout)
        self.tft = TemporalFusionTransformer(hidden_dim, hidden_dim, heads, int(model_cfg["num_layers"]), dropout, int(model_cfg["window_size"]))
        self.spatial_attention = SpatialAttention(hidden_dim, heads, float(model_cfg["attention_dropout"]))
        self.temporal_attention = TemporalAttention(hidden_dim, heads, float(model_cfg["attention_dropout"]))
        self.fusion = FeatureFusion(hidden_dim, node_dim, dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.horizon),
        )

    def _graph_encode(
        self,
        x: torch.Tensor,
        static_adjacency: Optional[torch.Tensor],
        edge_attr: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch, steps, nodes, hidden = x.shape
        adjacency = self.graph_learner(static_adjacency)
        edge_index, _ = AdaptiveGraphLearner.dense_to_edge_index(adjacency)
        if edge_attr is not None and edge_attr.shape[0] != edge_index.shape[1]:
            edge_attr = None
        outputs = []
        for t in range(steps):
            per_batch = []
            for b in range(batch):
                per_batch.append(self.gatv2(x[b, t], edge_index, edge_attr=edge_attr))
            outputs.append(torch.stack(per_batch, dim=0))
        return torch.stack(outputs, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        static_adjacency: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # x: [batch, window, nodes, features], output: [batch, horizon, nodes]
        projected = self.input_projection(x)
        graph_features = self._graph_encode(projected, static_adjacency, edge_attr)
        multiscale_features = self.multiscale(graph_features)
        tft_features = self.tft(multiscale_features)
        spatial_features = self.spatial_attention(tft_features)
        temporal_features = self.temporal_attention(spatial_features)
        fused = self.fusion(graph_features, tft_features, spatial_features, temporal_features, self.node_embeddings())
        last_state = fused[:, -1]
        prediction = self.head(last_state)
        return prediction.permute(0, 2, 1)

    def attention_maps(self) -> Dict[str, torch.Tensor]:
        maps = {}
        if hasattr(self.spatial_attention, "last_attention"):
            maps["spatial"] = self.spatial_attention.last_attention
        if hasattr(self.temporal_attention, "last_attention"):
            maps["temporal"] = self.temporal_attention.last_attention
        maps["adaptive_adjacency"] = self.graph_learner.adaptive_adjacency().detach()
        return maps