

from __future__ import annotations

import torch
import torch.nn as nn


class NodeEmbeddings(nn.Module):
    def __init__(self, num_nodes: int, embedding_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self) -> torch.Tensor:
        node_ids = torch.arange(self.embedding.num_embeddings, device=self.embedding.weight.device)
        return self.embedding(node_ids)