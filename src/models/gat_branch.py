"""
Layer 2b: Graph Attention Network (GAT) Branch.

Captures structural dependencies in the supply-chain material graph.
Uses multi-layer GAT with edge features (4 edge types: product_group,
sub_group, plant, storage).

Output: structural embedding [num_nodes, hidden_dim] + edge attention weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple

from torch_geometric.nn import GATConv


class GATBranch(nn.Module):
    """
    Graph Attention Network (GAT) Branch — Layer 2b.

    Architecture:
      - Multi-layer GATConv (PyTorch Geometric)
      - ELU activation between layers
      - Dropout for regularization
      - Final projection to match TFT embedding dimension
      - Edge attention weight extraction

    Node features:
      - recent_demand_avg (last 7 days)
      - price_level (current / average)
      - production_capacity_util
      - lead_time_days (normalized)

    Edge features:
      - edge_type one-hot (4 dimensions)

    Args:
        config: Configuration dictionary with 'model.gat' section.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        gat_cfg = config.get("model", {}).get("gat", {})
        self.hidden_dim = gat_cfg.get("hidden_dim", 64)
        self.num_layers = gat_cfg.get("num_layers", 2)
        self.heads = gat_cfg.get("heads", 4)
        self.dropout_rate = gat_cfg.get("dropout", 0.1)
        self.num_node_features = gat_cfg.get("num_node_features", 4)
        self.num_edge_features = gat_cfg.get("num_edge_features", 4)

        # ---- Input projection ----
        self.input_proj = nn.Linear(self.num_node_features, self.hidden_dim)

        # ---- GAT layers ----
        self.gat_layers = nn.ModuleList()
        self.gat_norms = nn.ModuleList()

        for i in range(self.num_layers):
            in_channels = self.hidden_dim if i == 0 else self.hidden_dim * self.heads
            concat = True if i < self.num_layers - 1 else False
            out_heads = self.heads if concat else 1

            self.gat_layers.append(
                GATConv(
                    in_channels=in_channels,
                    out_channels=self.hidden_dim,
                    heads=out_heads,
                    concat=concat,
                    dropout=self.dropout_rate,
                    edge_dim=self.num_edge_features,
                    add_self_loops=True,
                    fill_value="mean",  # CRITICAL: fills self-loop edge_attr with mean of existing
                )
            )
            out_dim = self.hidden_dim * out_heads if concat else self.hidden_dim
            self.gat_norms.append(nn.LayerNorm(out_dim))

        self.dropout = nn.Dropout(self.dropout_rate)

        # ---- Output projection (ensure output matches TFT hidden_dim) ----
        model_hidden = config.get("model", {}).get("hidden_dim", 64)
        self.output_proj = nn.Linear(self.hidden_dim, model_hidden)

        # ---- Node importance scorer ----
        self.importance_scorer = nn.Linear(model_hidden, 1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the GAT branch.

        Args:
            x: Node features [num_nodes, num_node_features].
            edge_index: Edge indices [2, num_edges].
            edge_attr: Edge features [num_edges, num_edge_features].

        Returns:
            Dictionary with keys:
              - 'structural_embedding': [num_nodes, hidden_dim]
              - 'edge_attention_weights': [num_edges, heads] (from last layer)
              - 'node_importance_scores': [num_nodes]
        """
        # Input projection
        h = self.input_proj(x)

        # Multi-layer GAT
        edge_attention = None
        for i, (gat_layer, norm) in enumerate(zip(self.gat_layers, self.gat_norms)):
            h = gat_layer(h, edge_index, edge_attr=edge_attr,
                          return_attention_weights=True if i == self.num_layers - 1 else None)

            if isinstance(h, tuple):
                # Last layer returns (output, (edge_index, attention_weights))
                h, (_, edge_attention) = h

            h = norm(h)
            if i < self.num_layers - 1:
                h = F.elu(h)
                h = self.dropout(h)

        # Project to model hidden dim
        structural_embedding = self.output_proj(h)  # [num_nodes, hidden_dim]

        # Node importance scores
        importance = torch.sigmoid(
            self.importance_scorer(structural_embedding)
        ).squeeze(-1)  # [num_nodes]

        return {
            "structural_embedding": structural_embedding,
            "edge_attention_weights": edge_attention if edge_attention is not None else torch.zeros(edge_index.shape[1]),
            "node_importance_scores": importance,
        }
