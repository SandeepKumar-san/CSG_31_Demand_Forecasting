"""
Layer 2b: Multi-Relational Graph Attention Network (GAT) Branch.

Captures structural dependencies in the supply-chain / material graph
using edge-type-aware attention. Supports variable numbers of edge types:
  - SupplyGraph: 4 types (product_group, sub_group, plant, storage)
  - USGS: 15 types (supply_chain_input, technology_cluster, alloy_components, ...)

Architecture:
  - Edge type embedding layer (num_edge_types)
  - Multi-layer GATConv with edge features
  - BatchNorm + ELU + Dropout
  - Multi-head attention (4 heads)
  - Edge type importance analysis support

Output: structural embedding [num_nodes, hidden_dim] + edge attention weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple

from torch_geometric.nn import GATConv


class EdgeTypeAwareGAT(nn.Module):
    """
    Multi-Relational Graph Attention Network (GAT) — Layer 2b.

    Supports variable number of edge types via one-hot encoding
    concatenated with edge weights. Automatically adapts to the
    dataset's edge type count (no hardcoding).

    Architecture:
      - Edge type -> one-hot encoding (num_edge_types dims)
      - Concatenate with edge weight (1 dim) -> edge_attr
      - Multi-layer GATConv with edge feature support
      - BatchNorm + ELU activation + Dropout
      - Output projection to model hidden_dim
      - Node importance scoring
      - Edge type importance analysis

    Args:
        config: Configuration dictionary with 'model.gat' section.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        gat_cfg = config["model"]["gat"]
        self.hidden_dim = gat_cfg["hidden_dim"]
        self.num_layers = gat_cfg["num_layers"]
        self.heads = gat_cfg["heads"]
        self.dropout_rate = gat_cfg["dropout"]
        self.num_node_features = gat_cfg["num_node_features"]

        # ---- Edge type configuration ----
        self.num_edge_types = gat_cfg["num_edge_types"]
        # Edge feature dim = one-hot(num_edge_types) + weight(1)
        self.edge_feature_dim = self.num_edge_types + 1

        # ---- Edge type embedding (learnable) ----
        self.edge_type_embedding = nn.Embedding(
            self.num_edge_types, self.num_edge_types
        )
        # Initialize as near-identity for interpretability
        nn.init.eye_(self.edge_type_embedding.weight)

        # ---- Input projection ----
        self.input_proj = nn.Linear(self.num_node_features, self.hidden_dim)

        # ---- GAT layers ----
        self.gat_layers = nn.ModuleList()
        self.gat_norms = nn.ModuleList()

        for i in range(self.num_layers):
            in_channels = (
                self.hidden_dim if i == 0
                else self.hidden_dim * self.heads
            )
            concat = True if i < self.num_layers - 1 else False
            out_heads = self.heads if concat else 1

            self.gat_layers.append(
                GATConv(
                    in_channels=in_channels,
                    out_channels=self.hidden_dim,
                    heads=out_heads,
                    concat=concat,
                    dropout=self.dropout_rate,
                    edge_dim=self.edge_feature_dim,
                    add_self_loops=True,
                    fill_value="mean",
                )
            )
            out_dim = self.hidden_dim * out_heads if concat else self.hidden_dim
            self.gat_norms.append(nn.BatchNorm1d(out_dim))

        self.dropout = nn.Dropout(self.dropout_rate)

        # ---- Output projection (ensure output matches TFT hidden_dim) ----
        model_hidden = config["model"]["hidden_dim"]
        self.output_proj = nn.Linear(self.hidden_dim, model_hidden)

        # ---- Node importance scorer ----
        self.importance_scorer = nn.Linear(model_hidden, 1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        edge_type: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the multi-relational GAT branch.

        Args:
            x: Node features [num_nodes, num_node_features].
            edge_index: Edge indices [2, num_edges].
            edge_attr: Pre-computed edge features [num_edges, edge_feature_dim].
                       If provided, used directly. If None, constructed from
                       edge_type and edge_weight.
            edge_type: Edge type IDs [num_edges] — integers 0 to num_edge_types-1.
            edge_weight: Edge confidence weights [num_edges] — floats 0.5-1.0.

        Returns:
            Dictionary with keys:
              - 'structural_embedding': [num_nodes, hidden_dim]
              - 'edge_attention_weights': [num_edges, heads] (from last layer)
              - 'node_importance_scores': [num_nodes]
              - 'edge_type_attention': Dict[int, float] - mean attention per type
        """
        # ---- Construct edge features if not provided ----
        if edge_attr is None and edge_type is not None:
            edge_attr = self._build_edge_features(edge_type, edge_weight)
        elif edge_attr is None:
            # Fallback: no edge features
            edge_attr = None

        # ---- Input projection ----
        # Lazy-init: if actual input size differs from config, we rebuild here.
        actual_in = x.shape[1]
        if self.input_proj.in_features != actual_in:
            self.input_proj = nn.Linear(actual_in, self.hidden_dim).to(x.device)
            self.num_node_features = actual_in
        h = self.input_proj(x)

        # ---- Multi-layer GAT ----
        edge_attention = None
        returned_edge_index = None
        for i, (gat_layer, norm) in enumerate(
            zip(self.gat_layers, self.gat_norms)
        ):
            is_last = (i == self.num_layers - 1)
            out = gat_layer(
                h, edge_index, edge_attr=edge_attr,
                return_attention_weights=True if is_last else None,
            )

            if isinstance(out, tuple):
                # Last layer returns (output, (edge_index_with_self_loops, attention_weights))
                h, (returned_edge_index, edge_attention) = out
            else:
                h = out

            h = norm(h)
            if not is_last:
                h = F.elu(h)
                h = self.dropout(h)

        # ---- Output projection ----
        structural_embedding = self.output_proj(h)  # [num_nodes, hidden_dim]

        # ---- Node importance scores ----
        importance = torch.sigmoid(
            self.importance_scorer(structural_embedding)
        ).squeeze(-1)  # [num_nodes]

        # ---- Edge type importance analysis ----
        edge_type_attention = {}
        if edge_attention is not None and edge_type is not None:
            edge_type_attention = self._compute_edge_type_attention(
                edge_attention, edge_type, returned_edge_index, edge_index
            )

        return {
            "structural_embedding": structural_embedding,
            "edge_attention_weights": (
                edge_attention if edge_attention is not None
                else torch.zeros(edge_index.shape[1], device=x.device)
            ),
            "node_importance_scores": importance,
            "edge_type_attention": edge_type_attention,
        }

    def _build_edge_features(
        self,
        edge_type: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Construct edge feature tensor from edge types and weights.

        Edge features = [one-hot(edge_type), edge_weight]
        Total dim = num_edge_types + 1

        Args:
            edge_type: [num_edges] integer tensor (0 to num_edge_types-1).
            edge_weight: [num_edges] float tensor (0.5 to 1.0). If None, defaults to 1.0.

        Returns:
            edge_attr: [num_edges, edge_feature_dim] tensor.
        """
        device = edge_type.device
        num_edges = edge_type.shape[0]

        # One-hot encoding of edge types
        one_hot = F.one_hot(
            edge_type.long(), num_classes=self.num_edge_types
        ).float()  # [num_edges, num_edge_types]

        # Edge weight (default 1.0 if not provided)
        if edge_weight is None:
            weight = torch.ones(num_edges, 1, device=device)
        else:
            weight = edge_weight.unsqueeze(-1).float()  # [num_edges, 1]

        # Concatenate: [one_hot, weight]
        edge_attr = torch.cat([one_hot, weight], dim=-1)  # [num_edges, num_edge_types + 1]
        return edge_attr

    def _compute_edge_type_attention(
        self,
        attention_weights: torch.Tensor,
        edge_type: torch.Tensor,
        returned_edge_index: Optional[torch.Tensor],
        original_edge_index: torch.Tensor,
    ) -> Dict[int, float]:
        """
        Compute mean attention weight per edge type.

        GAT with self-loops adds extra edges, so we only analyze
        the original edges (not self-loop additions).

        Args:
            attention_weights: [num_edges_with_self_loops, heads] from GAT.
            edge_type: [num_original_edges] type IDs.
            returned_edge_index: [2, num_edges_with_self_loops] from GAT.
            original_edge_index: [2, num_original_edges] original edges.

        Returns:
            Dict mapping edge_type_id -> mean attention weight.
        """
        try:
            # Mean across heads
            if attention_weights.dim() > 1:
                mean_attn = attention_weights.mean(dim=-1)  # [num_edges_with_self_loops]
            else:
                mean_attn = attention_weights

            # Only take attention for the original edges (skip self-loops)
            n_original = edge_type.shape[0]
            if mean_attn.shape[0] > n_original:
                mean_attn = mean_attn[:n_original]

            # Aggregate per edge type
            result = {}
            for t in range(self.num_edge_types):
                mask = (edge_type == t)
                if mask.any():
                    result[t] = mean_attn[mask].mean().item()
                else:
                    result[t] = 0.0
            return result
        except Exception:
            # If anything goes wrong, return empty dict rather than crash
            return {}

    def compute_edge_type_importance(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Compute importance ranking of edge types based on attention.

        Call this after training to analyze which relationship types
        the model considers most important.

        Args:
            x: Node features [num_nodes, num_node_features].
            edge_index: Edge indices [2, num_edges].
            edge_type: Edge type IDs [num_edges].
            edge_weight: Edge weights [num_edges].

        Returns:
            Dictionary with:
              - 'mean_attention': Dict[int, float] - mean attention per type
              - 'std_attention': Dict[int, float] - std per type
              - 'ranking': List[Tuple[int, float]] - sorted by importance desc
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(
                x, edge_index,
                edge_type=edge_type,
                edge_weight=edge_weight,
            )

        # Compute statistics per edge type
        attn = output["edge_attention_weights"]
        if attn.dim() > 1:
            mean_attn_all = attn.mean(dim=-1)
        else:
            mean_attn_all = attn

        n_original = edge_type.shape[0]
        if mean_attn_all.shape[0] > n_original:
            mean_attn_all = mean_attn_all[:n_original]

        mean_per_type = {}
        std_per_type = {}
        for t in range(self.num_edge_types):
            mask = (edge_type == t)
            if mask.any():
                vals = mean_attn_all[mask]
                mean_per_type[t] = vals.mean().item()
                std_per_type[t] = vals.std().item() if vals.shape[0] > 1 else 0.0
            else:
                mean_per_type[t] = 0.0
                std_per_type[t] = 0.0

        # Ranking (descending by mean attention)
        ranking = sorted(mean_per_type.items(), key=lambda x: x[1], reverse=True)

        return {
            "mean_attention": mean_per_type,
            "std_attention": std_per_type,
            "ranking": ranking,
        }


# Backward-compatible alias
GATBranch = EdgeTypeAwareGAT
