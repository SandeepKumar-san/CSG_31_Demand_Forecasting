"""
Complete end-to-end Adaptive Temporal-Structural Fusion Model.

Assembles all 5 layers into a single nn.Module:
  Layer 1: Raw Inputs (data preprocessing, external)
  Layer 2a: TFT Branch (temporal modeling)
  Layer 2b: GAT Branch (structural modeling)
  Layer 3: Adaptive Fusion (novel contribution)
  Layer 4: Fused Demand Representation
  Layer 5: Risk Scoring & Decision

Forward: raw batch -> {forecasts, alpha_weights, risk_assessment}
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional

from .tft_branch import TFTBranch
from .gat_branch import GATBranch
from .fusion_layer import AdaptiveFusionLayer
from .fused_representation import FusedDemandRepresentation
from .risk_decision_layer import RiskScoringLayer


class AdaptiveFusionForecaster(nn.Module):
    """
    End-to-end Adaptive Temporal-Structural Fusion Model.

    Combines TFT for temporal patterns, GAT for structural dependencies,
    and an adaptive fusion mechanism that learns when to trust each signal.

    Args:
        config: Full configuration dictionary.
        seed: Random seed for weight initialization reproducibility.
    """

    def __init__(self, config: dict, seed: int = 42) -> None:
        super().__init__()
        self.config = config
        self.hidden_dim = config["model"]["hidden_dim"]

        # Set seed before weight initialization
        torch.manual_seed(seed)

        # ---- Layer 2a: Temporal Branch (TFT) ----
        self.tft = TFTBranch(config)

        # ---- Layer 2b: Structural Branch (GAT) ----
        self.gat = GATBranch(config)

        # ---- Layer 3: Adaptive Fusion ----
        self.fusion = AdaptiveFusionLayer(config)

        # ---- Layer 4: Fused Demand Representation ----
        self.fused_repr = FusedDemandRepresentation(config)

        # ---- Layer 5: Risk Scoring & Decision ----
        self.risk_layer = RiskScoringLayer(config)

        # ---- Dimension Audit ----
        print(f"\n[Model Audit] Architecture build confirmed:")
        print(f"  - Hidden Dimension:    {self.hidden_dim}")
        print(f"  - TFT Branch:         {config['model']['tft']['num_layers']} layers, {config['model']['tft']['num_heads']} heads")
        print(f"  - GAT Branch:         {config['model']['gat']['num_layers']} layers, {config['model']['gat']['heads']} heads")
        print(f"  - Risk Layer:         ENABLED")

        # Apply deterministic weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Deterministic weight initialization for reproducibility."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.constant_(param, 0)

    def forward(
        self,
        batch: Dict[str, Any],
        compute_risk: bool = False,
    ) -> Dict[str, Any]:
        """
        Full forward pass through all 5 layers.

        Args:
            batch: Dictionary with keys:
              - 'time_series': [B, seq_len, n_features]
              - 'price_history': [B, seq_len]
              - 'static_features': [B, n_static]
              - 'graph_x': [n_nodes, node_features]
              - 'graph_edge_index': [2, n_edges]
              - 'graph_edge_attr': [n_edges, edge_features]
              - 'material_type': [B] int tensor (0-7)
              - 'horizon': [B] int tensor (0-3)
              - 'product_id': [B] int tensor
              Optional (for risk scoring):
              - 'prices': [B] current prices
              - 'budget': float
              - 'current_stock': [B] stock levels
              - 'lead_times': [B] lead times in days
            compute_risk: Whether to compute Layer 5 risk scoring.

        Returns:
            Dictionary with:
              - 'forecast': [B, horizons]
              - 'quantile_predictions': [B, horizons, quantiles]
              - 'forecast_std': [B, horizons]
              - 'alpha': [B, 1] fusion gate weights
              - 'temporal_embedding': [B, hidden_dim]
              - 'structural_embedding': [B, hidden_dim]
              - 'attention_weights': [B, heads, 1, seq_len]
              - 'variable_weights': [B, seq_len, num_vars]
              - 'risk_assessment': List[Dict] (if compute_risk=True)
        """
        device = next(self.parameters()).device

        # ---- Layer 2a: TFT Branch ----
        tft_out = self.tft(
            batch["time_series"].to(device),
            batch["price_history"].to(device),
            batch["static_features"].to(device),
        )
        h_temporal = tft_out["temporal_embedding"]  # [B, H]
        quantiles = tft_out["predictions"]  # [B, horizons, Q]

        # ---- Layer 2b: GAT Branch ----
        gat_out = self.gat(
            batch["graph_x"].to(device),
            batch["graph_edge_index"].to(device),
            edge_attr=batch.get("graph_edge_attr", None),
            edge_type=batch.get("graph_edge_type", None),
            edge_weight=batch.get("graph_edge_weight", None),
        )
        # Map structural embeddings from graph nodes to batch items
        product_ids = batch["product_id"]
        structural_all = gat_out["structural_embedding"]  # [n_nodes, H]
        h_structural = structural_all[product_ids.long()]  # [B, H]

        # ---- Layer 3: Adaptive Fusion ----
        material_types = batch["material_type"].to(device)
        horizons = batch.get("horizon", torch.zeros(
            material_types.shape[0], dtype=torch.long, device=device
        ))
        fused, alpha = self.fusion(
            h_temporal, h_structural, material_types, horizons
        )

        # ---- Layer 4: Fused Demand Representation ----
        forecast_dict = self.fused_repr(fused, quantiles)

        # ---- Build output ----
        output = {
            "forecast": forecast_dict["forecast"],
            "quantile_predictions": quantiles,
            "forecast_std": forecast_dict["std"],
            "lower": forecast_dict["lower"],
            "upper": forecast_dict["upper"],
            "alpha": alpha,
            "temporal_embedding": h_temporal,
            "structural_embedding": h_structural,
            "attention_weights": tft_out["attention_weights"],
            "variable_weights": tft_out["variable_weights"],
            "edge_attention": gat_out["edge_attention_weights"],
            "node_importance": gat_out["node_importance_scores"],
        }

        # ---- Layer 5: Risk Scoring (optional) ----
        if compute_risk and "prices" in batch:
            risk_results = self.risk_layer(
                forecast_dict,
                fused,
                batch["prices"].to(device),
                batch.get("budget", self.config["model"]["risk_scoring"]["budget_threshold"]),
                batch["current_stock"].to(device),
                batch["lead_times"].to(device),
                gat_out["edge_attention_weights"],
                batch["graph_edge_index"].to(device),
                batch["product_id"].to(device),
            )
            output["risk_assessment"] = risk_results

        return output

    def forward_with_fixed_alpha(
        self,
        batch: Dict[str, Any],
        fixed_alpha: float,
    ) -> Dict[str, Any]:
        """
        Forward pass with a fixed fusion weight (for baseline comparison).

        Used in evaluation to compare:
          - alpha=1.0 -> TFT-only
          - alpha=0.0 -> GAT-only
          - alpha=0.5 -> Fixed fusion
          - alpha=learned -> Adaptive fusion (normal forward)

        Args:
            batch: Same as forward().
            fixed_alpha: Fixed fusion weight ∈ [0, 1].

        Returns:
            Same output dict as forward().
        """
        device = next(self.parameters()).device

        # TFT Branch
        tft_out = self.tft(
            batch["time_series"].to(device),
            batch["price_history"].to(device),
            batch["static_features"].to(device),
        )
        h_temporal = tft_out["temporal_embedding"]
        quantiles = tft_out["predictions"]

        # GAT Branch
        gat_out = self.gat(
            batch["graph_x"].to(device),
            batch["graph_edge_index"].to(device),
            edge_attr=batch.get("graph_edge_attr", None),
            edge_type=batch.get("graph_edge_type", None),
            edge_weight=batch.get("graph_edge_weight", None),
        )
        product_ids = batch["product_id"]
        h_structural = gat_out["structural_embedding"][product_ids.long()]

        # Fixed alpha fusion (no gradient through alpha)
        alpha_tensor = torch.full(
            (h_temporal.size(0), 1), fixed_alpha, device=device
        )
        fused = alpha_tensor * h_temporal + (1.0 - alpha_tensor) * h_structural

        # Fused representation
        forecast_dict = self.fused_repr(fused, quantiles)

        return {
            "forecast": forecast_dict["forecast"],
            "quantile_predictions": quantiles,
            "forecast_std": forecast_dict["std"],
            "lower": forecast_dict["lower"],
            "upper": forecast_dict["upper"],
            "alpha": alpha_tensor,
        }
