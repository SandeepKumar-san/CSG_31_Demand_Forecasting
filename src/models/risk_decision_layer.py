"""
Layer 5: Risk Scoring & Decision Layer.

Interpretable rule-based risk assessment that translates forecasts
into actionable business decisions.

Three risk dimensions:
  1. Budget Stress — (forecasted_demand × price) / budget
  2. Lead-time Risk — (demand_during_leadtime) / current_stock
  3. Dependency Criticality — average GAT attention weight

Produces: Risk levels (Low/Medium/High/Critical) + action flags.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple


class RiskScoringLayer(nn.Module):
    """
    Risk Scoring & Decision Layer — Layer 5.

    Translates demand forecasts and graph attention weights into
    interpretable risk scores and actionable recommendations.

    This layer is primarily rule-based for interpretability, with
    configurable thresholds that can be tuned per domain.

    Args:
        config: Configuration dictionary with 'model.risk_scoring' section.
    """

    RISK_LEVELS = ["Low", "Medium", "High", "Critical"]

    def __init__(self, config: dict) -> None:
        super().__init__()
        risk_cfg = config.get("model", {}).get("risk_scoring", {})
        self.budget_threshold = risk_cfg.get("budget_threshold", 100000)
        self.lead_time_safety_factor = risk_cfg.get("lead_time_safety_factor", 1.5)
        self.criticality_threshold = risk_cfg.get("criticality_threshold", 0.7)

        # Optional: learned risk calibration layer
        hidden_dim = config.get("model", {}).get("hidden_dim", 64)
        self.risk_calibrator = nn.Sequential(
            nn.Linear(hidden_dim + 3, 32),  # embedding + 3 risk scores
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def compute_budget_stress(
        self,
        forecast: torch.Tensor,
        prices: torch.Tensor,
        budget: float,
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Budget stress = (forecasted_demand × price) / budget.

        Args:
            forecast: [batch] or scalar, forecasted demand.
            prices: [batch] or scalar, current prices.
            budget: Scalar budget threshold.

        Returns:
            stress_scores: [batch] tensor of stress values.
            risk_levels: List of risk level strings.
        """
        projected_cost = forecast * prices
        stress = projected_cost / (budget + 1e-8)

        risk_levels = []
        for s in stress.detach().cpu().tolist():
            if s < 0.7:
                risk_levels.append("Low")
            elif s < 0.9:
                risk_levels.append("Medium")
            elif s < 1.1:
                risk_levels.append("High")
            else:
                risk_levels.append("Critical")

        return stress, risk_levels

    def compute_leadtime_risk(
        self,
        forecast: torch.Tensor,
        current_stock: torch.Tensor,
        lead_time_days: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Lead-time risk = (forecasted_demand × lead_time / 30) / current_stock.

        High risk when demand during lead time exceeds stock.

        Args:
            forecast: [batch] forecasted demand.
            current_stock: [batch] current inventory levels.
            lead_time_days: [batch] lead time in days.

        Returns:
            risk_scores: [batch] tensor.
            risk_levels: List of strings.
        """
        demand_during_leadtime = forecast * (lead_time_days / 30.0)
        risk_score = demand_during_leadtime / (current_stock + 1e-6)

        risk_levels = []
        for s in risk_score.detach().cpu().tolist():
            if s < 0.5:
                risk_levels.append("Low")
            elif s < 1.0:
                risk_levels.append("Medium")
            elif s < 1.5:
                risk_levels.append("High")
            else:
                risk_levels.append("Critical")

        return risk_score, risk_levels

    def compute_dependency_criticality(
        self,
        edge_attention_weights: torch.Tensor,
        edge_index: torch.Tensor,
        node_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Dependency criticality = average incoming attention weight per node.

        High attention means many materials depend on this node.

        Args:
            edge_attention_weights: [num_edges, heads] or [num_edges].
            edge_index: [2, num_edges] edge index tensor.
            node_indices: [batch] node indices to score.

        Returns:
            criticality_scores: [batch] tensor.
            risk_levels: List of strings.
        """
        # Average attention across heads if multi-dimensional
        if edge_attention_weights.dim() > 1:
            attn = edge_attention_weights.mean(dim=-1)
        else:
            attn = edge_attention_weights

        criticality_scores = []
        risk_levels = []

        for node_idx in node_indices:
            node_idx_val = node_idx.item() if isinstance(node_idx, torch.Tensor) else node_idx
            # Get incoming edges (edges pointing to this node)
            mask = (edge_index[1] == node_idx_val)

            if mask.any() and attn.numel() > 0:
                incoming = attn[mask[:attn.shape[0]]] if mask.shape[0] > attn.shape[0] else attn[mask]
                criticality = incoming.mean().item() if incoming.numel() > 0 else 0.0
            else:
                criticality = 0.0

            criticality_scores.append(criticality)

            if criticality > self.criticality_threshold:
                risk_levels.append("Critical")
            elif criticality > 0.5:
                risk_levels.append("High")
            elif criticality > 0.3:
                risk_levels.append("Medium")
            else:
                risk_levels.append("Low")

        return torch.tensor(criticality_scores, dtype=torch.float), risk_levels

    def generate_action_flags(
        self,
        budget_risk: str,
        leadtime_risk: str,
        criticality_risk: str,
    ) -> List[str]:
        """
        Generate actionable recommendations based on risk scores.

        Args:
            budget_risk: Budget risk level string.
            leadtime_risk: Lead-time risk level string.
            criticality_risk: Dependency criticality risk level string.

        Returns:
            List of action recommendation strings.
        """
        actions = []

        if budget_risk in ["High", "Critical"]:
            actions.append(
                "⚠️ Review budget allocation - Projected cost exceeds threshold"
            )

        if leadtime_risk == "Critical":
            actions.append(
                "🚨 URGENT: Expedite supplier - Stock insufficient for lead time"
            )
        elif leadtime_risk == "High":
            actions.append("⚠️ Consider increasing order quantity")

        if criticality_risk == "Critical":
            actions.append("🔴 CRITICAL DEPENDENCY: Activate backup supplier")
        elif criticality_risk == "High":
            actions.append("🟡 High dependency material - Monitor closely")

        if not actions:
            actions.append("✅ All risk levels acceptable - Maintain current plan")

        return actions

    def forward(
        self,
        forecast_dict: Dict[str, torch.Tensor],
        fused_embedding: torch.Tensor,
        prices: torch.Tensor,
        budget: float,
        current_stock: torch.Tensor,
        lead_times: torch.Tensor,
        edge_attention: torch.Tensor,
        edge_index: torch.Tensor,
        node_indices: torch.Tensor,
    ) -> List[Dict[str, Any]]:
        """
        Complete risk assessment pipeline.

        Args:
            forecast_dict: From Layer 4 with 'forecast', 'std', 'lower', 'upper'.
            fused_embedding: [batch, hidden_dim] fused representation.
            prices: [batch] current prices per material.
            budget: Scalar budget threshold.
            current_stock: [batch] current inventory levels.
            lead_times: [batch] lead times in days.
            edge_attention: [num_edges, heads] edge attention from GAT.
            edge_index: [2, num_edges] edge index tensor.
            node_indices: [batch] node/product indices.

        Returns:
            List of risk assessment dictionaries, one per product.
        """
        forecast = forecast_dict["forecast"][:, 0]  # Use first horizon
        forecast_std = forecast_dict["std"][:, 0]

        # ---- Risk computations ----
        budget_stress, budget_risks = self.compute_budget_stress(
            forecast, prices, budget
        )
        lt_score, lt_risks = self.compute_leadtime_risk(
            forecast, current_stock, lead_times
        )
        dep_score, dep_risks = self.compute_dependency_criticality(
            edge_attention, edge_index, node_indices
        )

        # ---- Learned risk calibration ----
        risk_features = torch.stack(
            [budget_stress, lt_score, dep_score.to(fused_embedding.device)], dim=-1
        )  # [batch, 3]
        calibration_input = torch.cat(
            [fused_embedding, risk_features], dim=-1
        )  # [batch, hidden + 3]
        overall_risk = self.risk_calibrator(calibration_input).squeeze(-1)  # [batch]

        # ---- Assemble results ----
        results = []
        for i in range(len(node_indices)):
            idx = node_indices[i].item() if isinstance(node_indices[i], torch.Tensor) else int(node_indices[i])
            actions = self.generate_action_flags(
                budget_risks[i], lt_risks[i], dep_risks[i]
            )

            results.append({
                "product_id": idx,
                "forecast": forecast[i].item(),
                "forecast_std": forecast_std[i].item(),
                "budget_stress": budget_stress[i].item(),
                "budget_risk": budget_risks[i],
                "leadtime_risk_score": lt_score[i].item(),
                "leadtime_risk": lt_risks[i],
                "dependency_criticality": dep_score[i].item(),
                "dependency_risk": dep_risks[i],
                "overall_risk_score": overall_risk[i].item(),
                "actions": actions,
            })

        return results
