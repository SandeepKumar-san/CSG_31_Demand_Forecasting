"""
Layer 4: Fused Demand Representation.

Combines the adaptively fused embedding with TFT quantile estimates
to produce context-aware forecasts with uncertainty bounds.

Outputs:
  - Point forecast (per horizon)
  - Forecast uncertainty (standard deviation)
  - Lower and upper confidence bounds
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class FusedDemandRepresentation(nn.Module):
    """
    Fused Demand Representation — Layer 4.

    Takes the adaptively fused embedding and produces:
      1. Point demand forecasts for each horizon
      2. Uncertainty estimates (learned std deviation)
      3. Confidence intervals combining learned uncertainty
         with TFT quantile estimates

    Args:
        config: Configuration dictionary.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.hidden_dim = config.get("model", {}).get("hidden_dim", 64)
        data_cfg = config.get("data", {})
        self.num_horizons = len(data_cfg.get("forecast_horizons", [1, 3, 6, 12]))

        # ---- Forecast head ----
        self.forecast_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.num_horizons),
        )

        # ---- Uncertainty head (log std for numerical stability) ----
        self.uncertainty_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.num_horizons),
        )

        # ---- Residual connection from fused embedding ----
        self.residual_proj = nn.Linear(self.hidden_dim, self.num_horizons)
        self.layer_norm = nn.LayerNorm(self.num_horizons)

    def forward(
        self,
        fused_embedding: torch.Tensor,
        tft_quantiles: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Produce context-aware forecasts with uncertainty.

        Args:
            fused_embedding: [batch, hidden_dim] from Adaptive Fusion.
            tft_quantiles: Optional [batch, horizons, num_quantiles] from TFT.

        Returns:
            Dictionary with keys:
              - 'forecast': [batch, num_horizons] point forecast
              - 'std': [batch, num_horizons] forecast std deviation
              - 'lower': [batch, num_horizons] lower bound (10th percentile)
              - 'upper': [batch, num_horizons] upper bound (90th percentile)
              - 'fused_embedding': [batch, hidden_dim] (pass-through for Layer 5)
        """
        # Point forecast with residual connection
        forecast = self.forecast_head(fused_embedding)  # [B, horizons]
        residual = self.residual_proj(fused_embedding)   # [B, horizons]
        forecast = self.layer_norm(forecast + residual)

        # Uncertainty (exponentiate log-std for positive values)
        log_std = self.uncertainty_head(fused_embedding)
        forecast_std = torch.exp(log_std)  # [B, horizons]

        # Confidence intervals
        if tft_quantiles is not None and tft_quantiles.dim() == 3:
            # Use TFT quantile estimates (10th and 90th percentile)
            lower_bound = tft_quantiles[..., 0]   # [B, horizons]
            upper_bound = tft_quantiles[..., -1]   # [B, horizons]
        else:
            # Approximate from learned uncertainty (±1.645 for 90% CI)
            lower_bound = forecast - 1.645 * forecast_std
            upper_bound = forecast + 1.645 * forecast_std

        return {
            "forecast": forecast,
            "std": forecast_std,
            "lower": lower_bound,
            "upper": upper_bound,
            "fused_embedding": fused_embedding,
        }
