"""
Loss functions for the Adaptive Temporal-Structural Fusion model.

Includes:
- QuantileLoss: For uncertainty estimation (multi-horizon forecasting)
- AdaptiveFusionLoss: Combined loss with alpha regularization
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class QuantileLoss(nn.Module):
    """
    Quantile (pinball) loss for probabilistic forecasting.

    Produces asymmetric penalties depending on the quantile,
    enabling prediction intervals (e.g., 10th, 50th, 90th percentiles).

    Args:
        quantiles: List of quantile levels, e.g. [0.1, 0.5, 0.9].
    """

    def __init__(self, quantiles: list = None) -> None:
        super().__init__()
        self.quantiles = quantiles or [0.1, 0.5, 0.9]

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute quantile loss.

        Args:
            y_pred: Predictions [batch, horizons, num_quantiles].
            y_true: Targets [batch, horizons] or [batch, horizons, 1].

        Returns:
            Scalar quantile loss.
        """
        if y_true.dim() == 2:
            y_true = y_true.unsqueeze(-1)  # [batch, horizons, 1]

        losses = []
        for i, q in enumerate(self.quantiles):
            errors = y_true - y_pred[..., i : i + 1]
            losses.append(torch.max(q * errors, (q - 1.0) * errors))

        loss = torch.cat(losses, dim=-1)
        return loss.mean()


class AdaptiveFusionLoss(nn.Module):
    """
    Combined loss for the full Adaptive Fusion model.

    total_loss = forecast_loss + alpha_reg_weight * alpha_regularization

    where:
      - forecast_loss: QuantileLoss on predictions vs targets
      - alpha_regularization: Negative std of alpha (encourages diversity)

    Args:
        quantiles: Quantile levels for QuantileLoss.
        alpha_reg_weight: Weight for alpha regularization term (default: 0.01).
    """

    def __init__(
        self,
        quantiles: list = None,
        alpha_reg_weight: float = 0.01,
    ) -> None:
        super().__init__()
        self.quantile_loss = QuantileLoss(quantiles)
        self.mse_loss = nn.MSELoss()
        self.alpha_reg_weight = alpha_reg_weight

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            predictions: Dictionary with keys:
                - 'forecast': Point forecasts [batch, horizons]
                - 'quantile_predictions': [batch, horizons, quantiles]
                - 'alpha': Fusion gate weights [batch, 1]
            targets: Ground truth [batch, horizons].

        Returns:
            Dictionary with 'total_loss', 'forecast_loss', 'alpha_reg'.
        """
        # ---- Forecast loss (MSE on point forecast) ----
        forecast_loss = self.mse_loss(predictions["forecast"], targets)

        # ---- Quantile loss (if quantile predictions available) ----
        if "quantile_predictions" in predictions and predictions["quantile_predictions"] is not None:
            q_loss = self.quantile_loss(predictions["quantile_predictions"], targets)
            forecast_loss = forecast_loss + q_loss

        # ---- Alpha regularization: encourage diversity in alpha ----
        alpha = predictions.get("alpha")
        if alpha is not None and alpha.numel() > 1:
            # Negative std: high std = low penalty (encourages diversity)
            alpha_reg = -torch.std(alpha) * self.alpha_reg_weight
        else:
            alpha_reg = torch.tensor(0.0, device=targets.device)

        total_loss = forecast_loss + alpha_reg

        return {
            "total_loss": total_loss,
            "forecast_loss": forecast_loss,
            "alpha_reg": alpha_reg,
        }
