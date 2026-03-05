"""
Evaluation metrics for demand forecasting.

All metrics operate on torch tensors and return Python floats.
Used for model comparison and research reporting.
"""

from typing import Dict

import torch
import numpy as np


def mae(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Mean Absolute Error."""
    return torch.mean(torch.abs(y_true - y_pred)).item()


def rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Root Mean Squared Error."""
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()


def mape(y_true: torch.Tensor, y_pred: torch.Tensor, epsilon: float = 1e-8) -> float:
    """
    Mean Absolute Percentage Error.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        epsilon: Small constant to avoid division by zero.

    Returns:
        MAPE as a percentage (0-100 scale).
    """
    return (
        torch.mean(torch.abs((y_true - y_pred) / (torch.abs(y_true) + epsilon)))
        .item() * 100.0
    )


def smape(y_true: torch.Tensor, y_pred: torch.Tensor, epsilon: float = 1e-8) -> float:
    """
    Symmetric Mean Absolute Percentage Error.

    Returns:
        SMAPE as a percentage (0-200 scale).
    """
    numerator = torch.abs(y_true - y_pred)
    denominator = (torch.abs(y_true) + torch.abs(y_pred)) / 2.0 + epsilon
    return torch.mean(numerator / denominator).item() * 100.0


def r_squared(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    R-squared (coefficient of determination).

    Returns:
        R² score. 1.0 is perfect, 0.0 means no better than mean prediction.
    """
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return (1.0 - (ss_res / (ss_tot + 1e-8))).item()


def compute_all_metrics(
    y_true: torch.Tensor, y_pred: torch.Tensor
) -> Dict[str, float]:
    """
    Compute all forecasting metrics at once.

    Args:
        y_true: Ground truth tensor.
        y_pred: Prediction tensor.

    Returns:
        Dictionary with keys: MAE, RMSE, MAPE, SMAPE, R2.
    """
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "SMAPE": smape(y_true, y_pred),
        "R2": r_squared(y_true, y_pred),
    }
