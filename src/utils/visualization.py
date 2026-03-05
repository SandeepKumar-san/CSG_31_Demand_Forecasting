"""
Visualization module for the Adaptive Fusion model.

Generates 5 key plots for research paper:
  1. Training curves (train/val loss)
  2. Predictions vs Actual (time-series overlay)
  3. Alpha distribution by material type (boxplot)
  4. Risk dashboard (budget stress + lead-time risk)
  5. Model comparison (bar chart across baselines)
"""

import json
import os
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for reproducibility
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns


# Consistent style for research paper
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})


class ModelVisualizer:
    """
    Generate publication-quality plots for the Adaptive Fusion model.

    Args:
        output_dir: Directory to save generated plots.
    """

    def __init__(self, output_dir: str = "results/plots/") -> None:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Plot 1: Training Curves
    # ------------------------------------------------------------------
    def plot_training_curves(self, history: Dict[str, List[float]]) -> str:
        """
        Plot training and validation loss curves.

        Args:
            history: Dictionary with 'train_loss' and 'val_loss' lists.

        Returns:
            Path to saved figure.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        epochs = range(1, len(history["train_loss"]) + 1)

        # Loss curves
        ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#2196F3", linewidth=2)
        ax1.plot(epochs, history["val_loss"], label="Val Loss", color="#F44336", linewidth=2)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training & Validation Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Learning rate
        if "lr" in history:
            ax2.plot(epochs, history["lr"], color="#4CAF50", linewidth=2)
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Learning Rate")
            ax2.set_title("Learning Rate Schedule")
            ax2.set_yscale("log")
            ax2.grid(True, alpha=0.3)
        else:
            ax2.set_visible(False)

        plt.tight_layout()
        path = os.path.join(self.output_dir, "training_curves.png")
        fig.savefig(path)
        plt.close(fig)
        print(f"  📊 Saved: {path}")
        return path

    # ------------------------------------------------------------------
    # Plot 2: Predictions vs Actual
    # ------------------------------------------------------------------
    def plot_predictions_vs_actual(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        lower: Optional[np.ndarray] = None,
        upper: Optional[np.ndarray] = None,
        product_ids: Optional[List[int]] = None,
        n_products: int = 5,
    ) -> str:
        """
        Line plot of actual vs predicted demand for selected products.

        Args:
            actual: [n_samples, horizons] or [n_samples] actual values.
            predicted: [n_samples, horizons] or [n_samples] predictions.
            lower: Optional lower confidence bound.
            upper: Optional upper confidence bound.
            product_ids: Product IDs for each sample.
            n_products: Number of products to plot.

        Returns:
            Path to saved figure.
        """
        fig, axes = plt.subplots(n_products, 1, figsize=(12, 3 * n_products), sharex=True)
        if n_products == 1:
            axes = [axes]

        # If multi-dimensional, use first horizon
        if actual.ndim > 1:
            actual = actual[:, 0]
            predicted = predicted[:, 0]
            if lower is not None:
                lower = lower[:, 0]
            if upper is not None:
                upper = upper[:, 0]

        # Group by product
        if product_ids is not None:
            unique_pids = sorted(set(product_ids))[:n_products]
        else:
            # Split into n_products equal chunks
            chunk = len(actual) // n_products
            unique_pids = list(range(n_products))

        colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"]

        for idx, (ax, pid) in enumerate(zip(axes, unique_pids)):
            if product_ids is not None:
                mask = [i for i, p in enumerate(product_ids) if p == pid]
            else:
                mask = list(range(idx * chunk, min((idx + 1) * chunk, len(actual))))

            t = range(len(mask))
            act_vals = actual[mask]
            pred_vals = predicted[mask]

            ax.plot(t, act_vals, label="Actual", color="#333333", linewidth=1.5)
            ax.plot(t, pred_vals, label="Predicted", color=colors[idx % len(colors)],
                    linewidth=1.5, linestyle="--")

            if lower is not None and upper is not None:
                ax.fill_between(
                    t, lower[mask], upper[mask],
                    alpha=0.2, color=colors[idx % len(colors)], label="90% CI"
                )

            ax.set_ylabel("Demand")
            ax.set_title(f"Product {pid}")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Time Step")
        plt.suptitle("Predictions vs Actual Demand", fontsize=14, y=1.02)
        plt.tight_layout()

        path = os.path.join(self.output_dir, "predictions_vs_actual.png")
        fig.savefig(path)
        plt.close(fig)
        print(f"  📊 Saved: {path}")
        return path

    # ------------------------------------------------------------------
    # Plot 3: Alpha Distribution by Material Type
    # ------------------------------------------------------------------
    def plot_alpha_distribution(
        self,
        alphas: np.ndarray,
        material_types: np.ndarray,
        category_names: Optional[List[str]] = None,
    ) -> str:
        """
        Boxplot of alpha values grouped by material type.

        Shows which materials rely more on temporal vs structural signals.

        Args:
            alphas: [n_samples] alpha values.
            material_types: [n_samples] material type indices.
            category_names: Optional list of category name strings.

        Returns:
            Path to saved figure.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        unique_types = sorted(set(material_types.tolist()))
        data = []
        labels = []

        for mt in unique_types:
            mask = material_types == mt
            data.append(alphas[mask])
            if category_names and mt < len(category_names):
                labels.append(category_names[mt])
            else:
                labels.append(f"Type {mt}")

        bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True)

        # Color the boxes
        colors = sns.color_palette("Set2", len(unique_types))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Equal fusion (α=0.5)")
        ax.set_xlabel("Material Type")
        ax.set_ylabel("Alpha (α)")
        ax.set_title("Fusion Weight (α) Distribution by Material Type\n"
                      "(α→1: temporal-heavy, α→0: structural-heavy)")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        path = os.path.join(self.output_dir, "alpha_distribution_by_material.png")
        fig.savefig(path)
        plt.close(fig)
        print(f"  📊 Saved: {path}")
        return path

    # ------------------------------------------------------------------
    # Plot 3b: Alpha by Horizon
    # ------------------------------------------------------------------
    def plot_alpha_by_epoch(self, history: Dict[str, List[float]]) -> str:
        """
        Line plot of alpha mean and std over training epochs.

        Args:
            history: Training history with 'alpha_mean' and 'alpha_std'.

        Returns:
            Path to saved figure.
        """
        if "alpha_mean" not in history:
            return ""

        fig, ax = plt.subplots(figsize=(10, 5))
        epochs = range(1, len(history["alpha_mean"]) + 1)

        mean = np.array(history["alpha_mean"])
        std = np.array(history["alpha_std"])

        ax.plot(epochs, mean, color="#2196F3", linewidth=2, label="Mean α")
        ax.fill_between(epochs, mean - std, mean + std, alpha=0.2, color="#2196F3", label="±1 std")
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Equal fusion")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Alpha (α)")
        ax.set_title("Fusion Weight Evolution During Training")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        plt.tight_layout()
        path = os.path.join(self.output_dir, "alpha_by_horizon.png")
        fig.savefig(path)
        plt.close(fig)
        print(f"  📊 Saved: {path}")
        return path

    # ------------------------------------------------------------------
    # Plot 4: Risk Dashboard
    # ------------------------------------------------------------------
    def plot_risk_dashboard(
        self, risk_reports: List[Dict[str, Any]], top_n: int = 10
    ) -> str:
        """
        3-panel risk dashboard.

        Panels:
          - Budget stress scores (bar chart)
          - Lead-time risk scores (bar chart)
          - Overall risk distribution (pie chart)

        Args:
            risk_reports: List of risk assessment dicts from Layer 5.
            top_n: Number of top products to display.

        Returns:
            Path to saved figure.
        """
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

        # Sort by budget stress
        sorted_budget = sorted(risk_reports, key=lambda r: r.get("budget_stress", 0), reverse=True)

        # Panel 1: Budget Stress
        ax1 = fig.add_subplot(gs[0, 0])
        top_budget = sorted_budget[:top_n]
        pids = [f"P{r['product_id']}" for r in top_budget]
        stresses = [r.get("budget_stress", 0) for r in top_budget]
        colors = ["#F44336" if s >= 0.9 else "#FF9800" if s >= 0.7 else "#4CAF50" for s in stresses]
        ax1.barh(pids, stresses, color=colors)
        ax1.set_xlabel("Budget Stress Score")
        ax1.set_title("Top Budget Stress Products")
        ax1.axvline(x=0.9, color="red", linestyle="--", alpha=0.5, label="High threshold")
        ax1.axvline(x=0.7, color="orange", linestyle="--", alpha=0.5, label="Medium threshold")
        ax1.legend(fontsize=8)
        ax1.invert_yaxis()

        # Panel 2: Lead-time Risk
        ax2 = fig.add_subplot(gs[0, 1])
        sorted_lt = sorted(risk_reports, key=lambda r: r.get("leadtime_risk_score", 0), reverse=True)
        top_lt = sorted_lt[:top_n]
        pids_lt = [f"P{r['product_id']}" for r in top_lt]
        lt_scores = [r.get("leadtime_risk_score", 0) for r in top_lt]
        colors_lt = ["#F44336" if s >= 1.0 else "#FF9800" if s >= 0.5 else "#4CAF50" for s in lt_scores]
        ax2.barh(pids_lt, lt_scores, color=colors_lt)
        ax2.set_xlabel("Lead-time Risk Score")
        ax2.set_title("Top Lead-time Risk Products")
        ax2.axvline(x=1.0, color="red", linestyle="--", alpha=0.5)
        ax2.invert_yaxis()

        # Panel 3: Risk Distribution (Pie)
        ax3 = fig.add_subplot(gs[1, 0])
        risk_counts = {"Low": 0, "Medium": 0, "High": 0, "Critical": 0}
        for r in risk_reports:
            worst = max(
                ["Low", "Medium", "High", "Critical"].index(r.get("budget_risk", "Low")),
                ["Low", "Medium", "High", "Critical"].index(r.get("leadtime_risk", "Low")),
            )
            risk_counts[["Low", "Medium", "High", "Critical"][worst]] += 1

        pie_colors = ["#4CAF50", "#FF9800", "#F44336", "#B71C1C"]
        non_zero = {k: v for k, v in risk_counts.items() if v > 0}
        ax3.pie(
            non_zero.values(),
            labels=non_zero.keys(),
            colors=pie_colors[: len(non_zero)],
            autopct="%1.0f%%",
            startangle=90,
        )
        ax3.set_title("Overall Risk Distribution")

        # Panel 4: Alpha vs Risk
        ax4 = fig.add_subplot(gs[1, 1])
        alphas = [r.get("alpha", 0.5) for r in risk_reports]
        budget_s = [r.get("budget_stress", 0) for r in risk_reports]
        ax4.scatter(alphas, budget_s, alpha=0.6, c="#2196F3", edgecolors="white", s=50)
        ax4.set_xlabel("Fusion Weight (α)")
        ax4.set_ylabel("Budget Stress")
        ax4.set_title("Fusion Weight vs Budget Stress")
        ax4.grid(True, alpha=0.3)

        plt.suptitle("Risk Assessment Dashboard", fontsize=15, y=1.02)
        path = os.path.join(self.output_dir, "risk_dashboard.png")
        fig.savefig(path)
        plt.close(fig)
        print(f"  📊 Saved: {path}")
        return path

    # ------------------------------------------------------------------
    # Plot 5: Model Comparison
    # ------------------------------------------------------------------
    def plot_model_comparison(self, results: Dict[str, Dict[str, float]]) -> str:
        """
        Bar chart comparing RMSE across all model baselines.

        Args:
            results: Dict mapping model name → metrics dict.

        Returns:
            Path to saved figure.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        models = list(results.keys())
        rmse_vals = [results[m].get("RMSE", 0) for m in models]
        mae_vals = [results[m].get("MAE", 0) for m in models]
        r2_vals = [results[m].get("R2", 0) for m in models]

        x = np.arange(len(models))
        bar_width = 0.35

        # RMSE & MAE
        colors = ["#90CAF9", "#90CAF9", "#90CAF9", "#2196F3"]  # Highlight adaptive
        bars1 = ax1.bar(x - bar_width / 2, rmse_vals, bar_width, label="RMSE", color=colors)
        bars2 = ax1.bar(x + bar_width / 2, mae_vals, bar_width, label="MAE",
                        color=["#FFCC80", "#FFCC80", "#FFCC80", "#FF9800"])

        ax1.set_xlabel("Model")
        ax1.set_ylabel("Error")
        ax1.set_title("Model Comparison: RMSE & MAE")
        ax1.set_xticks(x)
        ax1.set_xticklabels([m.split("(")[0].strip() for m in models],
                            rotation=15, ha="right")
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis="y")

        # Add value labels
        for bar in bars1:
            ax1.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9
            )

        # R²
        r2_colors = ["#A5D6A7", "#A5D6A7", "#A5D6A7", "#4CAF50"]
        bars3 = ax2.bar(x, r2_vals, bar_width * 1.5, color=r2_colors)
        ax2.set_xlabel("Model")
        ax2.set_ylabel("R²")
        ax2.set_title("Model Comparison: R² Score")
        ax2.set_xticks(x)
        ax2.set_xticklabels([m.split("(")[0].strip() for m in models],
                            rotation=15, ha="right")
        ax2.grid(True, alpha=0.3, axis="y")

        for bar in bars3:
            ax2.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9
            )

        plt.tight_layout()
        path = os.path.join(self.output_dir, "model_comparison.png")
        fig.savefig(path)
        plt.close(fig)
        print(f"  📊 Saved: {path}")
        return path
