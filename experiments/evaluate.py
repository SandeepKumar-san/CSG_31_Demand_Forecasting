"""
Evaluation script comparing 4 model variants.

Baselines:
  1. TFT-only (alpha=1.0 fixed)
  2. GAT-only (alpha=0.0 fixed)
  3. Fixed fusion (alpha=0.5)
  4. Adaptive fusion (learned alpha)

Produces comparison table and generates plots.

Usage:
    python experiments/evaluate.py --config experiments/config.yaml
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from src.data.supplygraph_loader import SupplyGraphLoader
from src.models.complete_model import AdaptiveFusionForecaster
from src.training.metrics import compute_all_metrics
from src.utils.config import load_config
from src.utils.seed import SeedManager
from src.utils.visualization import ModelVisualizer


def collate_fn(batch):
    """Custom collate function for dict-based samples."""
    collated = {}
    for key in batch[0]:
        if isinstance(batch[0][key], torch.Tensor):
            collated[key] = torch.stack([b[key] for b in batch])
        elif isinstance(batch[0][key], (int, float)):
            collated[key] = torch.tensor([b[key] for b in batch])
        else:
            collated[key] = [b[key] for b in batch]
    return collated


def evaluate_model(
    model, loader, graph_data, device, fixed_alpha=None
) -> dict:
    """
    Evaluate model on a dataset.

    Args:
        model: AdaptiveFusionForecaster.
        loader: DataLoader.
        graph_data: PyTorch Geometric Data object.
        device: torch.device.
        fixed_alpha: If set, use fixed fusion weight.

    Returns:
        Dictionary of metrics.
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_alphas = []

    with torch.no_grad():
        for batch in loader:
            # Prepare batch — move ALL tensors to device
            batch["graph_x"] = graph_data.x.to(device)
            batch["graph_edge_index"] = graph_data.edge_index.to(device)
            if graph_data.edge_attr is not None:
                batch["graph_edge_attr"] = graph_data.edge_attr.to(device)
            batch["material_type"] = batch["material_type"].to(device)
            batch["product_id"] = batch["product_id"].to(device)
            if "horizon" in batch:
                batch["horizon"] = batch["horizon"].to(device)
            for key in ["time_series", "price_history", "static_features", "targets"]:
                if key in batch and isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            if fixed_alpha is not None:
                output = model.forward_with_fixed_alpha(batch, fixed_alpha)
            else:
                output = model(batch)

            all_preds.append(output["forecast"].cpu())
            all_targets.append(batch["targets"].cpu())
            all_alphas.append(output["alpha"].cpu())

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    alphas = torch.cat(all_alphas, dim=0)

    metrics = compute_all_metrics(targets, preds)
    metrics["alpha_mean"] = alphas.mean().item()
    metrics["alpha_std"] = alphas.std().item()

    return metrics


def main(config_path: str = "experiments/config.yaml") -> None:
    """
    Run comparative evaluation of all 4 model variants.
    """
    config = load_config(config_path)
    seed = config["reproducibility"]["seed"]
    seed_manager = SeedManager(seed=seed)
    seed_manager.set_seed()
    device = seed_manager.get_device()

    # ---- Load data ----
    loader = SupplyGraphLoader(config)
    _, _, test_ds, info = loader.prepare_datasets()
    graph_data = info["graph_data"]

    # ---- Dynamic config override from actual data ----
    actual_mat_types = info["metadata"]["n_material_types"]
    config["model"]["fusion"]["num_material_types"] = max(actual_mat_types, 2)
    config["model"]["tft"]["num_unknown_features"] = info["n_features"]

    test_loader = DataLoader(
        test_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # ---- Load trained model ----
    checkpoint_path = os.path.join(
        config.get("output", {}).get("checkpoints_dir", "results/checkpoints/"),
        "best_model.pt",
    )

    model = AdaptiveFusionForecaster(config, seed=seed).to(device)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"✅ Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"⚠️  No checkpoint found at {checkpoint_path}, using initialized model")

    # ---- Evaluate 4 variants ----
    print("\n" + "=" * 60)
    print("  BASELINE COMPARISON")
    print("=" * 60)

    variants = {
        "TFT-only (α=1.0)": 1.0,
        "GAT-only (α=0.0)": 0.0,
        "Fixed Fusion (α=0.5)": 0.5,
        "Adaptive Fusion (learned)": None,
    }

    all_results = {}

    for name, alpha in variants.items():
        metrics = evaluate_model(model, test_loader, graph_data, device, alpha)
        all_results[name] = metrics

        print(f"\n  {name}:")
        print(f"    RMSE:  {metrics['RMSE']:.4f}")
        print(f"    MAE:   {metrics['MAE']:.4f}")
        print(f"    MAPE:  {metrics['MAPE']:.2f}%")
        print(f"    SMAPE: {metrics['SMAPE']:.2f}%")
        print(f"    R²:    {metrics['R2']:.4f}")
        print(f"    α:     {metrics['alpha_mean']:.3f} ± {metrics['alpha_std']:.3f}")

    # ---- Summary table ----
    print("\n" + "=" * 60)
    print("  SUMMARY TABLE")
    print("=" * 60)
    print(f"  {'Model':<30} {'RMSE':>8} {'MAE':>8} {'MAPE':>8} {'R²':>8}")
    print("  " + "-" * 56)
    for name, metrics in all_results.items():
        marker = " ←" if "Adaptive" in name else ""
        print(
            f"  {name:<30} {metrics['RMSE']:>8.4f} {metrics['MAE']:>8.4f} "
            f"{metrics['MAPE']:>7.2f}% {metrics['R2']:>8.4f}{marker}"
        )

    # ---- Alpha statistics ----
    adaptive = all_results.get("Adaptive Fusion (learned)", {})
    print("\n  ALPHA STATISTICS (Adaptive Fusion):")
    print(f"    Mean alpha: {adaptive.get('alpha_mean', 0):.4f} (>0.5 leans temporal)")
    print(f"    Std alpha:  {adaptive.get('alpha_std', 0):.4f} (variation by material)")

    # ---- Save results ----
    results_dir = config.get("output", {}).get("results_dir", "results/")
    os.makedirs(results_dir, exist_ok=True)

    results_path = os.path.join(results_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Results saved to {results_path}")

    # ---- Generate plots ----
    plots_dir = config.get("output", {}).get("plots_dir", "results/plots/")
    os.makedirs(plots_dir, exist_ok=True)

    try:
        visualizer = ModelVisualizer(plots_dir)

        # Load training history if available
        history_path = os.path.join(
            config.get("output", {}).get("logs_dir", "results/logs/"),
            "training_history.json",
        )
        if os.path.exists(history_path):
            with open(history_path) as f:
                history = json.load(f)
            visualizer.plot_training_curves(history)
            visualizer.plot_alpha_by_epoch(history)

        # Model comparison
        visualizer.plot_model_comparison(all_results)

        print(f"✅ Plots saved to {plots_dir}")
    except Exception as e:
        print(f"⚠️  Plot generation failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model variants")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/config.yaml",
        help="Path to config YAML",
    )
    args = parser.parse_args()
    main(config_path=args.config)
