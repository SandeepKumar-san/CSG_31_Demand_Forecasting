"""
Edge Type Analysis for USGS Dataset.

Computes edge type importance rankings, runs ablation studies
(remove one edge type at a time), and generates visualizations.

Usage:
    python experiments/edge_type_analysis.py --config experiments/config.yaml
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.usgs_loader import USGSLoader, EDGE_TYPE_NAMES
from src.models.complete_model import AdaptiveFusionForecaster
from src.training.metrics import compute_all_metrics
from src.utils.config import load_dataset_config, update_dynamic_parameters
from src.utils.seed import SeedManager


def collate_fn(batch):
    """Custom collate for dict-based samples."""
    collated = {}
    for key in batch[0]:
        if isinstance(batch[0][key], torch.Tensor):
            collated[key] = torch.stack([b[key] for b in batch])
        elif isinstance(batch[0][key], (int, float)):
            collated[key] = torch.tensor([b[key] for b in batch])
        else:
            collated[key] = [b[key] for b in batch]
    return collated


def prepare_batch(batch, graph_data, device):
    """Prepare batch with graph data."""
    batch["graph_x"] = graph_data.x.to(device)
    batch["graph_edge_index"] = graph_data.edge_index.to(device)
    if graph_data.edge_attr is not None:
        batch["graph_edge_attr"] = graph_data.edge_attr.to(device)
    if hasattr(graph_data, "edge_type") and graph_data.edge_type is not None:
        batch["graph_edge_type"] = graph_data.edge_type.to(device)
    if hasattr(graph_data, "edge_weight_values") and graph_data.edge_weight_values is not None:
        batch["graph_edge_weight"] = graph_data.edge_weight_values.to(device)
    batch["material_type"] = batch["material_type"].to(device)
    batch["product_id"] = batch["product_id"].to(device)
    if "horizon" not in batch:
        batch["horizon"] = torch.zeros(
            batch["material_type"].shape[0], dtype=torch.long, device=device
        )
    else:
        batch["horizon"] = batch["horizon"].to(device)
    for key in ["time_series", "price_history", "static_features", "targets"]:
        if key in batch and isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
    return batch


def compute_edge_type_importance(model, test_loader, graph_data, device):
    """
    Compute mean attention weight per edge type from the trained model.

    Returns:
        Dict with edge_type_id → {mean, std, count}
    """
    model.eval()
    all_edge_type_attns = {t: [] for t in range(graph_data.num_edge_types)}

    with torch.no_grad():
        for batch in test_loader:
            batch = prepare_batch(batch, graph_data, device)
            output = model(batch)

            # Edge type attention from GAT output
            if "edge_attention" in output:
                edge_attn = output["edge_attention"]
                edge_type = graph_data.edge_type.to(device)

                if isinstance(edge_attn, torch.Tensor) and edge_attn.dim() >= 1:
                    # Mean across heads if multi-headed
                    if edge_attn.dim() > 1:
                        mean_attn = edge_attn.mean(dim=-1)
                    else:
                        mean_attn = edge_attn

                    n_original = edge_type.shape[0]
                    if mean_attn.shape[0] > n_original:
                        mean_attn = mean_attn[:n_original]

                    for t in range(graph_data.num_edge_types):
                        mask = (edge_type == t)
                        if mask.any():
                            vals = mean_attn[mask].cpu().numpy()
                            all_edge_type_attns[t].extend(vals.tolist())

    # Compute statistics
    results = {}
    for t_id in range(graph_data.num_edge_types):
        vals = all_edge_type_attns[t_id]
        name = EDGE_TYPE_NAMES.get(t_id, f"type_{t_id}")
        if len(vals) > 0:
            results[name] = {
                "type_id": t_id,
                "mean_attention": float(np.mean(vals)),
                "std_attention": float(np.std(vals)),
                "count": len(vals),
            }
        else:
            results[name] = {
                "type_id": t_id,
                "mean_attention": 0.0,
                "std_attention": 0.0,
                "count": 0,
            }

    return results


def run_ablation_study(model, config, test_loader, graph_data, device):
    """
    Run edge type ablation: remove one type at a time, measure RMSE degradation.

    Returns:
        Dict with edge_type_name → {rmse_full, rmse_ablated, degradation_pct}
    """
    model.eval()

    # First: compute full-graph RMSE
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            batch = prepare_batch(batch, graph_data, device)
            output = model(batch)
            all_preds.append(output["forecast"].cpu())
            all_targets.append(batch["targets"].cpu())

    preds_full = torch.cat(all_preds, dim=0)
    targets_full = torch.cat(all_targets, dim=0)
    metrics_full = compute_all_metrics(targets_full, preds_full)
    rmse_full = metrics_full["RMSE"]

    print(f"\n  Full graph RMSE: {rmse_full:.4f}")

    # Ablation: remove one edge type at a time
    ablation_results = {}
    num_edge_types = graph_data.num_edge_types

    for remove_type in range(num_edge_types):
        type_name = EDGE_TYPE_NAMES.get(remove_type, f"type_{remove_type}")

        # Create modified graph without this edge type
        mask = graph_data.edge_type != remove_type
        if not mask.any():
            print(f"    [{type_name}] No edges of this type, skipping")
            continue

        # Build ablated graph data
        from torch_geometric.data import Data
        ablated_graph = Data(
            x=graph_data.x.clone(),
            edge_index=graph_data.edge_index[:, mask],
            edge_attr=graph_data.edge_attr[mask] if graph_data.edge_attr is not None else None,
            num_nodes=graph_data.num_nodes,
        )
        ablated_graph.material_types = graph_data.material_types
        ablated_graph.lead_times = graph_data.lead_times
        ablated_graph.edge_type = graph_data.edge_type[mask]
        ablated_graph.edge_weight_values = graph_data.edge_weight_values[mask]
        ablated_graph.num_edge_types = num_edge_types

        # Evaluate on ablated graph
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in test_loader:
                batch = prepare_batch(batch, ablated_graph, device)
                output = model(batch)
                all_preds.append(output["forecast"].cpu())
                all_targets.append(batch["targets"].cpu())

        preds_abl = torch.cat(all_preds, dim=0)
        targets_abl = torch.cat(all_targets, dim=0)
        metrics_abl = compute_all_metrics(targets_abl, preds_abl)
        rmse_abl = metrics_abl["RMSE"]

        degradation_pct = ((rmse_abl - rmse_full) / rmse_full * 100) if rmse_full > 0 else 0.0

        ablation_results[type_name] = {
            "type_id": remove_type,
            "rmse_full": rmse_full,
            "rmse_ablated": rmse_abl,
            "degradation_pct": degradation_pct,
        }

        print(
            f"    [{type_name:30s}] RMSE: {rmse_abl:.4f} "
            f"(Δ = {degradation_pct:+.2f}%)"
        )

    return ablation_results


def save_results(importance, ablation, output_dir):
    """Save analysis results to CSV and JSON."""
    os.makedirs(output_dir, exist_ok=True)

    # Save importance ranking
    imp_path = os.path.join(output_dir, "edge_type_importance.csv")
    sorted_imp = sorted(
        importance.items(),
        key=lambda x: x[1]["mean_attention"],
        reverse=True,
    )
    with open(imp_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "edge_type", "type_id", "mean_attention", "std_attention", "count"])
        for rank, (name, data) in enumerate(sorted_imp, 1):
            writer.writerow([
                rank, name, data["type_id"],
                f"{data['mean_attention']:.6f}",
                f"{data['std_attention']:.6f}",
                data["count"],
            ])
    print(f"\n  [OK] Importance saved to {imp_path}")

    # Save ablation results
    if ablation:
        abl_path = os.path.join(output_dir, "edge_type_ablation.csv")
        sorted_abl = sorted(
            ablation.items(),
            key=lambda x: x[1]["degradation_pct"],
            reverse=True,
        )
        with open(abl_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["edge_type", "type_id", "rmse_full", "rmse_ablated", "degradation_pct"])
            for name, data in sorted_abl:
                writer.writerow([
                    name, data["type_id"],
                    f"{data['rmse_full']:.6f}",
                    f"{data['rmse_ablated']:.6f}",
                    f"{data['degradation_pct']:.4f}",
                ])
        print(f"  [OK] Ablation saved to {abl_path}")

    # Save combined JSON
    combined = {
        "importance": importance,
        "ablation": ablation,
    }
    json_path = os.path.join(output_dir, "edge_type_analysis.json")
    with open(json_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"  [OK] Combined analysis saved to {json_path}")


def main(config_path: str = "experiments/config.yaml"):
    """Run edge type analysis on trained USGS model."""
    config = load_dataset_config("usgs", config_path)

    seed = config["reproducibility"]["seed"]
    seed_manager = SeedManager(seed=seed)
    seed_manager.set_seed()
    device = seed_manager.get_device()

    print("=" * 60)
    print("  EDGE TYPE ANALYSIS — USGS DATASET")
    print("=" * 60)

    # Load data
    loader = USGSLoader(config)
    _, _, test_ds, info = loader.prepare_datasets()
    graph_data = info["graph_data"]

    # Update dynamic parameters (including num_edge_types, num_horizons, etc.)
    update_dynamic_parameters(config, info, graph_data)

    # Load model
    model = AdaptiveFusionForecaster(config, seed=seed)

    checkpoint_path = os.path.join(
        config.get("output", {}).get("checkpoints_dir", "results/checkpoints/"),
        "best_model.pt"
    )
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"  Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"  ⚠ No checkpoint found at {checkpoint_path}")
        print(f"  Using randomly initialized model (results won't be meaningful)")

    model = model.to(device)
    model.eval()

    # Create test loader
    test_loader = DataLoader(
        test_ds, batch_size=16, shuffle=False, collate_fn=collate_fn
    )

    # Run analyses
    print("\n📊 Computing edge type importance...")
    importance = compute_edge_type_importance(model, test_loader, graph_data, device)

    print("\n🔬 Running edge type ablation study...")
    ablation = run_ablation_study(model, config, test_loader, graph_data, device)

    # Save results
    base_results = config.get("output", {}).get("results_dir", "results/")
    output_dir = os.path.join(base_results, "usgs")
    save_results(importance, ablation, output_dir)

    print("\n[DONE] Edge type analysis complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edge Type Analysis for USGS")
    parser.add_argument(
        "--config", type=str, default="experiments/config.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()
    main(args.config)
