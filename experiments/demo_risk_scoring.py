"""
Risk Scoring Demo — shows business value of the model.

Generates a sample risk assessment report for top materials,
demonstrating the interpretable risk scoring layer (Layer 5).

Usage:
    python experiments/demo_risk_scoring.py --config experiments/config.yaml
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.supplygraph_loader import SupplyGraphLoader
from src.models.complete_model import AdaptiveFusionForecaster
from src.utils.config import load_config
from src.utils.seed import SeedManager


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


def main(config_path: str = "experiments/config.yaml") -> None:
    """Generate risk assessment report for top materials."""
    config = load_config(config_path)
    seed = config["reproducibility"]["seed"]
    seed_manager = SeedManager(seed=seed)
    seed_manager.set_seed()
    device = seed_manager.get_device()

    # ---- Load data ----
    loader = SupplyGraphLoader(config)
    _, _, test_ds, info = loader.prepare_datasets()
    graph_data = info["graph_data"]
    metadata = info["metadata"]

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

    # ---- Load model ----
    model = AdaptiveFusionForecaster(config, seed=seed).to(device)
    checkpoint_path = os.path.join(
        config.get("output", {}).get("checkpoints_dir", "results/checkpoints/"),
        "best_model.pt",
    )
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"✅ Loaded model from {checkpoint_path}")
    else:
        print("⚠️  No checkpoint found, using initialized model")

    model.eval()

    # ---- Get forecasts for all test products ----
    print("\n" + "=" * 60)
    print("  RISK ASSESSMENT REPORT")
    print("  Adaptive Temporal-Structural Fusion Model")
    print("=" * 60)

    all_forecasts = {}

    with torch.no_grad():
        for batch in test_loader:
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

            output = model(batch)

            for i in range(len(batch["product_id"])):
                pid = batch["product_id"][i].item()
                if pid not in all_forecasts:
                    all_forecasts[pid] = {
                        "forecasts": [],
                        "stds": [],
                        "alphas": [],
                    }
                all_forecasts[pid]["forecasts"].append(
                    output["forecast"][i, 0].item()
                )
                all_forecasts[pid]["stds"].append(
                    output["forecast_std"][i, 0].item()
                )
                all_forecasts[pid]["alphas"].append(
                    output["alpha"][i, 0].item()
                )

    # ---- Generate risk report for each product ----
    budget = config["model"]["risk_scoring"]["budget_threshold"]
    mat_props = metadata.get("material_properties", [])

    risk_reports = []

    for pid in sorted(all_forecasts.keys()):
        data = all_forecasts[pid]
        avg_forecast = np.mean(data["forecasts"])
        avg_std = np.mean(data["stds"])
        avg_alpha = np.mean(data["alphas"])

        # Get material properties
        if pid < len(mat_props):
            props = mat_props[pid]
            category = props.get("category", props.get("category_idx", "Unknown"))
            lead_time = props.get("lead_time", 10)
            base_price = props.get("base_price", 100)
        else:
            category = "Unknown"
            lead_time = 10
            base_price = 100

        # Compute risk scores
        projected_cost = avg_forecast * base_price
        budget_stress = projected_cost / budget

        # Estimate current stock (use recent demand as proxy)
        estimated_stock = avg_forecast * 1.2
        demand_during_lt = avg_forecast * (lead_time / 30.0)
        lt_risk_score = demand_during_lt / (estimated_stock + 1e-6)

        # Risk levels
        if budget_stress < 0.7:
            budget_risk = "Low"
        elif budget_stress < 0.9:
            budget_risk = "Medium"
        elif budget_stress < 1.1:
            budget_risk = "High"
        else:
            budget_risk = "Critical"

        if lt_risk_score < 0.5:
            lt_risk = "Low"
        elif lt_risk_score < 1.0:
            lt_risk = "Medium"
        elif lt_risk_score < 1.5:
            lt_risk = "High"
        else:
            lt_risk = "Critical"

        # Actions
        actions = []
        if budget_risk in ["High", "Critical"]:
            actions.append("⚠️ Review budget allocation - Projected cost exceeds threshold")
        if lt_risk == "Critical":
            actions.append("🚨 URGENT: Expedite supplier - Stock insufficient for lead time")
        elif lt_risk == "High":
            actions.append("⚠️ Consider increasing order quantity")
        if not actions:
            actions.append("✅ All risk levels acceptable - Maintain current plan")

        report = {
            "product_id": pid,
            "category": str(category),
            "forecast": round(avg_forecast, 2),
            "forecast_std": round(avg_std, 2),
            "alpha": round(avg_alpha, 3),
            "budget_stress": round(budget_stress, 3),
            "budget_risk": budget_risk,
            "leadtime_risk_score": round(lt_risk_score, 3),
            "leadtime_risk": lt_risk,
            "lead_time_days": lead_time,
            "actions": actions,
        }
        risk_reports.append(report)

    # ---- Print report ----
    # Sort by risk severity (Critical first)
    risk_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
    risk_reports.sort(
        key=lambda r: min(
            risk_order.get(r["budget_risk"], 3),
            risk_order.get(r["leadtime_risk"], 3),
        )
    )

    for report in risk_reports[:10]:  # Top 10 by risk
        print(f"\n  Product ID: {report['product_id']} (Category: {report['category']})")
        print(f"  Forecast (1-step): {report['forecast']:.1f} units ± {report['forecast_std']:.1f}")
        print(f"  Fusion α: {report['alpha']:.3f} ({'temporal-heavy' if report['alpha'] > 0.5 else 'structural-heavy'})")
        print(f"  Budget Stress: {report['budget_stress']:.3f} ({report['budget_risk']})")
        print(f"  Lead-time Risk: {report['leadtime_risk_score']:.3f} ({report['leadtime_risk']})")
        print(f"  Lead Time: {report['lead_time_days']} days")
        print(f"  ACTIONS:")
        for action in report["actions"]:
            print(f"    {action}")
        print("  " + "-" * 50)

    # ---- Save report ----
    reports_dir = config.get("output", {}).get("risk_reports_dir", "results/risk_reports/")
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, "risk_report.json")
    with open(report_path, "w") as f:
        json.dump(risk_reports, f, indent=2)
    print(f"\n✅ Full risk report saved to {report_path}")
    print(f"   Total products assessed: {len(risk_reports)}")

    critical = sum(1 for r in risk_reports if "Critical" in [r["budget_risk"], r["leadtime_risk"]])
    high = sum(1 for r in risk_reports if "High" in [r["budget_risk"], r["leadtime_risk"]])
    print(f"   Critical risk products: {critical}")
    print(f"   High risk products: {high}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Risk Scoring Demo")
    parser.add_argument(
        "--config", type=str, default="experiments/config.yaml"
    )
    args = parser.parse_args()
    main(config_path=args.config)
