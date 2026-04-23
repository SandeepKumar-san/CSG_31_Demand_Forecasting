"""
Generate USGS Risk Assessment Report

This script runs a forward pass of the best trained ATSF model 
over the USGS dataset and saves the output to a JSON file.
This JSON serves as a lightweight bridge to the Streamlit dashboard, 
avoiding expensive PyTorch loading on the web UI.
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.usgs_loader import USGSLoader
from src.models.complete_model import AdaptiveFusionForecaster
from src.utils.config import load_config, load_dataset_config, update_dynamic_parameters
from src.utils.seed import SeedManager

def collate_fn(batch):
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
    config = load_dataset_config("usgs", config_path)
    seed = config["reproducibility"]["seed"]
    seed_manager = SeedManager(seed=seed)
    seed_manager.set_seed()
    device = seed_manager.get_device()

    # Load Data
    loader = USGSLoader(config)
    _, _, test_ds, info = loader.prepare_datasets()
    graph_data = info["graph_data"]

    update_dynamic_parameters(config, info, graph_data)

    test_loader = DataLoader(
        test_ds, batch_size=config["training"]["batch_size"],
        shuffle=False, num_workers=0, collate_fn=collate_fn,
    )

    # Load Model
    model = AdaptiveFusionForecaster(config, seed=seed).to(device)
    base_results = config.get("output", {}).get("results_dir", "results/")
    checkpoint_path = os.path.join(base_results, "usgs", "checkpoints", f"best_model_seed{seed}.pt")
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"✅ Loaded model from {checkpoint_path}")
    else:
        print(f"⚠️ No checkpoint found at {checkpoint_path}, using randomized weights (unadvisable)")

    model.eval()

    all_forecasts = {}
    
    # Needs commodity names mapping
    node_map = {}
    nodes_csv_path = os.path.join("data", "processed", "usgs", "NodesIndex.csv")
    if os.path.exists(nodes_csv_path):
        import pandas as pd
        nodes_df = pd.read_csv(nodes_csv_path)
        node_map = dict(zip(nodes_df['node_id'], nodes_df['commodity_name']))

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
            preds = output["forecast"].cpu()
            alphas = output["alpha"].cpu()
            
            # Inverse transform
            product_ids = batch["product_id"].cpu()
            if hasattr(graph_data, "target_means") and hasattr(graph_data, "target_stds"):
                means = graph_data.target_means[product_ids].cpu().unsqueeze(-1)
                stds = graph_data.target_stds[product_ids].cpu().unsqueeze(-1)
                preds = (preds * stds) + means
            
            for i in range(len(product_ids)):
                pid = product_ids[i].item()
                if pid not in all_forecasts:
                    all_forecasts[pid] = {"forecasts": [], "alphas": []}
                all_forecasts[pid]["forecasts"].append(preds[i, 0].item())
                all_forecasts[pid]["alphas"].append(alphas[i, 0].item())

    # Build report
    risk_reports = {}
    budget = config["model"]["risk_scoring"]["budget_threshold"]
    
    for pid, data in all_forecasts.items():
        avg_forecast = np.mean(data["forecasts"])
        avg_alpha = np.mean(data["alphas"])
        
        commodity_name = node_map.get(pid, f"Commodity_{pid}")
        # Add some deterministic semi-synthetic budget logic based on the real prediction
        base_price = 1000 # Default unit estimate
        lead_time = 45 # Default days
        
        projected_cost = max(avg_forecast, 1) * base_price
        budget_stress = min(projected_cost / budget, 1.5) # clamp for display
        
        if budget_stress < 0.7: budget_risk = "Low"
        elif budget_stress < 0.9: budget_risk = "Medium"
        elif budget_stress < 1.1: budget_risk = "High"
        else: budget_risk = "Critical"

        estimated_stock = max(avg_forecast * 1.2, 1)
        demand_during_lt = max(avg_forecast * (lead_time / 30.0), 1)
        lt_risk_score = demand_during_lt / estimated_stock
        
        if lt_risk_score < 0.5: lt_risk = "Low"
        elif lt_risk_score < 1.0: lt_risk = "Medium"
        elif lt_risk_score < 1.3: lt_risk = "High"
        else: lt_risk = "Critical"

        actions = []
        if budget_risk in ["High", "Critical"]: actions.append("⚠️ Review budget allocation.")
        if lt_risk == "Critical": actions.append("🚨 URGENT: High Dependency Risk Detected.")
        if not actions: actions.append("✅ All risk levels acceptable.")

        risk_reports[commodity_name.upper()] = {
            "forecast_volume": round(avg_forecast, 2),
            "alpha": round(avg_alpha, 3),
            "budget_risk": budget_risk,
            "budget_stress": round(budget_stress, 3),
            "leadtime_score": round(lt_risk_score, 3),
            "leadtime_risk": lt_risk,
            "actions": actions
        }

    reports_dir = os.path.join(base_results, "risk_reports", "usgs")
    os.makedirs(reports_dir, exist_ok=True)
    out_path = os.path.join(reports_dir, "risk_report.json")
    with open(out_path, "w") as f:
        json.dump(risk_reports, f, indent=2)
        
    print(f"✅ Generated real USGS inference targets for {len(risk_reports)} nodes at {out_path}")

if __name__ == "__main__":
    main()
