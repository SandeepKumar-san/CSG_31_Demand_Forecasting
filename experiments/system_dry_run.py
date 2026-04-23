"""
SYSTEM DRY-RUN: One-Batch Validation
Checks: Preprocessing -> Loading -> Architecture -> Forward Pass -> Entropy Loss -> WAPE Metrics.
No model weights are updated.
"""
import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, os.getcwd())

from src.utils.config import load_config
from src.utils.seed import SeedManager
from src.data.usgs_loader import USGSLoader
from src.models.complete_model import AdaptiveFusionForecaster
from src.training.loss import AdaptiveFusionLoss
from src.training.metrics import compute_all_metrics

def run_system_check(dataset_name="usgs"):
    print(f"\n{'='*60}")
    print(f"  SYSTEM DRY-RUN: {dataset_name.upper()} DATASET")
    print(f"{'='*60}\n")
    
    # 1. Configuration & Seeding
    config = load_config("experiments/config.yaml")
    config["data"]["dataset"] = dataset_name
    seed_manager = SeedManager(seed=config["reproducibility"]["seed"])
    seed_manager.set_seed()
    
    # 2. Preprocessing & Loading
    print("STEP 1: Loading Data & Graph...")
    loader = USGSLoader(config)
    train_ds, val_ds, test_ds, info = loader.prepare_datasets()
    
    if len(train_ds) == 0:
        print("  [SKIP] No training samples found. Skipping model check.")
        return
    
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    batch = next(iter(train_loader))
    
    # Prep Graph Data for Batch (mock collate for dry-run)
    # The loader's collate_fn handles this in real training, 
    # but here we'll manually ensure dimensions are correct
    graph_data = info["graph_data"]
    batch["graph_x"] = graph_data.x
    batch["graph_edge_index"] = graph_data.edge_index
    batch["graph_edge_attr"] = graph_data.edge_attr
    batch["graph_edge_type"] = graph_data.edge_type
    batch["graph_edge_weight"] = graph_data.edge_weight_values

    # 3. Model Initialization
    print("STEP 2: Initializing Architecture...")
    # Ensure Architecture matches data dimensions
    config["model"]["gat"]["num_node_features"] = info["graph_data"].x.shape[1]
    config["model"]["gat"]["num_edge_types"] = info["num_edge_types"]
    config["model"]["gat"]["input_dim"] = info["graph_data"].x.shape[1]  # Ensure input_proj matches
    
    config["model"]["tft"]["num_unknown_features"] = info["n_features"]
    config["model"]["tft"]["num_static_features"] = info["graph_data"].x.shape[1]
    
    config["model"]["fusion"]["num_horizons"] = len(config["model"]["fusion"]["horizons"]) if dataset_name != "usgs" else 1
    config["model"]["fusion"]["num_material_types"] = info["metadata"]["n_material_types"]
    
    model = AdaptiveFusionForecaster(config)
    print(f"  [ARCH] Heads: {config['model']['gat']['heads']}, Hidden: {config['model']['tft']['hidden_dim']}")
    
    # 4. Forward Pass
    print("STEP 3: Executing One Forward Pass...")
    model.eval()
    with torch.no_grad():
        out = model(batch)
    
    forecasts = out["forecast"]
    alphas = out["alpha"]
    print(f"  [OUT] Forecast shape: {forecasts.shape}")
    print(f"  [OUT] Alpha shape: {alphas.shape}")
    print(f"  [OUT] Mean alpha: {alphas.mean().item():.4f}")
    
    # Check for NaNs
    if torch.isnan(forecasts).any():
        print("  [ERROR] NaNs found in forecasts!")
    else:
        print("  [PASS] No NaNs in outputs.")

    # 5. Loss & Metrics Calculation
    print("STEP 4: Verification of Math (Loss & Metrics)...")
    criterion = AdaptiveFusionLoss(
        quantiles=config["data"]["quantiles"],
        alpha_reg_weight=config["training"]["alpha_reg_weight"]
    )
    loss_dict = criterion(out, batch["targets"])
    loss_val = loss_dict["total_loss"]
    
    print(f"  [LOSS] Total Loss: {loss_val.item():.6f}")
    
    metrics = compute_all_metrics(batch["targets"], forecasts)
    print(f"  [METRICS] RMSE: {metrics['RMSE']:.4f} (expect ~1.0 for normalized targets)")
    print(f"  [METRICS] WAPE: {metrics['WAPE']:.2f}%")
    print(f"  [METRICS] SMAPE: {metrics['SMAPE']:.2f}%")
    
    print(f"\nDONE: {dataset_name.upper()} DRY-RUN COMPLETE: NO CRASHES DETECTED.")

if __name__ == "__main__":
    try:
        run_system_check("usgs")
    except Exception as e:
        print(f"\nFAIL: DRY-RUN FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
