"""
IEEE Classical & ML Baselines Evaluation Script.

Implements robust comparisons against:
  1. ARIMA (AutoRegressive Integrated Moving Average) — Classical baseline
  2. XGBoost (Extreme Gradient Boosting) — Tabular ML baseline

Both baselines use the exact same Train/Test splits and metrics
as the deep learning models to ensure fair IEEE-compliant ablation.

Usage:
    python experiments/run_classical_baselines.py --config experiments/config.yaml
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.simplefilter('ignore', ConvergenceWarning)
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.supplygraph_loader import SupplyGraphLoader
from src.data.usgs_loader import USGSLoader
from src.training.metrics import compute_all_metrics
from src.utils.config import load_config, load_dataset_config
from src.utils.seed import SeedManager


# ==============================================================================
# 1. Utility Functions
# ==============================================================================

def extract_tabular_data(dataset) -> tuple:
    """
    Flattens the temporal and static features into a 2D matrix for XGBoost.
    
    Returns:
        X (np.ndarray): [num_samples, flattened_features]
        Y (np.ndarray): [num_samples, num_horizons]
        PIDs (np.ndarray): [num_samples] product IDs
    """
    X_list, Y_list, pid_list = [], [], []
    
    for i in range(len(dataset)):
        sample = dataset[i]
        
        # Flatten temporal features [seq_length, n_features] -> [seq_length * n_features]
        time_series_flat = sample["time_series"].numpy().flatten()
        
        # Static features
        static = sample["static_features"].numpy().flatten()
        
        # Concatenate all features
        x = np.concatenate([time_series_flat, static])
        X_list.append(x)
        
        # Target array
        Y_list.append(sample["targets"].numpy())
        pid_list.append(sample["product_id"].item())
        
    return np.array(X_list), np.array(Y_list), np.array(pid_list)


# ==============================================================================
# 2. XGBoost Baseline
# ==============================================================================

def run_xgboost_baseline(train_ds, test_ds, graph_data) -> dict:
    """
    Train and evaluate XGBoost using MultiOutputRegressor for multiple horizons.
    """
    print("\n" + "-" * 50)
    print("  [TRAIN] Training XGBoost Baseline")
    print("-" * 50)

    X_train, Y_train, _ = extract_tabular_data(train_ds)
    X_test, Y_test, test_pids = extract_tabular_data(test_ds)

    print(f"  [XGBoost] Train shape: X={X_train.shape}, Y={Y_train.shape}")
    print(f"  [XGBoost] Test shape:  X={X_test.shape},  Y={Y_test.shape}")

    # Initialize Base Estimator
    base_estimator = xgb.XGBRegressor(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        objective='reg:squarederror'
    )
    
    # Use MultiOutputRegressor since our targets have multiple horizons
    model = MultiOutputRegressor(base_estimator)
    
    print("  [XGBoost] Fitting model... (this may take a moment)")
    model.fit(X_train, Y_train)
    
    print("  [XGBoost] Predicting on Test Dataset...")
    preds = model.predict(X_test)
    
    # Convert back to PyTorch tensors for evaluation function compatibility
    preds_tensor = torch.tensor(preds, dtype=torch.float32)
    targets_tensor = torch.tensor(Y_test, dtype=torch.float32)
    product_ids_tensor = torch.tensor(test_pids, dtype=torch.long)

    # --- Inverse Transform to Absolute Units ---
    if hasattr(graph_data, "target_means") and hasattr(graph_data, "target_stds"):
        means = graph_data.target_means[product_ids_tensor].cpu().unsqueeze(-1)
        stds = graph_data.target_stds[product_ids_tensor].cpu().unsqueeze(-1)
        targets_tensor = (targets_tensor * stds) + means
        preds_tensor = (preds_tensor * stds) + means
    
    metrics = compute_all_metrics(targets_tensor, preds_tensor)
    return metrics


# ==============================================================================
# 3. ARIMA Baseline (Statistical)
# ==============================================================================

def run_arima_baseline(test_ds, horizons, graph_data) -> dict:
    """
    Evaluate Walk-Forward ARIMA.
    
    ARIMA isn't trained on the global (X,Y) tabular set. 
    Instead, for EVERY sample in the test set, we fit a local ARIMA(1,1,1)
    on its historical demand (`price_history`), point-in-time, and forecast 
    the requested future horizons. This strictly mimics operational usage.
    """
    print("\n" + "-" * 50)
    print("  [EVAL] Evaluating Walk-Forward ARIMA Baseline")
    print("-" * 50)
    
    all_preds, all_targets, all_pids = [], [], []
    max_horizon = max(horizons)
    
    print(f"  [ARIMA] Generating predictions for {len(test_ds)} test samples...")
    
    success_count = 0
    fail_count = 0
    
    for i in range(len(test_ds)):
        sample = test_ds[i]
        
        # 1D array of historical demand values
        history = sample["price_history"].numpy()
        targets = sample["targets"].numpy()
        pid = sample["product_id"].item()
        
        try:
            # IEEE Standard: simple efficient ARIMA(1,1,1) for rapid walk-forward
            # If historical sequence is too short, fall back to Naive Forecast
            
            if len(history) < 3:
                forecast = np.full(max_horizon, history[-1])
            else:
                model = ARIMA(history, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False)
                fitted = model.fit()
                # Forecast 'max_horizon' steps into the future
                forecast = fitted.forecast(steps=max_horizon)
                # Anti-explosion safeguard: ARIMA(1,1,1) without stationarity can diverge exponentially.
                # Since inputs are Z-score normalized (and clipped [-5, 5]), we bound the output.
                forecast = np.clip(forecast, -5.0, 5.0)
            
            # Extract exactly the horizons we care about (0-indexed so h-1)
            pred_at_horizons = [forecast[h - 1] for h in horizons]
            
            all_preds.append(pred_at_horizons)
            all_targets.append(targets)
            all_pids.append(pid)
            success_count += 1
            
        except Exception as e:
            # Fallback to Naive Forecast (carry-forward last known value)
            pred_at_horizons = [history[-1]] * len(horizons)
            all_preds.append(pred_at_horizons)
            all_targets.append(targets)
            all_pids.append(pid)
            fail_count += 1
            
        if (i+1) % 500 == 0:
            print(f"    Processed {i+1}/{len(test_ds)} samples...")

    print(f"  [ARIMA] Succeeded: {success_count}, Fallback (Naive): {fail_count}")

    preds_tensor = torch.tensor(all_preds, dtype=torch.float32)
    targets_tensor = torch.tensor(np.array(all_targets), dtype=torch.float32)
    product_ids_tensor = torch.tensor(np.array(all_pids), dtype=torch.long)
    
    # --- Inverse Transform to Absolute Units ---
    if hasattr(graph_data, "target_means") and hasattr(graph_data, "target_stds"):
        means = graph_data.target_means[product_ids_tensor].cpu().unsqueeze(-1)
        stds = graph_data.target_stds[product_ids_tensor].cpu().unsqueeze(-1)
        targets_tensor = (targets_tensor * stds) + means
        preds_tensor = (preds_tensor * stds) + means
    
    metrics = compute_all_metrics(targets_tensor, preds_tensor)
    return metrics


# ==============================================================================
# Main Routine
# ==============================================================================

def main(config_path: str = "experiments/config.yaml", dataset_override: str = None) -> None:
    # 1. Peek at raw config to find default dataset
    if dataset_override is None:
        raw_config = load_config(config_path)
        dataset_name = raw_config.get("common", {}).get("dataset", "supplygraph")
    else:
        dataset_name = dataset_override

    # 2. Load bifurcated config
    config = load_dataset_config(dataset_name, config_path)
    
    seed_manager = SeedManager(seed=config["reproducibility"]["seed"])
    seed_manager.set_seed()
    
    print("============================================================")
    print(f"  IEEE CLASSICAL BASELINES: {dataset_name.upper()}")
    print("============================================================")

    # 3. Initialize correct loader
    if dataset_name == "usgs":
        loader = USGSLoader(config)
        train_ds, val_ds, test_ds, info = loader.prepare_datasets()
        horizons = config["data"]["forecast_horizons"]
    else:
        loader = SupplyGraphLoader(config)
        train_ds, val_ds, test_ds, info = loader.prepare_datasets()
        horizons = config["data"]["forecast_horizons"]

    print(f"\n  [Config] Horizons evaluated: {horizons}")

    # Combine Train and Val for XGBoost typical training (XGBoost does CV internally if needed)
    # But to match DL strictly, we'll train using train_ds and test on test_ds.
    
    # 1. Run XGBoost
    xgb_metrics = run_xgboost_baseline(train_ds, test_ds, info["graph_data"])
    
    # 2. Run ARIMA
    arima_metrics = run_arima_baseline(test_ds, horizons, info["graph_data"])

    all_results = {
        "ARIMA (Statistical)": arima_metrics,
        "XGBoost (ML)": xgb_metrics
    }

    # Print individual results
    for name, metrics in all_results.items():
        print(f"\n  {name}:")
        print(f"    RMSE:  {metrics['RMSE']:.4f}")
        print(f"    MAE:   {metrics['MAE']:.4f}")
        print(f"    WAPE:  {metrics['WAPE']:.2f}%")
        print(f"    SMAPE: {metrics['SMAPE']:.2f}%")
        print(f"    R²:    {metrics['R2']:.4f}")

    # ---- Summary table ----
    print("\n" + "=" * 60)
    print("  SUMMARY TABLE: Classical & ML Baselines")
    print("=" * 60)
    print(f"  {'Model':<30} {'RMSE':>8} {'MAE':>8} {'WAPE':>8} {'R2':>8}")
    print("  " + "-" * 56)
    for name, metrics in all_results.items():
        print(
            f"  {name:<30} {metrics['RMSE']:>8.4f} {metrics['MAE']:>8.4f} "
            f"{metrics['WAPE']:>7.2f}% {metrics['R2']:>8.4f}"
        )  

    # ---- SAVE RESULTS (ISSUE 15 ALIGNMENT) ----
    # Ensure results are nested in dataset-specific subfolders to avoid collision
    base_results = config.get("output", {}).get("results_dir", "results/")
    ds_results = os.path.join(base_results, dataset_name)
    os.makedirs(ds_results, exist_ok=True)
    
    out_file = os.path.join(ds_results, f"classical_baselines_{dataset_name}.json")
    with open(out_file, "w") as f:
        json.dump({
            "ARIMA": arima_metrics,
            "XGBoost": xgb_metrics
        }, f, indent=2)
    print(f"\n[DONE] Baselines saved to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="experiments/config.yaml")
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset (supplygraph or usgs)")
    args = parser.parse_args()
    main(args.config, args.dataset)
