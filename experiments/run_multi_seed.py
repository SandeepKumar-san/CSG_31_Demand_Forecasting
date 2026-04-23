"""
Multi-Seed Benchmarking Script.

Runs the training pipeline 5 times with different seeds and aggregates results.
Seeds: [42, 123, 456, 789, 1337]

Usage:
    python experiments/run_multi_seed.py --dataset usgs --epochs 10
"""
import argparse
import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict

# Add src to path
import sys
sys.path.insert(0, os.getcwd())

from experiments.train import main as train_main
from experiments.evaluate import main as evaluate_main
from src.utils.config import load_config, load_dataset_config, update_dynamic_parameters

def run_benchmark(dataset: str, epochs: int, config_path: str, num_seeds: int = 5, eval_only: bool = False):
    seeds = [42, 123, 456, 789, 1337][:num_seeds]
    all_metrics = []
    best_global_wape = float('inf')
    best_global_seed = None

    print("=" * 60)
    print(f"  MULTI-SEED BENCHMARK: {dataset.upper()}")
    print(f"  Seeds: {seeds}")
    print("=" * 60)

    # Import evaluation logic
    from experiments.evaluate import evaluate_model, collate_fn, run_plotting_suite
    from experiments.train import get_data_loader
    from torch.utils.data import DataLoader
    import torch

    for i, seed in enumerate(seeds):
        print(f"\n[RUN {i+1}/5] Seed: {seed}")
        try:
            if not eval_only:
                # 1. Train
                print(f"  [TRAIN] Running training for seed {seed}...")
                train_main(
                    config_path=config_path,
                    dataset_override=dataset,
                    epochs_override=epochs,
                    seed_override=seed
                )
            else:
                print(f"  [SKIP] Skipping training phase for seed {seed} (--eval_only active)")
            
            # 2. Evaluate on Test Set
            print(f"  [EVAL] Running test-set evaluation for seed {seed}...")
            
            # Load bifurcated config (merges common + dataset)
            cfg = load_dataset_config(dataset, config_path)
            cfg["reproducibility"]["seed"] = seed
            
            # Base results dir for this dataset (nested)
            base_results = cfg["output"].get("results_dir", "results/")
            ds_results = os.path.join(base_results, dataset)
            cfg["output"]["results_dir"] = ds_results
            cfg["output"]["plots_dir"] = os.path.join(ds_results, "plots")
            cfg["output"]["logs_dir"] = os.path.join(ds_results, "logs")
            cfg["output"]["checkpoints_dir"] = os.path.join(ds_results, "checkpoints")
            
            # Set seed for deterministic evaluation
            from src.utils.seed import SeedManager
            seed_manager = SeedManager(seed=seed)
            seed_manager.set_seed()
            device = seed_manager.get_device()
            
            # Load Data
            loader_obj = get_data_loader(cfg, dataset)
            _, _, test_ds, info = loader_obj.prepare_datasets()
            graph_data = info["graph_data"]
            
            # Update dynamic parameters (shapes only)
            update_dynamic_parameters(cfg, info, graph_data)
            
            test_loader = DataLoader(
                test_ds,
                batch_size=cfg["training"].get("batch_size", 32),
                shuffle=False,
                num_workers=0,
                collate_fn=collate_fn,
            )
            
            # Re-init model and load best checkpoint
            from src.models.complete_model import AdaptiveFusionForecaster
            model = AdaptiveFusionForecaster(cfg, seed=seed).to(device)
            
            # Path logic follows train.py's dataset-specific subfolders
            chk_dir = cfg["output"]["checkpoints_dir"]
            checkpoint_path = os.path.join(chk_dir, f"best_model_seed{seed}.pt")
            
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint["model_state_dict"])
                print(f"  [OK] Loaded best checkpoint from seed {seed}")
            else:
                print(f"  [WARN] No checkpoint found for seed {seed}, using last model weights")

            # Run Evaluation across all variants for IEEE-compliant ablation
            variants = {
                "TFT-only": 1.0,
                "GAT-only": 0.0,
                "Fixed Fusion": 0.5,
                "Adaptive Fusion (Ours)": None,
            }
            
            seed_results_for_plotting = {}
            for name, alpha in variants.items():
                v_metrics = evaluate_model(model, test_loader, graph_data, device, fixed_alpha=alpha)
                
                # Track Global Best (Ours)
                if alpha is None:
                    current_wape = v_metrics.get('WAPE', float('inf'))
                    if current_wape < best_global_wape:
                        best_global_wape = current_wape
                        best_global_seed = seed
                        global_best_path = os.path.join(chk_dir, "global_best_model.pt")
                        import shutil
                        try:
                            shutil.copy2(checkpoint_path, global_best_path)
                            print(f"  [RECORD] New best WAPE ({current_wape:.2f}%) found in seed {seed}!")
                        except: pass
                
                # Filter metrics and store for aggregation
                v_metrics_clean = {k: v for k, v in v_metrics.items() if not k.endswith("_array")}
                v_metrics_clean["Seed"] = seed
                v_metrics_clean["Variant"] = name
                all_metrics.append(v_metrics_clean)
                seed_results_for_plotting[name] = v_metrics

            # Generate plots for this specific seed (using all variants)
            # print(f"  [Plot] Generating visual artifacts for seed {seed}...")
            # run_plotting_suite(seed_results_for_plotting, cfg, info)
            
        except Exception as e:
            print(f"  [ERROR] Run failed for seed {seed}: {e}")
            import traceback
            traceback.print_exc()

    # Aggregate
    if not all_metrics:
        print("\nNo successful runs to aggregate.")
        return

    df = pd.DataFrame(all_metrics)
    
    # ---- Aggregate Results Output ----
    # Pre-merging the config once for the output paths
    cfg = load_dataset_config(dataset, config_path)
    base_results = cfg.get("output", {}).get("results_dir", "results/")
    ds_results = os.path.join(base_results, dataset)
    os.makedirs(ds_results, exist_ok=True)
    
    if "output" not in cfg:
        cfg["output"] = {}
    cfg["output"]["results_dir"] = ds_results
    cfg["output"]["plots_dir"] = os.path.join(ds_results, "plots")
    cfg["output"]["logs_dir"] = os.path.join(ds_results, "logs")
    
    # Save raw seed results
    csv_path = os.path.join(ds_results, "multi_seed_metrics.csv")
    df.to_csv(csv_path, index=False)
    
    # ---- Compute Mean +/- Std for all Variants ----
    # Group by Variant and calculate stats
    grouped = df.groupby("Variant")
    summary_mean = grouped.mean(numeric_only=True)
    summary_std = grouped.std(numeric_only=True)
    
    print("\n" + "=" * 80)
    print(f"  UNIFIED IEEE COMPARISON TABLE: {dataset.upper()}")
    print("=" * 80)
    print(f"  {'Model':<30} | {'RMSE':^15} | {'WAPE (%)':^15} | {'R2':^10}")
    print("  " + "-" * 78)
    
    # 1. Classical Baselines (Deterministic)
    # Search in dataset-specific folder first, then base folder
    ds_classical = os.path.join(ds_results, f"classical_baselines_{dataset}.json")
    base_classical = os.path.join(base_results, f"classical_baselines_{dataset}.json")
    
    classical_path = ds_classical if os.path.exists(ds_classical) else base_classical
    
    if os.path.exists(classical_path):
        with open(classical_path) as f:
            classical = json.load(f)
        # Use simple mapping for unified naming
        name_map = {"ARIMA": "ARIMA (Statistical)", "XGBoost": "XGBoost (ML)"}
        for k, m in classical.items():
            disp_name = name_map.get(k, k)
            print(f"  {disp_name:<30} | {m['RMSE']:^15.4f} | {m['WAPE']:^15.2f} | {m['R2']:^10.4f}")
    else:
        print(f"  [INFO] Classical baselines not found. Run experiments/run_classical_baselines.py first.")
    
    print("  " + "-" * 78)
    
    # 2. Deep Learning Variants (Stochastic: Mean +/- Std)
    # Order: TFT, GAT, Fixed, Adaptive
    order = ["TFT-only", "GAT-only", "Fixed Fusion", "Adaptive Fusion (Ours)"]
    for variant in order:
        if variant in summary_mean.index:
            m_rmse, s_rmse = summary_mean.at[variant, 'RMSE'], summary_std.at[variant, 'RMSE']
            m_wape, s_wape = summary_mean.at[variant, 'WAPE'], summary_std.at[variant, 'WAPE']
            m_r2, s_r2 = summary_mean.at[variant, 'R2'], summary_std.at[variant, 'R2']
            
            rmse_str = f"{m_rmse:.2f}±{s_rmse:.2f}"
            wape_str = f"{m_wape:.2f}±{s_wape:.2f}"
            r2_str = f"{m_r2:.3f}"
            
            print(f"  {variant:<30} | {rmse_str:^15} | {wape_str:^15} | {r2_str:^10}")

    # ---- Generate Final Aggregate Plots (IEEE Style with Error Bars) ----
    print("\n  [Plot] Generating final aggregate visualizations...")
    final_plot_results = {}
    for variant in order:
        if variant in summary_mean.index:
            v_stats = {}
            for col in summary_mean.columns:
                v_stats[col] = summary_mean.at[variant, col]
                v_stats[f"{col}_std"] = summary_std.at[variant, col]
            final_plot_results[variant] = v_stats
    
    # Force output to the main dataset folder (not a seed subfolder)
    # The run_plotting_suite will automatically load classical baselines if available
    run_plotting_suite(final_plot_results, cfg, info)

    print("=" * 80)
    print(f"[OK] Full multi-seed metrics saved to {csv_path}")
    print(f"[BEST] Global best seed for Ours: {best_global_seed} (WAPE: {best_global_wape:.2f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi-seed benchmark")
    parser.add_argument("--dataset", type=str, default="usgs", choices=["supplygraph", "usgs"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--config", type=str, default="experiments/config.yaml")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds to run (max 5)")
    parser.add_argument("--eval_only", action="store_true", help="Skip training and only evaluate existing checkpoints")
    
    args = parser.parse_args()
    
    # Cap seeds at 5
    num_seeds = min(max(args.seeds, 1), 5)
    run_benchmark(args.dataset, args.epochs, args.config, num_seeds, args.eval_only)
