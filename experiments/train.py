"""
Main training script for the Adaptive Fusion model.

Supports both SupplyGraph and USGS datasets via config-based switching.

Usage:
    python experiments/train.py --config experiments/config.yaml
    python experiments/train.py --config experiments/config.yaml --dataset usgs

Seeds everything, loads data, trains model, saves results.
Returns final metrics dictionary for reproducibility verification.
"""

import argparse
import os
import sys
from typing import Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from src.models.complete_model import AdaptiveFusionForecaster
from src.training.trainer import ReproducibleTrainer
from src.utils.config import load_config, load_dataset_config, update_dynamic_parameters
from src.utils.seed import SeedManager


def get_data_loader(config: dict, dataset_name: str):
    """
    Factory function to get the appropriate data loader.

    Args:
        config: Full configuration dictionary.
        dataset_name: "supplygraph" or "usgs".

    Returns:
        Loader instance with prepare_datasets() method.
    """
    if dataset_name == "usgs":
        from src.data.usgs_loader import USGSLoader
        return USGSLoader(config)
    else:
        from src.data.supplygraph_loader import SupplyGraphLoader
        return SupplyGraphLoader(config)


def collate_fn(batch):
    """Custom collate function to handle dict-based samples."""
    collated = {}
    for key in batch[0]:
        if isinstance(batch[0][key], torch.Tensor):
            collated[key] = torch.stack([b[key] for b in batch])
        elif isinstance(batch[0][key], (int, float)):
            collated[key] = torch.tensor([b[key] for b in batch])
        else:
            collated[key] = [b[key] for b in batch]
    return collated




def main(
    config_path: str = "experiments/config.yaml",
    dataset_override: str = None,
    epochs_override: int = None,
    seed_override: int = None,
) -> dict:
    """
    Main training entry point.

    Args:
        config_path: Path to YAML configuration file.
        dataset_override: Override dataset from CLI (None = use config).
        epochs_override: Override epochs from CLI (None = use config).

    Returns:
        Dictionary with final metrics (for reproducibility verification).
    """
    # ---- Load bifurcated configuration ----
    if dataset_override is None:
        # Peak at raw config to find default dataset
        raw_config = load_config(config_path)
        dataset_name = raw_config.get("common", {}).get("dataset", "supplygraph")
    else:
        dataset_name = dataset_override

    config = load_dataset_config(dataset_name, config_path)
    
    # ---- Dataset-Specific Output Organization ----
    # Base results dir for this dataset (now all in 'output' section)
    base_results = config["output"].get("results_dir", "results/")
    ds_results = os.path.join(base_results, dataset_name)
    
    # Update all output paths to be dataset-nested
    config["output"]["results_dir"] = ds_results
    config["output"]["plots_dir"] = os.path.join(ds_results, "plots")
    config["output"]["checkpoints_dir"] = os.path.join(ds_results, "checkpoints")
    config["output"]["logs_dir"] = os.path.join(ds_results, "logs")
    config["output"]["risk_reports_dir"] = os.path.join(ds_results, "risk_reports")

    # ---- Apply CLI Overrides ----
    if epochs_override is not None:
        config["training"]["epochs"] = epochs_override
        print(f"  [CLI] Epochs override: {epochs_override}")

    if seed_override is not None:
        config["reproducibility"]["seed"] = seed_override
        print(f"  [CLI] Seed override: {seed_override}")

    # ---- STEP 1: Set all seeds FIRST ----
    seed = config["reproducibility"]["seed"]
    seed_manager = SeedManager(seed=seed)
    seed_manager.set_seed()

    print("=" * 60)
    print("  ADAPTIVE TEMPORAL-STRUCTURAL FUSION MODEL")
    print("  Training Pipeline")
    print("=" * 60)
    print(f"  Dataset: {dataset_name}")
    print(f"  Master seed: {seed}")
    print(f"  Deterministic: {config['reproducibility']['deterministic']}")
    print("=" * 60)

    # ---- STEP 2: Get device ----
    device = seed_manager.get_device()

    # ---- STEP 3: Load data ----
    print("\n[LOAD] Loading data...")
    loader = get_data_loader(config, dataset_name)
    train_ds, val_ds, test_ds, info = loader.prepare_datasets()
    graph_data = info["graph_data"]

    # ---- Step 4: Dynamic Dimension Detection ----
    # This ONLY updates parameters that are determined by the data shape.
    # It NEVER overrides architectural hyperparameters (hidden_dim, dropout).
    update_dynamic_parameters(config, info, graph_data)

    # ---- Attach edge_type/edge_weight to graph_data for USGS ----
    # These are used by the multi-relational GAT
    if hasattr(graph_data, "edge_type"):
        print(f"  [Graph] edge_type tensor: {graph_data.edge_type.shape}")
    if hasattr(graph_data, "edge_weight_values"):
        print(f"  [Graph] edge_weight tensor: {graph_data.edge_weight_values.shape}")

    # ---- STEP 5: Create DataLoaders ----
    num_workers = config["reproducibility"].get("num_workers", 0)
    batch_size = config["training"]["batch_size"]

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=seed_manager.get_generator(),
        num_workers=num_workers,
        worker_init_fn=seed_manager.worker_init_fn,
        collate_fn=collate_fn,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_manager.worker_init_fn,
        collate_fn=collate_fn,
        drop_last=False,
    )

    print(f"\n  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # ---- STEP 6: Initialize model ----
    print("\n[STEP] Initializing model...")
    model = AdaptiveFusionForecaster(config, seed=seed)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,} total, {n_trainable:,} trainable")

    # ---- STEP 7: Train ----
    print("\n[STEP] Starting reproducible training...")
    trainer = ReproducibleTrainer(model, config, seed_manager, device)
    final_metrics = trainer.train(train_loader, val_loader, graph_data)

    # ---- STEP 8: Save results ----
    results_dir = config["output"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    import json
    results = {
        "dataset": dataset_name,
        "seed": seed,
        "best_val_loss": final_metrics["best_val_loss"],
        "train_loss": final_metrics["train_loss"],
        "val_loss": final_metrics["val_loss"],
        "epochs_trained": final_metrics["epochs_trained"],
        "total_time": final_metrics["total_time"],
        "n_parameters": n_params,
    }

    results_path = os.path.join(results_dir, "training_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[DONE] Results saved to {results_path}")
    print(f"   Best val loss: {results['best_val_loss']:.6f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Adaptive Temporal-Structural Fusion Model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/config.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["supplygraph", "usgs"],
        default=None,
        help="Override dataset (default: use config file setting)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()
    main(
        config_path=args.config,
        dataset_override=args.dataset,
        epochs_override=args.epochs,
        seed_override=args.seed,
    )
