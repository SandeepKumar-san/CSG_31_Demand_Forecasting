"""
Main training script for the Adaptive Fusion model.

Usage:
    python experiments/train.py --config experiments/config.yaml

Seeds everything, loads data, trains model, saves results.
Returns final metrics dictionary for reproducibility verification.
"""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from src.data.supplygraph_loader import SupplyGraphLoader
from src.models.complete_model import AdaptiveFusionForecaster
from src.training.trainer import ReproducibleTrainer
from src.utils.config import load_config
from src.utils.seed import SeedManager


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


def main(config_path: str = "experiments/config.yaml") -> dict:
    """
    Main training entry point.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Dictionary with final metrics (for reproducibility verification).
    """
    # ---- Load configuration ----
    config = load_config(config_path)

    # ---- STEP 1: Set all seeds FIRST ----
    seed = config["reproducibility"]["seed"]
    seed_manager = SeedManager(seed=seed)
    seed_manager.set_seed()

    print("=" * 60)
    print("  ADAPTIVE TEMPORAL-STRUCTURAL FUSION MODEL")
    print("  Training Pipeline")
    print("=" * 60)
    print(f"  Master seed: {seed}")
    print(f"  Deterministic: {config['reproducibility']['deterministic']}")
    print("=" * 60)

    # ---- STEP 2: Get device ----
    device = seed_manager.get_device()

    # ---- STEP 3: Load data ----
    print("\n📦 Loading data...")
    loader = SupplyGraphLoader(config)
    train_ds, val_ds, test_ds, info = loader.prepare_datasets()
    graph_data = info["graph_data"]

    # ---- Dynamic config override from actual data ----
    # Update num_material_types from real dataset (may differ from config default)
    actual_mat_types = info["metadata"]["n_material_types"]
    config["model"]["fusion"]["num_material_types"] = max(actual_mat_types, 2)
    print(f"  num_material_types set to {config['model']['fusion']['num_material_types']} (from data)")

    # Update num_unknown_features from actual feature count
    config["model"]["tft"]["num_unknown_features"] = info["n_features"]
    print(f"  num_unknown_features set to {info['n_features']} (from data)")

    # ---- STEP 4: Create DataLoaders (with seeded generators) ----
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

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # ---- STEP 5: Initialize model ----
    print("\n🏗️ Initializing model...")
    model = AdaptiveFusionForecaster(config, seed=seed)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,} total, {n_trainable:,} trainable")

    # ---- STEP 6: Train ----
    print("\n🔬 Starting reproducible training...")
    trainer = ReproducibleTrainer(model, config, seed_manager, device)
    final_metrics = trainer.train(train_loader, val_loader, graph_data)

    # ---- STEP 7: Save final results ----
    results_dir = config.get("output", {}).get("results_dir", "results/")
    os.makedirs(results_dir, exist_ok=True)

    import json
    results = {
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

    print(f"\n✅ Results saved to {results_path}")
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
    args = parser.parse_args()
    main(config_path=args.config)
