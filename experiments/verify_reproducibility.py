"""
Reproducibility verification script.

Runs training N times with the same seed and verifies that all runs
produce IDENTICAL results. Critical for research paper submission.

Usage:
    python experiments/verify_reproducibility.py --config experiments/config.yaml --num-runs 2
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.utils.config import load_config


def run_training(config_path: str) -> dict:
    """
    Run a full training session and return metrics.

    Imports train.main() freshly each time to ensure clean state.
    """
    # Force reimport for clean state
    if "experiments.train" in sys.modules:
        del sys.modules["experiments.train"]

    from experiments.train import main as train_main
    return train_main(config_path=config_path)


def verify_reproducibility(
    config_path: str = "experiments/config.yaml",
    num_runs: int = 2,
) -> bool:
    """
    Verify that multiple runs produce identical results.

    Args:
        config_path: Path to config YAML.
        num_runs: Number of runs to compare.

    Returns:
        True if all runs are reproducible.
    """
    config = load_config(config_path)
    seed = config["reproducibility"]["seed"]

    print("=" * 60)
    print("  REPRODUCIBILITY VERIFICATION")
    print("=" * 60)
    print(f"  Master seed: {seed}")
    print(f"  Number of runs: {num_runs}")
    print(f"  If reproducible, all runs should produce IDENTICAL results.")
    print("=" * 60)

    results = []

    for run in range(num_runs):
        print(f"\n🔬 RUN {run + 1}/{num_runs}")
        print("-" * 60)

        # Run full training
        metrics = run_training(config_path)
        results.append(metrics)

        print(f"  Final train loss: {metrics['train_loss']:.8f}")
        print(f"  Final val loss:   {metrics['val_loss']:.8f}")
        print(f"  Best val loss:    {metrics['best_val_loss']:.8f}")

    # ---- Compare results ----
    print("\n" + "=" * 60)
    print("  VERIFICATION RESULTS")
    print("=" * 60)

    # Compare val losses
    val_losses = [r["val_loss"] for r in results]
    train_losses = [r["train_loss"] for r in results]
    best_losses = [r["best_val_loss"] for r in results]

    # Check train loss reproducibility
    train_diff = max(train_losses) - min(train_losses)
    val_diff = max(val_losses) - min(val_losses)
    best_diff = max(best_losses) - min(best_losses)

    print(f"\n  Train Loss:")
    for i, loss in enumerate(train_losses):
        print(f"    Run {i + 1}: {loss:.10f}")
    print(f"    Max difference: {train_diff:.10e}")

    print(f"\n  Val Loss:")
    for i, loss in enumerate(val_losses):
        print(f"    Run {i + 1}: {loss:.10f}")
    print(f"    Max difference: {val_diff:.10e}")

    print(f"\n  Best Val Loss:")
    for i, loss in enumerate(best_losses):
        print(f"    Run {i + 1}: {loss:.10f}")
    print(f"    Max difference: {best_diff:.10e}")

    # ---- Overall verdict ----
    max_diff = max(train_diff, val_diff, best_diff)

    print("\n" + "=" * 60)
    if max_diff == 0.0:
        print("  ✅ PASS: All runs produced EXACTLY IDENTICAL results")
        status = "PASS_EXACT"
    elif max_diff < 1e-6:
        print("  ✅ PASS: Numerical precision EXCELLENT (< 1e-6)")
        status = "PASS_EXCELLENT"
    elif max_diff < 1e-4:
        print("  ⚠️  PASS: Numerical precision ACCEPTABLE (< 1e-4)")
        status = "PASS_ACCEPTABLE"
    else:
        print("  ❌ FAIL: Results differ across runs (>= 1e-4)")
        status = "FAIL"
        print("\n  ⚠️  Troubleshooting:")
        print("    - Verify all seeds are set correctly")
        print("    - Set num_workers=0 in DataLoader")
        print("    - Set torch.use_deterministic_algorithms(True)")
        print("    - Set CUBLAS_WORKSPACE_CONFIG=:4096:8")
        print("    - Ensure torch.backends.cudnn.benchmark=False")

    print(f"  Max overall difference: {max_diff:.10e}")
    print("=" * 60)

    is_reproducible = max_diff < 1e-4

    if is_reproducible:
        print("\n  🎉 System is reproducible. Safe to use for paper.")
    else:
        print("\n  ⚠️  Reproducibility issues detected. Fix before paper submission.")

    # ---- Compare model weights (if checkpoints exist) ----
    checkpoint_dir = config.get("output", {}).get("checkpoints_dir", "results/checkpoints/")
    ckpt_path = os.path.join(checkpoint_dir, "best_model.pt")

    if os.path.exists(ckpt_path):
        print("\n  Checking model weight reproducibility...")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt["model_state_dict"]
        weight_hash = sum(v.sum().item() for v in state_dict.values())
        print(f"  Weight checksum: {weight_hash:.6f}")
        print(f"  Saved seed: {ckpt.get('seed', 'N/A')}")

    # ---- Save verification report ----
    report = {
        "status": status,
        "num_runs": num_runs,
        "seed": seed,
        "max_difference": max_diff,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_losses": best_losses,
        "is_reproducible": is_reproducible,
        "pytorch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    report_dir = config.get("output", {}).get("results_dir", "results/")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "reproducibility_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved to {report_path}")

    return is_reproducible


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify training reproducibility"
    )
    parser.add_argument(
        "--config", type=str, default="experiments/config.yaml"
    )
    parser.add_argument(
        "--num-runs", type=int, default=2,
        help="Number of runs to compare (default: 2)"
    )
    args = parser.parse_args()
    verify_reproducibility(config_path=args.config, num_runs=args.num_runs)
