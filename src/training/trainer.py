"""
Reproducible training loop for the Adaptive Fusion model.

Features:
  - Deterministic seed management per epoch
  - Mixed loss: MSE (forecast) + QuantileLoss + alpha regularization
  - Early stopping with patience
  - Best model checkpointing
  - Epoch-level metric logging
  - Alpha statistics tracking
  - LR stats tracking

Designed for research reproducibility: identical loss curves across runs.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.loss import AdaptiveFusionLoss
from src.training.metrics import compute_all_metrics
from src.utils.seed import SeedManager


class ReproducibleTrainer:
    """
    Reproducible training loop with deterministic operations.

    Args:
        model: AdaptiveFusionForecaster instance.
        config: Configuration dictionary.
        seed_manager: SeedManager instance.
        device: torch.device.
    """

    def __init__(
        self,
        model: nn.Module,
        config: dict,
        seed_manager: SeedManager,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.seed_manager = seed_manager
        self.device = device

        train_cfg = config["training"]
        self.epochs = train_cfg["epochs"]
        self.lr = train_cfg["learning_rate"]
        self.weight_decay = train_cfg["weight_decay"]
        self.grad_clip = train_cfg["gradient_clip"]
        self.patience = train_cfg["early_stopping_patience"]
        alpha_reg = train_cfg["alpha_reg_weight"]
        q_weight = train_cfg.get("quantile_loss_weight", 0.3)
        quantiles = config["data"]["quantiles"]

        # ---- Loss function ----
        self.criterion = AdaptiveFusionLoss(
            quantiles=quantiles,
            alpha_reg_weight=alpha_reg,
            quantile_loss_weight=q_weight,
        )

        # ---- Optimizer (re-seed before creation for reproducibility) ----
        seed_manager.set_seed()
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # ---- Scheduler ----
        sched_cfg = train_cfg["scheduler"]
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=sched_cfg["factor"],
            patience=sched_cfg["patience"],
            min_lr=sched_cfg["min_lr"],
        )

        # ---- Training Audit ----
        print(f"\n[Training Audit] Strict configuration active:")
        print(f"  - Learning Rate:      {self.lr}")
        print(f"  - Alpha Reg Weight:   {alpha_reg}")
        print(f"  - Quantile Weight:    {q_weight}")
        print(f"  - Early Stopping:     {self.patience} epochs")
        print(f"  - Scheduler:          {sched_cfg['type']} (factor={sched_cfg['factor']}, patience={sched_cfg['patience']})")

        # ---- Output dirs ----
        out_cfg = config["output"]
        self.checkpoint_dir = out_cfg["checkpoints_dir"]
        self.logs_dir = out_cfg["logs_dir"]
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        # ---- Tracking ----
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_forecast_loss": [],
            "val_forecast_loss": [],
            "alpha_mean": [],
            "alpha_std": [],
            "lr": [],
            "alpha_by_category": [],
        }
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.epochs_without_improvement = 0

    def train_epoch(
        self,
        train_loader: DataLoader,
        graph_data,
        epoch: int,
    ) -> Dict[str, Any]:
        """
        Train for one epoch with deterministic seeding.

        Returns:
            Dictionary of epoch metrics.
        """
        # Re-seed at start of each epoch for consistency
        torch.manual_seed(self.seed_manager.seed + epoch)

        self.model.train()
        total_loss = 0.0
        total_forecast_loss = 0.0
        all_alphas = []
        all_mat_types = []
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            # Prepare batch with graph data
            batch = self._prepare_batch(batch, graph_data)

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(batch)

            # Compute loss
            loss_dict = self.criterion(
                {
                    "forecast": output["forecast"],
                    "quantile_predictions": output["quantile_predictions"],
                    "alpha": output["alpha"],
                },
                batch["targets"].to(self.device),
            )

            loss = loss_dict["total_loss"]

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )

            self.optimizer.step()

            total_loss += loss.item()
            total_forecast_loss += loss_dict["forecast_loss"].item()
            all_alphas.append(output["alpha"].detach().cpu())
            all_mat_types.append(batch["material_type"].detach().cpu())
            num_batches += 1

        # Compute epoch statistics
        avg_loss = total_loss / max(num_batches, 1)
        avg_forecast_loss = total_forecast_loss / max(num_batches, 1)

        alphas_cat = torch.cat(all_alphas, dim=0)
        types_cat = torch.cat(all_mat_types, dim=0)
        alpha_mean = alphas_cat.mean().item()
        alpha_std = alphas_cat.std().item()
        
        alpha_by_cat = {}
        if (epoch + 1) % 10 == 0:
            for c in torch.unique(types_cat):
                mask = types_cat == c
                alpha_by_cat[str(c.item())] = alphas_cat[mask].mean().item()

        return {
            "train_loss": avg_loss,
            "train_forecast_loss": avg_forecast_loss,
            "alpha_mean": alpha_mean,
            "alpha_std": alpha_std,
            "alpha_by_cat": alpha_by_cat,
        }

    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
        graph_data,
    ) -> Dict[str, Any]:
        """Validate model on validation set."""
        self.model.eval()
        total_loss = 0.0
        total_forecast_loss = 0.0
        all_preds = []
        all_targets = []
        num_batches = 0

        for batch in val_loader:
            batch = self._prepare_batch(batch, graph_data)
            output = self.model(batch)

            loss_dict = self.criterion(
                {
                    "forecast": output["forecast"],
                    "quantile_predictions": output["quantile_predictions"],
                    "alpha": output["alpha"],
                },
                batch["targets"].to(self.device),
            )

            total_loss += loss_dict["total_loss"].item()
            total_forecast_loss += loss_dict["forecast_loss"].item()
            all_preds.append(output["forecast"].cpu())
            all_targets.append(batch["targets"].cpu())
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        avg_forecast_loss = total_forecast_loss / max(num_batches, 1)

        # Compute metrics
        if all_preds:
            preds = torch.cat(all_preds, dim=0)
            targets = torch.cat(all_targets, dim=0)
            metrics = compute_all_metrics(targets, preds)
        else:
            metrics = {"MAE": 0.0, "RMSE": 0.0, "WAPE": 0.0, "SMAPE": 0.0, "R2": 0.0}

        metrics["val_loss"] = avg_loss
        metrics["val_forecast_loss"] = avg_forecast_loss
        return metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        graph_data,
    ) -> Dict[str, Any]:
        """Full training loop with early stopping and checkpointing."""
        print("=" * 60)
        print("TRAINING: Adaptive Temporal-Structural Fusion Model")
        print("=" * 60)
        print(f"  Epochs: {self.epochs}")
        print(f"  LR: {self.lr}")
        print(f"  Device: {self.device}")
        print(f"  Seed: {self.seed_manager.seed}")
        print("=" * 60)

        start_time = time.time()

        for epoch in range(self.epochs):
            epoch_start = time.time()

            # Train
            train_metrics = self.train_epoch(train_loader, graph_data, epoch)

            # Validate
            val_metrics = self.validate(val_loader, graph_data)

            # LR scheduling
            self.scheduler.step(val_metrics["val_loss"])
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Record history
            self.history["train_loss"].append(train_metrics["train_loss"])
            self.history["val_loss"].append(val_metrics["val_loss"])
            self.history["train_forecast_loss"].append(train_metrics["train_forecast_loss"])
            self.history["val_forecast_loss"].append(val_metrics["val_forecast_loss"])
            self.history["alpha_mean"].append(train_metrics["alpha_mean"])
            self.history["alpha_std"].append(train_metrics["alpha_std"])
            self.history["lr"].append(current_lr)

            # Print progress
            epoch_time = time.time() - epoch_start
            print(
                f"  Epoch {epoch + 1:3d}/{self.epochs} | "
                f"Train: {train_metrics['train_loss']:.4f} | "
                f"Val: {val_metrics['val_loss']:.4f} | "
                f"RMSE: {val_metrics['RMSE']:.4f} | "
                f"Alpha={train_metrics['alpha_mean']:.3f}+/-{train_metrics['alpha_std']:.3f} | "
                f"LR: {current_lr:.2e} | "
                f"{epoch_time:.1f}s"
            )

            # Alpha stats every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.history["alpha_by_category"].append(train_metrics["alpha_by_cat"])
                print(
                    f"    [Alpha Stats] Mean={train_metrics['alpha_mean']:.4f}, "
                    f"Std={train_metrics['alpha_std']:.4f}"
                )

            # Early stopping check
            if val_metrics["val_loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["val_loss"]
                self.best_epoch = epoch + 1
                self.epochs_without_improvement = 0
                self._save_checkpoint(epoch, val_metrics)
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.patience:
                    print(f"\n  [STOP] Early stopping at epoch {epoch + 1}")
                    break

        total_time = time.time() - start_time
        print("=" * 60)
        print(f"  Training complete in {total_time:.1f}s")
        print(f"  Best val loss: {self.best_val_loss:.4f}")
        print("=" * 60)

        # Save training history
        self._save_history()

        return {
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "train_loss": self.history["train_loss"][-1],
            "val_loss": self.history["val_loss"][-1],
            "total_time": total_time,
            "epochs_trained": len(self.history["train_loss"]),
            "history": self.history,
        }

    def _prepare_batch(self, batch: Dict, graph_data) -> Dict:
        """Add graph data to batch and move ALL tensors to device."""
        batch["graph_x"] = graph_data.x.to(self.device)
        batch["graph_edge_index"] = graph_data.edge_index.to(self.device)
        if graph_data.edge_attr is not None:
            batch["graph_edge_attr"] = graph_data.edge_attr.to(self.device)
            
        if hasattr(graph_data, "edge_type") and graph_data.edge_type is not None:
            batch["graph_edge_type"] = graph_data.edge_type.to(self.device)
        if hasattr(graph_data, "edge_weight_values") and graph_data.edge_weight_values is not None:
            batch["graph_edge_weight"] = graph_data.edge_weight_values.to(self.device)
            
        batch["material_type"] = batch["material_type"].to(self.device)
        batch["product_id"] = batch["product_id"].to(self.device)
        
        if "horizon" in batch:
            batch["horizon"] = batch["horizon"].to(self.device)
        else:
            batch["horizon"] = torch.zeros(batch["material_type"].shape[0], dtype=torch.long, device=self.device)
            
        for key in ["time_series", "price_history", "static_features", "targets"]:
            if key in batch and isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device)
        return batch

    def _save_checkpoint(self, epoch: int, metrics: Dict) -> None:
        """Save model checkpoint."""
        path = os.path.join(self.checkpoint_dir, f"best_model_seed{self.seed_manager.seed}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_loss": self.best_val_loss,
                "metrics": metrics,
                "config": self.config,
                "seed": self.seed_manager.seed,
            },
            path,
        )

    def _save_history(self) -> None:
        """Save training history as JSON."""
        path = os.path.join(self.logs_dir, f"training_history_seed{self.seed_manager.seed}.json")
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"  Training history saved to {path}")

    def load_checkpoint(self, path: Optional[str] = None) -> Dict:
        """Load model from checkpoint."""
        if path is None:
            path = os.path.join(self.checkpoint_dir, f"best_model_seed{self.seed_manager.seed}.pt")
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint
