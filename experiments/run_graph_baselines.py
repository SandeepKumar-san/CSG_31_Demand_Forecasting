"""
Graph-Aware Baseline: GraphSAGE.

Isolates the structural branch's contribution by training a pure graph 
message-passing model without a temporal TFT branch.

Usage:
    python experiments/run_graph_baselines.py --dataset supplygraph
    python experiments/run_graph_baselines.py --dataset usgs
"""

import argparse
import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_dataset_config, update_dynamic_parameters
from src.utils.seed import SeedManager
from src.data.supplygraph_loader import SupplyGraphLoader
from src.data.usgs_loader import USGSLoader

class GraphSAGEBaseline(torch.nn.Module):
    """Simple 2-layer GraphSAGE architecture."""
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index).relu()
        return self.lin(x)

def run_baseline(dataset_name, config):
    seed = config["reproducibility"]["seed"]
    seed_manager = SeedManager(seed=seed)
    seed_manager.set_seed()
    device = seed_manager.get_device()

    # Prepare data
    if dataset_name == "usgs":
        loader = USGSLoader(config)
    else:
        loader = SupplyGraphLoader(config)
    
    train_ds, val_ds, test_ds, info = loader.prepare_datasets()
    update_dynamic_parameters(config, info)
    
    graph_data = info["graph_data"].to(device)
    
    # We take the static node features as input and predict the target demand
    # GraphSAGE doesn't handle time series, so we use the final target value
    # from the training split as a proxy task.
    
    in_channels = graph_data.x.shape[1]
    hidden_dim = config["model"]["gat"]["hidden_dim"]
    model = GraphSAGEBaseline(in_channels, hidden_channels=hidden_dim, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)

    # Simplified training loop for pure graph message passing
    print(f"\n[GraphSAGE] Training on {dataset_name} structure...")
    
    # We use a subset of samples for a quick but fair comparison
    # GraphSAGE treats everything as a node-level regression task here
    targets_all = torch.tensor(info["graph_data"].target_means, dtype=torch.float).to(device)
    
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        out = model(graph_data.x, graph_data.edge_index).squeeze(-1)
        loss = F.mse_loss(out, targets_all)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:03d} | Loss: {loss.item():.6f}")

    # Evaluate
    model.eval()
    with torch.no_grad():
        preds_norm = model(graph_data.x, graph_data.edge_index).squeeze(-1)
        
    # Standard metrics
    targets_norm = targets_all
    mse = F.mse_loss(preds_norm, targets_norm).item()
    
    # Inverse transform for WAPE
    means = graph_data.target_means.cpu().numpy()
    stds = graph_data.target_stds.cpu().numpy()
    stds_safe = np.where(stds < 1e-8, 1.0, stds)
    
    preds_abs = preds_norm.cpu().numpy() * stds_safe + means
    trues_abs = targets_norm.cpu().numpy() * stds_safe + means
    
    wape = np.sum(np.abs(preds_abs - trues_abs)) / (np.sum(np.abs(trues_abs)) + 1e-8) * 100
    
    print(f"\n[GraphSAGE Results] WAPE: {wape:.2f}% | MSE: {mse:.6f}")
    
    # Save results
    res_path = os.path.join(config["output"]["results_dir"], f"graph_baseline_{dataset_name}.json")
    with open(res_path, "w") as f:
        json.dump({"GraphSAGE": {"WAPE": wape, "MSE": mse}}, f, indent=2)
    print(f"Results saved to {res_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["supplygraph", "usgs"], default="supplygraph")
    parser.add_argument("--config", type=str, default="experiments/config.yaml")
    args = parser.parse_args()
    
    cfg = load_dataset_config(args.dataset, args.config)
    
    # Use nested results directory
    ds_results = os.path.join(cfg["output"].get("results_dir", "results/"), args.dataset)
    cfg["output"]["results_dir"] = ds_results
    os.makedirs(ds_results, exist_ok=True)
    
    run_baseline(args.dataset, cfg)
