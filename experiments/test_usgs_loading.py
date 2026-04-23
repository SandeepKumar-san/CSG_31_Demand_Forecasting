"""Quick test to verify USGS data loading works end-to-end."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.data.usgs_loader import USGSLoader

# Step 1: Run preprocessing if needed
processed_path = "data/processed/usgs/NodesIndex.csv"
if not os.path.exists(processed_path):
    print("Running USGS preprocessing first...")
    from src.data.usgs_preprocessing import USGSPreprocessor
    preprocessor = USGSPreprocessor()
    preprocessor.run()

# Step 2: Load data
config = load_config("experiments/config.yaml")
config["data"]["dataset"] = "usgs"

loader = USGSLoader(config)
train_ds, val_ds, test_ds, info = loader.prepare_datasets()

print("\n=== USGS VERIFICATION ===")
print(f"Graph nodes: {info['n_nodes']}")
print(f"Graph edges: {info['graph_data'].edge_index.shape[1]}")
print(f"Edge attr shape: {info['graph_data'].edge_attr.shape}")
print(f"Node features shape: {info['graph_data'].x.shape}")
print(f"Edge types: {info['num_edge_types']}")
print(f"Timepoints: {info['n_timepoints']}")
print(f"Features: {info['feature_names']}")
print(f"Train samples: {len(train_ds)}")
print(f"Val samples: {len(val_ds)}")
print(f"Test samples: {len(test_ds)}")

# Check graph has edge_type and edge_weight
graph = info["graph_data"]
print(f"\nEdge type tensor: {graph.edge_type.shape}")
print(f"Edge weight tensor: {graph.edge_weight_values.shape}")
print(f"Num edge types: {graph.num_edge_types}")

# Test a single sample
if len(train_ds) > 0:
    sample = train_ds[0]
    print(f"\nSample keys: {list(sample.keys())}")
    for k, v in sample.items():
        if hasattr(v, 'shape'):
            print(f"  {k}: shape={v.shape} dtype={v.dtype}")
        else:
            print(f"  {k}: {v}")

print("\n[PASS] USGS data loading PASSED!")
