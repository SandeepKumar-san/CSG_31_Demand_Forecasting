"""Quick test to verify SupplyGraph data loading works."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.data.supplygraph_loader import SupplyGraphLoader

config = load_config("experiments/config.yaml")
loader = SupplyGraphLoader(config)
train_ds, val_ds, test_ds, info = loader.prepare_datasets()

print("\n=== VERIFICATION ===")
print(f"Graph nodes: {info['n_nodes']}")
print(f"Graph edges: {info['graph_data'].edge_index.shape[1]}")
print(f"Edge attr shape: {info['graph_data'].edge_attr.shape}")
print(f"Node features shape: {info['graph_data'].x.shape}")
print(f"Train samples: {len(train_ds)}")
print(f"Val samples: {len(val_ds)}")
print(f"Test samples: {len(test_ds)}")

# Test a single sample
sample = train_ds[0]
print(f"\nSample keys: {list(sample.keys())}")
for k, v in sample.items():
    if hasattr(v, 'shape'):
        print(f"  {k}: shape={v.shape} dtype={v.dtype}")
    else:
        print(f"  {k}: {v}")

print("\n✅ SupplyGraph data loading PASSED!")
