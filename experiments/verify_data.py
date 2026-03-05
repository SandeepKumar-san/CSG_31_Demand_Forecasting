"""Verify dataset files are present and readable (no PyTorch needed)."""
import pandas as pd
import numpy as np
import glob
import os

ROOT = "data/raw/supplygraph"

print("=" * 60)
print("  SUPPLYGRAPH DATASET VERIFICATION")
print("=" * 60)

# Nodes
ni = pd.read_csv(os.path.join(ROOT, "Nodes", "NodesIndex.csv"))
print(f"Nodes: {len(ni)}")

nt = pd.read_csv(os.path.join(ROOT, "Nodes", "Node Types (Product Group and Subgroup).csv"))
for col in nt.select_dtypes(include="object").columns:
    nt[col] = nt[col].str.strip()
print(f"Node types shape: {nt.shape}")
print(f"  Groups: {nt['Group'].unique().tolist()}")
print(f"  Sub-Groups: {nt['Sub-Group'].unique().tolist()}")

# Edges
print("\nEdge files:")
edge_dir = os.path.join(ROOT, "Edges", "EdgesIndex")
for f in sorted(glob.glob(os.path.join(edge_dir, "*.csv"))):
    df = pd.read_csv(f)
    name = os.path.basename(f)
    print(f"  {name}: {len(df)} edges, cols={list(df.columns)}")

# Temporal
print("\nTemporal (Weight):")
weight_dir = os.path.join(ROOT, "Temporal Data", "Weight")
for f in sorted(glob.glob(os.path.join(weight_dir, "*.csv"))):
    df = pd.read_csv(f)
    name = os.path.basename(f)
    print(f"  {name}: shape={df.shape}")

print("\nTemporal (Unit):")
unit_dir = os.path.join(ROOT, "Temporal Data", "Unit")
for f in sorted(glob.glob(os.path.join(unit_dir, "*.csv"))):
    df = pd.read_csv(f)
    name = os.path.basename(f)
    print(f"  {name}: shape={df.shape}")

# Show a sample temporal CSV
first_weight = sorted(glob.glob(os.path.join(weight_dir, "*.csv")))[0]
df = pd.read_csv(first_weight)
print(f"\nSample temporal CSV: {os.path.basename(first_weight)}")
print(f"  Date range: {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")
print(f"  Node columns: {len(df.columns) - 1}")
print(f"  Sample values (first 3 nodes, first 3 rows):")
print(df.iloc[:3, :4].to_string())

print("\n" + "=" * 60)
print("  ALL FILES PRESENT AND READABLE")
print("=" * 60)
