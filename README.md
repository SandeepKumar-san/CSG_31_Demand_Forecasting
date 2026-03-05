# Adaptive Temporal-Structural Fusion for Materials Demand Forecasting

## Overview

5-layer deep learning architecture combining **Temporal Fusion Transformer (TFT)** and **Graph Attention Network (GAT)** with an **adaptive fusion mechanism** for supply-chain materials demand forecasting and risk-aware decision support.

### Architecture

```
Layer 1: Raw Inputs (Demand, Prices, Lead Times, Material Properties)
Layer 2: Dual Branch Processing
  ├── 2a: TFT Branch (temporal patterns, quantile uncertainty)
  └── 2b: GAT Branch (structural dependencies, material graph)
Layer 3: Adaptive Fusion Layer ⭐ (Novel: material & horizon-specific weights)
Layer 4: Fused Demand Representation (context-aware forecast + confidence intervals)
Layer 5: Risk Scoring & Decision Layer (interpretable business logic)
```

### Key Innovation

The **Adaptive Fusion Layer** learns *when* to trust temporal vs structural signals, conditioned on:
- **Material type** (8 categories with distinct supply-chain dynamics)
- **Forecast horizon** (short-term vs long-term prediction)

When α → 1.0: model trusts temporal (TFT) signal more.  
When α → 0.0: model trusts structural (GAT) signal more.

## Project Structure

```
├── config.yaml / experiments/config.yaml   # All hyperparameters
├── requirements.txt                         # Exact pinned versions
├── src/
│   ├── data/
│   │   ├── supplygraph_loader.py           # SupplyGraph dataset
│   │   ├── data_generator.py               # (Removed — real data only)
│   │   ├── graph_builder.py                # Supply-chain graph
│   │   └── usgs_loader.py                  # USGS placeholder
│   ├── models/
│   │   ├── tft_branch.py                   # Layer 2a: TFT
│   │   ├── gat_branch.py                   # Layer 2b: GAT
│   │   ├── fusion_layer.py                 # Layer 3: Adaptive Fusion
│   │   ├── fused_representation.py         # Layer 4: Fused Representation
│   │   ├── risk_decision_layer.py          # Layer 5: Risk Scoring
│   │   └── complete_model.py               # End-to-end model
│   ├── training/
│   │   ├── trainer.py                      # Reproducible training loop
│   │   ├── loss.py                         # Quantile + fusion loss
│   │   └── metrics.py                      # MAE, RMSE, MAPE, SMAPE, R²
│   └── utils/
│       ├── seed.py                         # Deterministic seed manager
│       ├── config.py                       # Config loader
│       └── visualization.py               # 5 publication plots
├── experiments/
│   ├── train.py                            # Main training script
│   ├── evaluate.py                         # 4-variant comparison
│   ├── demo_risk_scoring.py                # Risk assessment demo
│   └── verify_reproducibility.py           # Reproducibility check
└── results/                                # Output directory
    ├── plots/
    ├── risk_reports/
    ├── checkpoints/
    └── logs/
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Training

```bash
# Train the complete model (requires SupplyGraph dataset)
python experiments/train.py --config experiments/config.yaml
```

### Evaluation

```bash
# Compare TFT-only, GAT-only, Fixed Fusion, and Adaptive Fusion
python experiments/evaluate.py --config experiments/config.yaml
```

### Risk Assessment Demo

```bash
# Generate risk reports with action flags
python experiments/demo_risk_scoring.py --config experiments/config.yaml
```

### Verify Reproducibility

```bash
# Run multiple times and verify identical results
python experiments/verify_reproducibility.py --config experiments/config.yaml --num-runs 2
```

## Datasets

> **⚠️ REAL DATA ONLY** — This is a research paper. No synthetic data is used.

### Primary: SupplyGraph (REQUIRED)
- **41 products**, 221 timepoints (2023-01-01 to 2023-08-09)
- Features: Sales Order, Production, Factory Issue, Delivery (Unit + Weight)
- Edge types: Product Group, Sub-Group, Plant, Storage Location
- **Download from Kaggle:** [SupplyGraph Dataset](https://www.kaggle.com/datasets/azminetoushikwasi/supplygraph-supply-chain-planning-using-gnns)
- Place data in `data/raw/supplygraph/` (with Edges/, Nodes/, Temporal Data/ subfolders)
- If data not found → prints download instructions and **exits**

### Future: USGS Minerals (Placeholder)
- 90 mineral commodities, 20+ years
- Placeholder loader implemented

## Reproducibility

All experiments use **seed=42** for complete reproducibility.

| Requirement | Implementation |
|---|---|
| Global seed | `SeedManager(seed=42).set_seed()` |
| PyTorch determinism | `torch.use_deterministic_algorithms(True)` |
| CUDA determinism | `CUBLAS_WORKSPACE_CONFIG=:4096:8` |
| DataLoader | `num_workers=0`, seeded generator |
| Weight init | `xavier_uniform_` with fixed seed |
| Verification | `verify_reproducibility.py` |

Running the same experiment multiple times with seed=42 produces:
- IDENTICAL training loss curves
- IDENTICAL validation metrics
- IDENTICAL alpha weight distributions

## Configuration

All hyperparameters are in `experiments/config.yaml`:

- `reproducibility.seed`: 42
- `model.tft.hidden_dim`: 64
- `model.gat.heads`: 4
- `training.epochs`: 50
- `training.learning_rate`: 0.001
- See `config.yaml` for full list

## Citation

```bibtex
@article{adaptivefusion2026,
  title={Adaptive Temporal-Structural Fusion for Supply Chain
         Demand Forecasting with Risk-Aware Decision Support},
  year={2026}
}
```
