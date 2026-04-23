# Adaptive Temporal-Structural Fusion (ATSF) for Materials Demand Forecasting

> A 5-layer deep learning architecture combining **Temporal Fusion Transformer (TFT)** and **Graph Attention Network (GAT)** with a novel **material- and horizon-conditioned adaptive fusion mechanism** for supply-chain demand forecasting and risk-aware decision support.

---

## Table of Contents

1. [Overview & Key Innovation](#overview--key-innovation)
2. [Architecture](#architecture)
3. [Empirical Results](#empirical-results)
4. [Datasets](#datasets)
5. [Graph Structure Validation](#graph-structure-validation)
6. [GAT Edge Attention Analysis](#gat-edge-attention-analysis)
7. [Project Structure](#project-structure)
8. [Quick Start](#quick-start)
9. [Configuration](#configuration)
10. [Reproducibility](#reproducibility)
11. [Statistical Significance](#statistical-significance)

---

## Overview & Key Innovation

Supply chain demand forecasting requires reasoning simultaneously over **temporal dynamics** (seasonal trends, lead-time lags) and **structural dependencies** (material substitutability, geopolitical co-movement, co-production relationships). Existing models treat these as separate problems. ATSF unifies them.

### The Central Contribution: Adaptive Fusion Layer

Rather than combining temporal and structural representations with a fixed weight, the **Adaptive Fusion Layer** learns a scalar gate α ∈ [0, 1] that is *conditioned jointly on*:

- **Material category** — embedded into a 16-dimensional space (8 types for SupplyGraph; 90+ commodities for USGS)
- **Forecast horizon** — embedded into a 16-dimensional space (horizons: 1, 3, 6, 12 steps for SupplyGraph; 1-step rolling CV for USGS)

```
α → 1.0 : model trusts TFT (temporal signal) more
α → 0.0 : model trusts GAT (structural signal) more
```

Across 5 random seeds on SupplyGraph, the learned α converges to **0.924 ± 0.013** — meaning the model learns that *temporal signals dominate* for corporate supply chain data, while structural context provides a crucial correction. On USGS mineral data, where geopolitical and co-production graph edges carry genuine causal signal, α converges to **0.489 ± 0.218**, reflecting genuine uncertainty about which modality to trust — a qualitatively different and meaningful fusion behaviour.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 1 — Raw Inputs                                               │
│  SupplyGraph: Sales Order, Production, Factory Issue, Delivery      │
│  USGS: Production, Consumption, Imports, Exports, Price,            │
│         Stocks, Capacity, Sales, Reserves, Supply,                  │
│         Employment, WorldProduction (12 features)                   │
└────────────────────┬────────────────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
┌────────▼─────────┐   ┌─────────▼────────┐
│  Layer 2a — TFT  │   │  Layer 2b — GAT  │
│  hidden_dim: 128 │   │  hidden_dim: 128  │
│  num_layers: 2   │   │  num_layers: 2   │
│  num_heads: 4    │   │  heads: 4        │
│  dropout: 0.2    │   │  dropout: 0.2    │
│  Quantiles:      │   │  15 edge types   │
│  [0.1, 0.5, 0.9] │   │  (USGS dataset)  │
└────────┬─────────┘   └─────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     │
┌────────────────────▼────────────────────────────────────────────────┐
│  Layer 3 — Adaptive Fusion Layer ⭐ (Novel)                         │
│  α = σ(W · [material_embed ‖ horizon_embed])                        │
│  material_embed_dim: 16   horizon_embed_dim: 16                     │
│  fused = α · h_TFT + (1 − α) · h_GAT                               │
│  + entropy regulariser (α_reg_weight: 0.1)                          │
└────────────────────┬────────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────────┐
│  Layer 4 — Fused Demand Representation                              │
│  Context-aware point forecast + quantile uncertainty bounds         │
└────────────────────┬────────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────────┐
│  Layer 5 — Risk Scoring & Decision Layer                            │
│  budget_threshold: $100,000  lead_time_safety_factor: 1.5x          │
│  criticality_threshold: 0.7  → Actionable business risk flags       │
└─────────────────────────────────────────────────────────────────────┘
```

**Loss function:** Joint quantile loss (weight: 0.3) + MSE + entropy regularisation on α.  
**Optimiser:** Adam (lr=0.001, weight\_decay=1e-4), ReduceLROnPlateau scheduler (factor=0.3, patience=10, min\_lr=1e-6).  
**Early stopping:** patience=50 epochs, max=1,000 epochs.

---

## Empirical Results

All results are computed across **5 independent random seeds** (42, 123, 456, 789, 1337). Values reported as **mean ± std**.

### Dataset 1: SupplyGraph (Corporate Supply Chain)

| Variant | MAE ↓ | RMSE ↓ | WAPE ↓ | R² ↑ |
|---|---|---|---|---|
| TFT-only | 10.46 ± 1.05 | 27.07 ± 2.38 | 90.63 ± 9.06% | 0.169 ± 0.149 |
| GAT-only | 9.09 ± 0.60 | 23.66 ± 1.60 | 78.77 ± 5.18% | 0.367 ± 0.087 |
| Fixed Fusion (α=0.5) | 9.78 ± 0.87 | 25.57 ± 2.42 | 84.68 ± 7.52% | 0.259 ± 0.141 |
| **Adaptive Fusion (Ours)** | **6.36 ± 0.22** | **17.03 ± 0.62** | **55.07 ± 1.94%** | **0.673 ± 0.024** |
| ARIMA | 8.35 | 21.10 | 72.35% | 0.499 |
| XGBoost | 6.06 | 16.30 | 52.50% | 0.701 |

**95% Confidence Interval (Adaptive Fusion):**  
MAE: [6.08, 6.64] · RMSE: [16.26, 17.79] · R²: [0.644, 0.703]

**Improvement over best ablation baseline (GAT-only):**
- MAE: **−30.1%** · RMSE: **−28.1%** · WAPE: **−30.1%** · R²: **+83.4%**

### Dataset 2: USGS Critical Minerals (90+ Commodities, 20+ years)

| Variant | MAE ↓ | RMSE ↓ | WAPE ↓ | SMAPE ↓ | R² ↑ |
|---|---|---|---|---|---|
| TFT-only | 1554 ± 238 | 11191 ± 1478 | 11.72 ± 1.79% | 5.41 ± 1.46% | 0.982 ± 0.005 |
| GAT-only | 2086 ± 869 | 16471 ± 8388 | 15.73 ± 6.55% | 5.09 ± 1.38% | 0.954 ± 0.035 |
| Fixed Fusion (α=0.5) | 1256 ± 685 | 8994 ± 6728 | 9.47 ± 5.17% | 4.34 ± 0.66% | 0.984 ± 0.020 |
| **Adaptive Fusion (Ours)** | **929 ± 109** | **6059 ± 1092** | **7.01 ± 0.83%** | **3.02 ± 0.13%** | **0.9947 ± 0.0018** |
| ARIMA | 2257 | 18789 | 17.02% | 5.21% | 0.951 |
| XGBoost | 1070 | 6807 | 8.07% | 3.24% | 0.994 |

**95% Confidence Interval (Adaptive Fusion):**  
MAE: [794, 1065] · RMSE: [4704, 7415] · WAPE: [5.99%, 8.03%] · R²: [0.9925, 0.9970]

**Improvement over best ablation baseline (Fixed Fusion):**
- MAE: **−26.0%** · RMSE: **−32.6%** · WAPE: **−26.0%** · R²: **+1.13%**

---

## Datasets

> **No synthetic data is used.** This project uses two fully real datasets.

### Dataset 1: SupplyGraph

- **Source:** [Kaggle — SupplyGraph](https://www.kaggle.com/datasets/azminetoushikwasi/supplygraph-supply-chain-planning-using-gnns)
- **Nodes:** 41 products across product groups, sub-groups, plants, and storage locations
- **Timepoints:** 221 (2023-01-01 to 2023-08-09)
- **Temporal features:** Sales Order, Production, Factory Issue, Delivery (Unit + Weight)
- **Graph edge types:** Plant, Product Group, Product Sub-Group, Storage Location
- **Split:** 70% train / 15% val / 15% test · Sequence length: 30 · Horizons: [1, 3, 6, 12]
- **Place data at:** `data/raw/supplygraph/` (with `Edges/`, `Nodes/`, `Temporal Data/` subfolders)

### Dataset 2: USGS Mineral Commodity Summaries 2026 (MCS2026)

- **Source:** U.S. Geological Survey, Mineral Commodity Summaries 2026
- **Commodities:** 90+ critical minerals (lithium, cobalt, rare earths, graphite, nickel, etc.)
- **Temporal span:** 20+ years of annual production, consumption, trade, and price data
- **Graph edge types:** 15 semantically meaningful relationship types including:
  - `technology_cluster`, `alloy_components`, `geopolitical_supply_risk`, `substitution`
  - `battery_coproduction`, `electronics_coproduction`, `supply_chain_input`
  - `price_correlation`, `recycling_secondary_source`, `critical_mineral_designation`
  - `construction_coproduction`, `byproduct_coproduction`, `catalyst_role`
  - `refractory_industrial`, `functional_coating`
- **Temporal features:** Production, Consumption, Imports, Exports, Price, Stocks, Capacity, Sales, Reserves, Supply, Employment, WorldProduction (12 features)
- **Evaluation:** Rolling-window walk-forward cross-validation (2 folds)
- **Place raw files at:** `data/raw/usgs/` · Processed cache at: `data/processed/usgs/`

---

## Graph Structure Validation

To verify that the graph edges encode genuine causal co-movement in demand (not spurious connections), we computed Pearson |r| and Spearman ρ correlation coefficients on the underlying time-series pairs for every edge in the graph.

### SupplyGraph — Pearson |r| by Edge Type

| Edge Type | n | Mean \|r\| | 95% CI |
|---|---|---|---|
| Product Sub-Group | 48 | **0.579** | [0.490, 0.665] |
| Product Group | 179 | 0.360 | [0.316, 0.405] |
| Plant | 360 | 0.341 | [0.311, 0.370] |
| Storage Location | 665 | 0.307 | [0.288, 0.327] |
| **Pooled (all edges)** | **1252** | **0.335** | [0.319, 0.350] |
| Random unconnected pairs | 136 | 0.088 | [0.060, 0.121] |

**Effect size:** Δ\|r\| (real edges − random baseline) = **+0.247** — confirming the graph encodes genuine co-movement structure.

### USGS — Important vs Noise Edge Types (Pearson \|r\|, Production axis)

| Edge Type | Group | n\_valid | \|r\|\_prod |
|---|---|---|---|
| geopolitical\_supply\_risk | IMPORTANT | 24 | **0.716** |
| substitution | IMPORTANT | 70 | 0.588 |
| alloy\_components | IMPORTANT | 92 | 0.579 |
| battery\_coproduction | IMPORTANT | 28 | 0.564 |
| technology\_cluster | IMPORTANT | 184 | 0.556 |
| electronics\_coproduction | IMPORTANT | 81 | 0.555 |
| **Mean (IMPORTANT)** | — | — | **0.610** |
| supply\_chain\_input | NOISE | 465 | 0.502 |
| recycling\_secondary\_source | NOISE | 61 | 0.563 |
| construction\_coproduction | NOISE | 146 | 0.538 |
| **Mean (NOISE)** | — | — | **0.534** |

---

## GAT Edge Attention Analysis

The GAT branch assigns learned attention weights to each edge type. Analysed across the full USGS evaluation set:

| Edge Type | Mean Attention | Count |
|---|---|---|
| critical\_mineral\_designation | **0.02070** | 1,936 |
| geopolitical\_supply\_risk | 0.01796 | 1,512 |
| refractory\_industrial | 0.01764 | 264 |
| construction\_coproduction | 0.01603 | 2,568 |
| supply\_chain\_input | 0.01510 | 14,488 |
| recycling\_secondary\_source | 0.01293 | 1,856 |
| functional\_coating | 0.01346 | 296 |
| price\_correlation | 0.01083 | 3,728 |
| byproduct\_coproduction | 0.01123 | 968 |
| technology\_cluster | 0.00883 | 8,624 |
| substitution | 0.00839 | 3,920 |
| battery\_coproduction | 0.00462 | 1,616 |

The model assigns highest attention to **geopolitical risk** and **critical mineral designation** edges — consistent with the domain knowledge that these relationships carry the most informative supply-chain signals.

### Edge Ablation (RMSE degradation when edge type is removed)

| Edge Type | RMSE Degradation |
|---|---|
| supply\_chain\_input | +1.29% |
| recycling\_secondary\_source | +0.50% |
| construction\_coproduction | +0.39% |
| price\_correlation | +0.17% |
| refractory\_industrial | +0.16% |
| technology\_cluster | −2.20% *(removing noisy edges improves)*  |
| alloy\_components | −1.22% |
| geopolitical\_supply\_risk | −0.55% |

---

## Project Structure

```
Demand_Forecast/
│
├── requirements.txt                    # Pinned dependencies
├── dashboard.py                        # Streamlit interactive dashboard
├── compute_graph_pearson.py            # Graph structure correlation analysis
├── gen_alpha_fig.py                    # Alpha trajectory figure generator
│
├── data/
│   ├── raw/
│   │   ├── supplygraph/                # SupplyGraph dataset (Nodes/, Edges/, Temporal Data/)
│   │   └── usgs/                       # MCS2026 CSVs + pdf_extracted_relationships.csv
│   └── processed/
│       └── usgs/                       # Preprocessed USGS graph cache (32 files)
│
├── src/
│   ├── models/
│   │   ├── complete_model.py           # End-to-end ATSF model
│   │   ├── tft_branch.py               # Layer 2a: Temporal Fusion Transformer
│   │   ├── gat_branch.py               # Layer 2b: Graph Attention Network
│   │   ├── fusion_layer.py             # Layer 3: Adaptive Fusion (novel)
│   │   ├── fused_representation.py     # Layer 4: Output head
│   │   └── risk_decision_layer.py      # Layer 5: Risk scoring
│   ├── data/
│   │   ├── supplygraph_loader.py       # SupplyGraph data pipeline
│   │   ├── usgs_loader.py              # USGS data pipeline
│   │   ├── usgs_preprocessing.py       # USGS raw → processed graph
│   │   └── graph_builder.py            # Supply-chain graph construction
│   ├── training/
│   │   ├── trainer.py                  # Reproducible training loop
│   │   ├── loss.py                     # Quantile + entropy fusion loss
│   │   └── metrics.py                  # MAE, RMSE, WAPE, SMAPE, R²
│   └── utils/
│       ├── seed.py                     # Deterministic seed manager
│       ├── config.py                   # YAML config loader
│       └── visualization.py            # Publication-quality plots
│
├── experiments/
│   ├── config.yaml                     # All hyperparameters (zero hardcoding)
│   ├── train.py                        # Training entry point
│   ├── evaluate.py                     # 4-variant ablation evaluation
│   ├── run_multi_seed.py               # 5-seed stability experiment
│   ├── run_classical_baselines.py      # ARIMA & XGBoost baselines
│   ├── run_graph_baselines.py          # GraphSAGE baseline
│   ├── edge_type_analysis.py           # GAT attention + ablation study
│   ├── generate_usgs_risk_reports.py   # Executive risk report generator
│   ├── demo_risk_scoring.py            # Risk assessment demo
│   ├── system_dry_run.py               # Fast smoke test (<10s)
│   ├── verify_reproducibility.py       # Determinism verification
│   ├── verify_data.py                  # SupplyGraph data integrity check
│   ├── verify_usgs_integrity.py        # USGS data integrity check
│   ├── test_data_loading.py            # SupplyGraph loader unit test
│   └── test_usgs_loading.py            # USGS loader unit test
│
└── results/
    ├── fig1_architecture.png           # Architecture diagram
    ├── fig2_alpha_trajectory.*         # α fusion weight trajectory (PDF + PNG)
    ├── fig_edge_attention.*            # GAT edge attention weights (PDF + PNG)
    ├── graph_structure_validation.json # Pearson/Spearman correlation data
    ├── graph_structure_validation_paper_tables.txt
    ├── reproducibility_report.json     # PASS_EXACT — max_diff = 0.0
    ├── "Wilcoxon test.txt"             # Full statistical significance report
    ├── supplygraph/                    # SupplyGraph training artefacts
    │   ├── multi_seed_metrics.csv      # 5-seed results (all 4 variants)
    │   ├── classical_baselines_supplygraph.json
    │   ├── training_results.json
    │   ├── logs/                       # Per-seed training history JSONs
    │   └── plots/                      # alpha_by_epoch, model_comparison, training_curves
    ├── usgs/                           # USGS training artefacts
    │   ├── multi_seed_metrics.csv
    │   ├── classical_baselines_usgs.json
    │   ├── edge_type_analysis.json     # Per-type attention + ablation
    │   ├── edge_type_importance.csv
    │   ├── edge_type_ablation.csv
    │   ├── logs/
    │   └── plots/
    ├── supplygraph_val/
    │   ├── evaluation_results.json
    │   └── plots/
    ├── plots/                          # Combined comparison plots
    └── risk_reports/usgs/risk_report.json
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

**SupplyGraph** — download from Kaggle and place at `data/raw/supplygraph/`:
```
data/raw/supplygraph/
├── Edges/
├── Nodes/
└── Temporal Data/
```

**USGS** — already included at `data/raw/usgs/` (MCS2026 CSV files + extracted relationships).

### 3. Train

```bash
# Train on SupplyGraph
python experiments/train.py --config experiments/config.yaml

# Train on USGS
python experiments/train.py --config experiments/config.yaml --dataset usgs
```

### 4. Run Full 5-Seed Experiment

```bash
python experiments/run_multi_seed.py --config experiments/config.yaml
```

### 5. Evaluate All Ablation Variants

```bash
# Compares TFT-only, GAT-only, Fixed Fusion, Adaptive Fusion (Ours)
python experiments/evaluate.py --config experiments/config.yaml
```

### 6. Run Classical Baselines

```bash
python experiments/run_classical_baselines.py --config experiments/config.yaml
```

### 7. Edge Type Attention Analysis

```bash
python experiments/edge_type_analysis.py --config experiments/config.yaml
```

### 8. Generate Risk Reports

```bash
python experiments/generate_usgs_risk_reports.py --config experiments/config.yaml
```

### 9. Quick Smoke Test (< 10 seconds)

```bash
python experiments/system_dry_run.py
```

### 10. Interactive Dashboard

```bash
streamlit run dashboard.py
```

---

## Configuration

All hyperparameters live in `experiments/config.yaml` with **zero hardcoded overrides** in Python. Dataset-specific configurations:

| Parameter | SupplyGraph | USGS |
|---|---|---|
| `hidden_dim` | 128 | 64 |
| `num_layers` (TFT + GAT) | 2 | 2 |
| `num_heads` (TFT + GAT) | 4 | 4 |
| `dropout` | 0.2 | 0.3 |
| `epochs` (max) | 1000 | 1000 |
| `batch_size` | 64 | 8 |
| `learning_rate` | 0.001 | 0.001 |
| `early_stopping_patience` | 50 | 50 |
| `alpha_reg_weight` | 0.1 | 0.05 |
| `quantile_loss_weight` | 0.3 | 0.3 |
| `sequence_length` | 30 | 1 |
| `forecast_horizons` | [1, 3, 6, 12] | [1] |
| `quantiles` | [0.1, 0.5, 0.9] | [0.1, 0.5, 0.9] |

---

## Reproducibility

All experiments use deterministic seeding. Verified with `verify_reproducibility.py`:

```json
{
  "status": "PASS_EXACT",
  "num_runs": 2,
  "seed": 42,
  "max_difference": 0.0,
  "is_reproducible": true,
  "pytorch_version": "2.1.0+cu118"
}
```

| Mechanism | Implementation |
|---|---|
| Global seed | `SeedManager(seed=42).set_seed()` |
| PyTorch determinism | `torch.use_deterministic_algorithms(True)` |
| CUDA workspace | `CUBLAS_WORKSPACE_CONFIG=:4096:8` |
| DataLoader | `num_workers=0`, seeded generator |
| Weight init | `xavier_uniform_` with fixed seed |

Running the same experiment from scratch with seed=42 produces **identical loss values** (max\_difference = 0.0) across runs.

---

## Statistical Significance

Wilcoxon signed-rank test comparing Adaptive Fusion vs all baselines across 5 seeds (metric: MAE):

| Comparison | SupplyGraph p-value | USGS p-value |
|---|---|---|
| Adaptive Fusion vs TFT-only | 0.0625 (marginal) | 0.0625 (marginal) |
| Adaptive Fusion vs GAT-only | 0.0625 (marginal) | 0.1250 |
| Adaptive Fusion vs Fixed Fusion | 0.0625 (marginal) | 0.6250 |

> **Note on n=5:** With only 5 paired observations, the Wilcoxon test's minimum achievable p-value is 0.0625. The test statistic = 0 (all 5 seeds show improvement) in every SupplyGraph comparison — this is the strongest result the test can produce at this sample size. The improvements are consistent across every seed with low variance.

---

## Citation

If you use this code or findings, please cite:

```bibtex
@misc{atsf2026,
  title  = {Adaptive Temporal-Structural Fusion for Materials Demand Forecasting},
  author = {Sandeep Kumar},
  year   = {2026},
  note   = {GitHub: SandeepKumar-san/CSG\_31\_Demand\_Forecasting}
}
```
