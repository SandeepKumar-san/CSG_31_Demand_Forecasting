# ATSF Project Report — Claude Code Master Instruction File

> **READ THIS ENTIRE FILE BEFORE WRITING A SINGLE WORD.**
> This file is the only source of truth for the report.
> The research paper (ATSF_Model_RP_refactored-1.pdf) and the project
> code directory are the only permitted sources of content.
> Nothing else. No internet. No training knowledge. No assumptions.

---

## SECTION 0 — ABSOLUTE RULES (READ FIRST, VIOLATING ANY ONE INVALIDATES THE REPORT)

1. **NO HALLUCINATION.** Do not invent, infer, estimate, or extrapolate any number, claim, result, or statement not explicitly in this file or the two permitted sources (paper + code).
2. **NO ROUNDING.** All metrics must appear exactly as in Section 4. 0.9947 stays 0.9947. 55.07% stays 55.07%.
3. **NO INVENTED INTENT.** Do not attribute goals, motivations, or conclusions beyond what is stated in Section 3.
4. **NO PADDING.** No generic filler sentences unless backed by a cited reference from Section 6.
5. **NO NEW TABLES OR FIGURES.** Only tables and figures listed in Section 5. No others.
6. **CITATIONS FROM SECTION 6 ONLY.** Do not add any reference not listed there.
7. **LIMITATIONS MUST APPEAR.** All 7 limitations in Section 3.5 must appear in the report.
8. **PLACEHOLDER OVER HALLUCINATION.** If any information needed for a section is not in this file or the two permitted sources — write `[INFO NEEDED]` and stop. Do not guess.

---

## SECTION 0B — PAUSE-AND-ASK PROTOCOL (CORE OPERATING RULE)

This is how Claude Code must work through every section and subsection:

```
STEP 1 — READ the section instructions below for that section/subsection.
STEP 2 — CHECK: does this file + the paper + the code directory contain
          everything needed to write this section completely?
STEP 3A — If YES: write the section. Then stop. Report completion.
           Ask: "Section X.Y done. Ready for X.Z — shall I proceed?"
STEP 3B — If NO: DO NOT WRITE ANYTHING YET.
           Instead, ask ONLY the specific questions needed for that
           section. One section at a time. Wait for answers.
           Only then write.
STEP 4 — Never proceed to the next section without explicit confirmation.
```

**What Claude Code may ask about:**
- Student names, IDs, guide name (front matter only)
- Specific code file contents it cannot read (ask user to paste the relevant portion)
- Clarification on a section instruction

**What Claude Code may NEVER ask about:**
- Any metric, result, or performance number → use Section 4 only
- Any model behaviour or claim → use Section 3 only
- Any reference → use Section 6 only
- Whether to round a number → never round

---

## SECTION 1 — DOCUMENT FORMATTING

### 1.1 Font and Size
- Font: **Times New Roman** throughout the entire document
- Chapter heading: **18pt, bold, center-justified**
- Section heading (e.g. 1.1): **16pt, bold, left-justified**
- Sub-section heading (e.g. 1.1.1): **14pt, bold, left-justified**
- Body text: **12pt, black, fully justified**

### 1.2 Page Numbering
- Front matter (Declaration through Abbreviations): Roman numerals (i, ii, iii...)
- Chapter pages: Arabic numerals starting from 1
- No page number on title page
- Page number in footer, centered

### 1.3 Structure Rules
- Each chapter begins on a new page
- Maximum 4 heading levels in table of contents
- Every figure must be cited in body text before it appears
- Every table must be cited in body text before it appears

### 1.4 Figure Format
- Caption: below the figure, centered
- Numbering: Figure X.Y (X = chapter, Y = order within chapter)
- Example: "As shown in Figure 3.1, the ATSF architecture consists of five layers."

### 1.5 Table Format
- Caption: above the table, centered
- Numbering: Table X.Y (X = chapter, Y = order within chapter)

### 1.6 References
- Harvard referencing style
- Ordered by appearance in text
- Every reference cited at least once in body

### 1.7 Target Length
- Total: 60-80 pages
- Chapters 1, 2, 3, 6, 7: approximately 10-12 pages each
- Chapters 4, 8: approximately 6-8 pages each
- Chapter 5: approximately 8-10 pages
- Chapter 9: approximately 4 pages

---

## SECTION 2 — IDENTITY PLACEHOLDERS

Claude Code: for all front matter sections, ask the user for these before writing.
Do not invent any of them. Ask once at the start of front matter, not repeatedly.

```
PROJECT TITLE:   Adaptive Temporal-Structural Fusion Model for
                 Mineral Commodity Demand Forecasting
STUDENT 1:       [ASK USER]
STUDENT 2:       [ASK USER]
STUDENT 3:       [ASK USER]
GUIDE:           [ASK USER]
DEPARTMENT:      School of Computer Science and Engineering (SoCSE)
UNIVERSITY:      Presidency University, Bengaluru
DEGREE:          Bachelor of Technology in Computer Science and Engineering
SUBMISSION DATE: December 2025
```

---

## SECTION 3 — VERIFIED CLAIMS (ONLY THESE MAY APPEAR IN THE REPORT)

### 3.1 The Core Problem (permitted claims)
- Mineral commodity demand is governed by two distinct signals: temporal patterns (historical trends, seasonality) and structural patterns (inter-commodity dependencies in supply-chain graphs).
- Transformer-based models capture temporal dynamics but ignore structural relationships.
- Graph neural networks model structure but rely on fixed temporal aggregation.
- Hybrid approaches combine both but impose a fixed fusion weight — assuming constant balance across datasets. This assumption fails across datasets with different structural characteristics.
- Daily FMCG data are dominated by temporal dynamics; annual mineral commodity data exhibit balanced or ambiguous modality contributions due to sparse temporal observations and stronger structural dependencies.

### 3.2 The ATSF Model (permitted claims)
- ATSF learns a per-instance scalar fusion weight alpha in (0,1), conditioned on material type and forecast horizon.
- alpha determines the optimal balance between the TFT branch (temporal) and GAT branch (structural) on a per-instance basis.
- The model has five layers: (1) TFT branch, (2) GAT branch, (3) Adaptive fusion layer, (4) Demand output head, (5) Rule-based risk scoring interface.
- The model contains approximately 198,000 trainable parameters.
- The scalar alpha is fully interpretable post-training: it quantifies each dataset's reliance on temporal vs. structural signal without additional probing.

### 3.3 The Four Contributions (use exactly these — do not add or remove)
1. A context-conditioned scalar fusion mechanism conditioned on material type and forecast horizon that replaces fixed-weight blending and entropy regularisation with one-sided boundary regularisation, enabling strong unimodal preference when empirically supported.
2. Empirical demonstration that alpha indicates learned modality preference: converging to 0.919+-0.010 on daily SupplyGraph (temporal dominance) and oscillating near 0.489+-0.197 on annual USGS mineral data (mixed regime) — a 43-percentage-point structural contribution gap that constitutes the paper's core finding.
3. Superior performance over all fixed-weight neural alternatives: 30.1% WAPE reduction over the best neural alternative on SupplyGraph, and a 95% confidence interval for USGS WAPE entirely below the strongest classical baseline.
4. A probabilistic uncertainty quantification layer with calibrated P10-P90 prediction intervals and an illustrative rule-based risk-scoring interface for procurement planning contexts.

### 3.4 What alpha Means (only these interpretations are permitted)
- alpha -> 1.0: full TFT dominance (temporal)
- alpha -> 0.0: full GAT dominance (structural)
- alpha = 0.919+-0.010 on SupplyGraph: consistent temporal dominance with negligible seed variance. SupplyGraph's four edge types encode taxonomic product classification, not causal demand dependencies.
- alpha = 0.489+-0.197 on USGS: near-balanced fusion with substantially higher variability, reflecting richer structural differentiation from 15 edge types.
- The 43.0 percentage-point structural contribution difference (8.1% vs 51.1%) demonstrates modality importance is dataset-dependent.
- This is a data-driven outcome of training — not a hyperparameter artifact — evidenced by consistent convergence across independent seeds.

### 3.5 All 7 Limitations (ALL must appear in the report — none may be omitted)
1. USGS contains only five annual observations per node, with 3-4 used for training — a low-data regime. External validation on longer time series is required.
2. Small sample size (n=5 seeds) limits statistical significance; minimum achievable Wilcoxon p-value is 0.0625.
3. The 222-day SupplyGraph window biases the test set toward late-period EEA behaviour.
4. Lead times in Layer 5 are synthetically assigned; the risk interface is a proof-of-concept and not validated against real procurement outcomes.
5. Experiments conducted on a single NVIDIA T4 GPU; scalability is not evaluated.
6. The supply-chain graph is static; dynamic edge modelling remains future work.
7. Generalisation beyond the two evaluated datasets remains to be validated.

### 3.6 XGBoost Framing (use exactly this — do not deviate)
- XGBoost's competitive performance on SupplyGraph is expected: a tabular model that ignores the graph entirely aligns with the dataset's lack of causal structural signal.
- This is consistent with prior findings that graph-based models rely on meaningful relational structure to provide benefit.
- The ATSF model provides this structural diagnosis automatically, alongside calibrated prediction intervals and interpretable fusion weights unavailable from tabular approaches.
- Do NOT frame XGBoost performance as a weakness of ATSF.

### 3.7 SupplyGraph Edge Types (permitted framing)
- SupplyGraph's four edge types (product group, sub-group, plant, storage location) encode taxonomic product classification, not causal demand dependencies.
- They indicate product co-membership in organisational categories rather than direct supply-chain substitution or co-production relationships.
- This structural shallowness is reflected in the model's converged alpha approx 0.919.

### 3.8 Fixed Fusion Failure (permitted framing)
- Fixed Fusion (alpha=0.5) achieves competitive mean WAPE on USGS (9.47%+-5.17%) but exhibits 5x higher variance than ATSF (+-0.83%).
- A fixed ratio may approximate the optimum on average but cannot adapt to context, leading to inconsistent performance across seeds.

### 3.9 Wilcoxon Framing (use exactly this — do not write "statistically significant")
- Across both datasets, ATSF outperforms every deep learning ablation variant on all five independently seeded runs.
- Wilcoxon signed-rank: stat=0, p=0.0625 — the minimum achievable for n=5.
- Write: "strong empirical consistency across seeds" — NOT "statistically significant."

### 3.10 Future Work (only these three — do not add others)
1. Multi-relational graph construction with dynamic edge weights.
2. Adaptive fusion on longer USGS time series as more annual data become available.
3. Prospective validation of the risk scoring layer against actual procurement outcomes.

---

## SECTION 4 — VERIFIED NUMBERS (CHARACTER-FOR-CHARACTER — DO NOT ALTER)

### 4.1 Primary Metrics

| Dataset | Metric | Exact Value |
|---|---|---|
| USGS | WAPE | 7.01%+-0.83% |
| USGS | RMSE | 6,059+-1,092 metric tonnes |
| USGS | R2 | 0.9947+-0.002 |
| USGS | SMAPE | 3.02%+-0.13% |
| USGS | 95% CI WAPE | [5.99%, 8.03%] |
| SupplyGraph | WAPE | 55.07%+-1.94% |
| SupplyGraph | RMSE | 17.03+-0.62 kg |
| SupplyGraph | R2 | 0.673+-0.024 |
| SupplyGraph | SMAPE | 124.39%+-0.47% |

### 4.2 Fusion Weight alpha by Seed

| Dataset | Seed 42 | Seed 123 | Seed 456 | Seed 789 | Seed 1337 | Mean+-Std |
|---|---|---|---|---|---|---|
| SupplyGraph | 0.932 | 0.918 | 0.919 | 0.905 | 0.922 | 0.919+-0.010 |
| USGS | 0.340 | 0.617 | 0.245 | 0.725 | 0.520 | 0.489+-0.197 |

### 4.3 Uncertainty Quantification

| Dataset | P10-P90 Coverage | Target | Mean Width | P50 WAPE |
|---|---|---|---|---|
| USGS | 79.1%+-4.0% | 80% | 2,185+-464 metric tonnes | 8.30%+-0.88% |
| SupplyGraph | 84.5%+-5.96% | 80% | 23.04+-2.18 kg | 42.54%+-2.97% |

### 4.4 USGS Full Baseline Table

| Model | RMSE (t) | WAPE (%) | SMAPE (%) | R2 |
|---|---|---|---|---|
| ARIMA | 18,789 | 17.02 | 5.21 | 0.951 |
| XGBoost | 6,807 | 8.07 | 3.24 | 0.994 |
| TFT-only | 11,190+-1,478 | 11.72+-1.79 | 5.41+-1.46 | 0.982+-0.005 |
| GAT-only | 16,471+-8,388 | 15.73+-6.55 | 5.09+-1.38 | 0.954+-0.035 |
| Fixed Fusion | 8,994+-6,728 | 9.47+-5.17 | 4.34+-0.66 | 0.984+-0.020 |
| ATSF (Ours) | 6,059+-1,092 | 7.01+-0.83 | 3.02+-0.13 | 0.9947+-0.002 |

### 4.5 SupplyGraph Full Baseline Table

| Model | RMSE (kg) | WAPE (%) | SMAPE (%) | R2 |
|---|---|---|---|---|
| ARIMA | 21.10 | 72.35 | 76.14 | 0.499 |
| XGBoost | 16.31 | 52.50 | 125.21 | 0.701 |
| TFT-only | 27.07+-2.38 | 90.63+-9.06 | 138.39+-4.91 | 0.169+-0.149 |
| GAT-only | 23.66+-1.60 | 78.77+-5.18 | 131.79+-3.83 | 0.367+-0.087 |
| Fixed Fusion | 25.57+-2.42 | 84.68+-7.52 | 134.82+-3.32 | 0.259+-0.141 |
| ATSF (Ours) | 17.03+-0.62 | 55.07+-1.94 | 124.39+-0.47 | 0.673+-0.024 |

### 4.6 Edge Ablation (USGS — full-graph RMSE baseline = 0.6317)

| Edge Type | Delta RMSE (%) | Type |
|---|---|---|
| technology_cluster | -2.20 | Causally important |
| alloy_components | -1.22 | Causally important |
| geopolitical_supply_risk | -0.55 | Causally important |
| substitution | -0.39 | Causally important |
| supply_chain_input | +1.29 | Noise at annual resolution |
| recycling_secondary_source | +0.50 | Noise at annual resolution |
| construction_coproduction | +0.39 | Noise at annual resolution |

### 4.7 Ablation Key Numbers

| Finding | Exact Value |
|---|---|
| Fixed Fusion WAPE variance (USGS) | +-5.17% |
| ATSF WAPE variance (USGS) | +-0.83% |
| Variance ratio | 5x |
| TFT-only WAPE (SupplyGraph) | 90.63%+-9.06% |
| GAT-only WAPE (SupplyGraph) | 78.77%+-5.18% |
| Per-category alpha std (USGS) | 0.215 |
| Per-category alpha std (SupplyGraph) | 0.021 |
| WAPE reduction vs best neural (SupplyGraph) | 30.1% |
| WAPE reduction vs Fixed Fusion (USGS) | 26.0% |
| Wilcoxon stat | 0 |
| Wilcoxon p | 0.0625 |

### 4.8 Model Configuration

| Parameter | SupplyGraph | USGS |
|---|---|---|
| Hidden dimension d | 128 | 64 |
| LSTM layers | 2 | 2 |
| Attention heads H | 4 | 4 |
| Temperature tau | 2.0 | 2.0 |
| Regularisation lambda | 0.1 | 0.05 |
| Dropout | 0.2 | 0.3 |
| Epochs | 70 | 70 |
| Learning rate | 1e-3 | 1e-3 |
| LR decay epochs | 27, 47, 63 | 27, 47, 63 |
| Optimizer | Adam | Adam |
| Total parameters | ~198,000 | ~198,000 |

Seeds used (both datasets): 42, 123, 456, 789, 1337
Hardware: Single NVIDIA T4 GPU, GCP Vertex AI Workbench
Framework: PyTorch + PyTorch Geometric

### 4.9 Dataset Statistics

| Property | SupplyGraph | USGS |
|---|---|---|
| Nodes | 40 | 127 |
| Edge types | 4 | 15 |
| Time coverage | 222 trading days (2022-2023) | 5 years (2019-2023) |
| Lookback window T | 30 days | 3 years |
| Forecast horizon h | {1, 3, 6, 12} days | 1 year |
| Train/val/test split | 70/10/20 chronological | 70/10/20 chronological |
| Training samples/node/seed | ~155 days | 3-4 years |

---

## SECTION 5 — PERMITTED FIGURES AND TABLES ONLY

### 5.1 Figures (exactly these 7 — no others)

| Figure ID | Caption | Source |
|---|---|---|
| Figure 1.1 | Sustainable Development Goals | UN SDG image (template) |
| Figure 3.1 | ATSF system architecture (5-layer pipeline) | Paper Figure 1 |
| Figure 3.2 | Agile/iterative development methodology | Draw fresh |
| Figure 5.1 | System block diagram (expanded architecture) | Paper Figure 1 + code structure |
| Figure 5.2 | System flow chart (data to forecast to risk) | Code pipeline |
| Figure 7.1 | Mean structural attention weights across 15 USGS edge types | Paper Figure 2 |
| Figure 7.2 | Fusion weight alpha evolution across five random seeds | Paper Figure 3 |

### 5.2 Tables (exactly these 11 — no others)

| Table ID | Caption | Source |
|---|---|---|
| Table 2.1 | Summary of literature reviews | Section 6 references |
| Table 3.1 | ATSF model configuration summary | Section 4.8 |
| Table 4.1 | Project timeline | Constructed from project phases |
| Table 4.2 | Risk analysis (PESTLE) | Section 3.5 limitations |
| Table 5.1 | Dataset description comparison | Section 4.9 |
| Table 6.1 | Software requirements | config.yaml + src/ directory |
| Table 7.1 | Converged alpha by dataset and seed | Section 4.2 |
| Table 7.2 | USGS baseline comparison | Section 4.4 |
| Table 7.3 | SupplyGraph baseline comparison | Section 4.5 |
| Table 7.4 | Uncertainty quantification results | Section 4.3 |
| Table 7.5 | Edge ablation results (USGS) | Section 4.6 |

---

## SECTION 6 — ALL 26 REFERENCES (HARVARD STYLE — USE ONLY THESE)

[1]  Lim, B., Arik, S.O., Loeff, N. and Pfister, T. (2021) 'Temporal fusion transformers for interpretable multi-horizon time series forecasting', International Journal of Forecasting, 37(4), pp. 1748-1764.

[2]  Velickovic, P., Cucurull, G., Casanova, A., Romero, A., Lio, P. and Bengio, Y. (2018) 'Graph attention networks', Proceedings of the 6th International Conference on Learning Representations (ICLR). Available at: https://openreview.net/forum?id=rJXMpikCZ

[3]  Wu, Z., Pan, S., Chen, F., Long, G., Zhang, C. and Philip, S.Y. (2021) 'A comprehensive survey on graph neural networks', IEEE Transactions on Neural Networks and Learning Systems, 32(1), pp. 4-24.

[4]  Ding, C., Sun, S. and Zhao, J. (2023) 'MST-GAT: A multimodal spatial-temporal graph attention network for time series anomaly detection', Information Fusion, 89, pp. 527-536.

[5]  Hochreiter, S. and Schmidhuber, J. (1997) 'Long short-term memory', Neural Computation, 9(8), pp. 1735-1780.

[6]  Makridakis, S., Spiliotis, E. and Assimakopoulos, V. (2020) 'The M4 competition: 100,000 time series and 61 forecasting methods', International Journal of Forecasting, 36(1), pp. 54-74.

[7]  Hosseinnia Shavaki, F. and Ebrahimi Ghahnavieh, A. (2023) 'Applications of deep learning into supply chain management: A systematic literature review and a framework for future research', Artificial Intelligence Review, 56, pp. 4447-4489.

[8]  Kipf, T.N. and Welling, M. (2017) 'Semi-supervised classification with graph convolutional networks', Proceedings of the 5th International Conference on Learning Representations (ICLR). Available at: https://openreview.net/forum?id=SJU4ayYgl

[9]  Hamilton, W.L., Ying, Z. and Leskovec, J. (2017) 'Inductive representation learning on large graphs', Advances in Neural Information Processing Systems (NeurIPS), vol. 30.

[10] Jin, M., Koh, H.Y., Wen, Q., Zambon, D., Alippi, C., Webb, G.I., King, I. and Pan, S. (2024) 'A survey on graph neural networks for time series: Forecasting, classification, imputation, and anomaly detection', IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(12), pp. 10466-10485.

[11] Jiang, W. and Luo, J. (2022) 'Graph neural network for traffic forecasting: A survey', Expert Systems with Applications, 207, p. 117921.

[12] Kosasih, E.E. and Brintrup, A. (2022) 'A machine learning approach for predicting hidden links in supply chain with graph neural networks', International Journal of Production Research, 60(17), pp. 5380-5393.

[13] Wu, Z., Pan, S., Long, G., Jiang, J. and Zhang, C. (2019) 'Graph WaveNet for deep spatial-temporal graph modeling', Proceedings of the 28th International Joint Conference on Artificial Intelligence (IJCAI), pp. 1534-1540.

[14] Wen, R., Torkkola, K., Narayanaswamy, B. and Madeka, D. (2017) 'A multi-horizon quantile recurrent forecaster', arXiv preprint arXiv:1711.11053.

[15] Gneiting, T. and Raftery, A.E. (2007) 'Strictly proper scoring rules, prediction, and estimation', Journal of the American Statistical Association, 102(477), pp. 359-378.

[16] Lundberg, S.M. and Lee, S.-I. (2017) 'A unified approach to interpreting model predictions', Advances in Neural Information Processing Systems (NeurIPS), vol. 30.

[17] International Energy Agency (2024) Global Critical Minerals Outlook 2024. Paris: IEA. Available at: https://www.iea.org/reports/global-critical-minerals-outlook-2024

[18] Wasi, A.T., Islam, M.S., Akib, A.R. and Bappy, M.M. (2024) 'Graph neural networks in supply chain analytics and optimization: Concepts, perspectives, dataset and benchmarks', arXiv preprint arXiv:2411.08550.

[19] Wasi, A.T., Islam, M.S. and Akib, A.R. (2024) 'SupplyGraph: A benchmark dataset for supply chain planning using graph neural networks', Proceedings of the 4th Workshop on Graphs and More Complex Structures for Learning and Reasoning, 38th AAAI Conference on Artificial Intelligence.

[20] U.S. Geological Survey (2024) Mineral Commodity Summaries 2024. Reston, VA: U.S. Geological Survey.

[21] Kingma, D.P. and Ba, J. (2015) 'Adam: A method for stochastic optimization', Proceedings of the 3rd International Conference on Learning Representations (ICLR).

[22] Paszke, A., Gross, S., Massa, F. et al. (2019) 'PyTorch: An imperative style, high-performance deep learning library', Advances in Neural Information Processing Systems (NeurIPS), vol. 32.

[23] Fey, M. and Lenssen, J.E. (2019) 'Fast graph representation learning with PyTorch Geometric', Proceedings of the ICLR Workshop on Representation Learning on Graphs and Manifolds. Available at: https://arxiv.org/abs/1903.02428

[24] Hyndman, R.J. and Athanasopoulos, G. (2018) Forecasting: Principles and Practice. 2nd edn. Melbourne: OTexts. Available at: https://otexts.com/fpp2/

[25] Chen, T. and Guestrin, C. (2016) 'XGBoost: A scalable tree boosting system', Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 785-794.

[26] Wilcoxon, F. (1945) 'Individual comparisons by ranking methods', Biometrics Bulletin, 1(6), pp. 80-83.

---

## SECTION 7 — CHAPTER-BY-CHAPTER INSTRUCTIONS WITH PAUSE POINTS

---

### FRONT MATTER

**PAUSE BEFORE STARTING. Ask the user:**
```
Before I begin the front matter, I need:
1. Student 1: Full name and student ID
2. Student 2: Full name and student ID
3. Student 3: Full name and student ID
4. Guide: Full name and designation
I will not write the title page or declaration until I have these.
```

**Title Page** — University name, logo placeholder, project title, "A PROJECT REPORT", "Submitted by", student names and IDs, "Under the guidance of", guide name, degree, department, university, city, month and year.

**Bonafide Certificate** — University header, school name, "BONAFIDE CERTIFICATE", examiner signature table (2 rows, blank).

**Declaration** — Standard declaration. Student names. Place: Bengaluru. Date: December 2025.

**Acknowledgement** — Thank guide, department, Presidency University. End with student names. Do not add unverifiable external acknowledgements.

**Abstract** — Adapt from the paper abstract. Must include:
- Problem: fixed fusion weight assumption fails across datasets
- Approach: ATSF with learnable scalar alpha conditioned on material type and forecast horizon
- Results (use exact values from Section 4.1 and 4.2):
  - SupplyGraph WAPE: 55.07%+-1.94% (30.1% reduction over best neural alternative)
  - USGS WAPE: 7.01%+-0.83% (95% CI [5.99%, 8.03%] entirely below XGBoost 8.07%)
  - alpha=0.919+-0.010 (SupplyGraph) vs alpha=0.489+-0.197 (USGS): 43-point structural contribution gap
- Maximum one page.

**List of Figures** — All 7 figures from Section 5.1. Page numbers filled after full document completion.

**List of Tables** — All 11 tables from Section 5.2. Page numbers filled after full document completion.

**Abbreviations** — Use only the table below. Add only abbreviations that appear in the report body.

| Abbreviation | Full Form |
|---|---|
| ATSF | Adaptive Temporal-Structural Fusion |
| TFT | Temporal Fusion Transformer |
| GAT | Graph Attention Network |
| GNN | Graph Neural Network |
| LSTM | Long Short-Term Memory |
| VSN | Variable Selection Network |
| WAPE | Weighted Absolute Percentage Error |
| RMSE | Root Mean Squared Error |
| SMAPE | Symmetric Mean Absolute Percentage Error |
| R2 | Coefficient of Determination |
| UQ | Uncertainty Quantification |
| MSE | Mean Squared Error |
| MLP | Multi-Layer Perceptron |
| SDG | Sustainable Development Goal |
| USGS | United States Geological Survey |
| FMCG | Fast-Moving Consumer Goods |
| EEA | European Economic Area |
| CI | Confidence Interval |
| GCP | Google Cloud Platform |
| GPU | Graphics Processing Unit |
| AI | Artificial Intelligence |
| ML | Machine Learning |
| DL | Deep Learning |
| IEA | International Energy Agency |
| ARIMA | Autoregressive Integrated Moving Average |
| SHAP | SHapley Additive exPlanations |
| P10/P50/P90 | 10th / 50th / 90th percentile prediction intervals |

**After front matter complete — ask:**
```
Front matter complete. Ready to begin Chapter 1 (Introduction). Shall I proceed?
```

---

### CHAPTER 1 — INTRODUCTION

**1.1 Background**
- Source: Section 3.1 of this file + paper Introduction section
- Cite: [1][2][3][4] as relevant
- Do NOT invent market size figures, commodity price data, or supply chain revenue statistics

**PAUSE after 1.1. Ask:**
```
Section 1.1 (Background) done. Shall I proceed to 1.2?
```

**1.2 Statistics of the Project**
- Source: Section 4.9 of this file only
- SupplyGraph: 40 nodes, 222 trading days (2022-2023), 4 typed edge relations — cite [19]
- USGS: 127 nodes, 5 years (2019-2023), 15 typed edge relations — cite [20]
- Do NOT cite or mention any other dataset
- Do NOT invent global market share percentages or industry statistics

**PAUSE after 1.2. Ask:**
```
Section 1.2 (Statistics) done. Shall I proceed to 1.3?
```

**1.3 Prior Existing Technologies**
- Three streams only:
  - Temporal: TFT [1], LSTM [5] — limitation: ignore structural relationships
  - Graph: GAT [2], GCN [8], GraphSAGE [9] — limitation: fixed temporal aggregation
  - Hybrid: Graph WaveNet [13], MST-GAT [4] — limitation: fixed fusion weights
- Source: Related Work section of the research paper + Section 3.1 of this file
- Do NOT claim any model is "state-of-the-art" without a direct citation supporting that claim

**PAUSE after 1.3. Ask:**
```
Section 1.3 (Prior Technologies) done. Shall I proceed to 1.4?
```

**1.4 Proposed Approach**
- Use Section 3.2 (ATSF description) and Section 3.3 (contributions summary)
- Motivation: from Section 3.1
- Applications: mineral commodity demand forecasting; supply chain procurement planning
- Limitations: briefly reference Section 3.5 (full list appears in Chapter 9)

**PAUSE after 1.4. Ask:**
```
Section 1.4 (Proposed Approach) done. Shall I proceed to 1.5?
```

**1.5 Objectives (write exactly these 5 — do not modify wording)**
1. To design and implement a per-instance scalar fusion mechanism (alpha) conditioned on material type and forecast horizon for adaptive temporal-structural integration.
2. To empirically demonstrate that learned modality preference (alpha) varies significantly across datasets with different structural characteristics.
3. To evaluate ATSF performance against six baseline models (ARIMA, XGBoost, TFT-only, GAT-only, Fixed Fusion) across two structurally distinct datasets using WAPE, RMSE, R2, and SMAPE.
4. To implement a probabilistic uncertainty quantification layer producing calibrated P10-P90 prediction intervals.
5. To analyse structural attention through edge ablation, quantifying the causal contribution of each USGS edge type to forecasting performance.

**PAUSE after 1.5. Ask:**
```
Section 1.5 (Objectives) done. Shall I proceed to 1.6 (SDGs)?
```

**1.6 SDGs**
- SDG 9 (Industry, Innovation and Infrastructure): supports efficient industrial supply chain planning
- SDG 12 (Responsible Consumption and Production): improved forecasting reduces over-procurement waste in mineral supply chains
- SDG 13 (Climate Action): critical mineral demand forecasting supports energy transition planning — cite [17]
- Include Figure 1.1 (SDG image)
- Do NOT align to any SDG not listed above

**PAUSE after 1.6. Ask:**
```
Section 1.6 (SDGs) done. Shall I proceed to 1.7?
```

**1.7 Overview of Project Report**
- Single paragraph only, summarising all 9 chapters in order
- Do not introduce new content here

**PAUSE after 1.7. Ask:**
```
Chapter 1 complete. Ready to begin Chapter 2 (Literature Review). Shall I proceed?
```

---

### CHAPTER 2 — LITERATURE REVIEW

**Operating rule for this chapter:**
Write one paragraph (approximately 150-200 words) per reference.
Summarise: the method used, results reported, limitations, future work noted.
Do NOT copy sentences from the research paper verbatim.
Do NOT generate generic AI summaries from training knowledge.

If Claude Code cannot access the full text of a reference, it must PAUSE and ask:
```
I need the content of reference [N] ([Author Year]) to write the literature
paragraph. Can you paste the abstract or key findings from that paper?
```

**References to cover in this order:**
[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16], [17], [19], [20]

**After each paragraph — PAUSE and ask:**
```
Literature paragraph for [N] ([Author]) done. Shall I proceed to [N+1]?
```

**End Chapter 2 with Table 2.1** — summary of all reviewed literature.
Columns: Ref No. | Title and Year | Methods | Key Features | Merits | Demerits
Use only information from the paragraphs written above. Do not invent content.

**PAUSE after Table 2.1. Ask:**
```
Chapter 2 complete. Ready to begin Chapter 3 (Methodology). Shall I proceed?
```

---

### CHAPTER 3 — METHODOLOGY

**3.1 Development Methodology**
- Describe an Agile/iterative methodology mapped to this project's phases
- Phases: requirements -> data preparation -> model design -> implementation -> training -> evaluation -> report
- Include Figure 3.2 (iterative pipeline diagram)
- PAUSE: ask user if they have a preferred methodology style, otherwise proceed with Agile iterative

**PAUSE after 3.1. Ask:**
```
Section 3.1 (Development Methodology) done. Shall I proceed to 3.2?
```

**3.2 System Architecture Overview**
- 5-layer ATSF pipeline using Section 3.2
- Reference Figure 3.1 (paper Figure 1)
- Source: paper Section IV + Figure 1

**PAUSE after 3.2. Ask:**
```
Section 3.2 (Architecture Overview) done. Shall I proceed to 3.3?
```

**3.3 TFT Branch (Layer 1)**
- VSN: soft learned feature gating
- Two-layer LSTM (d=128 SupplyGraph, d=64 USGS)
- Multi-head self-attention H=4
- Three quantile heads (P10, P50, P90) via pinball loss
- Cite [1]
- Source: paper Section IV.A

**PAUSE after 3.3. Ask:**
```
Section 3.3 (TFT Branch) done. Shall I proceed to 3.4?
```

**3.4 GAT Branch (Layer 2)**
- Two-layer multi-relational GAT
- Node features: recent demand average, price-level proxy, production utilisation, normalised lead time
- USGS: 15 typed edges. SupplyGraph: 4 typed edges.
- Typed attention gates prevent spurious cross-relation aggregation
- Cite [2][18]
- Source: paper Section IV.B

**PAUSE after 3.4. Ask:**
```
Section 3.4 (GAT Branch) done. Shall I proceed to 3.5?
```

**3.5 Adaptive Fusion Layer (Layer 3)**
- Scalar alpha learned end-to-end
- Residual LayerNorm on temporal embedding; concatenated with structural embedding + 16-dim context embeddings
- Three-layer MLP -> temperature-scaled sigmoid tau=2.0
- Write all three equations:
  - Eq (1): alpha = sigmoid(MLP([h't, hs, zm, zh]) / tau)
  - Eq (2): h-hat = alpha * ht + (1 - alpha) * hs
  - Eq (3): L_reg = lambda * (E[ReLU(alpha - 0.95)] + E[ReLU(0.05 - alpha)])
- lambda=0.1 SupplyGraph, lambda=0.05 USGS (from Section 4.8)
- Explain: replaces entropy regularisation; permits strong unimodal preference
- Source: paper Section IV.C

**PAUSE after 3.5. Ask:**
```
Section 3.5 (Adaptive Fusion Layer) done. Shall I proceed to 3.6?
```

**3.6 Output and Risk Layers (Layers 4-5)**
- Layer 4: linear projection -> h-step point forecasts
- Layer 5: configurable rule-based criteria (budget stress, lead-time risk, dependency criticality) + lightweight calibrator MLP
- Disclose: thresholds are user-defined, not learned; risk module is proof-of-concept only
- Source: paper Section IV.D

**PAUSE after 3.6. Ask:**
```
Section 3.6 (Output and Risk Layers) done. Shall I proceed to 3.7?
```

**3.7 Training Objective**
- Combined loss: L = L_pinball + L_MSE + L_reg
- Adam [21], lr=1e-3, step-decay at epochs 27, 47, 63
- LayerNorm (not batch normalisation)
- 70 epochs, ~198,000 parameters
- Include Table 3.1 (model configuration — use Section 4.8 exactly)
- Source: paper Section IV training objective paragraph

**PAUSE after 3.7. Ask:**
```
Chapter 3 complete. Ready to begin Chapter 4 (Project Management). Shall I proceed?
```

---

### CHAPTER 4 — PROJECT MANAGEMENT

**4.1 Project Timeline — Table 4.1**

**PAUSE before 4.1. Ask:**
```
For the project timeline (Table 4.1), do you have actual start/end months
for each phase? If yes, provide them. If no, I will use generic
Month 1-2 style ranges — shall I proceed with generic ranges?
```

Phases to include:
1. Literature review and problem definition
2. Dataset acquisition and preprocessing (SupplyGraph + USGS)
3. Model design and architecture implementation
4. Training and multi-seed experimentation (5 seeds x 2 datasets)
5. Baseline comparison and ablation studies
6. Results analysis and visualisation
7. IEEE paper writing and submission
8. Project report writing

**PAUSE after 4.1. Ask:**
```
Section 4.1 (Timeline) done. Shall I proceed to 4.2?
```

**4.2 Risk Analysis — Table 4.2 (PESTLE)**
Use ONLY these verified risks from Section 3.5:

| Category | Risk | Mitigation |
|---|---|---|
| Technical | USGS low-data regime (3-4 training samples per node) | 5-seed replication + Wilcoxon test |
| Technical | Static graph — no dynamic edge modelling | Disclosed as limitation; dynamic edges as future work |
| Technical | Single GPU — scalability not evaluated | Disclosed; GCP Vertex AI used |
| Data | SupplyGraph 222-day test set biased toward EEA group | Transparently reported in results |
| Operational | Lead-time proxies in Layer 5 — not real logistics data | Disclosed as proof-of-concept only |
| Statistical | n=5 seeds limits Wilcoxon significance | p=0.0625 reported as minimum achievable |

Do NOT invent additional risks.

**PAUSE after 4.2. Ask:**
```
Section 4.2 (Risk Analysis) done. Shall I proceed to 4.3?
```

**4.3 Project Budget**
- GCP Vertex AI Workbench: NVIDIA T4 GPU (cloud compute cost)
- All software: open-source (PyTorch [22], PyTorch Geometric [23], Python 3.10)
- Datasets: publicly available at no cost (SupplyGraph [19], USGS MCS 2024 [20])
- No private procurement data used

**PAUSE after 4.3. Ask:**
```
Chapter 4 complete. Ready to begin Chapter 5 (Analysis and Design). Shall I proceed?
```

---

### CHAPTER 5 — ANALYSIS AND DESIGN

**5.1 System Requirements — Table 6.1**

**PAUSE before 5.1. Ask:**
```
For Section 5.1 (Software Requirements), I need to read
experiments/config.yaml from your project directory.
Please paste its contents here.
I will not write software versions or hyperparameters without seeing the actual file.
```

Once pasted, extract Python version, PyTorch version, batch size, hidden dims, and all training hyperparameters. These become Table 6.1.

**PAUSE after 5.1. Ask:**
```
Section 5.1 (System Requirements) done. Shall I proceed to 5.2?
```

**5.2 Block Diagram — Figure 5.1**
Expanded ATSF block diagram showing:
- Data inputs (SupplyGraph CSV / USGS CSV) -> preprocessing -> graph construction
- TFT branch and GAT branch in parallel
- Adaptive fusion layer -> output head (point + quantiles) -> risk scoring
- Source: paper Figure 1 + src/models/ file names from Project_Directory_Structure.md

**PAUSE after 5.2. Ask:**
```
Section 5.2 (Block Diagram) done. Shall I proceed to 5.3?
```

**5.3 System Flow Chart — Figure 5.2**
End-to-end pipeline flow:
Raw data (data/raw/) ->
Preprocessing (supplygraph_loader.py / usgs_loader.py / usgs_preprocessing.py) ->
Graph construction (graph_builder.py) ->
Training (trainer.py via train.py) ->
Multi-seed runs (run_multi_seed.py) ->
Baseline comparison (run_classical_baselines.py / run_graph_baselines.py) ->
Evaluation (evaluate.py) ->
Risk scoring (demo_risk_scoring.py) ->
Results (results/ directory)

Source: Project_Directory_Structure.md (already uploaded)

**PAUSE after 5.3. Ask:**
```
Section 5.3 (Flow Chart) done. Shall I proceed to 5.4?
```

**5.4 Dataset Description — Table 5.1**
Use Section 4.9 exactly. Columns: Property | SupplyGraph | USGS
Cite [19] for SupplyGraph, [20] for USGS.

**PAUSE after 5.4. Ask:**
```
Section 5.4 (Dataset Description) done. Shall I proceed to 5.5?
```

**5.5 Graph Construction Design**
- Source: src/data/graph_builder.py from Project_Directory_Structure.md
- USGS: 127 nodes, 15 edge types (use edge type names from Section 4.6 + paper Section V.A)
- SupplyGraph: 40 nodes, 4 edge types: product group, sub-group, plant, storage location
- Use framing from Section 3.7 for SupplyGraph edge type clarification

**PAUSE after 5.5. Ask:**
```
Chapter 5 complete. Ready to begin Chapter 6 (Implementation). Shall I proceed?
```

---

### CHAPTER 6 — IMPLEMENTATION

**Operating rule for this entire chapter:**
Every sub-section maps to actual files in the project directory.
If Claude Code needs the content of a specific file, it must PAUSE and ask
the user to paste it. Do NOT describe file functionality beyond what is in
Project_Directory_Structure.md or the file contents pasted by the user.

**6.1 Development Environment**
- Source: README.md + config.yaml (ask user to paste if not already provided in 5.1)
- Hardware: Single NVIDIA T4 GPU, GCP Vertex AI Workbench
- Framework: PyTorch [22], PyTorch Geometric [23]

**PAUSE after 6.1. Ask:**
```
Section 6.1 (Development Environment) done. Shall I proceed to 6.2?
```

**6.2 Data Preprocessing Implementation**
- supplygraph_loader.py: normalises and loads supply chain CSVs; handles zero-inflation; 70/10/20 chronological split
- usgs_preprocessing.py: cleans raw USGS text (MCS PDF) into structured tabular data
- usgs_loader.py: loads cleaned USGS data into tensors; applies normalisation
- verify_data.py and verify_usgs_integrity.py: QA checks for missing values and broken tensors
- Source: Project_Directory_Structure.md descriptions

**PAUSE after 6.2. Ask:**
```
Section 6.2 (Data Preprocessing) done. Shall I proceed to 6.3?
```

**6.3 Model Implementation**
- complete_model.py: orchestrates all 5 layers
- tft_branch.py: TFT branch (VSN, LSTM, MH self-attention, quantile heads)
- gat_branch.py: GAT branch (multi-relational, typed attention gates)
- fusion_layer.py: adaptive fusion (MLP -> sigmoid -> alpha -> h-hat)
- fused_representation.py: maps h-hat to point forecast
- risk_decision_layer.py: rule-based risk scoring
- Source: Project_Directory_Structure.md + paper Sections IV.A-IV.D

**PAUSE before writing. Ask:**
```
For Section 6.3 (Model Implementation), would you like me to include code
snippets from any of these files? If yes, please paste the relevant
portions — I will only include code you provide directly.
```

**PAUSE after 6.3. Ask:**
```
Section 6.3 (Model Implementation) done. Shall I proceed to 6.4?
```

**6.4 Training Implementation**
- trainer.py: main training loop; combined loss L = L_pinball + L_MSE + L_reg
- loss.py: pinball, MSE, L_reg implementations
- metrics.py: WAPE, RMSE, SMAPE, R2, P10-P90 coverage
- run_multi_seed.py: 5 seeds (42, 123, 456, 789, 1337)
- seed.py: locks system randomness for reproducibility
- train.py: entry point for single training run
- config.yaml: all hyperparameters (use Section 4.8 values)

**PAUSE after 6.4. Ask:**
```
Section 6.4 (Training Implementation) done. Shall I proceed to 6.5?
```

**6.5 Baseline Implementation**
- run_classical_baselines.py: ARIMA [24] and XGBoost [25] using same chronological split
- run_graph_baselines.py: graph-only baseline models
- evaluate.py: final evaluation on held-out test set

**PAUSE after 6.5. Ask:**
```
Section 6.5 (Baseline Implementation) done. Shall I proceed to 6.6?
```

**6.6 Risk Scoring Implementation**
- demo_risk_scoring.py: generates risk assessment JSON output
- risk_decision_layer.py: budget stress, lead-time risk, dependency criticality thresholds
- Proxy lead times: SupplyGraph — group-level constants sampled from [3, 30] days; USGS — fixed 1.0
- Disclose explicitly: risk outputs are representative only, not operationally validated

**PAUSE after 6.6. Ask:**
```
Section 6.6 (Risk Scoring) done. Shall I proceed to 6.7?
```

**6.7 Reproducibility and Verification**
- verify_reproducibility.py: proves identical outputs across two runs to 6 decimal places
- system_dry_run.py: 10-second fast test before overnight training
- reproducibility_report.json: automated verification output in results/
- stats_analysis.py: Wilcoxon tests and confidence interval computation
- gen_alpha_fig.py: generates the 2-panel alpha trajectory plot

**PAUSE after 6.7. Ask:**
```
Chapter 6 complete. Ready to begin Chapter 7 (Results and Discussion). Shall I proceed?
```

---

### CHAPTER 7 — RESULTS AND DISCUSSION

**CRITICAL RULE FOR THIS ENTIRE CHAPTER:**
Every number must trace to Section 4 of this file.
No exceptions. If a number is needed and not in Section 4 — write [INFO NEEDED] and PAUSE.

**7.1 Experiment Setup**
- 5 seeds: 42, 123, 456, 789, 1337
- 70 epochs per run
- Hardware: single NVIDIA T4 GPU, GCP Vertex AI
- DL models: mean +- std across 5 seeds
- Classical models: deterministic single run

**PAUSE after 7.1. Ask:**
```
Section 7.1 (Experiment Setup) done. Shall I proceed to 7.2?
```

**7.2 Adaptive Fusion Behaviour (alpha Analysis)**
- Present Table 7.1 (copy exactly from Section 4.2)
- SupplyGraph: alpha=0.919+-0.010
- USGS: alpha=0.489+-0.197
- 43.0 percentage-point structural contribution difference (8.1% vs 51.1%)
- Cite Figure 7.2
- Use framing from Section 3.4 and Section 3.8 only

**PAUSE after 7.2. Ask:**
```
Section 7.2 (alpha Analysis) done. Shall I proceed to 7.3?
```

**7.3 Forecasting Performance**
- Present Table 7.2 (copy exactly from Section 4.4)
- Present Table 7.3 (copy exactly from Section 4.5)
- USGS key claims (exact values from Section 4.1 and 4.4):
  - WAPE 7.01%+-0.83%
  - 95% CI [5.99%, 8.03%] entirely below XGBoost baseline (8.07%)
  - 26.0% WAPE reduction over Fixed Fusion (9.47%)
  - R2=0.9947+-0.002
- SupplyGraph key claims:
  - WAPE 55.07%+-1.94%
  - 30.1% WAPE reduction over best neural alternative (GAT-only 78.77%)
  - XGBoost framing: use Section 3.6 exactly
  - Higher WAPE reflects zero-inflated daily data difficulty, not model underperformance

**PAUSE after 7.3. Ask:**
```
Section 7.3 (Forecasting Performance) done. Shall I proceed to 7.4?
```

**7.4 Uncertainty Quantification**
- Present Table 7.4 (copy exactly from Section 4.3)
- Both datasets: near-nominal calibration (target 80%)
- SupplyGraph slightly overcovered (84.5%) due to conservative widths on zero-inflated data
- Cite [14][15]

**PAUSE after 7.4. Ask:**
```
Section 7.4 (UQ) done. Shall I proceed to 7.5?
```

**7.5 Ablation Study**
- Use Section 4.7 numbers only
- Fixed Fusion variance 5x higher (+-5.17% vs +-0.83%) on USGS
- TFT-only (90.63%+-9.06%) worse than GAT-only (78.77%) on SupplyGraph
- Per-category alpha std: 0.215 USGS vs 0.021 SupplyGraph

**PAUSE after 7.5. Ask:**
```
Section 7.5 (Ablation) done. Shall I proceed to 7.6?
```

**7.6 Structural Attention Interpretability**
- Present Table 7.5 (copy exactly from Section 4.6)
- Full-graph RMSE baseline: 0.6317
- Cite Figure 7.1
- technology_cluster: -2.20% (largest single-edge effect)
- supply_chain_input: +1.29% (noise at annual resolution)
- Attention ordering consistent with ablation results

**PAUSE after 7.6. Ask:**
```
Section 7.6 (Structural Attention) done. Shall I proceed to 7.7?
```

**7.7 Statistical Significance**
- Wilcoxon signed-rank: stat=0, p=0.0625
- Minimum achievable for n=5
- Write "strong empirical consistency across seeds" — NOT "statistically significant"
- Use framing from Section 3.9 exactly

**PAUSE after 7.7. Ask:**
```
Chapter 7 complete. Ready to begin Chapter 8 (SLESS). Shall I proceed?
```

---

### CHAPTER 8 — SOCIAL, LEGAL, ETHICAL, SUSTAINABILITY AND SAFETY

**8.1 Privacy Considerations**
- No private procurement data used
- Both datasets publicly available: SupplyGraph [19], USGS MCS 2024 [20]
- Risk scoring output must not be deployed without domain expert validation

**PAUSE after 8.1. Ask:**
```
Section 8.1 done. Shall I proceed to 8.2?
```

**8.2 Algorithmic Bias and Fairness**
- SupplyGraph test set biased toward late-period EEA group (222-day window — from Section 3.5)
- USGS low-data regime (3-4 training samples per node) — results should be interpreted cautiously
- Generalisation beyond two evaluated datasets not validated

**PAUSE after 8.2. Ask:**
```
Section 8.2 done. Shall I proceed to 8.3?
```

**8.3 Potential for Misuse**
- Risk layer uses synthetic proxy lead times — not validated against real procurement outcomes
- Risk outputs are representative only — must not be used for real-world decisions without expert validation
- The model must not be interpreted as a geopolitical prediction system — geopolitical_supply_risk is an edge label, not a prediction target

**PAUSE after 8.3. Ask:**
```
Section 8.3 done. Shall I proceed to 8.4?
```

**8.4 Regulatory Compliance**
- Consistent with IEA critical minerals reporting guidelines [17]
- USGS data from official U.S. Geological Survey publications [20]
- No proprietary or classified data used

**PAUSE after 8.4. Ask:**
```
Section 8.4 done. Shall I proceed to 8.5?
```

**8.5 Sustainability and SDG Impact**
- SDG 9: efficient supply chain intelligence reduces waste and optimises resource allocation
- SDG 12: better forecasting reduces over-procurement in mineral supply chains
- SDG 13: supports critical mineral planning for energy transition — cite [17]

**PAUSE after 8.5. Ask:**
```
Chapter 8 complete. Ready to begin Chapter 9 (Conclusion). Shall I proceed?
```

---

### CHAPTER 9 — CONCLUSION

**9.1 Contributions to the Field**
- Summarise the four contributions from Section 3.3
- Use exact numbers: 30.1% WAPE reduction, 43-point alpha gap, 95% CI [5.99%, 8.03%]
- Do not add a fifth contribution

**PAUSE after 9.1. Ask:**
```
Section 9.1 done. Shall I proceed to 9.2?
```

**9.2 Limitations and Constraints**
- List all 7 limitations from Section 3.5 — every one, in full — none may be omitted

**PAUSE after 9.2. Ask:**
```
Section 9.2 done. Shall I proceed to 9.3?
```

**9.3 Future Work Directions**
- Only these three (from Section 3.10):
  1. Dynamic edge weight modelling
  2. Extended USGS time series as more annual data become available
  3. Prospective validation of the risk scoring layer against real procurement outcomes
- Do not add others

**PAUSE after 9.3. Ask:**
```
Section 9.3 done. Shall I proceed to 9.4?
```

**9.4 Closing Remarks**
- Core message: modality balance is data-dependent and should be learned rather than assumed
- The scalar alpha serves not merely as a tuning parameter but as a dataset-level structural diagnostic
- Source: paper Section VIII (Conclusion) and Section VII (Discussion, first paragraph)

**PAUSE after 9.4. Ask:**
```
Chapter 9 complete. Ready to compile Appendix. Shall I proceed?
```

---

### APPENDIX

**PAUSE before Appendix. Ask:**
```
For the Appendix I need the actual contents of:
1. experiments/config.yaml — please paste it here
2. A sample output from results/reproducibility_report.json — please paste it here
3. A sample output from demo_risk_scoring.py — please paste it here
I will only include what you paste. I will not invent config values or output formats.
```

Once provided, include:
- Full config.yaml contents
- Sample reproducibility_report.json output
- Sample risk scoring JSON output
- Project directory structure tree (from Project_Directory_Structure.md already uploaded)

---

## SECTION 8 — FALLBACK TABLE (WHEN UNCERTAIN)

| Situation | Action |
|---|---|
| Number needed but not in Section 4 | Write [INFO NEEDED: metric] and PAUSE |
| Claim needed but not in Section 3 | Write [INFO NEEDED: claim] and PAUSE |
| Figure not in Section 5.1 | Do NOT add it. Flag it to user. |
| Reference not in Section 6 | Do NOT add it. Flag it to user. |
| Student or guide info needed | PAUSE and ask — never invent |
| File content needed | PAUSE and ask user to paste — never infer |
| Uncertain about formatting | Re-read Section 1 — do not improvise |

**The only permitted sources are:**
1. This instruction file
2. The research paper: ATSF_Model_RP_refactored-1.pdf
3. The project code files (must be pasted by user when needed)

**Nothing else is permitted.**

---

*Master instruction file for ATSF Project Report*
*Presidency University, B.Tech CSE, SoCSE, December 2025*
*All verified numbers sourced from the IEEE conference paper and*
*project training logs, CSV results files, and code directory.*
